"""Geometry-aware smooth gating: use phase-space velocity to detect bursts.

When visible trajectory moves fast (high |v|) -> relax smooth constraint on h
When visible trajectory is calm (low |v|) -> enforce smooth constraint

Also: reward h for being large at high-speed timesteps.
Combined with alternating training (5:1).
"""
from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from models.cvhi_residual import CVHI_Residual
from scripts.load_beninca import load_beninca
from scripts.cvhi_residual_L1L3_diagnostics import evaluate


SEEDS = [42, 123, 456]
EPOCHS = 500
HP = dict(lr=6e-4, lam_kl=0.017, lam_cf=9.5, min_energy=0.14)

SPECIES_ORDER = ["Cyclopoids", "Calanoids", "Rotifers", "Nanophyto",
                 "Picophyto", "Filam_diatoms", "Ostracods",
                 "Harpacticoids", "Bacteria"]


def make_model(N, device):
    return CVHI_Residual(
        num_visible=N, encoder_d=96, encoder_blocks=3, encoder_heads=4,
        takens_lags=(1,2,4,8), encoder_dropout=0.1,
        d_species_f=20, f_visible_layers=2, f_visible_top_k=4,
        d_species_G=12, G_field_layers=1, G_field_top_k=3,
        prior_std=1.0, gnn_backbone="mlp", use_formula_hints=True,
        use_G_field=True, num_mixture_components=1,
        G_anchor_first=True, G_anchor_sign=+1,
    ).to(device)


def get_param_groups(model):
    enc_ids = set()
    for name, p in model.named_parameters():
        if "encoder" in name or "readout" in name:
            enc_ids.add(id(p))
    fvis = [p for p in model.parameters() if id(p) not in enc_ids]
    enc = [p for p in model.parameters() if id(p) in enc_ids]
    return fvis, enc


def compute_phase_velocity(x_np):
    """Phase-space velocity from visible data."""
    safe = np.maximum(x_np, 1e-6)
    dlog = np.log(safe[1:] / safe[:-1])
    speed = np.linalg.norm(dlog, axis=-1)
    speed_full = np.concatenate([[speed[0]], speed])
    return speed_full


def make_gate(speed, percentile=70, sharpness=10):
    """Smooth gate: 1=calm (enforce smooth), 0=turbulent (relax smooth)."""
    speed_norm = (speed - speed.min()) / (speed.max() - speed.min() + 1e-8)
    threshold = np.percentile(speed_norm, percentile)
    gate = 1.0 / (1.0 + np.exp(sharpness * (speed_norm - threshold)))
    return gate


def train_one(visible, hidden, seed, device, use_geo, epochs=EPOCHS):
    torch.manual_seed(seed)
    T, N = visible.shape
    x_full = torch.tensor(visible, dtype=torch.float32, device=device).unsqueeze(0)

    speed = compute_phase_velocity(visible)
    gate = make_gate(speed)
    gate_t = torch.tensor(gate, dtype=torch.float32, device=device)

    model = make_model(N, device)
    fvis_p, enc_p = get_param_groups(model)
    opt = torch.optim.AdamW(model.parameters(), lr=HP["lr"], weight_decay=1e-4)
    warmup = 100
    train_end = int(0.75 * T)
    best_val = float("inf")
    best_state = None

    for epoch in range(epochs):
        if hasattr(model, "G_anchor_alpha"):
            f = epoch / epochs
            model.G_anchor_alpha = max(0, 1 - max(0, (f - 0.5) / 0.45))
        h_w = 0.0 if epoch < warmup else min(1.0, (epoch - warmup) / 100)
        K_r = 0 if epoch < warmup else min(3, int((epoch - warmup) / (epochs - warmup) * 6))

        # Alternating 5:1
        if epoch >= warmup:
            cyc = (epoch - warmup) % 6
            if cyc < 5:
                for p in enc_p: p.requires_grad_(False)
                for p in fvis_p: p.requires_grad_(True)
            else:
                for p in fvis_p: p.requires_grad_(False)
                for p in enc_p: p.requires_grad_(True)

        model.train()
        opt.zero_grad()
        if epoch > warmup:
            mask = (torch.rand(1, T, 1, device=device) > 0.05).float()
            x_in = x_full * mask + (1 - mask) * x_full.mean(1, keepdim=True)
        else:
            x_in = x_full

        out = model(x_in, n_samples=2, rollout_K=K_r)
        tr = model.slice_out(out, 0, train_end)
        tr["visible"] = out["visible"][:, :train_end]
        tr["G"] = out["G"][:, :train_end]

        # Standard loss with LOW smooth/hf (we replace with geo version)
        if use_geo:
            lam_s = 0.005   # reduced base smooth
            lam_h = 0.05    # reduced base hf
        else:
            lam_s = 0.02
            lam_h = 0.2

        losses = model.loss(
            tr, beta_kl=HP["lam_kl"], free_bits=0.02,
            margin_null=0.002, margin_shuf=0.001,
            lam_necessary=HP["lam_cf"], lam_shuffle=HP["lam_cf"] * 0.6,
            lam_energy=2.0, min_energy=HP["min_energy"],
            lam_smooth=lam_s, lam_sparse=0.02,
            h_weight=h_w, lam_rollout=0.5 * h_w,
            rollout_weights=(1.0, 0.5, 0.25),
            lam_hf=lam_h, lowpass_sigma=6.0, lam_rmse_log=0.1,
        )
        total = losses["total"]

        # Geometry-aware additions
        if use_geo and h_w > 0:
            h_samples = out["h_samples"]  # (S, B, T)

            # 1. Gated smooth: less penalty at high-speed timesteps
            dh = h_samples[:, :, 2:] - 2 * h_samples[:, :, 1:-1] + h_samples[:, :, :-2]
            gate_w = gate_t[1:-1].unsqueeze(0).unsqueeze(0)
            geo_smooth = (dh ** 2 * gate_w).mean()
            total = total + 0.02 * geo_smooth

            # 2. Speed-correlated h reward: encourage |h| when speed is high
            burst_indicator = (1 - gate_t).unsqueeze(0).unsqueeze(0)  # high at burst
            speed_reward = (h_samples.abs() * burst_indicator).mean()
            total = total - 0.01 * h_w * speed_reward

        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        with torch.no_grad():
            vo = model.slice_out(out, train_end, T)
            vo["visible"] = out["visible"][:, train_end:T]
            vo["G"] = out["G"][:, train_end:T]
            vl = model.loss(
                vo, h_weight=1, margin_null=0.002, margin_shuf=0.001,
                lam_energy=2, min_energy=HP["min_energy"],
                lam_rollout=0.5, rollout_weights=(1, 0.5, 0.25),
                lam_hf=lam_h, lowpass_sigma=6,
            )
            vr = vl["recon_full"].item()
        if epoch > warmup + 15 and vr < best_val:
            best_val = vr
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    for p in model.parameters():
        p.requires_grad_(True)
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        out = model(x_full, n_samples=30, rollout_K=3)
        h_mean = out["h_samples"].mean(0)[0].cpu().numpy()
    pear, _ = evaluate(h_mean, hidden)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return h_mean, pear, speed, gate


def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"runs/{ts}_beninca_geo_gating")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_dir}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    full, species, _ = load_beninca()
    species = [str(s) for s in species]

    print("\n=== Geometry-aware smooth gating ===\n")

    fig, axes = plt.subplots(len(SPECIES_ORDER), 3, figsize=(22, 28),
                              constrained_layout=True)

    results = {"baseline": {}, "geo": {}}

    for row, sp in enumerate(SPECIES_ORDER):
        h_idx = species.index(sp)
        vis = np.delete(full, h_idx, axis=1).astype(np.float32)
        hid = full[:, h_idx].astype(np.float32)
        t = np.arange(len(hid))

        ps_base = []
        ps_geo = []
        best_hb = None; best_pb = -999
        best_hg = None; best_pg = -999
        best_speed = None; best_gate = None

        for seed in SEEDS:
            hb, pb, _, _ = train_one(vis, hid, seed, device, use_geo=False)
            hg, pg, spd, gat = train_one(vis, hid, seed, device, use_geo=True)
            ps_base.append(pb)
            ps_geo.append(pg)
            if pb > best_pb:
                best_pb = pb; best_hb = hb
            if pg > best_pg:
                best_pg = pg; best_hg = hg; best_speed = spd; best_gate = gat

        results["baseline"][sp] = np.mean(ps_base)
        results["geo"][sp] = np.mean(ps_geo)
        d = np.mean(ps_geo) - np.mean(ps_base)
        print(f"  {sp:<16} baseline={np.mean(ps_base):+.3f}  "
              f"geo={np.mean(ps_geo):+.3f}  D={d:+.3f}")

        L = min(len(best_hg), len(hid))

        ax = axes[row, 0]
        a, b = np.polyfit(best_hb[:L], hid[:L], 1)
        ax.plot(t[:L], hid[:L], "k-", lw=1, alpha=0.5, label="true")
        ax.plot(t[:L], a * best_hb[:L] + b, "b-", lw=0.8,
                label=f"base P={best_pb:+.3f}")
        ax.set_title(f"{sp} / baseline"); ax.legend(fontsize=7); ax.grid(alpha=0.2)

        ax = axes[row, 1]
        a, b = np.polyfit(best_hg[:L], hid[:L], 1)
        ax.plot(t[:L], hid[:L], "k-", lw=1, alpha=0.5, label="true")
        ax.plot(t[:L], a * best_hg[:L] + b, "r-", lw=0.8,
                label=f"geo P={best_pg:+.3f}")
        ax.set_title(f"{sp} / geo-gated"); ax.legend(fontsize=7); ax.grid(alpha=0.2)

        ax = axes[row, 2]
        ax.plot(t[:L], best_speed[:L] / best_speed.max(), "g-", lw=0.7,
                alpha=0.7, label="speed")
        ax.fill_between(t[:L], 0, best_gate[:L], alpha=0.15, color="blue",
                          label="smooth gate")
        ax.plot(t[:L], hid[:L] / (hid.max() + 1e-8), "k-", lw=0.7, alpha=0.5,
                label="true (norm)")
        ax.set_title(f"{sp} / speed vs true"); ax.legend(fontsize=7); ax.grid(alpha=0.2)

    fig.suptitle("Geometry-aware smooth gating", fontweight="bold", fontsize=14)
    fig.savefig(out_dir / "fig_geo_gating.png", dpi=130)
    plt.close()

    # Summary
    overall_base = np.mean(list(results["baseline"].values()))
    overall_geo = np.mean(list(results["geo"].values()))
    print(f"\n  Overall: baseline={overall_base:+.4f}  geo={overall_geo:+.4f}  "
          f"D={overall_geo - overall_base:+.4f}")

    with open(out_dir / "summary.md", "w", encoding="utf-8") as f:
        f.write("# Geometry-aware smooth gating\n\n")
        f.write("| Species | baseline | geo | D |\n|---|---|---|---|\n")
        for sp in SPECIES_ORDER:
            d = results["geo"][sp] - results["baseline"][sp]
            f.write(f"| {sp} | {results['baseline'][sp]:+.3f} | "
                    f"{results['geo'][sp]:+.3f} | {d:+.3f} |\n")
        f.write(f"\n**Overall**: baseline={overall_base:+.4f}, "
                f"geo={overall_geo:+.4f}, D={overall_geo - overall_base:+.4f}\n")

    print(f"\n[OK] {out_dir}")


if __name__ == "__main__":
    main()
