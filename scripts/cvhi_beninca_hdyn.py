"""Beninca Stage 1 with h-dynamics loss (V1 style, proven winner on Huisman).

Config: Stage 1b (best baseline) + NEW L_h_dyn.
All 9 species rotation × 5 seeds × 500 epochs.

V1 Huisman results (previous):
  sp1: 0.18 → 0.30 (+0.12) at λ=0.3
  sp2: 0.68 → 0.81 (+0.13)
  sp4: 0.36 → 0.61 (+0.25)

Goal: replicate on real Beninca chaos. Baseline Stage 1b = 0.132 overall.
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
import torch.nn as nn
import torch.nn.functional as F

from models.cvhi_residual import CVHI_Residual
from scripts.load_beninca import load_beninca
from scripts.cvhi_residual_L1L3_diagnostics import evaluate, hidden_true_substitution


SEEDS = [42, 123, 456, 789, 2024]
EPOCHS = 500
LAMBDA_HDYN = 0.3   # sweet spot from Huisman V1
DETACH_UNTIL = 150   # first 150 epochs: g learns encoder's h w/o backprop
SEGMENT_LEN = 15   # Not used in V1, but available

BEST_HP = dict(
    encoder_d=96, encoder_blocks=3, encoder_dropout=0.1,
    takens_lags=(1, 2, 4, 8),
    lr=0.0006033475528697158,
    lam_smooth=0.02, lam_kl=0.017251789430967935,
    lam_hf=0.2, min_energy=0.14353013693386804,
    lam_cf=9.517725868477207,
)

SPECIES_ORDER = ["Cyclopoids", "Calanoids", "Rotifers", "Nanophyto",
                 "Picophyto", "Filam_diatoms", "Ostracods",
                 "Harpacticoids", "Bacteria"]


class HiddenDynamicsNet(nn.Module):
    """Small MLP: (h_t, x_t) → h_{t+1}."""
    def __init__(self, n_visible, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1 + n_visible, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, h_t, x_t):
        if h_t.dim() == 1:
            h_t = h_t.unsqueeze(-1)
        elif h_t.dim() == 2:
            h_t = h_t.unsqueeze(-1)
        inp = torch.cat([h_t, x_t], dim=-1)
        return self.net(inp).squeeze(-1)


def make_model(N, device):
    return CVHI_Residual(
        num_visible=N,
        encoder_d=BEST_HP["encoder_d"], encoder_blocks=BEST_HP["encoder_blocks"],
        encoder_heads=4,
        takens_lags=BEST_HP["takens_lags"], encoder_dropout=BEST_HP["encoder_dropout"],
        d_species_f=20, f_visible_layers=2, f_visible_top_k=4,
        d_species_G=12, G_field_layers=1, G_field_top_k=3,
        prior_std=1.0, gnn_backbone="mlp",
        use_formula_hints=True, use_G_field=True,
        num_mixture_components=1,
        G_anchor_first=True, G_anchor_sign=+1,
        hierarchical_h=False,
    ).to(device)


def alpha_schedule(epoch, epochs, start=0.5, end=0.95):
    f = epoch / max(1, epochs)
    if f <= start: return 1.0
    if f >= end: return 0.0
    return 1.0 - (f - start) / (end - start)


def train_one(visible, hidden, seed, device, lam_hdyn, epochs=EPOCHS):
    torch.manual_seed(seed)
    T, N = visible.shape
    x_full = torch.tensor(visible, dtype=torch.float32, device=device).unsqueeze(0)

    model = make_model(N, device)
    g = HiddenDynamicsNet(n_visible=N, hidden_dim=32).to(device)
    params = list(model.parameters()) + list(g.parameters())
    opt = torch.optim.AdamW(params, lr=BEST_HP["lr"], weight_decay=1e-4)

    warmup = int(0.2 * epochs)
    ramp = max(1, int(0.2 * epochs))
    def lr_lambda(step):
        if step < 50: return step / 50
        p = (step - 50) / max(1, epochs - 50)
        return 0.5 * (1 + np.cos(np.pi * p))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    train_end = int(0.75 * T)
    best_val = float("inf"); best_state = None

    for epoch in range(epochs):
        if hasattr(model, "G_anchor_alpha"):
            model.G_anchor_alpha = alpha_schedule(epoch, epochs)

        if epoch < warmup:
            h_w, K_r, lam_r = 0.0, 0, 0.0
        else:
            post = epoch - warmup
            h_w = min(1.0, post / ramp)
            k_ramp = min(1.0, post / max(1, epochs - warmup) * 2)
            K_r = max(1 if h_w > 0 else 0, int(round(k_ramp * 3)))
            lam_r = 0.5 * h_w

        if epoch > warmup:
            mask = (torch.rand(1, T, 1, device=device) > 0.05).float()
            x_train = x_full * mask + (1 - mask) * x_full.mean(dim=1, keepdim=True)
        else:
            x_train = x_full

        model.train(); g.train(); opt.zero_grad()
        out = model(x_train, n_samples=2, rollout_K=K_r)
        tr_out = model.slice_out(out, 0, train_end)
        tr_out["visible"] = out["visible"][:, 0:train_end]
        tr_out["G"] = out["G"][:, 0:train_end]

        losses = model.loss(
            tr_out,
            beta_kl=BEST_HP["lam_kl"], free_bits=0.02,
            margin_null=0.002, margin_shuf=0.001,
            lam_necessary=BEST_HP["lam_cf"],
            lam_shuffle=BEST_HP["lam_cf"]*0.6,
            lam_energy=2.0, min_energy=BEST_HP["min_energy"],
            lam_smooth=BEST_HP["lam_smooth"], lam_sparse=0.02,
            h_weight=h_w, lam_rollout=lam_r,
            rollout_weights=(1.0, 0.5, 0.25),
            lam_hf=BEST_HP["lam_hf"], lowpass_sigma=6.0,
            lam_rmse_log=0.1,
        )
        total = losses["total"]

        # === h-dynamics consistency (V1 style) ===
        if lam_hdyn > 0 and h_w > 0:
            h_mean = tr_out["h_samples"].mean(dim=0)      # (B, T_tr)
            T_h = h_mean.shape[-1]
            x_slice = x_train[:, :T_h]
            h_prev = h_mean[:, :-1]
            x_prev = x_slice[:, :-1]
            h_next_pred = g(h_prev, x_prev)
            h_next_actual = h_mean[:, 1:]

            if epoch < DETACH_UNTIL + warmup:
                target = h_next_actual.detach()
            else:
                target = h_next_actual

            L_hdyn = F.mse_loss(h_next_pred, target)
            total = total + lam_hdyn * h_w * L_hdyn

        total.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step()

        with torch.no_grad():
            val_out = model.slice_out(out, train_end, T)
            val_out["visible"] = out["visible"][:, train_end:T]
            val_out["G"] = out["G"][:, train_end:T]
            val_losses = model.loss(
                val_out, h_weight=1.0,
                margin_null=0.002, margin_shuf=0.001,
                lam_energy=2.0, min_energy=BEST_HP["min_energy"],
                lam_rollout=0.5, rollout_weights=(1.0, 0.5, 0.25),
                lam_hf=BEST_HP["lam_hf"], lowpass_sigma=6.0,
            )
            val_recon = val_losses["recon_full"].item()
        if epoch > warmup + 15 and val_recon < best_val:
            best_val = val_recon
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval(); g.eval()
    with torch.no_grad():
        out_eval = model(x_full, n_samples=30, rollout_K=3)
        h_mean = out_eval["h_samples"].mean(dim=0)[0].cpu().numpy()
        # Diagnostic: how well g fits encoder's h
        h_t = torch.tensor(h_mean, dtype=torch.float32, device=device).unsqueeze(0)
        h_pred_next = g(h_t[:, :-1], x_full[:, :-1])
        hdyn_corr = F.cosine_similarity(h_pred_next.flatten(),
                                           h_t[0, 1:].flatten(), dim=0).item()

    pear, _ = evaluate(h_mean, hidden)
    d = hidden_true_substitution(model, visible, hidden, device)

    del model, g
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "pearson": pear,
        "d_ratio": d["recon_true_scaled"] / d["recon_encoder"],
        "val_recon": best_val,
        "h_mean": h_mean,
        "hdyn_corr": hdyn_corr,
    }


def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"runs/{ts}_beninca_hdyn")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_dir}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    full, species, _ = load_beninca()
    species = [str(s) for s in species]

    print(f"\n=== Beninca h_dyn V1 (λ_hdyn={LAMBDA_HDYN}) ===")
    print(f"Config: Stage 1b + L_h_dyn; 5 seeds × 9 species × {EPOCHS} epochs")

    all_results = {}
    for h_name in SPECIES_ORDER:
        h_idx = species.index(h_name)
        visible = np.delete(full, h_idx, axis=1)
        hidden = full[:, h_idx]
        print(f"\n--- hidden={h_name} ---")
        rs = []
        for s in SEEDS:
            t0 = datetime.now()
            try:
                r = train_one(visible, hidden, s, device, LAMBDA_HDYN)
                dt = (datetime.now() - t0).total_seconds()
                print(f"  seed={s}  P={r['pearson']:+.3f}  d_r={r['d_ratio']:.2f}  "
                      f"hdyn_corr={r['hdyn_corr']:+.3f}  ({dt:.1f}s)")
            except Exception as e:
                print(f"  seed={s}  FAILED: {e}")
                import traceback; traceback.print_exc()
                r = {"pearson": float("nan"), "d_ratio": float("nan"),
                     "val_recon": float("nan"), "h_mean": None,
                     "hdyn_corr": float("nan")}
            r["seed"] = s
            rs.append(r)
        all_results[h_name] = rs

    # Baselines
    phase2 = {"Cyclopoids": 0.064, "Calanoids": 0.159, "Rotifers": 0.056,
              "Nanophyto": 0.204, "Picophyto": 0.062, "Filam_diatoms": 0.056,
              "Ostracods": 0.125, "Harpacticoids": 0.129, "Bacteria": 0.171}
    stage1b = {"Cyclopoids": 0.055, "Calanoids": 0.173, "Rotifers": 0.080,
               "Nanophyto": 0.141, "Picophyto": 0.116, "Filam_diatoms": 0.031,
               "Ostracods": 0.272, "Harpacticoids": 0.120, "Bacteria": 0.197}

    print(f"\n{'='*100}")
    print(f"BENINCA h_dyn V1 RESULTS (λ={LAMBDA_HDYN})")
    print('='*100)
    print(f"{'Species':<18}{'P2':<10}{'S1b':<10}{'hdyn':<12}{'D vs S1b':<12}{'hdyn_corr':<12}")
    sum_s1b = 0; sum_hdyn = 0; cnt = 0
    max_pearsons = {}
    for h in SPECIES_ORDER:
        rs = all_results[h]
        P = np.array([r["pearson"] for r in rs if not np.isnan(r["pearson"])])
        C = np.array([r["hdyn_corr"] for r in rs if not np.isnan(r.get("hdyn_corr", np.nan))])
        hdyn_mean = float(P.mean()) if len(P) else float("nan")
        max_pearsons[h] = float(P.max()) if len(P) else float("nan")
        hdyn_corr_mean = float(C.mean()) if len(C) else float("nan")
        print(f"{h:<18}{phase2[h]:<+10.3f}{stage1b[h]:<+10.3f}"
              f"{hdyn_mean:<+12.3f}{hdyn_mean - stage1b[h]:<+12.3f}"
              f"{hdyn_corr_mean:<+12.3f}")
        if not np.isnan(hdyn_mean):
            sum_s1b += stage1b[h]; sum_hdyn += hdyn_mean; cnt += 1

    avg_s1b = sum_s1b/cnt; avg_hdyn = sum_hdyn/cnt
    print(f"\nOverall: S1b={avg_s1b:+.4f}  hdyn={avg_hdyn:+.4f}  Δ={avg_hdyn-avg_s1b:+.4f}")

    if avg_hdyn > avg_s1b + 0.01:
        verdict = f"[WIN] hdyn beats S1b by {avg_hdyn-avg_s1b:+.3f}"
    elif avg_hdyn > avg_s1b - 0.01:
        verdict = "[NEUTRAL] hdyn ≈ S1b"
    else:
        verdict = "[LOSS] hdyn worse than S1b"
    print(f"\n{verdict}\n{'='*100}")

    # Plot recovery per species (best seed)
    fig, axes = plt.subplots(3, 3, figsize=(16, 9), constrained_layout=True)
    for ax, h in zip(axes.flat, SPECIES_ORDER):
        h_idx = species.index(h)
        true_h = full[:, h_idx]
        t_axis = np.arange(len(true_h))
        ax.plot(t_axis, true_h, color="black", lw=1.5, label="true", zorder=10)
        rs = all_results[h]
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(rs)))
        for r, c in zip(rs, colors):
            hm = r.get("h_mean")
            if hm is None:
                continue
            L = min(len(hm), len(true_h))
            a, b = np.polyfit(hm[:L], true_h[:L], 1)
            ax.plot(t_axis[:L], a * hm[:L] + b, color=c, lw=0.8, alpha=0.7)
        P = np.array([r["pearson"] for r in rs if not np.isnan(r["pearson"])])
        mP = P.mean() if len(P) else float("nan")
        s1b_val = stage1b.get(h, float("nan"))
        delta = mP - s1b_val
        color_ind = "green" if delta > 0.01 else ("red" if delta < -0.01 else "gray")
        ax.set_title(f"{h}  P={mP:+.3f} (S1b={s1b_val:+.3f}, Δ{delta:+.3f})",
                     color=color_ind, fontsize=10)
        ax.grid(alpha=0.25)
    fig.suptitle(f"Beninca hidden recovery: hdyn λ={LAMBDA_HDYN} vs true (scale-aligned)",
                 fontweight="bold")
    fig.savefig(out_dir / "fig_beninca_recovery.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # Summary bar chart
    fig, ax = plt.subplots(figsize=(13, 6), constrained_layout=True)
    x = np.arange(len(SPECIES_ORDER))
    w = 0.25
    p2_vals = [phase2[h] for h in SPECIES_ORDER]
    s1b_vals = [stage1b[h] for h in SPECIES_ORDER]
    hdyn_vals = [np.mean([r["pearson"] for r in all_results[h] if not np.isnan(r["pearson"])])
                  for h in SPECIES_ORDER]
    ax.bar(x - w, p2_vals, w, label="Phase 2", color="#90a4ae")
    ax.bar(x, s1b_vals, w, label="Stage 1b", color="#1976d2")
    ax.bar(x + w, hdyn_vals, w, label=f"+ h_dyn (λ={LAMBDA_HDYN})", color="#c62828")
    for i, v in enumerate(hdyn_vals):
        ax.text(i + w, v + 0.01, f"{v:.2f}", ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(SPECIES_ORDER, rotation=20, fontsize=10)
    ax.set_ylabel("Pearson")
    ax.set_title(f"Beninca: h_dyn vs baselines (overall S1b={avg_s1b:+.3f}, "
                 f"hdyn={avg_hdyn:+.3f}, Δ={avg_hdyn-avg_s1b:+.3f})",
                 fontweight="bold")
    ax.legend(); ax.grid(alpha=0.25, axis="y")
    fig.savefig(out_dir / "fig_bar_comparison.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    with open(out_dir / "summary.md", "w", encoding="utf-8") as f:
        f.write(f"# Beninca h_dyn V1 (λ={LAMBDA_HDYN})\n\n")
        f.write(f"Seeds: {SEEDS}, Epochs: {EPOCHS}\n\n")
        f.write("| Species | P2 | S1b | **hdyn** | Δ vs S1b | hdyn_corr |\n")
        f.write("|---|---|---|---|---|---|\n")
        for h in SPECIES_ORDER:
            rs = all_results[h]
            P = np.array([r["pearson"] for r in rs if not np.isnan(r["pearson"])])
            C = np.array([r["hdyn_corr"] for r in rs
                           if not np.isnan(r.get("hdyn_corr", np.nan))])
            hm = float(P.mean()) if len(P) else float("nan")
            cm = float(C.mean()) if len(C) else float("nan")
            f.write(f"| {h} | {phase2[h]:+.3f} | {stage1b[h]:+.3f} | "
                    f"**{hm:+.3f}** | {hm-stage1b[h]:+.3f} | {cm:+.3f} |\n")
        f.write(f"\n**Overall**: S1b={avg_s1b:+.4f}, hdyn={avg_hdyn:+.4f}, "
                f"Δ={avg_hdyn-avg_s1b:+.4f}\n\n")
        f.write(f"**{verdict}**\n")

    def to_ser(v):
        if isinstance(v, (int, float, np.floating)):
            return float(v)
        if isinstance(v, np.ndarray):
            return v.tolist()
        return v
    dump = {h: [{k: to_ser(v) for k, v in r.items()} for r in rs]
            for h, rs in all_results.items()}
    with open(out_dir / "raw.json", "w") as f:
        json.dump(dump, f, indent=2, default=float)
    print(f"\n[OK] {out_dir}")


if __name__ == "__main__":
    main()
