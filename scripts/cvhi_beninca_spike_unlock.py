"""Spike unlock experiment: remove/reduce smoothness biases to let h capture bursts.

Diagnosis: model outputs smooth h because 3 forces suppress spikes:
  1. KL to Gaussian N(0,1) — penalizes large |h| quadratically
  2. lam_smooth=0.02 — second-order diff smoothness
  3. lam_hf=0.2 — explicit high-frequency penalty (lowpass sigma=6)

Configs:
  A. baseline (Stage 1b as-is)
  B. no_hf: lam_hf=0 (remove HF penalty, keep smooth)
  C. low_reg: lam_smooth=0.005, lam_hf=0.02 (reduce both)
  D. laplace: replace Gaussian KL with L1 on h (Laplace prior)
  E. laplace+low_reg: D + C combined
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

from models.cvhi_residual import CVHI_Residual
from scripts.load_beninca import load_beninca
from scripts.cvhi_residual_L1L3_diagnostics import evaluate, hidden_true_substitution


SEEDS = [42, 123, 456]
EPOCHS = 500

BEST_HP = dict(
    encoder_d=96, encoder_blocks=3, encoder_dropout=0.1,
    takens_lags=(1, 2, 4, 8),
    lr=0.0006033475528697158,
    lam_kl=0.017251789430967935,
    lam_cf=9.517725868477207,
    min_energy=0.14353013693386804,
)

SPECIES_ORDER = ["Cyclopoids", "Calanoids", "Rotifers", "Nanophyto",
                 "Picophyto", "Filam_diatoms", "Ostracods",
                 "Harpacticoids", "Bacteria"]

# (name, lam_smooth, lam_hf, beta_kl, lam_l1_h, lowpass_sigma)
CONFIGS = [
    ("baseline",       0.02,  0.2,   BEST_HP["lam_kl"], 0.0,   6.0),
    ("no_hf",          0.02,  0.0,   BEST_HP["lam_kl"], 0.0,   6.0),
    ("low_reg",        0.005, 0.02,  BEST_HP["lam_kl"], 0.0,   6.0),
    ("laplace",        0.02,  0.2,   0.0,               0.03,  6.0),
    ("laplace+low",    0.005, 0.02,  0.0,               0.03,  6.0),
]


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


def train_one(visible, hidden, seed, device, cfg, epochs=EPOCHS):
    cfg_name, lam_smooth, lam_hf, beta_kl, lam_l1_h, lp_sigma = cfg
    torch.manual_seed(seed)
    T, N = visible.shape
    x_full = torch.tensor(visible, dtype=torch.float32, device=device).unsqueeze(0)

    model = make_model(N, device)
    opt = torch.optim.AdamW(model.parameters(), lr=BEST_HP["lr"], weight_decay=1e-4)

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
        model.train(); opt.zero_grad()
        out = model(x_train, n_samples=2, rollout_K=K_r)
        tr_out = model.slice_out(out, 0, train_end)
        tr_out["visible"] = out["visible"][:, 0:train_end]
        tr_out["G"] = out["G"][:, 0:train_end]
        losses = model.loss(
            tr_out, beta_kl=beta_kl, free_bits=0.02,
            margin_null=0.002, margin_shuf=0.001,
            lam_necessary=BEST_HP["lam_cf"], lam_shuffle=BEST_HP["lam_cf"]*0.6,
            lam_energy=2.0, min_energy=BEST_HP["min_energy"],
            lam_smooth=lam_smooth, lam_sparse=0.02,
            h_weight=h_w, lam_rollout=lam_r,
            rollout_weights=(1.0, 0.5, 0.25),
            lam_hf=lam_hf, lowpass_sigma=lp_sigma,
            lam_rmse_log=0.1,
        )
        total = losses["total"]

        # Laplace prior: L1 penalty on h_samples (replaces Gaussian KL)
        if lam_l1_h > 0 and h_w > 0:
            h_samples = out["h_samples"]  # (S, B, T)
            l1_h = h_samples.abs().mean()
            total = total + lam_l1_h * h_w * l1_h

        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
                lam_hf=lam_hf, lowpass_sigma=lp_sigma,
            )
            val_recon = val_losses["recon_full"].item()
        if epoch > warmup + 15 and val_recon < best_val:
            best_val = val_recon
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        out_eval = model(x_full, n_samples=30, rollout_K=3)
        h_mean = out_eval["h_samples"].mean(dim=0)[0].cpu().numpy()
    pear, _ = evaluate(h_mean, hidden)
    # Also compute LP Pearson
    from scipy.ndimage import gaussian_filter1d
    h_lp = gaussian_filter1d(h_mean, sigma=8)
    pear_lp, _ = evaluate(h_lp, hidden)
    # h stats
    h_std = float(h_mean.std())
    h_kurtosis = float(((h_mean - h_mean.mean()) / (h_std + 1e-8)) ** 4).mean() - 3  # excess kurtosis
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"pearson": pear, "pearson_lp": pear_lp, "val_recon": best_val,
            "h_std": h_std, "h_kurtosis": h_kurtosis, "h_mean": h_mean}


def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"runs/{ts}_beninca_spike_unlock")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_dir}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    full, species, _ = load_beninca()
    species = [str(s) for s in species]

    print(f"\n=== Spike unlock experiment ===")
    for cfg in CONFIGS:
        print(f"  {cfg[0]}: smooth={cfg[1]}, hf={cfg[2]}, kl={cfg[3]:.4f}, l1_h={cfg[4]}, lp_s={cfg[5]}")
    total_runs = len(SPECIES_ORDER) * len(CONFIGS) * len(SEEDS)
    print(f"Total: {total_runs} runs\n")

    results = {c[0]: {sp: [] for sp in SPECIES_ORDER} for c in CONFIGS}
    run_i = 0

    for h_name in SPECIES_ORDER:
        h_idx = species.index(h_name)
        visible = np.delete(full, h_idx, axis=1).astype(np.float32)
        hidden_raw = full[:, h_idx].astype(np.float32)
        print(f"\n--- hidden={h_name} ---")
        for cfg in CONFIGS:
            for seed in SEEDS:
                run_i += 1
                t0 = datetime.now()
                try:
                    r = train_one(visible, hidden_raw, seed, device, cfg)
                    dt = (datetime.now() - t0).total_seconds()
                    print(f"  [{run_i}/{total_runs}] {cfg[0]:<16} seed={seed}  "
                          f"P={r['pearson']:+.3f}  P_lp={r['pearson_lp']:+.3f}  "
                          f"kurt={r['h_kurtosis']:+.1f}  ({dt:.1f}s)")
                except Exception as e:
                    print(f"  [{run_i}/{total_runs}] FAILED: {e}")
                    import traceback; traceback.print_exc()
                    r = {"pearson": float("nan"), "pearson_lp": float("nan"),
                         "val_recon": float("nan"), "h_std": 0, "h_kurtosis": 0,
                         "h_mean": None}
                r["seed"] = seed; r["config"] = cfg[0]
                results[cfg[0]][h_name].append(r)

    # Summary
    print(f"\n{'='*100}")
    print("SPIKE UNLOCK RESULTS")
    print('='*100)
    header = f"{'Species':<16}"
    for cfg in CONFIGS:
        header += f"{cfg[0]:<16}"
    print(header)
    print('-'*100)

    config_means = {c[0]: [] for c in CONFIGS}
    config_lp_means = {c[0]: [] for c in CONFIGS}
    for sp in SPECIES_ORDER:
        line = f"{sp:<16}"
        for cfg in CONFIGS:
            rs = results[cfg[0]][sp]
            P = np.array([r["pearson"] for r in rs if not np.isnan(r["pearson"])])
            m = float(P.mean()) if len(P) else float("nan")
            if not np.isnan(m):
                config_means[cfg[0]].append(m)
            P_lp = np.array([r["pearson_lp"] for r in rs if not np.isnan(r["pearson_lp"])])
            m_lp = float(P_lp.mean()) if len(P_lp) else float("nan")
            if not np.isnan(m_lp):
                config_lp_means[cfg[0]].append(m_lp)
            line += f"{m:<+16.3f}"
        print(line)

    print('-'*100)
    avg_line = f"{'Overall':<16}"
    for cfg in CONFIGS:
        avg = np.mean(config_means[cfg[0]]) if config_means[cfg[0]] else float("nan")
        avg_line += f"{avg:<+16.4f}"
    print(avg_line)
    avg_lp_line = f"{'Overall(LP)':<16}"
    for cfg in CONFIGS:
        avg = np.mean(config_lp_means[cfg[0]]) if config_lp_means[cfg[0]] else float("nan")
        avg_lp_line += f"{avg:<+16.4f}"
    print(avg_lp_line)

    # Kurtosis summary (higher = more spikey)
    print(f"\nKurtosis (excess, >0 means heavier tails than Gaussian):")
    kurt_line = f"{'Mean kurt':<16}"
    for cfg in CONFIGS:
        kurts = []
        for sp in SPECIES_ORDER:
            for r in results[cfg[0]][sp]:
                if not np.isnan(r.get("h_kurtosis", float("nan"))):
                    kurts.append(r["h_kurtosis"])
        kurt_line += f"{np.mean(kurts) if kurts else float('nan'):<+16.1f}"
    print(kurt_line)

    # Save
    best_cfg = max(CONFIGS, key=lambda c: np.mean(config_means[c[0]]) if config_means[c[0]] else -999)
    best_avg = np.mean(config_means[best_cfg[0]])
    print(f"\nBest: {best_cfg[0]} = {best_avg:+.4f}")

    with open(out_dir / "summary.md", "w", encoding="utf-8") as f:
        f.write("# Spike unlock experiment\n\n")
        f.write("Goal: remove smoothness biases to let h capture bursts.\n\n")
        f.write(f"Seeds: {SEEDS}, Epochs: {EPOCHS}\n\n")
        f.write("| Config | lam_smooth | lam_hf | beta_kl | lam_l1_h |\n")
        f.write("|---|---|---|---|---|\n")
        for cfg in CONFIGS:
            f.write(f"| {cfg[0]} | {cfg[1]} | {cfg[2]} | {cfg[3]:.4f} | {cfg[4]} |\n")
        f.write(f"\n| Species |" + "|".join(f" {c[0]} " for c in CONFIGS) + "|\n")
        f.write("|---|" + "---|" * len(CONFIGS) + "\n")
        for sp in SPECIES_ORDER:
            row = f"| {sp}"
            for cfg in CONFIGS:
                rs = results[cfg[0]][sp]
                P = np.array([r["pearson"] for r in rs if not np.isnan(r["pearson"])])
                m = float(P.mean()) if len(P) else float("nan")
                row += f" | {m:+.3f}"
            f.write(row + " |\n")
        f.write(f"\n**Overall**: " + ", ".join(
            f"{c[0]}={np.mean(config_means[c[0]]):+.4f}" for c in CONFIGS) + "\n")
        f.write(f"\n**Overall(LP)**: " + ", ".join(
            f"{c[0]}={np.mean(config_lp_means[c[0]]):+.4f}" for c in CONFIGS) + "\n")
        f.write(f"\nBest: **{best_cfg[0]}** = {best_avg:+.4f}\n")

    def to_ser(v):
        if isinstance(v, (int, float, np.floating)):
            return float(v)
        if isinstance(v, np.ndarray):
            return v.tolist()
        return v
    dump = {cfg: {sp: [{k: to_ser(v) for k, v in r.items()}
                        for r in rs] for sp, rs in d.items()}
            for cfg, d in results.items()}
    with open(out_dir / "raw.json", "w") as f:
        json.dump(dump, f, default=float)

    print(f"\n[OK] {out_dir}")


if __name__ == "__main__":
    main()
