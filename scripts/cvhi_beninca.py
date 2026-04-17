"""Beninca 2008 mesocosm 实验: 测试完整封闭系统下 Pearson 能上多少.

9 species → 隐藏 1 种, 可见 8 种.
用 G_anchor_first + anneal_late config.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import torch

from models.cvhi_residual import CVHI_Residual
from scripts.cvhi_residual_L1L3_diagnostics import (
    evaluate, _configure_matplotlib, hidden_true_substitution,
)
from scripts.load_beninca import load_beninca


SEEDS_20 = [42, 123, 456, 789, 2024, 31415, 27182, 65537,
             7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]


def alpha_schedule(epoch, epochs, start=0.5, end=0.95):
    f = epoch / max(1, epochs)
    if f <= start: return 1.0
    if f >= end: return 0.0
    return 1.0 - (f - start) / (end - start)


def make_model(N, device):
    return CVHI_Residual(
        num_visible=N,
        encoder_d=48, encoder_blocks=2, encoder_heads=4,
        takens_lags=(1, 2, 4, 8, 12), encoder_dropout=0.2,
        d_species_f=20, f_visible_layers=2, f_visible_top_k=4,
        d_species_G=12, G_field_layers=1, G_field_top_k=3,
        prior_std=1.0, gnn_backbone="mlp",
        use_formula_hints=True, use_G_field=True,
        num_mixture_components=1,
        G_anchor_first=True, G_anchor_sign=+1,
    ).to(device)


def train_one(visible, hidden_eval, device, seed, epochs=300):
    torch.manual_seed(seed)
    T, N = visible.shape
    x = torch.tensor(visible, dtype=torch.float32, device=device).unsqueeze(0)
    model = make_model(N, device)
    opt = torch.optim.AdamW(model.parameters(), lr=0.0008, weight_decay=1e-4)
    warmup = int(0.2 * epochs); ramp = max(1, int(0.2 * epochs))
    def lr_lambda(step):
        if step < 50: return step / 50
        p = (step - 50) / max(1, epochs - 50)
        return 0.5 * (1 + np.cos(np.pi * p))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    train_end = int(0.75 * T)
    best_val = float("inf"); best_state = None
    m_null, m_shuf, min_e = 0.002, 0.001, 0.05
    for epoch in range(epochs):
        model.G_anchor_alpha = alpha_schedule(epoch, epochs)
        if epoch < warmup:
            h_w, K_r, lam_r = 0.0, 0, 0.0
        else:
            post = epoch - warmup
            h_w = min(1.0, post / ramp)
            k_ramp = min(1.0, post / (epochs - warmup) * 2)
            K_r = max(1 if h_w > 0 else 0, int(round(k_ramp * 3)))
            lam_r = 0.5 * h_w
        model.train(); opt.zero_grad()
        out = model(x, n_samples=2, rollout_K=K_r)
        tr_out = model.slice_out(out, 0, train_end)
        losses = model.loss(tr_out, beta_kl=0.03, free_bits=0.02,
            margin_null=m_null, margin_shuf=m_shuf,
            lam_necessary=5.0, lam_shuffle=3.0,
            lam_energy=2.0, min_energy=min_e,
            lam_smooth=0.02, lam_sparse=0.02,
            h_weight=h_w, lam_rollout=lam_r,
            rollout_weights=(1.0, 0.5, 0.25),
            lam_hf=0.0, lowpass_sigma=6.0)
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        with torch.no_grad():
            val_out = model.slice_out(out, train_end, T)
            val_losses = model.loss(val_out, h_weight=1.0,
                margin_null=m_null, margin_shuf=m_shuf,
                lam_necessary=5.0, lam_shuffle=3.0,
                lam_energy=2.0, min_energy=min_e,
                lam_rollout=lam_r, rollout_weights=(1.0, 0.5, 0.25),
                lam_hf=0.0, lowpass_sigma=6.0)
            val_recon = val_losses["recon_full"].item()
        if epoch > warmup + 15 and val_recon < best_val:
            best_val = val_recon
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        out_eval = model(x, n_samples=30, rollout_K=3)
        h_mean = out_eval["h_samples"].mean(dim=0)[0].cpu().numpy()
    pear, _ = evaluate(h_mean, hidden_eval)
    diag = hidden_true_substitution(model, visible, hidden_eval, device)
    d_ratio = diag["recon_true_scaled"] / diag["recon_encoder"]
    return {"seed": seed, "pearson": pear, "d_ratio": d_ratio, "val_recon": best_val}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--hidden_species", type=str, default="all",
                     help="hidden species name or 'all' to loop all 9")
    args = ap.parse_args()

    _configure_matplotlib()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"runs/{ts}_beninca")
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    full, species, days = load_beninca()
    print(f"Beninca data: {full.shape}, species: {species}")
    seeds = SEEDS_20[:args.n_seeds]

    if args.hidden_species == "all":
        hidden_list = species
    else:
        hidden_list = [args.hidden_species]

    all_results = {}
    for h_name in hidden_list:
        if h_name not in species:
            print(f"  skip (not in species): {h_name}"); continue
        h_idx = species.index(h_name)
        visible = np.delete(full, h_idx, axis=1)   # (T, 8)
        hidden = full[:, h_idx]
        print(f"\n{'='*70}\nHidden: {h_name} (idx {h_idx})  visible shape: {visible.shape}\n{'='*70}")
        rs = []
        for s in seeds:
            t0 = datetime.now()
            r = train_one(visible, hidden, device, s, epochs=args.epochs)
            dt = (datetime.now() - t0).total_seconds()
            rs.append(r)
            print(f"  seed={s}  P={r['pearson']:+.3f}  d_r={r['d_ratio']:.2f}  ({dt:.1f}s)")
        all_results[h_name] = rs

    # summary
    summary = {}
    for h in all_results:
        P = np.array([r["pearson"] for r in all_results[h]])
        D = np.array([r["d_ratio"] for r in all_results[h]])
        summary[h] = {"mean_P": float(P.mean()), "std_P": float(P.std(ddof=1) if len(P)>1 else 0),
                       "max_P": float(P.max()), "mean_d_r": float(D.mean())}
    print(f"\n{'='*80}\nSUMMARY ({args.n_seeds} seeds, {args.epochs} epochs)\n{'='*80}")
    print(f"{'hidden':<20s}{'mean P':<12s}{'std':<10s}{'max P':<10s}{'d_ratio':<8s}")
    for h in all_results:
        s = summary[h]
        print(f"{h:<20s}{s['mean_P']:+.4f}      {s['std_P']:.4f}    {s['max_P']:+.3f}     {s['mean_d_r']:.3f}")

    with open(out_dir / "summary.md", "w", encoding="utf-8") as f:
        f.write(f"# Beninca 2008 mesocosm 实验\n\n- seeds: {args.n_seeds}  epochs: {args.epochs}\n\n")
        f.write("| hidden | mean P | std | max | d_ratio |\n|---|---|---|---|---|\n")
        for h in all_results:
            s = summary[h]
            f.write(f"| {h} | {s['mean_P']:+.4f} | {s['std_P']:.4f} | {s['max_P']:+.3f} | {s['mean_d_r']:.3f} |\n")

    with open(out_dir / "raw.json", "w") as f:
        json.dump({h: [{k: float(v) if isinstance(v,(int,float,np.floating)) else v for k,v in r.items()} for r in rs]
                    for h, rs in all_results.items()}, f, indent=2, default=float)
    print(f"\n[OK] {out_dir}")


if __name__ == "__main__":
    main()
