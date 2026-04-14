"""Anchor anneal: 早期强约束 (alpha=1 softplus), 后期放松 (alpha→0 identity).

目标: 保持 G_anchor_first 的 Pearson 提升, 同时让 d_ratio 回归 ~1.
5 seeds × Portal × 3 configs:
  hard:     alpha=1 全程 (G_anchor_first 基线)
  anneal:   alpha=1 → 0 线性退火 (从 30% 到 90% epochs)
  anneal_late: alpha=1 → 0 更晚退火 (从 50% 到 95%)
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
    load_portal, evaluate, _configure_matplotlib, hidden_true_substitution,
)
from scripts.cvhi_residual_mendota import load_lake_mendota


def load_mendota_xtroph():
    """Mendota: hidden = Microcystis aeruginosa (idx 1)."""
    matrix, _, _, _, _ = load_lake_mendota()
    hid = matrix[:, 1]
    vis = np.delete(matrix, 1, axis=1)
    return vis.astype(np.float32), hid.astype(np.float32)


SEEDS = [42, 123, 456, 789, 2024, 31415, 27182, 65537,
          7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]


def make_portal(N, device):
    return CVHI_Residual(
        num_visible=N,
        encoder_d=48, encoder_blocks=2, encoder_heads=4,
        takens_lags=(1, 2, 4, 8, 12), encoder_dropout=0.2,
        d_species_f=20, f_visible_layers=2, f_visible_top_k=4,
        d_species_G=12, G_field_layers=1, G_field_top_k=3,
        prior_std=1.0, gnn_backbone="mlp",
        use_formula_hints=True, use_G_field=True,
        num_mixture_components=1,
        G_anchor_first=True,
    ).to(device)


def alpha_schedule(epoch, epochs, start_frac, end_frac):
    """alpha: 1.0 for t<start_frac, linear → 0 in [start, end], 0 for t>end."""
    f = epoch / max(1, epochs)
    if f <= start_frac: return 1.0
    if f >= end_frac: return 0.0
    return 1.0 - (f - start_frac) / (end_frac - start_frac)


def train_one(visible, hidden_eval, device, seed, schedule_args, epochs=300):
    torch.manual_seed(seed)
    T, N = visible.shape
    x = torch.tensor(visible, dtype=torch.float32, device=device).unsqueeze(0)
    model = make_portal(N, device)
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

    alpha_log = []
    for epoch in range(epochs):
        # 更新 alpha
        if schedule_args is None:
            alpha = 1.0
        else:
            alpha = alpha_schedule(epoch, epochs, *schedule_args)
        model.G_anchor_alpha = alpha
        alpha_log.append(alpha)

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
        losses = model.loss(
            tr_out, beta_kl=0.03, free_bits=0.02,
            margin_null=m_null, margin_shuf=m_shuf,
            lam_necessary=5.0, lam_shuffle=3.0,
            lam_energy=2.0, min_energy=min_e,
            lam_smooth=0.02, lam_sparse=0.02,
            h_weight=h_w, lam_rollout=lam_r,
            rollout_weights=(1.0, 0.5, 0.25),
            lam_hf=0.0, lowpass_sigma=6.0,
        )
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
                lam_hf=0.0, lowpass_sigma=6.0,
            )
            val_recon = val_losses["recon_full"].item()
        if epoch > warmup + 15 and val_recon < best_val:
            best_val = val_recon
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    # 评测时保持最终 alpha (通常是 0, 若 anneal end 已过)
    model.eval()
    with torch.no_grad():
        out_eval = model(x, n_samples=30, rollout_K=3)
        h_mean = out_eval["h_samples"].mean(dim=0)[0].cpu().numpy()
    pear, _ = evaluate(h_mean, hidden_eval)
    diag = hidden_true_substitution(model, visible, hidden_eval, device)
    d_ratio = diag["recon_true_scaled"] / diag["recon_encoder"]
    final_alpha = model.G_anchor_alpha
    return {"seed": seed, "pearson": pear, "d_ratio": d_ratio, "final_alpha": final_alpha,
            "val_recon": best_val}


def aggregate(pears):
    arr = np.asarray(pears); n = len(arr)
    mean = float(arr.mean()); std = float(arr.std(ddof=1)) if n > 1 else 0.0
    return dict(n=n, mean=mean, std=std, max=float(arr.max()), min=float(arr.min()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--schedules", nargs="+", default=["hard", "anneal_early", "anneal_mid", "anneal_late"])
    ap.add_argument("--dataset", type=str, default="Portal", choices=["Portal", "Mendota"])
    args = ap.parse_args()

    _configure_matplotlib()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"runs/{ts}_anchor_anneal_{args.dataset}")
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.dataset == "Portal":
        vis, hid = load_portal("OT")
    else:
        vis, hid = load_mendota_xtroph()
    print(f"Dataset: {args.dataset}, visible shape: {vis.shape}")
    seeds = SEEDS[:args.n_seeds]

    # schedules:
    # hard:       None (alpha=1 all time)
    # anneal_mid: decay 30% → 90%
    # anneal_early: decay 20% → 70%  (更激进早放松)
    # anneal_late:  decay 50% → 95%  (保持更久再放松)
    SCHEDULES_ALL = {
        "hard":           None,
        "anneal_early":   (0.20, 0.70),
        "anneal_mid":     (0.30, 0.90),
        "anneal_late":    (0.50, 0.95),
    }
    SCHEDULES = {k: SCHEDULES_ALL[k] for k in args.schedules}

    results = {}
    for sname, sargs in SCHEDULES.items():
        print(f"\n{'='*70}\n{sname}  schedule_args={sargs}\n{'='*70}")
        results[sname] = []
        for s in seeds:
            t0 = datetime.now()
            r = train_one(vis, hid, device, s, sargs, epochs=args.epochs)
            dt = (datetime.now() - t0).total_seconds()
            results[sname].append(r)
            print(f"  seed={s}  P={r['pearson']:+.3f}  d_r={r['d_ratio']:.2f}  final_α={r['final_alpha']:.2f}  ({dt:.1f}s)")

    summary = {}
    for sname in SCHEDULES:
        summary[sname] = {
            "pearson": aggregate([r["pearson"] for r in results[sname]]),
            "d_ratio": aggregate([r["d_ratio"] for r in results[sname]]),
        }

    print(f"\n{'='*90}")
    print(f"SUMMARY")
    print(f"{'='*90}")
    print(f"{'schedule':<15s}{'mean P':<12s}{'std':<8s}{'max':<8s}{'mean d_r':<12s}{'std d_r':<10s}")
    for sname in SCHEDULES:
        p = summary[sname]["pearson"]; d = summary[sname]["d_ratio"]
        print(f"{sname:<15s}{p['mean']:<+12.4f}{p['std']:<8.4f}{p['max']:<+8.3f}{d['mean']:<12.3f}{d['std']:<10.3f}")

    with open(out_dir / "summary.md", "w", encoding="utf-8") as f:
        f.write("# Anchor Annealing 实验\n\n")
        f.write(f"- seeds: {len(seeds)}  epochs: {args.epochs}\n\n")
        f.write("| schedule | mean P | std | max | mean d_r | std d_r |\n|---|---|---|---|---|---|\n")
        for sname in SCHEDULES:
            p = summary[sname]["pearson"]; d = summary[sname]["d_ratio"]
            f.write(f"| {sname} | {p['mean']:+.4f} | {p['std']:.4f} | {p['max']:+.3f} | {d['mean']:.3f} | {d['std']:.3f} |\n")

    raw = {s: [{k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                 for k, v in r.items()} for r in rs] for s, rs in results.items()}
    with open(out_dir / "raw_results.json", "w") as f:
        json.dump({"summary": summary, "raw": raw}, f, indent=2, default=float)
    print(f"\n[OK] {out_dir}")


if __name__ == "__main__":
    main()
