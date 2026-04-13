"""Compare GNN backbones on LV and Holling:
  - SoftForms (current): 5 preset forms with soft gates (fixed mixture)
  - MLP with formula hints: MLP learns messages, presets are hints (not forced)
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from models.cvhi_residual import CVHI_Residual
from scripts.cvhi_residual_synthetic_comparison import load_lv, load_holling
from scripts.cvhi_residual_L1L3_diagnostics import (
    evaluate, _configure_matplotlib, hidden_true_substitution,
)


def make_cfg_model(N, backbone, device):
    return CVHI_Residual(
        num_visible=N,
        encoder_d=64, encoder_blocks=2, encoder_heads=4,
        takens_lags=(1, 2, 4, 8), encoder_dropout=0.1,
        d_species_f=24, f_visible_layers=2, f_visible_top_k=4,
        d_species_G=16, G_field_layers=1, G_field_top_k=3,
        prior_std=1.0,
        gnn_backbone=backbone,
    ).to(device)


def train_one(visible, hidden_eval, device, seed, backbone, epochs=300):
    torch.manual_seed(seed)
    T, N = visible.shape
    x = torch.tensor(visible, dtype=torch.float32, device=device).unsqueeze(0)
    model = make_cfg_model(N, backbone, device)
    opt = torch.optim.AdamW(model.parameters(), lr=0.0008, weight_decay=1e-4)
    warmup_epochs = int(0.2 * epochs)
    ramp_epochs = max(1, int(0.2 * epochs))

    def lr_lambda(step):
        if step < 50: return step / 50
        p = (step - 50) / max(1, epochs - 50)
        return 0.5 * (1 + np.cos(np.pi * p))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    train_end = int(0.75 * T)
    best_val = float("inf"); best_state = None; best_epoch = -1

    for epoch in range(epochs):
        if epoch < warmup_epochs:
            h_weight, rollout_K, lam_rollout = 0.0, 0, 0.0
        else:
            post = epoch - warmup_epochs
            h_weight = min(1.0, post / ramp_epochs)
            k_ramp = min(1.0, post / (epochs - warmup_epochs) * 2)
            rollout_K = max(1 if h_weight > 0 else 0, int(round(k_ramp * 3)))
            lam_rollout = 0.5 * h_weight

        model.train(); opt.zero_grad()
        out = model(x, n_samples=2, rollout_K=rollout_K)
        train_out = model.slice_out(out, 0, train_end)
        losses = model.loss(
            train_out, beta_kl=0.03, free_bits=0.02,
            margin_null=0.003, margin_shuf=0.002,
            lam_necessary=5.0, lam_shuffle=3.0,
            lam_energy=2.0, min_energy=0.02,
            lam_smooth=0.02, lam_sparse=0.02,
            h_weight=h_weight, lam_rollout=lam_rollout,
            rollout_weights=(1.0, 0.5, 0.25),
            lam_hf=0.0, lowpass_sigma=6.0,
        )
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()

        with torch.no_grad():
            val_out = model.slice_out(out, train_end, T)
            val_losses = model.loss(
                val_out, h_weight=1.0,
                margin_null=0.003, margin_shuf=0.002,
                lam_energy=2.0, min_energy=0.02,
                lam_rollout=0.5, rollout_weights=(1.0, 0.5, 0.25),
                lam_hf=0.0, lowpass_sigma=6.0,
            )
            val_recon = val_losses["recon_full"].item()
        if epoch > warmup_epochs + 15 and val_recon < best_val:
            best_val = val_recon
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        out_eval = model(x, n_samples=30, rollout_K=3)
        h_mean = out_eval["h_samples"].mean(dim=0)[0].cpu().numpy()
        full = model.loss(
            out_eval, h_weight=1.0,
            margin_null=0.003, margin_shuf=0.002,
            lam_energy=2.0, min_energy=0.02,
            lam_rollout=0.5, rollout_weights=(1.0, 0.5, 0.25),
            lam_hf=0.0, lowpass_sigma=6.0,
        )

    pear, h_scaled = evaluate(h_mean, hidden_eval)
    d = hidden_true_substitution(model, visible, hidden_eval, device)
    return {
        "seed": seed, "backbone": backbone,
        "pearson": pear, "h_mean": h_mean, "h_scaled": h_scaled,
        "val_recon": float(full["recon_full"]),
        "h_var": float(full["h_var"]),
        "d_ratio": d["recon_true_scaled"] / d["recon_encoder"],
        "num_params": sum(p.numel() for p in model.parameters()),
        "best_epoch": best_epoch,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_seeds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=300)
    args = parser.parse_args()
    _configure_matplotlib()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"runs/{ts}_backbone_compare")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_dir}\n")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    seeds = [42, 123, 456, 789, 2024][:args.n_seeds]
    datasets = {"LV": load_lv, "Holling": load_holling}
    backbones = ["softforms", "mlp"]

    results = {b: {ds: [] for ds in datasets} for b in backbones}

    total = len(backbones) * len(datasets) * len(seeds)
    run_i = 0
    for backbone in backbones:
        for ds_key, load_fn in datasets.items():
            visible, hidden = load_fn()
            for seed in seeds:
                run_i += 1
                print(f"[{run_i}/{total}] backbone={backbone}  ds={ds_key}  seed={seed}")
                r = train_one(visible, hidden, device, seed, backbone, args.epochs)
                results[backbone][ds_key].append(r)
                print(f"    P={r['pearson']:+.4f}  d_ratio={r['d_ratio']:.3f}  "
                      f"val={r['val_recon']:.4f}  params={r['num_params']:,}")

    # Summary table
    print(f"\n{'='*90}")
    print(f"{'Backbone':<15}{'Dataset':<12}{'mean P':<12}{'max P':<12}{'std P':<10}"
          f"{'mean d_ratio':<15}{'params':<10}")
    print("="*90)
    for backbone in backbones:
        for ds_key in datasets:
            rs = results[backbone][ds_key]
            pears = np.array([r["pearson"] for r in rs])
            ratios = np.array([r["d_ratio"] for r in rs])
            print(f"{backbone:<15}{ds_key:<12}{pears.mean():<+12.4f}{pears.max():<+12.4f}"
                  f"{pears.std():<10.4f}{ratios.mean():<15.3f}{rs[0]['num_params']:<10,}")

    # Comparison
    print(f"\n{'='*72}\nBackbone comparison deltas\n{'='*72}")
    for ds_key in datasets:
        sf = np.array([r["pearson"] for r in results["softforms"][ds_key]])
        ml = np.array([r["pearson"] for r in results["mlp"][ds_key]])
        print(f"\n{ds_key}:")
        print(f"  SoftForms  mean={sf.mean():+.3f}  max={sf.max():+.3f}")
        print(f"  MLP+hints  mean={ml.mean():+.3f}  max={ml.max():+.3f}")
        print(f"  Δ mean  = {ml.mean() - sf.mean():+.3f}")
        print(f"  Δ max   = {ml.max() - sf.max():+.3f}")

    # Plots
    def save_single(title, plot_fn, path, figsize=(13, 5)):
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        plot_fn(ax)
        fig.suptitle(title, fontsize=13, fontweight="bold")
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)

    for ds_key in datasets:
        _, hidden = datasets[ds_key]()
        def plot_both_backbones(ax, ds_key=ds_key, hidden=hidden):
            ht = hidden
            t_axis = np.arange(len(ht))
            ax.plot(t_axis, ht, color="black", linewidth=2.0, label="真实 hidden", zorder=10)
            sf_rs = results["softforms"][ds_key]
            ml_rs = results["mlp"][ds_key]
            for r in sf_rs:
                h = r["h_scaled"]; L = min(len(h), len(t_axis))
                ax.plot(t_axis[:L], h[:L], color="#1565c0", linewidth=0.8, alpha=0.35)
            for r in ml_rs:
                h = r["h_scaled"]; L = min(len(h), len(t_axis))
                ax.plot(t_axis[:L], h[:L], color="#c62828", linewidth=0.8, alpha=0.35)
            from matplotlib.lines import Line2D
            ax.legend(handles=[
                Line2D([0],[0], color="black", lw=2, label="真实"),
                Line2D([0],[0], color="#1565c0", lw=1.5, label=f"SoftForms (5 seeds)"),
                Line2D([0],[0], color="#c62828", lw=1.5, label=f"MLP+hints (5 seeds)"),
            ], fontsize=11)
            ax.set_xlabel("time"); ax.set_ylabel("hidden")
            ax.grid(alpha=0.3)
        save_single(f"{ds_key}: SoftForms vs MLP+hints", plot_both_backbones,
                     out_dir / f"fig_{ds_key}_compare.png", figsize=(14, 5))

    # Bar plot
    def plot_bars(ax, metric="mean"):
        bks = backbones
        x = np.arange(len(bks))
        w = 0.35
        colors = ["#1565c0", "#c62828"]
        for i, ds_key in enumerate(datasets):
            vals = []
            for b in bks:
                pears = np.array([r["pearson"] for r in results[b][ds_key]])
                vals.append(pears.mean() if metric == "mean" else pears.max())
            ax.bar(x + (i - 0.5) * w, vals, w, color=colors[i], label=ds_key)
            for j, v in enumerate(vals):
                ax.text(x[j] + (i - 0.5) * w, v + 0.02, f"{v:+.3f}",
                        ha="center", fontsize=10)
        ax.set_xticks(x); ax.set_xticklabels(bks)
        ax.set_ylabel(f"{metric} Pearson")
        ax.axhline(0, color="grey", linewidth=0.5)
        ax.legend(fontsize=11); ax.grid(alpha=0.25, axis="y")
    save_single("Mean Pearson", lambda ax: plot_bars(ax, "mean"),
                 out_dir / "fig_mean_bars.png", figsize=(9, 5))
    save_single("Max Pearson", lambda ax: plot_bars(ax, "max"),
                 out_dir / "fig_max_bars.png", figsize=(9, 5))

    # Summary
    with open(out_dir / "summary.md", "w", encoding="utf-8") as f:
        f.write("# SoftForms vs MLP+hints backbone 对比\n\n")
        f.write(f"seeds: {seeds}, epochs={args.epochs}\n\n")
        f.write("## 结果表\n\n")
        f.write("| backbone | dataset | mean P | max P | std P | d_ratio | params |\n|---|---|---|---|---|---|---|\n")
        for backbone in backbones:
            for ds_key in datasets:
                rs = results[backbone][ds_key]
                pears = np.array([r["pearson"] for r in rs])
                ratios = np.array([r["d_ratio"] for r in rs])
                f.write(f"| {backbone} | {ds_key} | {pears.mean():+.4f} | {pears.max():+.4f} | "
                        f"{pears.std():.4f} | {ratios.mean():.3f} | {rs[0]['num_params']:,} |\n")
        f.write("\n## 各 seed 详细\n\n")
        for backbone in backbones:
            for ds_key in datasets:
                f.write(f"### {backbone} / {ds_key}\n\n")
                f.write("| seed | Pearson | d_ratio | val_recon |\n|---|---|---|---|\n")
                for r in results[backbone][ds_key]:
                    f.write(f"| {r['seed']} | {r['pearson']:+.4f} | {r['d_ratio']:.3f} | {r['val_recon']:.4f} |\n")
                f.write("\n")

    print(f"\n[OK] saved to: {out_dir}")


if __name__ == "__main__":
    main()
