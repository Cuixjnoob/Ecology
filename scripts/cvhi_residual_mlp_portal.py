"""MLP backbone (with formula hints) on Portal OT, vs previous SoftForms results.

红线: 无 hidden supervision, 无 anchor.
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
from scipy.stats import spearmanr

from models.cvhi_residual import CVHI_Residual
from scripts.cvhi_residual_L1L3_diagnostics import (
    load_portal, evaluate, _configure_matplotlib, hidden_true_substitution,
)


def make_model(N, backbone, device):
    return CVHI_Residual(
        num_visible=N,
        encoder_d=48, encoder_blocks=2, encoder_heads=4,
        takens_lags=(1, 2, 4, 8, 12), encoder_dropout=0.2,
        d_species_f=20, f_visible_layers=2, f_visible_top_k=4,
        d_species_G=12, G_field_layers=1, G_field_top_k=3,
        prior_std=1.0,
        gnn_backbone=backbone,
    ).to(device)


def train_one(visible, hidden_eval, device, seed, backbone, epochs=300):
    torch.manual_seed(seed)
    T, N = visible.shape
    x = torch.tensor(visible, dtype=torch.float32, device=device).unsqueeze(0)
    model = make_model(N, backbone, device)
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

    margin_null, margin_shuf, min_energy = 0.002, 0.001, 0.05
    history = {"recon": [], "m_null": [], "h_var": []}

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
            margin_null=margin_null, margin_shuf=margin_shuf,
            lam_necessary=5.0, lam_shuffle=3.0,
            lam_energy=2.0, min_energy=min_energy,
            lam_smooth=0.02, lam_sparse=0.02,
            h_weight=h_weight, lam_rollout=lam_rollout,
            rollout_weights=(1.0, 0.5, 0.25),
            lam_hf=0.0, lowpass_sigma=6.0,
        )
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        history["recon"].append(losses["recon_full"].item())
        history["m_null"].append(losses["margin_null_obs"].item())
        history["h_var"].append(losses["h_var"].item())

        with torch.no_grad():
            val_out = model.slice_out(out, train_end, T)
            val_losses = model.loss(
                val_out, h_weight=1.0,
                margin_null=margin_null, margin_shuf=margin_shuf,
                lam_energy=2.0, min_energy=min_energy,
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
            margin_null=margin_null, margin_shuf=margin_shuf,
            lam_energy=2.0, min_energy=min_energy,
            lam_rollout=0.5, rollout_weights=(1.0, 0.5, 0.25),
            lam_hf=0.0, lowpass_sigma=6.0,
        )
    pear, h_scaled = evaluate(h_mean, hidden_eval)
    d = hidden_true_substitution(model, visible, hidden_eval, device)
    return {
        "seed": seed, "backbone": backbone, "pearson": pear,
        "h_mean": h_mean, "h_scaled": h_scaled,
        "val_recon": float(full["recon_full"]),
        "m_null": float(full["margin_null_obs"]),
        "h_var": float(full["h_var"]),
        "d_ratio": d["recon_true_scaled"] / d["recon_encoder"],
        "recon_null": d["recon_null"],
        "recon_encoder": d["recon_encoder"],
        "num_params": sum(p.numel() for p in model.parameters()),
        "best_epoch": best_epoch,
        "history": history,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_seeds", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--hidden", type=str, default="OT")
    args = parser.parse_args()
    _configure_matplotlib()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"runs/{ts}_mlp_portal_{args.hidden}")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_dir}\n")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    seeds = [42, 123, 456, 789, 2024, 31415, 27182, 65537][:args.n_seeds]
    visible, hidden = load_portal(args.hidden)
    print(f"Portal hidden={args.hidden}, T={visible.shape[0]}, N_visible={visible.shape[1]}\n")

    results = {"softforms": [], "mlp": []}
    backbones = ["softforms", "mlp"]
    total = len(backbones) * len(seeds)
    i = 0
    for backbone in backbones:
        print(f"\n=== backbone = {backbone} ===")
        for seed in seeds:
            i += 1
            print(f"[{i}/{total}] seed={seed}", end="  ")
            r = train_one(visible, hidden, device, seed, backbone, args.epochs)
            results[backbone].append(r)
            print(f"P={r['pearson']:+.4f}  d_ratio={r['d_ratio']:.3f}  "
                  f"val={r['val_recon']:.4f}  h_var={r['h_var']:.3f}")

    # Summary
    print(f"\n{'='*85}")
    print(f"{'Backbone':<15}{'mean P':<12}{'max P':<12}{'std P':<10}"
          f"{'mean d_ratio':<15}{'params':<10}")
    print("="*85)
    for backbone in backbones:
        pears = np.array([r["pearson"] for r in results[backbone]])
        ratios = np.array([r["d_ratio"] for r in results[backbone]])
        print(f"{backbone:<15}{pears.mean():<+12.4f}{pears.max():<+12.4f}"
              f"{pears.std():<10.4f}{ratios.mean():<15.3f}"
              f"{results[backbone][0]['num_params']:<10,}")

    # Spearman (val_recon vs Pearson)
    print(f"\n{'Spearman ρ (val_recon vs Pearson)':=^60}")
    for backbone in backbones:
        vals = np.array([r["val_recon"] for r in results[backbone]])
        pears = np.array([r["pearson"] for r in results[backbone]])
        rho, p = spearmanr(vals, pears)
        print(f"  {backbone:<15s}  ρ={rho:+.3f}  (p={p:.3f})")

    # Reference baselines on Portal
    print(f"\n{'Reference'.center(60, '=')}")
    print(f"  Linear Sparse+EM (supervised projection):  0.353")
    print(f"  CVHI original (anchor-based):              0.33 ± 0.21 (5 seeds)")
    print(f"  CVHI_Residual SoftForms (pre-L1):          0.146 ± 0.081")
    print(f"  CVHI_Residual SoftForms + L1only_K3:       0.180 ± 0.063")

    # Plot
    def save_single(title, plot_fn, path, figsize=(13, 6)):
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        plot_fn(ax)
        fig.suptitle(title, fontsize=13, fontweight="bold")
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)

    def plot_both(ax):
        t_axis = np.arange(len(hidden))
        ax.plot(t_axis, hidden, color="black", linewidth=2.0, label=f"真实 {args.hidden}", zorder=10)
        for r in results["softforms"]:
            h = r["h_scaled"]; L = min(len(h), len(t_axis))
            ax.plot(t_axis[:L], h[:L], color="#1565c0", linewidth=0.7, alpha=0.35)
        for r in results["mlp"]:
            h = r["h_scaled"]; L = min(len(h), len(t_axis))
            ax.plot(t_axis[:L], h[:L], color="#c62828", linewidth=0.7, alpha=0.35)
        from matplotlib.lines import Line2D
        ax.legend(handles=[
            Line2D([0],[0], color="black", lw=2, label=f"真实 {args.hidden}"),
            Line2D([0],[0], color="#1565c0", lw=1.5, label=f"SoftForms ({len(seeds)} seeds)"),
            Line2D([0],[0], color="#c62828", lw=1.5, label=f"MLP+hints ({len(seeds)} seeds)"),
        ], fontsize=11)
        ax.set_xlabel("time (months)"); ax.set_ylabel("OT abundance")
        ax.grid(alpha=0.25)
    save_single(f"Portal {args.hidden}: SoftForms vs MLP+hints", plot_both,
                 out_dir / "fig_compare_overlay.png")

    # Bar
    def plot_bar(ax):
        x = np.arange(2)
        means = [np.mean([r["pearson"] for r in results[b]]) for b in backbones]
        maxes = [np.max([r["pearson"] for r in results[b]]) for b in backbones]
        stds = [np.std([r["pearson"] for r in results[b]]) for b in backbones]
        ax.bar(x - 0.2, means, 0.4, yerr=stds, capsize=4, color="#1565c0", label="mean ± std")
        ax.bar(x + 0.2, maxes, 0.4, color="#ff7f0e", label="max")
        ax.set_xticks(x); ax.set_xticklabels(backbones)
        ax.set_ylabel("Pearson to hidden_true")
        ax.axhline(0.353, color="red", linestyle="--", alpha=0.5,
                    label="Linear supervised baseline = 0.353")
        ax.legend(fontsize=11); ax.grid(alpha=0.25, axis="y")
        for i, (m, mx) in enumerate(zip(means, maxes)):
            ax.text(i - 0.2, m + 0.01, f"{m:+.3f}", ha="center", fontsize=10)
            ax.text(i + 0.2, mx + 0.01, f"{mx:+.3f}", ha="center", fontsize=10)
    save_single(f"Portal {args.hidden} — {len(seeds)} seeds", plot_bar,
                 out_dir / "fig_bars.png", figsize=(10, 6))

    # Summary md
    with open(out_dir / "summary.md", "w", encoding="utf-8") as f:
        f.write(f"# Portal {args.hidden} — MLP vs SoftForms\n\n")
        f.write(f"seeds: {seeds}, epochs={args.epochs}\n\n")
        f.write("## 结果\n\n")
        f.write("| Backbone | mean P | max P | std P | mean d_ratio |\n|---|---|---|---|---|\n")
        for backbone in backbones:
            pears = np.array([r["pearson"] for r in results[backbone]])
            ratios = np.array([r["d_ratio"] for r in results[backbone]])
            f.write(f"| {backbone} | {pears.mean():+.4f} | {pears.max():+.4f} | "
                    f"{pears.std():.4f} | {ratios.mean():.3f} |\n")
        f.write(f"\n## 各 seed 详细\n\n")
        for backbone in backbones:
            f.write(f"### {backbone}\n\n")
            f.write("| seed | Pearson | d_ratio | val_recon | h_var |\n|---|---|---|---|---|\n")
            for r in results[backbone]:
                f.write(f"| {r['seed']} | {r['pearson']:+.4f} | {r['d_ratio']:.3f} | "
                        f"{r['val_recon']:.4f} | {r['h_var']:.3f} |\n")
            f.write("\n")

    print(f"\n[OK] saved to: {out_dir}")


if __name__ == "__main__":
    main()
