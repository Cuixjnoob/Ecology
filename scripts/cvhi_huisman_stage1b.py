"""CVHI-Residual Stage 1b clean on Huisman 1999 chaos data.

Goal: fill the "synthetic chaos" data point between LV/Holling (0.75-0.84)
and Beninca real chaos (0.13). Validate paper narrative that
hidden-recovery Pearson degrades with attractor chaos strength.

Config: same as cvhi_synthetic_stage1b.py (RMSE log + input dropout +
G_anchor_first, no eco priors). Rotate each of 6 species as hidden.
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
from scripts.train_utils_fast import train_one_fast


SEEDS = [42, 123, 456]
EPOCHS = 300
HUISMAN_PATH = Path("runs/huisman1999_chaos/trajectories.npz")


def load_huisman():
    """Load Huisman 1999 data, per-channel normalized."""
    d = np.load(HUISMAN_PATH)
    N_all = d["N_all"].astype(np.float32)     # (T, 6)
    R_all = d["resources"].astype(np.float32)  # (T, 5)
    # Concat: species 1-6 + resources 1-5 → total 11 channels
    full = np.concatenate([N_all, R_all], axis=1)   # (T, 11)
    # Per-channel normalize to mean = 1 (match Beninca convention)
    full = full / (full.mean(axis=0, keepdims=True) + 1e-8)
    return full.astype(np.float32)


def make_model(N, device):
    return CVHI_Residual(
        num_visible=N,
        encoder_d=64, encoder_blocks=2, encoder_heads=4,
        takens_lags=(1, 2, 4, 8), encoder_dropout=0.1,
        d_species_f=20, f_visible_layers=2, f_visible_top_k=4,
        d_species_G=12, G_field_layers=1, G_field_top_k=3,
        prior_std=1.0, gnn_backbone="mlp",
        use_formula_hints=True, use_G_field=True,
        num_mixture_components=1,
        G_anchor_first=True, G_anchor_sign=+1,
        hierarchical_h=False,
    ).to(device)


def run_hidden(full, hidden_idx, device, out_dir):
    """Train CVHI with hidden = channel hidden_idx, rotate over seeds."""
    visible = np.delete(full, hidden_idx, axis=1)
    hidden = full[:, hidden_idx]
    results = []
    for s in SEEDS:
        torch.manual_seed(s)
        model = make_model(visible.shape[1], device)
        t0 = datetime.now()
        try:
            r = train_one_fast(
                model, visible, hidden, device=device,
                epochs=EPOCHS, lr=0.0008,
                beta_kl=0.03, free_bits=0.02,
                margin_null=0.003, margin_shuf=0.002,
                lam_necessary=5.0, lam_shuffle=3.0,
                lam_energy=2.0, min_energy=0.02,
                lam_smooth=0.02, lam_sparse=0.02,
                lam_rollout=0.5, rollout_weights=(1.0, 0.5, 0.25),
                lam_hf=0.0,
                lam_rmse_log=0.1, input_dropout_prob=0.05,
                use_compile=True, use_ema=False, use_snapshot_ensemble=False,
            )
            dt = (datetime.now() - t0).total_seconds()
            print(f"    seed={s}  P={r['pearson']:+.4f}  d_r={r['d_ratio']:.2f}  ({dt:.1f}s)")
        except Exception as e:
            print(f"    seed={s}  FAILED: {e}")
            import traceback; traceback.print_exc()
            r = {"pearson": float("nan"), "d_ratio": float("nan"),
                 "val_recon": float("nan"), "h_mean": None}
        r["seed"] = s
        # Keep h_mean as np array for plotting
        results.append(r)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return results


def plot_recovery_per_species(all_results, full, species_names, out_dir):
    """Plot each hidden rotation: true vs recovered h(t), overlaid seeds."""
    n = len(species_names)
    fig, axes = plt.subplots(n, 1, figsize=(13, 2.2 * n), constrained_layout=True)
    if n == 1:
        axes = [axes]
    for i, (name, ax) in enumerate(zip(species_names, axes)):
        true_h = full[:, i]
        t_axis = np.arange(len(true_h))
        ax.plot(t_axis, true_h, color="black", lw=1.8, label="true hidden", zorder=10)
        rs = all_results[name]
        P_list = [r["pearson"] for r in rs if not np.isnan(r["pearson"])]
        colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(rs)))
        for r, c in zip(rs, colors):
            h = r.get("h_mean")
            if h is None:
                continue
            h = np.asarray(h)
            L = min(len(h), len(true_h))
            # Scale-invariant align via linear regression
            x_, y_ = h[:L], true_h[:L]
            a, b = np.polyfit(x_, y_, 1)
            h_sc = a * x_ + b
            ax.plot(t_axis[:L], h_sc, color=c, lw=1.0, alpha=0.75,
                    label=f"seed {r['seed']}  P={r['pearson']:+.3f}")
        mean_P = np.mean(P_list) if P_list else float("nan")
        ax.set_title(f"{name}  (mean Pearson = {mean_P:+.3f})", fontsize=11)
        ax.set_ylabel(f"{name} abundance")
        ax.legend(fontsize=8, ncol=len(rs)+1, loc="upper right")
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("time (day)")
    fig.suptitle("Huisman 1999 chaos: CVHI hidden recovery (6 rotations)",
                 fontsize=13, fontweight="bold")
    fig.savefig(out_dir / "fig_recovery_per_species.png", dpi=130,
                bbox_inches="tight")
    plt.close(fig)


def plot_raw_abundance(full, species_names, out_dir):
    """Plot the raw Huisman trajectories (species + resources)."""
    T, N = full.shape
    t_axis = np.arange(T)
    fig, axes = plt.subplots(2, 1, figsize=(13, 6.5), constrained_layout=True)

    ax = axes[0]
    for i in range(len(species_names)):
        ax.plot(t_axis, full[:, i], label=species_names[i], lw=1.1, alpha=0.85)
    ax.set_ylabel("species abundance (normalized)")
    ax.set_title("Huisman 1999: 6 species dynamics (chaos)")
    ax.legend(fontsize=9, ncol=6); ax.grid(alpha=0.25)

    ax = axes[1]
    for j in range(N - len(species_names)):
        ax.plot(t_axis, full[:, len(species_names) + j], label=f"R{j+1}",
                lw=1.1, alpha=0.85)
    ax.set_ylabel("resource (normalized)")
    ax.set_xlabel("time (day)")
    ax.set_title("Huisman 1999: 5 resource dynamics")
    ax.legend(fontsize=9, ncol=5); ax.grid(alpha=0.25)

    fig.savefig(out_dir / "fig_raw_abundance.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_chaos_gradient(overall_mean, overall_max, out_dir):
    """Plot chaos gradient across datasets."""
    datasets = ["LV\n(synthetic)", "Holling\n(synthetic)",
                "Huisman 1999\n(synthetic chaos)", "Beninca\n(real chaos)"]
    lambdas = [0.00, 0.01, 0.04, 0.06]
    mean_p = [0.755, 0.843, overall_mean, 0.132]
    max_p = [0.81, 0.90, overall_max, 0.272]

    fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    x = np.arange(len(datasets))
    w = 0.35
    b1 = ax.bar(x - w/2, mean_p, w, color="#1976d2", label="mean Pearson")
    b2 = ax.bar(x + w/2, max_p, w, color="#c62828", label="max Pearson")
    for bar, v in zip(b1, mean_p):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.015, f"{v:.3f}",
                ha="center", fontsize=10)
    for bar, v in zip(b2, max_p):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.015, f"{v:.3f}",
                ha="center", fontsize=10)
    ax.set_xticks(x); ax.set_xticklabels(datasets, fontsize=10)
    ax.set_ylabel("Pearson correlation")
    ax.set_title("Hidden recovery performance vs chaos strength\n"
                 "(Huisman fills synthetic-chaos gap between Holling and Beninca)",
                 fontweight="bold")
    ax.set_ylim(0, 1.0)
    # Annotate Lyapunov
    for i, lam in enumerate(lambdas):
        ax.text(i, -0.055, f"λ≈{lam}", ha="center", fontsize=9,
                color="#555", style="italic")
    ax.grid(alpha=0.25, axis="y")
    ax.legend(fontsize=11)
    fig.savefig(out_dir / "fig_chaos_gradient.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"runs/{ts}_huisman_stage1b")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_dir}")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    full = load_huisman()
    print(f"Huisman data: T={full.shape[0]}, channels={full.shape[1]} "
          f"(6 species + 5 resources)")

    # Rotate through 6 species (indices 0..5) as hidden
    species_names = [f"sp{i+1}" for i in range(6)]
    all_results = {}
    for i, name in enumerate(species_names):
        print(f"\n--- hidden = {name} (channel idx {i}) ---")
        all_results[name] = run_hidden(full, i, device, out_dir)

    # Summary
    print(f"\n{'='*72}\nHUISMAN 1999 CHAOS RESULTS\n{'='*72}")
    print(f"{'species':<10}{'mean P':<12}{'max P':<12}{'std':<10}{'mean d_r':<12}")
    pears_all = []
    for name in species_names:
        rs = all_results[name]
        P = np.array([r["pearson"] for r in rs if not np.isnan(r["pearson"])])
        D = np.array([r["d_ratio"] for r in rs if not np.isnan(r["d_ratio"])])
        print(f"{name:<10}{P.mean():<+12.4f}{P.max():<+12.4f}"
              f"{P.std():<10.4f}{D.mean():<12.2f}")
        pears_all.extend(P.tolist())

    overall_mean = np.mean(pears_all)
    overall_max = np.max(pears_all)
    print(f"\nOverall mean Pearson: {overall_mean:+.4f}")
    print(f"Overall max Pearson:  {overall_max:+.4f}")

    # Context comparison
    print(f"\n{'='*72}\nContext (chaos gradient):\n{'='*72}")
    print(f"  Synthetic LV (low chaos):       mean 0.755, max 0.81")
    print(f"  Synthetic Holling (mild chaos): mean 0.843, max 0.90")
    print(f"  Huisman 1999 (strong chaos):    mean {overall_mean:+.3f}, max {overall_max:+.3f}")
    print(f"  Real Beninca plankton:          mean 0.132, max 0.27")

    # Plots
    print(f"\nGenerating figures...")
    plot_raw_abundance(full, species_names, out_dir)
    print(f"  [OK] fig_raw_abundance.png")
    plot_recovery_per_species(all_results, full, species_names, out_dir)
    print(f"  [OK] fig_recovery_per_species.png")
    plot_chaos_gradient(overall_mean, overall_max, out_dir)
    print(f"  [OK] fig_chaos_gradient.png")

    # Save
    with open(out_dir / "summary.md", "w", encoding="utf-8") as f:
        f.write("# Huisman 1999 chaos - CVHI Stage 1b clean\n\n")
        f.write(f"Seeds: {SEEDS}, Epochs: {EPOCHS}\n\n")
        f.write("Config: RMSE log (0.1) + input dropout (0.05) + G_anchor_first, ")
        f.write("NO MTE/stoich priors.\n\n")
        f.write("## Results per hidden species\n\n")
        f.write("| species | mean P | max P | std | mean d_ratio |\n|---|---|---|---|---|\n")
        for name in species_names:
            rs = all_results[name]
            P = np.array([r["pearson"] for r in rs if not np.isnan(r["pearson"])])
            D = np.array([r["d_ratio"] for r in rs if not np.isnan(r["d_ratio"])])
            f.write(f"| {name} | {P.mean():+.4f} | {P.max():+.4f} | "
                    f"{P.std():.4f} | {D.mean():.2f} |\n")
        f.write(f"\n**Overall**: mean P = {overall_mean:+.4f}, max = {overall_max:+.4f}\n\n")
        f.write("## Chaos gradient (cross-dataset)\n\n")
        f.write("| Dataset | chaos strength | mean Pearson | max Pearson |\n")
        f.write("|---|---|---|---|\n")
        f.write("| LV (synthetic) | λ~0 | 0.755 | 0.81 |\n")
        f.write("| Holling (synthetic) | λ~0.01 | 0.843 | 0.90 |\n")
        f.write(f"| **Huisman 1999** | λ~0.03-0.05 | **{overall_mean:+.4f}** | **{overall_max:+.4f}** |\n")
        f.write("| Beninca (real) | λ~0.05-0.07 | 0.132 | 0.27 |\n")

    def to_serial(v):
        if isinstance(v, (int, float, np.floating)):
            return float(v)
        if isinstance(v, np.ndarray):
            return v.tolist()
        return v
    dump = {name: [{k: to_serial(v)
                    for k, v in r.items() if k != "h_mean"}
                   for r in rs] for name, rs in all_results.items()}
    with open(out_dir / "raw.json", "w") as f:
        json.dump(dump, f, indent=2, default=float)
    print(f"\n[OK] {out_dir}")


if __name__ == "__main__":
    main()
