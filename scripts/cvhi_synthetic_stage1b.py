"""Stage 1b config on synthetic LV + Holling — clean CVHI, no eco priors.

Config: Stage 1b clean (RMSE log + input dropout + G_anchor_first), NO MTE, NO stoich sign.
Purpose: sanity check current method on synthetic; compare to 2026-04-13 20-seed baseline
  (LV mean 0.727 / Holling mean 0.738).
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
from scripts.cvhi_residual_synthetic_comparison import load_lv, load_holling
from scripts.train_utils_fast import train_one_fast


SEEDS = [42, 123, 456]
EPOCHS = 300


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


def run_dataset(name, visible, hidden, device, out_dir):
    print(f"\n{'='*72}\n{name}  (T={visible.shape[0]}, N={visible.shape[1]})\n{'='*72}")
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
                # Stage 1b clean additions
                lam_rmse_log=0.1,
                input_dropout_prob=0.05,
                # No eco priors
                lam_mte_prior=0.0,
                lam_mte_shape=0.0,
                lam_stoich_sign=0.0,
                # No EMA/snapshot for pure config
                use_compile=True,
                use_ema=False,
                use_snapshot_ensemble=False,
            )
            dt = (datetime.now() - t0).total_seconds()
            print(f"  seed={s}  P={r['pearson']:+.4f}  d_ratio={r['d_ratio']:.3f}  "
                  f"val={r['val_recon']:.4f}  ({dt:.1f}s)")
        except Exception as e:
            print(f"  seed={s}  FAILED: {e}")
            import traceback; traceback.print_exc()
            r = {"pearson": float("nan"), "val_recon": float("nan"),
                 "d_ratio": float("nan"), "h_var": float("nan"), "h_mean": None}
        r["seed"] = s
        results.append(r)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Plot
    fig, ax = plt.subplots(figsize=(13, 4.5), constrained_layout=True)
    t_axis = np.arange(len(hidden))
    ax.plot(t_axis, hidden, color="black", linewidth=2.0, label="true hidden", zorder=10)
    colors = plt.cm.viridis(np.linspace(0, 1, len(results)))
    for i, r in enumerate(results):
        if r.get("h_mean") is None:
            continue
        h = np.asarray(r["h_mean"])
        # scale-invariant align via linear regression
        L = min(len(h), len(hidden))
        x, y = h[:L], hidden[:L]
        a, b = np.polyfit(x, y, 1)
        h_sc = a * x + b
        ax.plot(t_axis[:L], h_sc, color=colors[i], linewidth=1.1, alpha=0.85,
                label=f"seed {r['seed']} (P={r['pearson']:+.3f})")
    pears = np.array([r["pearson"] for r in results if not np.isnan(r["pearson"])])
    ax.set_title(f"{name}: CVHI-Residual Stage 1b clean  (mean P={pears.mean():+.3f})")
    ax.set_xlabel("time"); ax.set_ylabel("hidden")
    ax.legend(fontsize=10); ax.grid(alpha=0.25)
    fig.savefig(out_dir / f"fig_{name}_overlay.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    return results


def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"runs/{ts}_synthetic_stage1b_clean")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_dir}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    lv_vis, lv_hid = load_lv()
    ho_vis, ho_hid = load_holling()

    all_results = {
        "LV": run_dataset("LV", lv_vis, lv_hid, device, out_dir),
        "Holling": run_dataset("Holling", ho_vis, ho_hid, device, out_dir),
    }

    # Summary
    OLD_BASELINE = {"LV": (0.727, 0.919), "Holling": (0.738, 0.955)}

    print(f"\n{'='*72}\nSUMMARY (vs 2026-04-13 20-seed baseline)\n{'='*72}")
    print(f"{'Dataset':<12}{'mean P':<12}{'max P':<12}{'std P':<10}"
          f"{'mean d_r':<12}{'old mean':<12}{'old max':<12}")
    for ds, rs in all_results.items():
        P = np.array([r["pearson"] for r in rs if not np.isnan(r["pearson"])])
        D = np.array([r["d_ratio"] for r in rs if not np.isnan(r["d_ratio"])])
        om, oM = OLD_BASELINE[ds]
        print(f"{ds:<12}{P.mean():<+12.4f}{P.max():<+12.4f}{P.std():<10.4f}"
              f"{D.mean():<12.3f}{om:<+12.3f}{oM:<+12.3f}")

    with open(out_dir / "summary.md", "w", encoding="utf-8") as f:
        f.write("# Synthetic LV + Holling — CVHI Stage 1b clean (no eco priors)\n\n")
        f.write(f"Seeds: {SEEDS}, epochs={EPOCHS}\n\n")
        f.write("Config: RMSE log (0.1) + input dropout (0.05) + G_anchor_first(+1), ")
        f.write("NO MTE, NO stoich sign, NO EMA, NO snapshot.\n\n")
        f.write("## Results\n\n")
        f.write("| Dataset | mean P | max P | std P | mean d_ratio | "
                "old 20-seed mean | old 20-seed max |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for ds, rs in all_results.items():
            P = np.array([r["pearson"] for r in rs if not np.isnan(r["pearson"])])
            D = np.array([r["d_ratio"] for r in rs if not np.isnan(r["d_ratio"])])
            om, oM = OLD_BASELINE[ds]
            f.write(f"| {ds} | {P.mean():+.4f} | {P.max():+.4f} | {P.std():.4f} | "
                    f"{D.mean():.3f} | {om:+.3f} | {oM:+.3f} |\n")
        f.write("\n## Per-seed\n\n")
        for ds, rs in all_results.items():
            f.write(f"### {ds}\n\n| seed | Pearson | d_ratio | val_recon | h_var |\n")
            f.write("|---|---|---|---|---|\n")
            for r in rs:
                f.write(f"| {r['seed']} | {r['pearson']:+.4f} | {r['d_ratio']:.3f} | "
                        f"{r['val_recon']:.4f} | {r.get('h_var', float('nan')):.3f} |\n")
            f.write("\n")

    # JSON dump (strip h_mean arrays for size)
    serializable = {}
    for ds, rs in all_results.items():
        serializable[ds] = []
        for r in rs:
            rs2 = {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                   for k, v in r.items() if k not in ("h_mean",)}
            serializable[ds].append(rs2)
    with open(out_dir / "raw.json", "w") as f:
        json.dump(serializable, f, indent=2, default=float)

    print(f"\n[OK] saved to: {out_dir}")


if __name__ == "__main__":
    main()
