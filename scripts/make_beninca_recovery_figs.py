"""Post-hoc: generate recovery overlay figures for best config.

Reruns each species with best config (found from event_weight or log_huber),
single best seed, saves h_mean. Then plots:
  - 3×3 grid of true vs recovered h per species (scale-aligned)
  - Per-species Pearson bar chart with improvement over S1b
  - Event weight heatmap (when event weighting is used)

Usage:
  python -m scripts.make_beninca_recovery_figs \\
      --exp-dir runs/XXXX_beninca_event_weight \\
      [--config "log+huber+event 1"]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.load_beninca import load_beninca
from scripts.cvhi_beninca_event_weight import (
    train_one, SPECIES_ORDER, BEST_HP
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-dir", type=str, required=True,
                     help="Path to experiment dir with raw.json")
    ap.add_argument("--config", type=str, default=None,
                     help="Config name; if None, auto-pick best")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42],
                     help="Seeds to generate overlay for")
    args = ap.parse_args()

    exp_dir = Path(args.exp_dir)
    with open(exp_dir / "raw.json") as f:
        results = json.load(f)

    # Pick best config if not specified
    if args.config is None:
        config_means = {}
        for cfg, sp_data in results.items():
            vals = []
            for sp, rs in sp_data.items():
                Ps = [r["pearson"] for r in rs if not np.isnan(r.get("pearson", np.nan))]
                if Ps:
                    vals.append(np.mean(Ps))
            config_means[cfg] = np.mean(vals) if vals else float("nan")
        args.config = max(config_means, key=lambda c: config_means[c])
        print(f"Auto-picked best config: {args.config} (mean P={config_means[args.config]:+.3f})")

    # Config settings — parse from config name
    use_log = "log" in args.config
    use_huber = "huber" in args.config
    event_alpha = 0.0
    if "event 2" in args.config:
        event_alpha = 2.0
    elif "event 1" in args.config or "event a=1" in args.config:
        event_alpha = 1.0

    print(f"Settings: log={use_log}, huber={use_huber}, event_alpha={event_alpha}")

    out_dir = exp_dir / "figures"
    out_dir.mkdir(exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    full, species, _ = load_beninca()
    species = [str(s) for s in species]

    # Rerun for each species with seed 42 to get h_mean
    recovered = {}
    for sp in SPECIES_ORDER:
        h_idx = species.index(sp)
        visible_raw = np.delete(full, h_idx, axis=1).astype(np.float32)
        hidden_raw = full[:, h_idx].astype(np.float32)
        print(f"Rerunning {sp}...", end=" ")
        r = train_one(visible_raw, hidden_raw, args.seeds[0], device,
                       use_log, use_huber, event_alpha)
        print(f"P={r['pearson']:+.3f}")
        recovered[sp] = (r["h_mean"], r["pearson"], hidden_raw)

    # Figure 1: 3x3 grid of true vs recovered h
    fig, axes = plt.subplots(3, 3, figsize=(16, 10), constrained_layout=True)
    stage1b_ref = {"Cyclopoids": 0.055, "Calanoids": 0.173, "Rotifers": 0.080,
                    "Nanophyto": 0.141, "Picophyto": 0.116, "Filam_diatoms": 0.031,
                    "Ostracods": 0.272, "Harpacticoids": 0.120, "Bacteria": 0.197}
    for ax, sp in zip(axes.flat, SPECIES_ORDER):
        h_mean, pearson, true_h = recovered[sp]
        t_axis = np.arange(len(true_h))
        ax.plot(t_axis, true_h, color="black", lw=1.5, label="true", zorder=10)
        # Scale-aligned
        L = min(len(h_mean), len(true_h))
        a, b = np.polyfit(h_mean[:L], true_h[:L], 1)
        h_scaled = a * h_mean[:L] + b
        ax.plot(t_axis[:L], h_scaled, color="#c62828", lw=1.0, alpha=0.85,
                label=f"recovered P={pearson:+.3f}")
        s1b = stage1b_ref.get(sp, float("nan"))
        delta = pearson - s1b
        color = "green" if delta > 0.01 else ("red" if delta < -0.01 else "gray")
        ax.set_title(f"{sp}  P={pearson:+.3f}  (S1b={s1b:+.3f}, Δ{delta:+.3f})",
                     color=color, fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
    fig.suptitle(f"Beninca hidden recovery: {args.config}",
                 fontweight="bold", fontsize=13)
    fig.savefig(out_dir / f"fig_recovery_{args.config.replace(' ', '_')}.png",
                dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out_dir / f'fig_recovery_{args.config.replace(chr(32), chr(95))}.png'}")

    # Figure 2: Event weight visualization (if used)
    if event_alpha > 0:
        fig, axes = plt.subplots(3, 3, figsize=(16, 10), constrained_layout=True)
        for ax, sp in zip(axes.flat, SPECIES_ORDER):
            h_idx = species.index(sp)
            visible = np.delete(full, h_idx, axis=1)
            # Compute log-ratio magnitudes (what event_weight uses)
            safe = np.maximum(visible, 1e-6)
            lr = np.log(safe[1:] / safe[:-1])
            mag = np.linalg.norm(lr, axis=-1)   # (T-1,)
            w = (mag + 1e-6) ** event_alpha
            w = w / w.mean()   # normalize to mean=1
            t_axis = np.arange(len(w))
            ax.fill_between(t_axis, 0, w, color="#c62828", alpha=0.4)
            ax.plot(t_axis, w, color="#c62828", lw=0.8)
            ax.axhline(1.0, color="k", ls="--", alpha=0.5, label="uniform=1")
            ax.set_title(f"{sp}  max weight={w.max():.1f}  frac>2={(w>2).mean():.1%}",
                         fontsize=10)
            ax.set_ylabel("loss weight")
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8)
        axes[2, 1].set_xlabel("time")
        fig.suptitle(f"Event weights (α={event_alpha}): where gradient is focused",
                     fontweight="bold")
        fig.savefig(out_dir / f"fig_event_weights_{args.config.replace(' ', '_')}.png",
                    dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] event weight figure")

    # Figure 3: summary bar — best config vs baselines across species
    fig, ax = plt.subplots(figsize=(12, 5.5), constrained_layout=True)
    x = np.arange(len(SPECIES_ORDER))
    w = 0.4
    phase2 = {"Cyclopoids": 0.064, "Calanoids": 0.159, "Rotifers": 0.056,
              "Nanophyto": 0.204, "Picophyto": 0.062, "Filam_diatoms": 0.056,
              "Ostracods": 0.125, "Harpacticoids": 0.129, "Bacteria": 0.171}
    pearsons = [recovered[sp][1] for sp in SPECIES_ORDER]
    ax.bar(x - w/2, [stage1b_ref[sp] for sp in SPECIES_ORDER], w,
           color="#90a4ae", label="Stage 1b baseline")
    ax.bar(x + w/2, pearsons, w, color="#c62828",
           label=f"{args.config}")
    for i, p in enumerate(pearsons):
        ax.text(i + w/2, p + 0.01, f"{p:.2f}", ha="center", fontsize=9)
    overall_base = np.mean([stage1b_ref[sp] for sp in SPECIES_ORDER])
    overall_ours = np.mean(pearsons)
    ax.set_xticks(x); ax.set_xticklabels(SPECIES_ORDER, rotation=25, fontsize=10)
    ax.set_ylabel("Pearson")
    ax.set_title(f"Beninca: S1b={overall_base:+.3f} → {args.config}={overall_ours:+.3f} "
                 f"(Δ={overall_ours - overall_base:+.3f})",
                 fontweight="bold")
    ax.legend(); ax.grid(alpha=0.25, axis="y")
    fig.savefig(out_dir / f"fig_bar_{args.config.replace(' ', '_')}.png",
                dpi=130, bbox_inches="tight")
    plt.close(fig)

    print(f"\n[OK] All figures in {out_dir}")


if __name__ == "__main__":
    main()
