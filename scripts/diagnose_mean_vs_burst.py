"""Quantitatively diagnose: is model learning mean or bursts?

Decomposes recovered h(t) into low-freq (trend) and high-freq (bursts),
compares separately with true h's components. Answers user's question:
"mean 学得基本对" — is this really what's happening?

Method:
  true_h = LP(true_h) + HP(true_h)          [low-pass + high-pass]
  rec_h  = LP(rec_h) + HP(rec_h)
  Compute:
    Pearson(LP_rec, LP_true)   — trend alignment
    Pearson(HP_rec, HP_true)   — detail/burst alignment
    Pearson(rec@flat_t, true@flat_t)    — flat-region only
    Pearson(rec@burst_t, true@burst_t)  — burst-region only
    std(rec)/std(true)          — amplitude retention
    mean(rec) vs mean(true)     — baseline alignment

If LP_corr > HP_corr substantially: confirms "learning mean/trend only".
If flat_corr > burst_corr substantially: confirms "ignoring bursts".
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter1d

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.load_beninca import load_beninca


SPECIES_ORDER = ["Cyclopoids", "Calanoids", "Rotifers", "Nanophyto",
                 "Picophyto", "Filam_diatoms", "Ostracods",
                 "Harpacticoids", "Bacteria"]
STAGE1B_REF = {"Cyclopoids": 0.055, "Calanoids": 0.173, "Rotifers": 0.080,
               "Nanophyto": 0.141, "Picophyto": 0.116, "Filam_diatoms": 0.031,
               "Ostracods": 0.272, "Harpacticoids": 0.120, "Bacteria": 0.197}


def pearson(a, b):
    a = np.asarray(a) - np.mean(a); b = np.asarray(b) - np.mean(b)
    denom = np.sqrt((a*a).sum() * (b*b).sum())
    return float((a * b).sum() / (denom + 1e-12))


def decompose_freq(x, sigma=20):
    """Return (low_freq, high_freq) via Gaussian smoothing."""
    lp = gaussian_filter1d(x.astype(float), sigma=sigma, mode="nearest")
    hp = x - lp
    return lp, hp


def analyze_species(h_mean_list, true_h, name):
    """Multi-faceted diagnosis. Returns dict of metrics."""
    # Average h_mean across seeds (scale-aligned first)
    # Actually, we process each seed separately and average metrics.
    metrics_per_seed = []
    for h_raw in h_mean_list:
        h_arr = np.asarray(h_raw)
        L = min(len(h_arr), len(true_h))
        h = h_arr[:L]
        t = true_h[:L]

        # Scale-align: fit h to t linearly
        a, b = np.polyfit(h, t, 1)
        h_aligned = a * h + b

        # Overall Pearson
        p_all = pearson(h_aligned, t)

        # Frequency decomposition
        t_lp, t_hp = decompose_freq(t, sigma=20)
        h_lp, h_hp = decompose_freq(h_aligned, sigma=20)
        p_lp = pearson(h_lp, t_lp)
        p_hp = pearson(h_hp, t_hp)

        # Amplitude
        std_ratio = h_aligned.std() / (t.std() + 1e-8)

        # Burst / flat separation
        threshold = np.percentile(np.abs(t - t.mean()), 80)   # top 20% = burst
        burst_mask = np.abs(t - t.mean()) > threshold
        flat_mask = ~burst_mask

        p_flat = pearson(h_aligned[flat_mask], t[flat_mask]) if flat_mask.sum() > 10 else float("nan")
        p_burst = pearson(h_aligned[burst_mask], t[burst_mask]) if burst_mask.sum() > 10 else float("nan")

        # RMSE per segment
        rmse_flat = np.sqrt(((h_aligned[flat_mask] - t[flat_mask])**2).mean())
        rmse_burst = np.sqrt(((h_aligned[burst_mask] - t[burst_mask])**2).mean())

        # Constant baseline: predict h = mean(t) always
        p_baseline = pearson(np.ones_like(t) * t.mean(), t)   # = 0 mathematically, sanity check

        metrics_per_seed.append({
            "p_all": p_all,
            "p_low_freq": p_lp,      # trend match
            "p_high_freq": p_hp,     # burst match
            "p_flat_seg": p_flat,    # flat region
            "p_burst_seg": p_burst,  # burst region
            "std_ratio": std_ratio,
            "rmse_flat": rmse_flat,
            "rmse_burst": rmse_burst,
        })

    # Average across seeds
    keys = metrics_per_seed[0].keys()
    avg = {k: float(np.nanmean([m[k] for m in metrics_per_seed]))
           for k in keys}
    return avg


def main():
    exp_dir = Path("runs/20260415_214355_beninca_hdyn")
    out_dir = Path("runs/mean_vs_burst_diagnosis")
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(exp_dir / "raw.json") as f:
        data = json.load(f)

    full, species, _ = load_beninca()
    species = [str(s) for s in species]

    print("="*110)
    print("MEAN vs BURST DIAGNOSIS (h_dyn λ=0.3 experiment)")
    print("="*110)
    print(f"{'Species':<14}{'P_all':<9}{'P_LP':<9}{'P_HP':<9}{'P_flat':<9}"
          f"{'P_burst':<9}{'std_r':<9}{'rmse_F':<10}{'rmse_B':<10}")
    print("-"*110)

    all_metrics = {}
    for sp in SPECIES_ORDER:
        h_idx = species.index(sp)
        true_h = full[:, h_idx]
        h_list = [r["h_mean"] for r in data[sp] if "h_mean" in r and r["h_mean"]]
        if not h_list:
            print(f"{sp:<14} (no h_mean)")
            continue
        metrics = analyze_species(h_list, true_h, sp)
        all_metrics[sp] = metrics
        print(f"{sp:<14}{metrics['p_all']:<+9.3f}"
              f"{metrics['p_low_freq']:<+9.3f}{metrics['p_high_freq']:<+9.3f}"
              f"{metrics['p_flat_seg']:<+9.3f}{metrics['p_burst_seg']:<+9.3f}"
              f"{metrics['std_ratio']:<9.3f}"
              f"{metrics['rmse_flat']:<10.3f}{metrics['rmse_burst']:<10.3f}")

    # Overall summary
    p_lp_mean = np.mean([m["p_low_freq"] for m in all_metrics.values()])
    p_hp_mean = np.mean([m["p_high_freq"] for m in all_metrics.values()])
    p_flat_mean = np.mean([m["p_flat_seg"] for m in all_metrics.values() if not np.isnan(m["p_flat_seg"])])
    p_burst_mean = np.mean([m["p_burst_seg"] for m in all_metrics.values() if not np.isnan(m["p_burst_seg"])])
    std_ratio_mean = np.mean([m["std_ratio"] for m in all_metrics.values()])

    print("-"*110)
    print(f"{'MEAN':<14}{'':<9}"
          f"{p_lp_mean:<+9.3f}{p_hp_mean:<+9.3f}"
          f"{p_flat_mean:<+9.3f}{p_burst_mean:<+9.3f}"
          f"{std_ratio_mean:<9.3f}")

    print(f"\n{'='*70}")
    print("DIAGNOSIS")
    print('='*70)
    print(f"  Low-freq (trend) match:  P_LP   = {p_lp_mean:+.3f}")
    print(f"  High-freq (burst) match: P_HP   = {p_hp_mean:+.3f}")
    print(f"  Flat-segment match:       P_flat = {p_flat_mean:+.3f}")
    print(f"  Burst-segment match:      P_burst = {p_burst_mean:+.3f}")
    print(f"  Amplitude retained:       {std_ratio_mean:.1%} of true amplitude")
    print()
    if p_lp_mean > p_hp_mean + 0.1:
        print(f"  ==> Model predominantly captures TREND (low-freq) but misses BURSTS")
        print(f"  ==> User's observation CONFIRMED: learning mean, not burst")
    elif p_lp_mean < p_hp_mean + 0.05:
        print(f"  ==> Model captures trend AND bursts similarly (not mean-only)")
    else:
        print(f"  ==> Mixed: trend slightly dominates burst recovery")

    # Plot per-species breakdown
    fig, ax = plt.subplots(figsize=(12, 5.5), constrained_layout=True)
    x = np.arange(len(SPECIES_ORDER))
    w = 0.2
    p_lps = [all_metrics[sp]["p_low_freq"] for sp in SPECIES_ORDER]
    p_hps = [all_metrics[sp]["p_high_freq"] for sp in SPECIES_ORDER]
    p_flats = [all_metrics[sp]["p_flat_seg"] for sp in SPECIES_ORDER]
    p_bursts = [all_metrics[sp]["p_burst_seg"] for sp in SPECIES_ORDER]

    ax.bar(x - 1.5*w, p_lps, w, color="#1976d2", label="LP corr (trend)")
    ax.bar(x - 0.5*w, p_hps, w, color="#c62828", label="HP corr (bursts)")
    ax.bar(x + 0.5*w, p_flats, w, color="#4caf50", label="flat-segment corr", alpha=0.7)
    ax.bar(x + 1.5*w, p_bursts, w, color="#ff9800", label="burst-segment corr", alpha=0.7)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xticks(x); ax.set_xticklabels(SPECIES_ORDER, rotation=25, fontsize=9)
    ax.set_ylabel("Pearson")
    ax.set_title(f"Mean vs burst diagnosis: LP>HP means model learns mean only\n"
                 f"Overall: LP={p_lp_mean:+.3f}, HP={p_hp_mean:+.3f}, "
                 f"flat={p_flat_mean:+.3f}, burst={p_burst_mean:+.3f}",
                 fontweight="bold")
    ax.legend(fontsize=9, ncol=2); ax.grid(alpha=0.25, axis="y")
    fig.savefig(out_dir / "fig_mean_vs_burst.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # Per-species time-series decomposition
    fig, axes = plt.subplots(3, 3, figsize=(16, 10), constrained_layout=True)
    for ax, sp in zip(axes.flat, SPECIES_ORDER):
        h_idx = species.index(sp)
        true_h = full[:, h_idx]
        h_list = [r["h_mean"] for r in data[sp] if "h_mean" in r and r["h_mean"]]
        if not h_list:
            continue
        # Pick best seed
        best_pearson = -1
        best_h = None
        for h_raw in h_list:
            h_arr = np.asarray(h_raw)
            L = min(len(h_arr), len(true_h))
            p = pearson(h_arr[:L], true_h[:L])
            if abs(p) > best_pearson:
                best_pearson = abs(p)
                best_h = h_arr

        L = min(len(best_h), len(true_h))
        h = best_h[:L]; t = true_h[:L]
        a, b = np.polyfit(h, t, 1)
        h_aligned = a * h + b

        # Decompose
        t_lp, t_hp = decompose_freq(t, sigma=20)
        h_lp, h_hp = decompose_freq(h_aligned, sigma=20)

        time = np.arange(L)
        ax.plot(time, t, color="black", lw=1.2, label="true", alpha=0.8)
        ax.plot(time, h_aligned, color="#c62828", lw=0.9, label="recovered", alpha=0.8)
        ax.plot(time, t_lp, color="#1976d2", lw=1.5, linestyle="--",
                label="true LP (trend)", alpha=0.9)
        ax.plot(time, h_lp, color="#ff9800", lw=1.0, linestyle="--",
                label="rec LP (trend)", alpha=0.9)
        p = all_metrics[sp]
        ax.set_title(f"{sp}  P_all={p['p_all']:+.2f}  P_LP={p['p_low_freq']:+.2f}  "
                     f"P_HP={p['p_high_freq']:+.2f}", fontsize=10)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.25)
    fig.suptitle("Low-freq (dashed) vs full: is model just tracking the trend?",
                 fontweight="bold")
    fig.savefig(out_dir / "fig_lp_hp_decomposition.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # Save
    with open(out_dir / "diagnosis.md", "w", encoding="utf-8") as f:
        f.write("# Mean vs burst diagnosis\n\n")
        f.write("Decomposing recovered h(t) into low-pass (trend) + high-pass (bursts).\n\n")
        f.write("| Species | P_all | P_LP | P_HP | P_flat | P_burst | std ratio |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for sp in SPECIES_ORDER:
            m = all_metrics[sp]
            f.write(f"| {sp} | {m['p_all']:+.3f} | {m['p_low_freq']:+.3f} | "
                    f"{m['p_high_freq']:+.3f} | {m['p_flat_seg']:+.3f} | "
                    f"{m['p_burst_seg']:+.3f} | {m['std_ratio']:.2f} |\n")
        f.write(f"\n**Overall**: LP={p_lp_mean:+.3f}, HP={p_hp_mean:+.3f}, "
                f"flat={p_flat_mean:+.3f}, burst={p_burst_mean:+.3f}, "
                f"amplitude={std_ratio_mean:.1%}\n")

    print(f"\n[OK] {out_dir}")


if __name__ == "__main__":
    main()
