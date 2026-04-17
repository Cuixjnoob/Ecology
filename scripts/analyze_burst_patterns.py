"""Analyze burst patterns: why Beninca is so spiky vs synthetic data.

Compares:
  - Beninca (real chaos)
  - Huisman (synthetic chaos)
  - LV (stable synthetic)

Metrics:
  - Coefficient of variation (CV = std/mean)
  - Skewness (asymmetry)
  - Max/mean ratio (burst intensity)
  - 95th percentile / 50th percentile (upper tail heaviness)
  - Autocorrelation decay (how quickly bursts are independent)
  - Zero-crossing rate (how often signal rises/falls)
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats


def burst_stats(x: np.ndarray) -> dict:
    """Compute burst-related statistics on a time series."""
    x = np.asarray(x, dtype=float)
    return dict(
        mean=float(x.mean()),
        std=float(x.std()),
        cv=float(x.std() / (x.mean() + 1e-8)),
        skewness=float(stats.skew(x)),
        kurtosis=float(stats.kurtosis(x)),
        max_over_mean=float(x.max() / (x.mean() + 1e-8)),
        p95_over_p50=float(np.percentile(x, 95) / (np.percentile(x, 50) + 1e-8)),
        zero_crossings=int(np.sum(np.diff(np.sign(x - x.mean())) != 0)),
    )


def main():
    out_dir = Path("runs/burst_analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load Beninca
    from scripts.load_beninca import load_beninca
    beninca_full, beninca_species, _ = load_beninca()
    beninca_species = [str(s) for s in beninca_species]

    # Load Huisman
    huisman_data = np.load("runs/huisman1999_chaos/trajectories.npz")
    huisman_full = huisman_data["N_all"]  # (T, 6)
    huisman_names = [f"sp{i+1}" for i in range(6)]

    # Load LV
    lv_data = np.load("runs/analysis_5vs6_species/trajectories.npz")
    lv_vis = lv_data["states_B_5species"]
    lv_hidden = lv_data["hidden_B"]
    lv_full = np.concatenate([lv_vis, lv_hidden[:, None]], axis=1)
    lv_names = [f"sp{i+1}" for i in range(6)]

    # Compute stats per channel
    print("="*90)
    print("BURST STATISTICS: how spiky is each system?")
    print("="*90)
    print(f"{'System':<10}{'Species':<16}{'CV':<8}{'skew':<8}{'kurt':<8}"
          f"{'max/mean':<12}{'p95/p50':<10}")

    all_stats = {}

    print("\n--- BENINCA (real) ---")
    beninca_stats = {}
    for i, sp in enumerate(beninca_species[:9]):   # just species, skip nutrients
        s = burst_stats(beninca_full[:, i])
        beninca_stats[sp] = s
        print(f"{'Beninca':<10}{sp:<16}{s['cv']:<8.2f}{s['skewness']:<8.2f}"
              f"{s['kurtosis']:<8.2f}{s['max_over_mean']:<12.2f}"
              f"{s['p95_over_p50']:<10.2f}")

    print("\n--- HUISMAN (synthetic chaos) ---")
    huisman_stats = {}
    for i, sp in enumerate(huisman_names):
        s = burst_stats(huisman_full[:, i])
        huisman_stats[sp] = s
        print(f"{'Huisman':<10}{sp:<16}{s['cv']:<8.2f}{s['skewness']:<8.2f}"
              f"{s['kurtosis']:<8.2f}{s['max_over_mean']:<12.2f}"
              f"{s['p95_over_p50']:<10.2f}")

    print("\n--- LV (stable) ---")
    lv_stats = {}
    for i, sp in enumerate(lv_names):
        s = burst_stats(lv_full[:, i])
        lv_stats[sp] = s
        print(f"{'LV':<10}{sp:<16}{s['cv']:<8.2f}{s['skewness']:<8.2f}"
              f"{s['kurtosis']:<8.2f}{s['max_over_mean']:<12.2f}"
              f"{s['p95_over_p50']:<10.2f}")

    # Compute mean stats per system
    print("\n" + "="*90)
    print("MEAN PER SYSTEM")
    print("="*90)
    print(f"{'System':<10}{'mean CV':<10}{'mean skew':<12}{'mean kurt':<12}"
          f"{'mean max/mean':<15}")
    for name, data_stats in [("LV", lv_stats), ("Huisman", huisman_stats),
                              ("Beninca", beninca_stats)]:
        cvs = [s["cv"] for s in data_stats.values()]
        skews = [s["skewness"] for s in data_stats.values()]
        kurts = [s["kurtosis"] for s in data_stats.values()]
        mxs = [s["max_over_mean"] for s in data_stats.values()]
        print(f"{name:<10}{np.mean(cvs):<10.2f}{np.mean(skews):<12.2f}"
              f"{np.mean(kurts):<12.2f}{np.mean(mxs):<15.2f}")

    # Visualizations: side-by-side
    fig, axes = plt.subplots(3, 3, figsize=(15, 9), constrained_layout=True)

    # Row 0: Beninca samples
    for j, sp in enumerate(["Bacteria", "Ostracods", "Nanophyto"]):
        idx = beninca_species.index(sp)
        x = beninca_full[:, idx]
        ax = axes[0, j]
        ax.plot(x, color="#c62828", lw=0.9)
        ax.axhline(x.mean(), color="black", ls=":", alpha=0.5, label=f"mean")
        s = beninca_stats[sp]
        ax.set_title(f"Beninca {sp}  CV={s['cv']:.1f}  skew={s['skewness']:.1f}",
                     fontsize=10)
        ax.set_ylabel("abundance")
        ax.grid(alpha=0.25)

    # Row 1: Huisman samples
    for j, sp_i in enumerate([1, 3, 5]):   # sp2, sp4, sp6
        x = huisman_full[:, sp_i]
        ax = axes[1, j]
        ax.plot(x, color="#1976d2", lw=0.9)
        ax.axhline(x.mean(), color="black", ls=":", alpha=0.5)
        s = huisman_stats[huisman_names[sp_i]]
        ax.set_title(f"Huisman {huisman_names[sp_i]}  CV={s['cv']:.1f}  "
                     f"skew={s['skewness']:.1f}", fontsize=10)
        ax.set_ylabel("abundance")
        ax.grid(alpha=0.25)

    # Row 2: LV samples
    for j, sp_i in enumerate([0, 2, 4]):
        x = lv_full[:, sp_i]
        ax = axes[2, j]
        ax.plot(x, color="#2e7d32", lw=0.9)
        ax.axhline(x.mean(), color="black", ls=":", alpha=0.5)
        s = lv_stats[lv_names[sp_i]]
        ax.set_title(f"LV {lv_names[sp_i]}  CV={s['cv']:.1f}  "
                     f"skew={s['skewness']:.1f}", fontsize=10)
        ax.set_ylabel("abundance")
        ax.set_xlabel("time")
        ax.grid(alpha=0.25)

    fig.suptitle("Burst patterns: Beninca (real) vs Huisman (synthetic chaos) vs LV",
                 fontweight="bold")
    fig.savefig(out_dir / "fig_burst_comparison.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # Distribution comparison
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    for ax, (name, data_arr, color) in zip(axes, [
        ("Beninca", beninca_full[:, :9].flatten() / beninca_full[:, :9].mean(), "#c62828"),
        ("Huisman", huisman_full.flatten() / huisman_full.mean(), "#1976d2"),
        ("LV", lv_full.flatten() / lv_full.mean(), "#2e7d32"),
    ]):
        ax.hist(data_arr, bins=80, color=color, alpha=0.7,
                density=True, range=(0, max(5, np.percentile(data_arr, 99))))
        sk = stats.skew(data_arr); krt = stats.kurtosis(data_arr)
        ax.set_title(f"{name}: skew={sk:.1f}, kurt={krt:.1f}")
        ax.set_xlabel("abundance / mean")
        ax.set_ylabel("density")
        ax.grid(alpha=0.25)
    fig.suptitle("Distribution shape: Beninca is heavy-tailed (bursty)",
                 fontweight="bold")
    fig.savefig(out_dir / "fig_distribution.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    print(f"\n[OK] {out_dir}")


if __name__ == "__main__":
    main()
