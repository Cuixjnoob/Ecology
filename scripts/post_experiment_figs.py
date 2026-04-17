"""Generate comprehensive figures after ANY Beninca experiment.

Two categories:
  A. Intuitive / viewing figures:
      - fig_recovery_overlay.png      (3x3 grid: true vs recovered h per species)
      - fig_summary_bar.png           (config comparison bar chart)
      - fig_stage1b_delta.png         (bar chart of Δ vs Stage 1b per species)
  B. Analytical / data figures:
      - fig_pearson_boxplot.png       (seed distribution per config×species)
      - fig_config_heatmap.png        (mean Pearson matrix)
      - fig_delta_heatmap.png         (improvement over baseline, colored)
      - fig_flatness_correlation.png  (does flatness explain gains/losses?)
      - fig_seed_stability.png        (std across seeds per config)

Usage:
  python -m scripts.post_experiment_figs --exp-dir runs/XXX_beninca_YYY [--rerun-recovery]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


SPECIES_ORDER = ["Cyclopoids", "Calanoids", "Rotifers", "Nanophyto",
                 "Picophyto", "Filam_diatoms", "Ostracods",
                 "Harpacticoids", "Bacteria"]
STAGE1B_REF = {"Cyclopoids": 0.055, "Calanoids": 0.173, "Rotifers": 0.080,
               "Nanophyto": 0.141, "Picophyto": 0.116, "Filam_diatoms": 0.031,
               "Ostracods": 0.272, "Harpacticoids": 0.120, "Bacteria": 0.197}
STAGE1B_MEAN = np.mean(list(STAGE1B_REF.values()))


def load_results(exp_dir: Path):
    """Load raw.json from experiment dir. Normalize structure."""
    with open(exp_dir / "raw.json") as f:
        d = json.load(f)
    # d is {config: {species: [seed_results]}}
    return d


def pearson_matrix(results):
    """Returns (configs, species, mean_matrix, std_matrix)."""
    configs = list(results.keys())
    species_list = SPECIES_ORDER
    M_mean = np.zeros((len(configs), len(species_list)))
    M_std = np.zeros((len(configs), len(species_list)))
    M_all = {}
    for i, cfg in enumerate(configs):
        for j, sp in enumerate(species_list):
            rs = results[cfg].get(sp, [])
            vals = [r["pearson"] for r in rs if r.get("pearson") is not None
                     and not np.isnan(r.get("pearson", np.nan))]
            M_mean[i, j] = np.mean(vals) if vals else np.nan
            M_std[i, j] = np.std(vals) if vals else 0
            M_all[(cfg, sp)] = vals
    return configs, species_list, M_mean, M_std, M_all


def fig_summary_bar(results, out_dir):
    """Bar chart: each config's mean Pearson per species + S1b ref."""
    configs, species_list, M_mean, M_std, _ = pearson_matrix(results)

    fig, ax = plt.subplots(figsize=(13, 6), constrained_layout=True)
    x = np.arange(len(species_list))
    n_cfg = len(configs)
    total_w = 0.85
    w = total_w / (n_cfg + 1)   # +1 for S1b ref

    ref_vals = [STAGE1B_REF[sp] for sp in species_list]
    ax.bar(x - total_w/2 + w * 0.5, ref_vals, w,
           color="#90a4ae", label="Stage 1b ref", alpha=0.85)
    cmap = plt.cm.Set1(np.linspace(0, 1, n_cfg))
    for i, cfg in enumerate(configs):
        pos = x - total_w/2 + w * (i + 1.5)
        ax.bar(pos, M_mean[i], w, yerr=M_std[i], color=cmap[i], label=cfg,
               capsize=2, alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(species_list, rotation=25, fontsize=10)
    ax.set_ylabel("Pearson")
    overall = {cfg: np.nanmean(M_mean[i]) for i, cfg in enumerate(configs)}
    best_cfg = max(overall, key=overall.get)
    ax.set_title(f"Pearson by config × species (S1b ref={STAGE1B_MEAN:+.3f}, "
                 f"best={best_cfg}={overall[best_cfg]:+.3f})",
                 fontweight="bold", fontsize=12)
    ax.legend(fontsize=9, loc="upper left", ncol=2); ax.grid(alpha=0.25, axis="y")
    ax.axhline(STAGE1B_MEAN, color="black", ls=":", alpha=0.5, lw=1)
    fig.savefig(out_dir / "fig_summary_bar.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_stage1b_delta(results, out_dir):
    """Bar chart: Δ over S1b baseline per species, per config."""
    configs, species_list, M_mean, _, _ = pearson_matrix(results)
    delta = np.array([[M_mean[i, j] - STAGE1B_REF[sp]
                        for j, sp in enumerate(species_list)]
                       for i in range(len(configs))])

    fig, ax = plt.subplots(figsize=(13, 6), constrained_layout=True)
    x = np.arange(len(species_list))
    n_cfg = len(configs)
    total_w = 0.80
    w = total_w / n_cfg
    cmap = plt.cm.Set1(np.linspace(0, 1, n_cfg))
    for i, cfg in enumerate(configs):
        pos = x - total_w/2 + w * (i + 0.5)
        colors = ["#2e7d32" if d > 0.005 else ("#c62828" if d < -0.005 else "#9e9e9e")
                  for d in delta[i]]
        ax.bar(pos, delta[i], w, color=colors, edgecolor=cmap[i],
               linewidth=2, label=cfg)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(species_list, rotation=25, fontsize=10)
    ax.set_ylabel("Δ Pearson (vs Stage 1b)")
    ax.set_title("Improvement over Stage 1b per species (green=win, red=loss)",
                 fontweight="bold")
    ax.legend(fontsize=9, ncol=2); ax.grid(alpha=0.25, axis="y")
    fig.savefig(out_dir / "fig_stage1b_delta.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_pearson_boxplot(results, out_dir):
    """Box plot: per config, per species distribution across seeds."""
    configs, species_list, _, _, M_all = pearson_matrix(results)
    n_sp = len(species_list)
    n_cfg = len(configs)

    fig, axes = plt.subplots(3, 3, figsize=(14, 9), constrained_layout=True)
    for ax, sp in zip(axes.flat, species_list):
        data = []
        for cfg in configs:
            data.append(M_all[(cfg, sp)])
        bp = ax.boxplot(data, widths=0.5, patch_artist=True)
        cmap = plt.cm.Set1(np.linspace(0, 1, n_cfg))
        for patch, c in zip(bp['boxes'], cmap):
            patch.set_facecolor(c); patch.set_alpha(0.6)
        ax.axhline(STAGE1B_REF[sp], color="black", ls="--", alpha=0.6,
                   label=f"S1b={STAGE1B_REF[sp]:+.3f}")
        ax.set_xticklabels(configs, rotation=30, fontsize=8)
        ax.set_title(sp, fontsize=10, fontweight="bold")
        ax.set_ylabel("Pearson")
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(alpha=0.25)
    fig.suptitle("Pearson seed distributions per species × config",
                 fontweight="bold", fontsize=12)
    fig.savefig(out_dir / "fig_pearson_boxplot.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_heatmaps(results, out_dir):
    """Two heatmaps: abs mean Pearson + Δ vs S1b."""
    configs, species_list, M_mean, _, _ = pearson_matrix(results)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5), constrained_layout=True)

    # Heatmap 1: absolute Pearson
    ax = axes[0]
    im = ax.imshow(M_mean, aspect="auto", cmap="RdYlGn",
                   vmin=0, vmax=max(0.3, np.nanmax(M_mean)))
    ax.set_xticks(range(len(species_list)))
    ax.set_xticklabels(species_list, rotation=40, fontsize=9)
    ax.set_yticks(range(len(configs))); ax.set_yticklabels(configs, fontsize=9)
    for i in range(len(configs)):
        for j in range(len(species_list)):
            val = M_mean[i, j]
            ax.text(j, i, f"{val:+.2f}", ha="center", va="center",
                    color="black" if abs(val) < 0.15 else "white", fontsize=8)
    ax.set_title("Mean Pearson (absolute)", fontweight="bold")
    plt.colorbar(im, ax=ax, label="Pearson")

    # Heatmap 2: Δ vs S1b
    ax = axes[1]
    delta = np.array([[M_mean[i, j] - STAGE1B_REF[sp]
                        for j, sp in enumerate(species_list)]
                       for i in range(len(configs))])
    vmax = max(0.1, np.nanmax(np.abs(delta)))
    im = ax.imshow(delta, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(species_list)))
    ax.set_xticklabels(species_list, rotation=40, fontsize=9)
    ax.set_yticks(range(len(configs))); ax.set_yticklabels(configs, fontsize=9)
    for i in range(len(configs)):
        for j in range(len(species_list)):
            d = delta[i, j]
            ax.text(j, i, f"{d:+.2f}", ha="center", va="center",
                    color="black" if abs(d) < vmax * 0.5 else "white", fontsize=8)
    ax.set_title("Δ vs Stage 1b (blue=loss, red=win)", fontweight="bold")
    plt.colorbar(im, ax=ax, label="Δ Pearson")

    fig.savefig(out_dir / "fig_heatmaps.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_flatness_correlation(results, out_dir):
    """Scatter: species flatness metric vs best config gain."""
    from scripts.load_beninca import load_beninca
    full, species_all, _ = load_beninca()
    species_all = [str(s) for s in species_all]

    configs, species_list, M_mean, _, _ = pearson_matrix(results)
    overall = {cfg: np.nanmean(M_mean[i]) for i, cfg in enumerate(configs)}
    best_idx = np.argmax([overall[c] for c in configs])
    best_cfg = configs[best_idx]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)

    # Scatter 1: flatness (% time < 10% peak) vs delta over S1b
    flatness = []
    delta = []
    for sp in species_list:
        idx = species_all.index(sp)
        x = full[:, idx]
        flat = (x < 0.10 * x.max()).mean()
        flatness.append(flat)
        j = species_list.index(sp)
        delta.append(M_mean[best_idx, j] - STAGE1B_REF[sp])

    ax = axes[0]
    colors = ["#2e7d32" if d > 0.005 else ("#c62828" if d < -0.005 else "#9e9e9e")
              for d in delta]
    ax.scatter(flatness, delta, c=colors, s=80, edgecolors="black")
    for f, d, sp in zip(flatness, delta, species_list):
        ax.annotate(sp, (f, d), fontsize=8, xytext=(5, 3),
                    textcoords="offset points")
    if len(flatness) >= 3:
        corr = np.corrcoef(flatness, delta)[0, 1]
        ax.set_title(f"Flatness vs Δ (best={best_cfg})  corr={corr:+.2f}",
                     fontweight="bold")
    ax.set_xlabel("flatness: fraction of time <10% peak")
    ax.set_ylabel("Δ Pearson vs S1b (best config)")
    ax.axhline(0, color="k", lw=0.5); ax.grid(alpha=0.3)

    # Scatter 2: CV vs delta
    cvs = []
    for sp in species_list:
        idx = species_all.index(sp)
        x = full[:, idx]
        cvs.append(x.std() / x.mean())

    ax = axes[1]
    ax.scatter(cvs, delta, c=colors, s=80, edgecolors="black")
    for c, d, sp in zip(cvs, delta, species_list):
        ax.annotate(sp, (c, d), fontsize=8, xytext=(5, 3),
                    textcoords="offset points")
    if len(cvs) >= 3:
        corr = np.corrcoef(cvs, delta)[0, 1]
        ax.set_title(f"CV vs Δ (best={best_cfg})  corr={corr:+.2f}",
                     fontweight="bold")
    ax.set_xlabel("CV (std/mean)")
    ax.set_ylabel("Δ Pearson vs S1b (best config)")
    ax.axhline(0, color="k", lw=0.5); ax.grid(alpha=0.3)

    fig.suptitle("Does species burst/flatness explain method improvement?",
                 fontweight="bold", fontsize=12)
    fig.savefig(out_dir / "fig_flatness_correlation.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_seed_stability(results, out_dir):
    """Bar chart of std across seeds (shows method stability)."""
    configs, species_list, _, M_std, _ = pearson_matrix(results)

    fig, ax = plt.subplots(figsize=(13, 5), constrained_layout=True)
    x = np.arange(len(species_list))
    w = 0.8 / len(configs)
    cmap = plt.cm.Set1(np.linspace(0, 1, len(configs)))
    for i, cfg in enumerate(configs):
        pos = x + (i - (len(configs)-1)/2) * w
        ax.bar(pos, M_std[i], w, color=cmap[i], label=cfg, alpha=0.75)
    ax.set_xticks(x); ax.set_xticklabels(species_list, rotation=25, fontsize=10)
    ax.set_ylabel("std across seeds")
    ax.set_title("Method stability: σ(Pearson) across seeds per config × species",
                 fontweight="bold")
    ax.legend(fontsize=9, ncol=2); ax.grid(alpha=0.25, axis="y")
    fig.savefig(out_dir / "fig_seed_stability.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_recovery_overlays(results, out_dir, needs_rerun=False):
    """Generate 3x3 recovery overlay per config using saved h_mean if available.
    If h_mean missing, skip with warning (user can run make_beninca_recovery_figs).
    """
    from scripts._recovery_plot import make_recovery_grid
    from scripts.load_beninca import load_beninca
    full, species, _ = load_beninca()
    species = [str(s) for s in species]

    made_any = False
    for cfg_name, sp_data in results.items():
        # Check if h_mean present in ANY species
        sample = next(iter(sp_data.values()))
        if not sample or not isinstance(sample, list) or not sample[0].get("h_mean"):
            print(f"  [skip] {cfg_name}: no h_mean in raw.json")
            continue

        # Convert to required format
        results_cvt = {}
        for sp, rs in sp_data.items():
            converted = []
            for r in rs:
                rc = dict(r)
                if isinstance(rc.get("h_mean"), list):
                    rc["h_mean"] = np.asarray(rc["h_mean"])
                converted.append(rc)
            results_cvt[sp] = converted

        safe_name = cfg_name.replace(" ", "_").replace("/", "_")
        out_path = out_dir / f"fig_recovery_{safe_name}.png"
        make_recovery_grid(
            results_cvt, full, species, out_path,
            title=f"Beninca recovery: {cfg_name} (3 seeds overlaid, scale-aligned)",
            time_unit="time step (dt=4 days)",
        )
        print(f"  [OK] {out_path.name}")
        made_any = True

    if not made_any:
        print(f"  [!] No config had h_mean saved. To generate overlays:")
        print(f"      python -m scripts.make_beninca_recovery_figs --exp-dir {out_dir.parent}")


def fig_overall_summary(results, out_dir):
    """Single-panel summary for the paper: overall Pearson per config with CI."""
    configs, species_list, M_mean, _, M_all = pearson_matrix(results)

    # Per-config: overall mean + bootstrap CI
    cfg_summary = []
    for cfg in configs:
        all_vals = []
        for sp in species_list:
            all_vals.extend(M_all.get((cfg, sp), []))
        all_vals = np.array(all_vals)
        mu = all_vals.mean()
        # bootstrap 95% CI
        n_boot = 500
        boot = np.array([np.random.choice(all_vals, size=len(all_vals),
                                            replace=True).mean()
                         for _ in range(n_boot)])
        ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
        cfg_summary.append((cfg, mu, ci_lo, ci_hi))

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    labels = [s[0] for s in cfg_summary]
    means = [s[1] for s in cfg_summary]
    ci_lo = [s[2] for s in cfg_summary]
    ci_hi = [s[3] for s in cfg_summary]
    yerr = [[m - lo for m, lo in zip(means, ci_lo)],
            [hi - m for m, hi in zip(means, ci_hi)]]
    cmap = plt.cm.Set1(np.linspace(0, 1, len(labels)))
    ax.bar(range(len(labels)), means, yerr=yerr, color=cmap,
           capsize=5, alpha=0.85, edgecolor="black")
    for i, m in enumerate(means):
        ax.text(i, m + 0.008, f"{m:+.3f}", ha="center", fontweight="bold")
    ax.axhline(STAGE1B_MEAN, color="black", ls="--", alpha=0.7,
               label=f"Stage 1b ref = {STAGE1B_MEAN:+.3f}")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=15, fontsize=10)
    ax.set_ylabel("Mean Pearson (95% bootstrap CI)")
    ax.set_title("Overall configuration comparison", fontweight="bold")
    ax.legend(); ax.grid(alpha=0.25, axis="y")
    fig.savefig(out_dir / "fig_overall_summary.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-dir", type=str, required=True)
    args = ap.parse_args()

    exp_dir = Path(args.exp_dir)
    out_dir = exp_dir / "figures"
    out_dir.mkdir(exist_ok=True)

    print(f"Processing {exp_dir}")
    results = load_results(exp_dir)
    configs = list(results.keys())
    print(f"Configs found: {configs}")
    print(f"Generating 7 figures in {out_dir}...")

    # Essential figures only (per user's directive)
    fig_summary_bar(results, out_dir)
    print(f"  [1/3] fig_summary_bar.png (config x species bar)")
    fig_flatness_correlation(results, out_dir)
    print(f"  [2/3] fig_flatness_correlation.png (explains who wins/loses)")
    print(f"  [3/3] recovery overlays:")
    fig_recovery_overlays(results, out_dir)

    print(f"\n[OK] {out_dir}")


if __name__ == "__main__":
    main()
