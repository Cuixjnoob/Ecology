"""Reusable recovery overlay plotting utility.

Standard plot: 3x3 grid showing true h vs recovered h per species,
scale-aligned via linear regression. Color-coded title shows Δ vs Stage 1b.

Usage in any experiment script:
  from scripts._recovery_plot import make_recovery_grid
  make_recovery_grid(results, full_data, species_list, out_path,
                      title="My experiment name")
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


SPECIES_ORDER_DEFAULT = ["Cyclopoids", "Calanoids", "Rotifers", "Nanophyto",
                         "Picophyto", "Filam_diatoms", "Ostracods",
                         "Harpacticoids", "Bacteria"]
STAGE1B_REF = {"Cyclopoids": 0.055, "Calanoids": 0.173, "Rotifers": 0.080,
               "Nanophyto": 0.141, "Picophyto": 0.116, "Filam_diatoms": 0.031,
               "Ostracods": 0.272, "Harpacticoids": 0.120, "Bacteria": 0.197}


def make_recovery_grid(
    results: Dict,
    full_data: np.ndarray,
    species_names: List[str],
    out_path: Union[str, Path],
    title: str = "",
    stage1b_ref: Dict = None,
    time_unit: str = "time step",
    grid_shape: tuple = (3, 3),
):
    """Create standard recovery overlay plot (3x3 default).

    Args:
      results: {species_name: [seed_run_dict, ...]}
               Each seed dict must have 'h_mean' (np.ndarray) and 'pearson' (float)
      full_data: (T, N_total) all channels including hidden
      species_names: list of species in data column order (for finding each hidden)
      out_path: output png path
      title: super-title
      stage1b_ref: dict mapping species->S1b Pearson, default Beninca values
      time_unit: xlabel
      grid_shape: (nrows, ncols)
    """
    if stage1b_ref is None:
        stage1b_ref = STAGE1B_REF

    species_list = list(results.keys())
    nr, nc = grid_shape
    fig, axes = plt.subplots(nr, nc, figsize=(16, 9), constrained_layout=True)
    axes_flat = axes.flat if nr * nc > 1 else [axes]

    for ax, sp in zip(axes_flat, species_list):
        # Find species column in full_data
        if sp in species_names:
            h_idx = species_names.index(sp)
            true_h = full_data[:, h_idx]
        else:
            ax.text(0.5, 0.5, f"missing {sp}", ha="center", va="center",
                    transform=ax.transAxes)
            continue

        t_axis = np.arange(len(true_h))
        ax.plot(t_axis, true_h, color="black", lw=1.2, label="true", zorder=10)

        rs = results[sp]
        pearsons = []
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, max(1, len(rs))))
        for r, c in zip(rs, colors):
            hm = r.get("h_mean")
            if hm is None:
                continue
            hm = np.asarray(hm)
            if not isinstance(hm, np.ndarray):
                continue
            L = min(len(hm), len(true_h))
            if L < 10:
                continue
            # Scale-aligned (linear regression)
            try:
                a, b = np.polyfit(hm[:L], true_h[:L], 1)
            except Exception:
                continue
            h_scaled = a * hm[:L] + b
            p = r.get("pearson", float("nan"))
            pearsons.append(p)
            ax.plot(t_axis[:L], h_scaled, color=c, lw=0.8, alpha=0.7)

        mP = float(np.mean(pearsons)) if pearsons else float("nan")
        s1b = stage1b_ref.get(sp, float("nan"))
        delta = mP - s1b
        if delta > 0.01:
            tc = "green"
        elif delta < -0.01:
            tc = "red"
        else:
            tc = "gray"
        ax.set_title(f"{sp}  P={mP:+.3f}  (S1b={s1b:+.3f}, Δ{delta:+.3f})",
                     color=tc, fontsize=10, fontweight="bold")
        ax.grid(alpha=0.25)
        ax.set_ylabel(sp)

    # X label only on bottom row
    for ax in axes_flat:
        ax.set_xlabel(time_unit)

    if title:
        fig.suptitle(title, fontweight="bold", fontsize=13)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def make_recovery_grid_from_raw_json(
    raw_path: Union[str, Path],
    config_name: str,
    full_data: np.ndarray,
    species_names: List[str],
    out_path: Union[str, Path],
    title: str = "",
):
    """Load from raw.json and make grid. Raw.json format: {config: {species: [runs]}}."""
    import json
    with open(raw_path) as f:
        d = json.load(f)
    if config_name not in d:
        raise KeyError(f"config '{config_name}' not in raw.json "
                       f"(available: {list(d.keys())})")
    # Convert h_mean lists back to arrays
    results = {}
    for sp, rs in d[config_name].items():
        converted = []
        for r in rs:
            rc = dict(r)
            if "h_mean" in rc and isinstance(rc["h_mean"], list):
                rc["h_mean"] = np.asarray(rc["h_mean"])
            converted.append(rc)
        results[sp] = converted

    make_recovery_grid(results, full_data, species_names, out_path,
                        title=title or f"Recovery: {config_name}")
