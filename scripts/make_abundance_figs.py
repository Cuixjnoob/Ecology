"""Generate abundance-over-time figures for all datasets.

Produces:
  1. Beninca: 9 species in one figure (bloom-bust clear)
  2. Huisman: 6 species + 5 resources
  3. LV: 6 species (hidden highlighted)
  4. Holling: 6 species (hidden highlighted)
  5. Side-by-side: Beninca vs Huisman vs LV (one species each, show dynamics style)
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_beninca():
    from scripts.load_beninca import load_beninca
    full, species, _ = load_beninca()
    species = [str(s) for s in species]
    T = full.shape[0]
    t_axis = np.arange(T) * 4.0  # dt=4 days

    # Separate species and nutrients
    sp_names = ["Cyclopoids", "Calanoids", "Rotifers", "Nanophyto",
                "Picophyto", "Filam_diatoms", "Ostracods",
                "Harpacticoids", "Bacteria"]
    nut_names = ["NO2", "NO3", "NH4", "SRP"]

    fig, axes = plt.subplots(3, 3, figsize=(16, 9), constrained_layout=True,
                               sharex=True)
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for ax, sp in zip(axes.flat, sp_names):
        idx = species.index(sp)
        x = full[:, idx]
        ax.plot(t_axis, x, color=colors[sp_names.index(sp)], lw=0.8)
        ax.set_title(f"{sp}  (max/mean={x.max()/x.mean():.0f}×, "
                     f"<10%peak: {(x < 0.1*x.max()).mean():.0%} of time)",
                     fontsize=10)
        ax.set_ylabel("abundance (norm, mean=1)")
        ax.grid(alpha=0.25)
    axes[2, 1].set_xlabel("day")
    fig.suptitle("Beninca 2008 plankton: 9 species over 7.3 years (bloom-bust dynamics)",
                 fontweight="bold", fontsize=13)

    out_dir = Path("runs/abundance_figs")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "fig_beninca_9species.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # Also: nutrients panel
    fig, axes = plt.subplots(2, 2, figsize=(12, 6), constrained_layout=True,
                               sharex=True)
    for ax, nm in zip(axes.flat, nut_names):
        idx = species.index(nm)
        x = full[:, idx]
        ax.plot(t_axis, x, color="#1976d2", lw=0.8)
        ax.set_title(f"{nm}  CV={x.std()/x.mean():.2f}", fontsize=10)
        ax.set_ylabel("concentration (norm)")
        ax.grid(alpha=0.25)
    for ax in axes[1, :]: ax.set_xlabel("day")
    fig.suptitle("Beninca nutrients (dissolved)", fontweight="bold")
    fig.savefig(out_dir / "fig_beninca_nutrients.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] Beninca figures in {out_dir}")


def plot_huisman():
    data = np.load("runs/huisman1999_chaos/trajectories.npz")
    N_all = data["N_all"]   # (T, 6)
    R = data["resources"]   # (T, 5)
    t_axis = np.arange(N_all.shape[0]) * 2.0   # dt=2 days

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), constrained_layout=True,
                               sharex=True)
    colors_sp = plt.cm.tab10(np.linspace(0, 1, 6))
    ax = axes[0]
    for i in range(6):
        x = N_all[:, i]
        ax.plot(t_axis, x, color=colors_sp[i], lw=1.0, alpha=0.85,
                label=f"sp{i+1}  max/mean={x.max()/x.mean():.1f}")
    ax.set_ylabel("abundance")
    ax.set_title("Huisman 1999 (K41=0.26, λ_Lyap≈0.04): 6 species competing for 5 resources",
                 fontweight="bold", fontsize=12)
    ax.legend(fontsize=9, ncol=3)
    ax.grid(alpha=0.25)

    ax = axes[1]
    colors_r = plt.cm.Set2(np.linspace(0, 1, 5))
    for j in range(5):
        ax.plot(t_axis, R[:, j], color=colors_r[j], lw=1.0, alpha=0.85,
                label=f"R{j+1}")
    ax.set_ylabel("resource concentration")
    ax.set_xlabel("day")
    ax.set_title("Huisman resources (5 limiting)", fontsize=11)
    ax.legend(fontsize=9, ncol=5)
    ax.grid(alpha=0.25)

    out_dir = Path("runs/abundance_figs")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "fig_huisman_full.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Huisman figure")


def plot_synthetic(name: str, path: str, out_dir: Path):
    data = np.load(path)
    visible = data["states_B_5species"]  # (T, 5)
    hidden = data["hidden_B"]             # (T,)
    T = visible.shape[0]
    t_axis = np.arange(T)

    full = np.concatenate([visible, hidden[:, None]], axis=1)   # (T, 6)

    fig, ax = plt.subplots(figsize=(13, 5), constrained_layout=True)
    colors = plt.cm.tab10(np.linspace(0, 1, 6))
    for i in range(6):
        label = f"sp{i+1} (HIDDEN)" if i == 5 else f"sp{i+1}"
        lw = 2.0 if i == 5 else 1.0
        ax.plot(t_axis, full[:, i], color=colors[i], lw=lw,
                alpha=0.9 if i == 5 else 0.7,
                label=label, zorder=5 if i == 5 else 3)
    ax.set_xlabel("time step"); ax.set_ylabel("abundance")
    ax.set_title(f"{name}: 5 visible + 1 hidden (bold)", fontweight="bold")
    ax.legend(fontsize=9, ncol=3)
    ax.grid(alpha=0.25)
    fig.savefig(out_dir / f"fig_{name.lower()}_full.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {name} figure")


def plot_side_by_side():
    """One species from each dataset, same y-scale style, show dynamics type."""
    from scripts.load_beninca import load_beninca
    beninca_full, beninca_species, _ = load_beninca()
    beninca_species = [str(s) for s in beninca_species]

    huisman = np.load("runs/huisman1999_chaos/trajectories.npz")["N_all"]
    lv = np.load("runs/analysis_5vs6_species/trajectories.npz")["states_B_5species"]

    fig, axes = plt.subplots(3, 1, figsize=(13, 7.5), constrained_layout=True)

    # LV (stable)
    ax = axes[0]
    x = lv[:, 0]
    ax.plot(np.arange(len(x)), x, color="#2e7d32", lw=1.0)
    ax.set_title(f"LV (stable, synthetic) — CV={x.std()/x.mean():.2f}",
                 fontweight="bold", fontsize=11, loc="left")
    ax.set_ylabel("abundance"); ax.grid(alpha=0.25)

    # Huisman (synthetic chaos)
    ax = axes[1]
    x = huisman[:, 1]   # sp2 (representative)
    t_h = np.arange(len(x)) * 2.0
    ax.plot(t_h, x, color="#1976d2", lw=1.0)
    ax.set_title(f"Huisman 1999 (synthetic chaos, λ≈0.04) — CV={x.std()/x.mean():.2f}",
                 fontweight="bold", fontsize=11, loc="left")
    ax.set_ylabel("abundance"); ax.grid(alpha=0.25)

    # Beninca (real chaos)
    ax = axes[2]
    idx = beninca_species.index("Nanophyto")   # classic blooming species
    x = beninca_full[:, idx]
    t_b = np.arange(len(x)) * 4.0
    ax.plot(t_b, x, color="#c62828", lw=1.0)
    ax.set_title(f"Beninca 2008 Nanophytoplankton (real chaos, λ≈0.06) — "
                 f"CV={x.std()/x.mean():.2f}, {(x<0.1*x.max()).mean():.0%} <10%peak",
                 fontweight="bold", fontsize=11, loc="left")
    ax.set_ylabel("abundance")
    ax.set_xlabel("day"); ax.grid(alpha=0.25)

    fig.suptitle("Dynamics gradient: stable → synthetic chaos → real chaos (bloom-bust)",
                 fontweight="bold", fontsize=13)

    out_dir = Path("runs/abundance_figs")
    fig.savefig(out_dir / "fig_dynamics_gradient.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] side-by-side figure")


def main():
    out_dir = Path("runs/abundance_figs")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Generating abundance figures...")
    plot_beninca()
    plot_huisman()
    plot_synthetic("LV", "runs/analysis_5vs6_species/trajectories.npz", out_dir)
    plot_synthetic("Holling", "runs/20260413_100414_5vs6_holling/trajectories.npz", out_dir)
    plot_side_by_side()
    print(f"\n[OK] All figures in {out_dir}")


if __name__ == "__main__":
    main()
