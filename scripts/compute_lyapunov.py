"""Compute maximum Lyapunov exponent for our synthetic datasets.

Two methods:
  1. Benettin-style perturbation: run ODE twice with 1e-8 initial difference,
     log-slope of |diff(t)| gives λ_max.
  2. Rosenstein (data-only): from single time series, via delay embedding.

Reference: Beninca 2008 Nature reports λ ≈ 0.051-0.066/day for 9 plankton
species. We check if our Huisman chaos has similar magnitude.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
from scipy.integrate import odeint

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# Method 1: Benettin-style perturbation on ODE
# ============================================================

def lyapunov_benettin_ode(derivs_fn, y0, t_max, dt=0.1, eps=1e-8,
                           renorm_every=10, n_skip=500):
    """Run two trajectories, renormalize perturbation, track log-divergence.

    Returns: λ_max (average log-divergence rate per unit time)
    """
    y0 = np.asarray(y0, dtype=float)
    y0_p = y0 + eps * np.random.randn(len(y0))
    # Normalize initial perturbation
    delta0 = y0_p - y0
    delta0 = delta0 / np.linalg.norm(delta0) * eps
    y0_p = y0 + delta0

    t_vec = np.arange(0, t_max + dt, dt)
    # Integrate both
    sol1 = odeint(derivs_fn, y0, t_vec, rtol=1e-9, atol=1e-11, mxstep=5000)
    sol2 = odeint(derivs_fn, y0_p, t_vec, rtol=1e-9, atol=1e-11, mxstep=5000)

    # Use renorm-based Benettin algorithm via post-hoc reconstruction
    # Track divergence, renormalize periodically
    lyap_log = []
    log_sum = 0.0
    n_steps = len(t_vec)
    current_y = y0.copy()
    current_yp = y0_p.copy()

    # Simpler approach: just compute log of distance and fit slope after transient
    dist = np.linalg.norm(sol1 - sol2, axis=1)   # (T,)
    log_dist = np.log(dist + 1e-20)

    # Skip transient n_skip steps, fit linear on log
    t_fit = t_vec[n_skip:]
    log_fit = log_dist[n_skip:]
    # Fit over region BEFORE saturation (saturation = attractor width ~ O(1))
    saturated = log_fit > np.log(0.5)   # saturated when distance reaches ~0.5
    if saturated.any():
        end_idx = np.argmax(saturated)
        if end_idx < 30:
            end_idx = min(len(log_fit), 200)
    else:
        end_idx = len(log_fit)

    t_use = t_fit[:end_idx]
    log_use = log_fit[:end_idx]

    if len(t_use) < 10:
        return float("nan"), t_vec, log_dist

    # Linear fit: log(dist) = λ*t + const
    coeffs = np.polyfit(t_use, log_use, 1)
    lam = float(coeffs[0])
    return lam, t_vec, log_dist


# ============================================================
# LV derivs (5+1)
# ============================================================

def make_lv_derivs_from_data():
    """We don't have LV ODE, but we can estimate λ from data via Rosenstein."""
    return None


# ============================================================
# Holling derivs (not easily re-created from data; use Rosenstein)
# ============================================================


# ============================================================
# Huisman derivs (reuse from generate script)
# ============================================================

from scripts.generate_huisman1999 import derivs as huisman_derivs, S_VEC, N_SPECIES


def huisman_y0():
    N0 = np.array([0.1 + (i + 1) / 100.0 for i in range(N_SPECIES)])
    R0 = S_VEC.copy().astype(float)
    # Run transient first
    y0 = np.concatenate([N0, R0])
    t_transient = np.arange(0, 1000, 0.5)
    sol = odeint(huisman_derivs, y0, t_transient, rtol=1e-8, atol=1e-10,
                  mxstep=5000)
    return sol[-1]   # post-transient point on attractor


# ============================================================
# Method 2: Rosenstein (data only)
# ============================================================

def lyapunov_rosenstein(x, dt=1.0, lag=5, emb_dim=6, n_iter=40,
                         n_neighbors=5):
    """Rosenstein's algorithm for Lyapunov from univariate time series.

    Returns: λ_max estimate
    """
    x = np.asarray(x, dtype=float)
    T = len(x)
    embed_len = T - (emb_dim - 1) * lag
    if embed_len < 100:
        return float("nan"), None, None

    # Build delay embedding
    embedded = np.zeros((embed_len, emb_dim))
    for i in range(emb_dim):
        embedded[:, i] = x[i * lag : i * lag + embed_len]

    # For each point, find nearest neighbor that's temporally separated
    # (Theiler window = mean period or ~10 steps)
    theiler = 10

    # For speed, only sample 500 points
    n_samples = min(500, embed_len - n_iter)
    sample_idx = np.random.choice(embed_len - n_iter, n_samples, replace=False)

    # Compute divergences
    log_div = np.zeros(n_iter)
    counts = np.zeros(n_iter)

    from scipy.spatial.distance import cdist
    for i in sample_idx:
        # Distances from point i to all others
        dists = cdist(embedded[i:i+1], embedded).flatten()
        # Exclude temporal neighbors
        dists[max(0, i - theiler):min(embed_len, i + theiler + 1)] = np.inf
        # Also exclude points that can't propagate n_iter steps
        dists[embed_len - n_iter:] = np.inf
        j = int(np.argmin(dists))
        if dists[j] == np.inf or dists[j] < 1e-10:
            continue
        # Track divergence
        for k in range(n_iter):
            d = np.linalg.norm(embedded[i + k] - embedded[j + k])
            if d > 1e-10:
                log_div[k] += np.log(d)
                counts[k] += 1

    avg_log_div = log_div / np.maximum(counts, 1)
    # Fit λ as slope of avg_log_div vs time
    t_axis = np.arange(n_iter) * dt
    # Use early portion (before saturation)
    end = n_iter // 2
    coeffs = np.polyfit(t_axis[:end], avg_log_div[:end], 1)
    lam = float(coeffs[0])
    return lam, t_axis, avg_log_div


# ============================================================
# Main
# ============================================================

def main():
    out_dir = Path("runs/lyapunov_analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # ---- Huisman 1999: Benettin-style ----
    print("="*60)
    print("HUISMAN 1999 (Benettin perturbation method)")
    print("="*60)
    np.random.seed(42)
    y0 = huisman_y0()
    lam_h, t_h, log_d_h = lyapunov_benettin_ode(
        huisman_derivs, y0, t_max=150, dt=0.1, eps=1e-8, n_skip=50)
    print(f"  Benettin λ_max = {lam_h:+.4f} /day")
    results["Huisman_benettin"] = lam_h

    # ---- Huisman 1999: Rosenstein on our saved trajectories ----
    print("\nHUISMAN 1999 (Rosenstein from saved time series)")
    h_data = np.load("runs/huisman1999_chaos/trajectories.npz")
    # Use species 1 time series (dt=2 in saved data)
    x1 = h_data["N_all"][:, 0]   # (T,)
    lam_h_r, t_r_h, div_r_h = lyapunov_rosenstein(x1, dt=2.0, lag=3,
                                                    emb_dim=6, n_iter=30)
    print(f"  Rosenstein λ_max (sp1) = {lam_h_r:+.4f} /day")
    results["Huisman_rosenstein"] = lam_h_r

    # ---- LV synthetic ----
    print("\nLV synthetic (Rosenstein)")
    lv_data = np.load("runs/analysis_5vs6_species/trajectories.npz")
    x_lv = lv_data["states_B_5species"][:, 0]
    # dt unknown; assume 1 per step
    lam_lv, t_r_lv, div_r_lv = lyapunov_rosenstein(x_lv, dt=1.0, lag=2,
                                                     emb_dim=6, n_iter=40)
    print(f"  Rosenstein λ_max (sp0) = {lam_lv:+.4f} /step")
    results["LV_rosenstein"] = lam_lv

    # ---- Holling synthetic ----
    print("\nHolling synthetic (Rosenstein)")
    ho_data = np.load("runs/20260413_100414_5vs6_holling/trajectories.npz")
    x_ho = ho_data["states_B_5species"][:, 0]
    lam_ho, t_r_ho, div_r_ho = lyapunov_rosenstein(x_ho, dt=1.0, lag=2,
                                                     emb_dim=6, n_iter=40)
    print(f"  Rosenstein λ_max (sp0) = {lam_ho:+.4f} /step")
    results["Holling_rosenstein"] = lam_ho

    # ---- Beninca ----
    print("\nBeninca real (Rosenstein on Cyclopoids)")
    from scripts.load_beninca import load_beninca
    full, species, _ = load_beninca()
    species = [str(s) for s in species]
    # Use Cyclopoids time series (first species), dt=4 days
    x_b = full[:, 0]
    lam_b, t_r_b, div_r_b = lyapunov_rosenstein(x_b, dt=4.0, lag=2,
                                                  emb_dim=6, n_iter=30)
    print(f"  Rosenstein λ_max (Cyclopoids) = {lam_b:+.4f} /day")
    print(f"  [reference: Beninca 2008 reported 0.051-0.066/day]")
    results["Beninca_rosenstein"] = lam_b

    # Summary table
    print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
    print(f"{'Dataset':<20}{'Method':<15}{'λ_max':<15}{'Pearson (Stage 1b)':<20}")
    print('-'*70)
    pearson_map = {
        "LV_rosenstein": ("LV (synthetic)", "Rosenstein", 0.755),
        "Holling_rosenstein": ("Holling (synthetic)", "Rosenstein", 0.843),
        "Huisman_benettin": ("Huisman (synthetic)", "Benettin-ODE", 0.343),
        "Huisman_rosenstein": ("Huisman (synthetic)", "Rosenstein", 0.343),
        "Beninca_rosenstein": ("Beninca (real)", "Rosenstein", 0.132),
    }
    for key in ["LV_rosenstein", "Holling_rosenstein", "Huisman_rosenstein",
                "Huisman_benettin", "Beninca_rosenstein"]:
        name, method, P = pearson_map[key]
        lam = results[key]
        print(f"{name:<20}{method:<15}{lam:<+15.4f}{P:<20.3f}")

    # Plot: divergence curves
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    # Benettin Huisman
    ax = axes[0, 0]
    ax.plot(t_h, log_d_h, 'b-', linewidth=1.2)
    ax.set_xlabel("time")
    ax.set_ylabel("log |diff|")
    ax.set_title(f"Huisman Benettin: λ = {results['Huisman_benettin']:+.4f} /day")
    ax.grid(alpha=0.3)
    # Rosenstein Huisman
    ax = axes[0, 1]
    ax.plot(t_r_h, div_r_h, 'r-', marker='o', markersize=3)
    ax.set_xlabel("time"); ax.set_ylabel("avg log divergence")
    ax.set_title(f"Huisman Rosenstein: λ = {results['Huisman_rosenstein']:+.4f} /day")
    ax.grid(alpha=0.3)
    # Rosenstein LV/Holling/Beninca
    ax = axes[1, 0]
    ax.plot(t_r_lv, div_r_lv, 'g-', marker='o', markersize=3, label=f"LV λ={lam_lv:+.3f}")
    ax.plot(t_r_ho, div_r_ho, 'purple', marker='s', markersize=3, label=f"Holling λ={lam_ho:+.3f}")
    ax.set_xlabel("time"); ax.set_ylabel("avg log divergence")
    ax.set_title("LV + Holling Rosenstein")
    ax.legend(); ax.grid(alpha=0.3)
    ax = axes[1, 1]
    ax.plot(t_r_b, div_r_b, 'orange', marker='D', markersize=3)
    ax.set_xlabel("time (days)"); ax.set_ylabel("avg log divergence")
    ax.set_title(f"Beninca real: λ = {results['Beninca_rosenstein']:+.4f} /day\n"
                  f"[published: 0.051-0.066]")
    ax.grid(alpha=0.3)

    fig.suptitle("Maximum Lyapunov exponent across datasets", fontsize=13, fontweight="bold")
    fig.savefig(out_dir / "lyapunov_diagnostic.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # Save summary
    with open(out_dir / "summary.md", "w", encoding="utf-8") as f:
        f.write("# Lyapunov exponent analysis\n\n")
        f.write("## Results\n\n")
        f.write("| Dataset | Method | λ_max | Stage 1b Pearson |\n|---|---|---|---|\n")
        for key in ["LV_rosenstein", "Holling_rosenstein", "Huisman_rosenstein",
                    "Huisman_benettin", "Beninca_rosenstein"]:
            name, method, P = pearson_map[key]
            lam = results[key]
            f.write(f"| {name} | {method} | {lam:+.4f} | {P:.3f} |\n")
        f.write(f"\nReference: Beninca 2008 published λ ≈ 0.051-0.066 /day for 9 plankton species.\n\n")
        f.write(f"See `lyapunov_diagnostic.png` for divergence curves.\n")

    print(f"\n[OK] {out_dir}")


if __name__ == "__main__":
    main()
