"""MI ceiling experiment: estimate info-theoretic upper bound on Pearson.

For each dataset and each hidden species:
  1. Build feature X = delay embedding of visible channels (captures "visible history")
  2. Estimate I(X; hidden_true) via KSG (kNN-based MI estimator)
  3. Upper bound Pearson: ρ_max = sqrt(1 - exp(-2·I))   (Gaussian equality)
  4. Compare to actual Stage 1b Pearson

Any unsupervised method that computes h from visible_history cannot
exceed ρ_max. This tests whether our method is at info-theoretic ceiling.

Theory:
  For Gaussian: I(X,Y) = -0.5 log(1-ρ²)  →  ρ_max = sqrt(1-exp(-2I))
  For non-Gaussian: I(X,Y) ≥ -0.5 log(1-ρ²), so ρ² ≤ 1 - exp(-2I) always.
"""
from __future__ import annotations

from pathlib import Path
import json
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.feature_selection import mutual_info_regression
from sklearn.decomposition import PCA


def build_delay_embedding(data: np.ndarray, lags: list = (0, 1, 2, 4, 8)) -> np.ndarray:
    """data: (T, N). Returns (T_valid, N * len(lags)) with delay-stacked features.
    Rows earlier than max(lags) are dropped.
    """
    T, N = data.shape
    max_lag = max(lags)
    T_valid = T - max_lag
    feats = []
    for L in lags:
        # offset: data[max_lag-L : max_lag-L+T_valid]
        start = max_lag - L
        feats.append(data[start:start + T_valid])
    return np.concatenate(feats, axis=1)  # (T_valid, N * len(lags))


def mi_with_pca(visible: np.ndarray, hidden: np.ndarray,
                 lags=(0, 1, 2, 4, 8), pca_dim: int = 5,
                 n_neighbors: int = 4, n_repeats: int = 3) -> tuple:
    """Estimate MI(visible_history, hidden) via KSG after PCA reduction.

    PCA keeps top pca_dim components to make KSG reliable.
    n_repeats: run multiple KSG estimates (different random state) for CI.
    """
    # Align hidden with max-lag truncation
    max_lag = max(lags)
    hidden_trim = hidden[max_lag:]

    # Delay embedding of visible
    emb = build_delay_embedding(visible, lags)        # (T_valid, N*|lags|)

    # PCA reduction (KSG is sensitive to high dim)
    if emb.shape[1] > pca_dim:
        pca = PCA(n_components=pca_dim)
        emb_pca = pca.fit_transform(emb)
        var_explained = pca.explained_variance_ratio_.sum()
    else:
        emb_pca = emb
        var_explained = 1.0

    # KSG MI estimation (sklearn uses KSG-like kNN estimator)
    mis = []
    for seed in range(n_repeats):
        mi = mutual_info_regression(emb_pca, hidden_trim,
                                      n_neighbors=n_neighbors,
                                      random_state=seed)
        # mi is array (1,) since hidden is 1-D, but we have multivariate X
        # mutual_info_regression returns per-feature MI. For multivariate X we
        # need different approach. Let's use the mean of per-feature MIs as
        # a conservative **lower bound** on joint MI.
        mis.append(float(np.mean(mi)))

    # Better: estimate MI(X, Y) where X is multivariate using KSG directly
    # sklearn's mutual_info_regression treats each feature separately.
    # We need joint MI. Use a workaround: create a single summary feature.
    # Actually best: use sklearn's mutual_info_score with binned joint.
    # For now, use mutual_info_regression feature-wise and take max / sum.
    # Max is ≤ joint MI (true joint ≥ each marginal MI).
    # So max(mi_per_feat) is a **lower bound** on joint MI.
    # This gives CONSERVATIVE upper bound on Pearson (we under-estimate I).

    mi_mean = float(np.mean(mis))
    mi_std = float(np.std(mis))
    return mi_mean, mi_std, var_explained


def ksg_joint_mi(X: np.ndarray, y: np.ndarray, k: int = 4) -> float:
    """KSG estimator for I(X; y) with multivariate X, univariate y.

    Ref: Kraskov, Stögbauer, Grassberger (2004).
    """
    from scipy.spatial import cKDTree
    from scipy.special import digamma

    N = X.shape[0]
    Z = np.concatenate([X, y.reshape(-1, 1)], axis=1)

    # Find k-th neighbor distances in joint space (Chebyshev/max norm)
    tree_Z = cKDTree(Z)
    dists, _ = tree_Z.query(Z, k=k + 1, p=np.inf)
    eps = dists[:, -1]   # k-th neighbor distance

    # For each point, count points in X-space and y-space within eps
    tree_X = cKDTree(X)
    tree_y = cKDTree(y.reshape(-1, 1))

    n_X = np.array([len(tree_X.query_ball_point(X[i], eps[i] - 1e-10, p=np.inf))
                     for i in range(N)])
    n_y = np.array([len(tree_y.query_ball_point(y[i:i+1], eps[i] - 1e-10, p=np.inf))
                     for i in range(N)])

    # Exclude self
    n_X = n_X - 1
    n_y = n_y - 1
    # Clip to avoid digamma(0)
    n_X = np.clip(n_X, 1, None)
    n_y = np.clip(n_y, 1, None)

    mi = digamma(k) + digamma(N) - np.mean(digamma(n_X + 1) + digamma(n_y + 1))
    return float(max(0.0, mi))


def mi_ceiling(visible: np.ndarray, hidden: np.ndarray,
                lags=(0, 1, 2, 4, 8), pca_dim: int = 4,
                k_neighbors: int = 4):
    """Full pipeline: embed visible → PCA → KSG → Pearson ceiling."""
    max_lag = max(lags)
    hidden_trim = hidden[max_lag:]
    emb = build_delay_embedding(visible, lags)

    # PCA reduce
    if emb.shape[1] > pca_dim:
        pca = PCA(n_components=pca_dim)
        X = pca.fit_transform(emb)
        var_frac = float(pca.explained_variance_ratio_.sum())
    else:
        X = emb
        var_frac = 1.0

    # Standardize X and y (helps KSG)
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    y = (hidden_trim - hidden_trim.mean()) / (hidden_trim.std() + 1e-8)

    # KSG joint MI
    mi = ksg_joint_mi(X, y, k=k_neighbors)
    # Pearson ceiling
    rho_max = float(np.sqrt(1.0 - np.exp(-2.0 * mi))) if mi > 0 else 0.0
    return dict(mi=mi, pca_var_frac=var_frac, pearson_ceiling=rho_max)


def analyze_beninca():
    from scripts.load_beninca import load_beninca
    full, species, _ = load_beninca()
    species = [str(s) for s in species]
    print(f"\n{'='*70}\nBENINCA (real data, T={full.shape[0]})\n{'='*70}")

    SPECIES_TO_TEST = ["Cyclopoids", "Calanoids", "Rotifers", "Nanophyto",
                        "Picophyto", "Filam_diatoms", "Ostracods",
                        "Harpacticoids", "Bacteria"]
    stage1b = {"Cyclopoids": 0.055, "Calanoids": 0.173, "Rotifers": 0.080,
               "Nanophyto": 0.141, "Picophyto": 0.116, "Filam_diatoms": 0.031,
               "Ostracods": 0.272, "Harpacticoids": 0.120, "Bacteria": 0.197}

    results = {}
    print(f"{'Species':<18}{'MI (nats)':<12}{'ρ_ceiling':<12}"
          f"{'ρ (Stage 1b)':<15}{'ratio':<10}")
    for sp in SPECIES_TO_TEST:
        idx = species.index(sp)
        visible = np.delete(full, idx, axis=1)
        hidden = full[:, idx]
        r = mi_ceiling(visible, hidden, lags=(0, 1, 2, 4, 8), pca_dim=4, k_neighbors=4)
        rho_ceil = r["pearson_ceiling"]
        actual = stage1b[sp]
        ratio = actual / rho_ceil if rho_ceil > 0.01 else float("inf")
        print(f"{sp:<18}{r['mi']:<12.3f}{rho_ceil:<+12.3f}"
              f"{actual:<+15.3f}{ratio:<10.2f}")
        results[sp] = dict(mi=r["mi"], rho_ceiling=rho_ceil,
                            rho_actual=actual, ratio=ratio,
                            pca_var_frac=r["pca_var_frac"])
    return results


def analyze_huisman():
    print(f"\n{'='*70}\nHUISMAN 1999 (synthetic chaos, K41=0.26)\n{'='*70}")
    data = np.load("runs/huisman1999_chaos/trajectories.npz")
    N_all = data["N_all"]        # (T, 6)
    R = data["resources"]         # (T, 5)
    full = np.concatenate([N_all, R], axis=1)
    # Normalize per-channel
    full = full / (full.mean(axis=0, keepdims=True) + 1e-8)
    T = full.shape[0]
    print(f"T = {T}")

    pearson_huisman = {"sp1": 0.122, "sp2": 0.623, "sp3": 0.273, "sp4": 0.516,
                       "sp5": 0.268, "sp6": 0.256}  # stage1b from K41=0.30 run
    # Note: new K41=0.26 run pearsons not yet available. Use old as reference.

    results = {}
    print(f"{'Species':<10}{'MI (nats)':<12}{'ρ_ceiling':<12}"
          f"{'ρ (K41=0.30)':<15}{'ratio':<10}")
    for i in range(6):
        sp = f"sp{i+1}"
        visible = np.delete(full, i, axis=1)
        hidden = full[:, i]
        r = mi_ceiling(visible, hidden, lags=(0, 1, 2, 4, 8), pca_dim=4)
        rho_ceil = r["pearson_ceiling"]
        actual = pearson_huisman.get(sp, float("nan"))
        ratio = actual / rho_ceil if rho_ceil > 0.01 and not np.isnan(actual) else float("nan")
        print(f"{sp:<10}{r['mi']:<12.3f}{rho_ceil:<+12.3f}"
              f"{actual:<+15.3f}{ratio:<10.2f}")
        results[sp] = dict(mi=r["mi"], rho_ceiling=rho_ceil,
                            rho_actual=actual, ratio=ratio)
    return results


def analyze_synthetic(name: str, path: str, pearson_map: dict):
    print(f"\n{'='*70}\n{name}\n{'='*70}")
    data = np.load(path)
    visible = data["states_B_5species"].astype(np.float32)   # (T, 5)
    hidden = data["hidden_B"].astype(np.float32)

    # Normalize
    visible = visible / (visible.mean(axis=0, keepdims=True) + 1e-8)
    hidden_norm = hidden / (hidden.mean() + 1e-8)

    r = mi_ceiling(visible, hidden_norm, lags=(0, 1, 2, 4, 8), pca_dim=4)
    rho_ceil = r["pearson_ceiling"]
    actual = pearson_map.get(name, float("nan"))
    ratio = actual / rho_ceil if rho_ceil > 0.01 else float("inf")
    print(f"  MI = {r['mi']:.3f} nats")
    print(f"  ρ_ceiling = {rho_ceil:+.3f}")
    print(f"  ρ_actual  = {actual:+.3f}")
    print(f"  ratio     = {ratio:.2f}")
    return dict(mi=r["mi"], rho_ceiling=rho_ceil, rho_actual=actual, ratio=ratio)


def main():
    out_dir = Path("runs/mi_ceiling_analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    # LV
    all_results["LV"] = analyze_synthetic(
        "LV", "runs/analysis_5vs6_species/trajectories.npz",
        {"LV": 0.755})

    # Holling
    all_results["Holling"] = analyze_synthetic(
        "Holling", "runs/20260413_100414_5vs6_holling/trajectories.npz",
        {"Holling": 0.843})

    # Huisman (per species)
    all_results["Huisman"] = analyze_huisman()

    # Beninca (per species)
    all_results["Beninca"] = analyze_beninca()

    # Summary plot: ρ_actual vs ρ_ceiling for all datasets
    fig, ax = plt.subplots(figsize=(10, 6.5), constrained_layout=True)

    colors = {"LV": "#66c2a5", "Holling": "#fc8d62",
              "Huisman": "#8da0cb", "Beninca": "#e78ac3"}

    pts_x, pts_y, pts_color, pts_label = [], [], [], []
    # Scalars
    for name in ["LV", "Holling"]:
        r = all_results[name]
        pts_x.append(r["rho_ceiling"]); pts_y.append(r["rho_actual"])
        pts_color.append(colors[name]); pts_label.append(name)

    # Huisman per-species
    for sp, r in all_results["Huisman"].items():
        pts_x.append(r["rho_ceiling"]); pts_y.append(r["rho_actual"])
        pts_color.append(colors["Huisman"]); pts_label.append(f"Huisman_{sp}")

    # Beninca per-species
    for sp, r in all_results["Beninca"].items():
        pts_x.append(r["rho_ceiling"]); pts_y.append(r["rho_actual"])
        pts_color.append(colors["Beninca"]); pts_label.append(f"Beninca_{sp}")

    for i, (x_, y_, c_, l_) in enumerate(zip(pts_x, pts_y, pts_color, pts_label)):
        ax.scatter(x_, y_, s=80, c=c_, alpha=0.8, edgecolors="black", zorder=5)
        ax.annotate(l_, (x_, y_), xytext=(5, 5),
                    textcoords="offset points", fontsize=8)

    # Diagonal: y = x (ceiling achieved)
    lims = [0, max(max(pts_x), max(pts_y)) * 1.1]
    ax.plot(lims, lims, "k--", alpha=0.4, label="ρ = ceiling (optimal)")
    ax.set_xlabel("ρ_ceiling (information-theoretic upper bound)")
    ax.set_ylabel("ρ_actual (Stage 1b)")
    ax.set_title("Are we at information-theoretic ceiling?\n"
                 "Points on diagonal = optimal; below = room for improvement",
                 fontweight="bold")
    ax.grid(alpha=0.3)
    # Create custom legend
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="o", markersize=10, linewidth=0,
                       markerfacecolor=c, markeredgecolor="black", label=name)
                for name, c in colors.items()]
    handles.append(Line2D([0], [0], linestyle="--", color="k",
                           alpha=0.4, label="ρ = ceiling"))
    ax.legend(handles=handles, loc="lower right")
    fig.savefig(out_dir / "fig_mi_ceiling.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # Write summary
    with open(out_dir / "summary.md", "w", encoding="utf-8") as f:
        f.write("# MI ceiling analysis\n\n")
        f.write("Estimates Pearson_max = sqrt(1 - exp(-2·MI(visible_history, hidden))) ")
        f.write("via KSG kNN estimator with PCA-reduced visible delay embedding.\n\n")
        f.write("## Summary\n\n")

        for name in ["LV", "Holling"]:
            r = all_results[name]
            f.write(f"- **{name}**: MI={r['mi']:.3f} nats, "
                    f"ρ_ceiling={r['rho_ceiling']:+.3f}, "
                    f"ρ_actual={r['rho_actual']:+.3f}, "
                    f"ratio={r['ratio']:.2f}\n")

        f.write("\n### Huisman (synthetic chaos, per species)\n\n")
        f.write("| Species | MI | ρ_ceiling | ρ_actual | ratio |\n|---|---|---|---|---|\n")
        for sp, r in all_results["Huisman"].items():
            f.write(f"| {sp} | {r['mi']:.3f} | {r['rho_ceiling']:+.3f} | "
                    f"{r['rho_actual']:+.3f} | {r['ratio']:.2f} |\n")

        f.write("\n### Beninca (real chaos, per species)\n\n")
        f.write("| Species | MI | ρ_ceiling | ρ_actual | ratio |\n|---|---|---|---|---|\n")
        for sp, r in all_results["Beninca"].items():
            f.write(f"| {sp} | {r['mi']:.3f} | {r['rho_ceiling']:+.3f} | "
                    f"{r['rho_actual']:+.3f} | {r['ratio']:.2f} |\n")

    # JSON dump
    with open(out_dir / "raw.json", "w") as f:
        json.dump(all_results, f, indent=2, default=float)

    print(f"\n[OK] {out_dir}")


if __name__ == "__main__":
    main()
