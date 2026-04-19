"""EDM baselines for hidden species recovery.

Baseline 1: Univariate EDM Simplex (Sugihara & May 1990)
  - Simplex projection on each visible species
  - Prediction residual PCA = hidden proxy

Baseline 2: Multiview Embedding (Ye & Sugihara 2016, Science)
  - Combine multiple embedding views from different variable combinations
  - Top-k views weighted by in-sample forecast skill
  - Residual PCA = hidden proxy

Reference:
  Sugihara & May 1990 Nature; Sugihara et al. 2012 Science
  Ye & Sugihara 2016 Science "Information leverage in interconnected ecosystems"
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
from itertools import combinations
from sklearn.decomposition import PCA
from pathlib import Path
from datetime import datetime


def pearson(a, b):
    L = min(len(a), len(b))
    a, b = a[:L], b[:L]
    X = np.column_stack([a, np.ones(L)])
    coef, _, _, _ = np.linalg.lstsq(X, b, rcond=None)
    a_sc = X @ coef
    ac = a_sc - a_sc.mean()
    bc = b - b.mean()
    return float(np.sum(ac * bc) / (np.sqrt(np.sum(ac**2) * np.sum(bc**2)) + 1e-8))


def simplex_predict(lib_data, pred_data, E, nn_k=None):
    """Simplex projection: nearest-neighbor weighted prediction.

    lib_data: (T_lib, D) library points
    pred_data: (T_pred, D) prediction points
    nn_k: number of nearest neighbors (default E+1)
    Returns: predictions (T_pred,) for the first column
    """
    if nn_k is None:
        nn_k = E + 1
    T_lib = len(lib_data)
    T_pred = len(pred_data)
    predictions = np.full(T_pred, np.nan)

    for i in range(T_pred):
        # Find nearest neighbors in library
        dists = np.sqrt(((lib_data[:-1] - pred_data[i]) ** 2).sum(axis=1))
        idx = np.argsort(dists)[:nn_k]
        d_nn = dists[idx]

        # Exponential weighting
        d_min = d_nn[0]
        if d_min < 1e-10:
            weights = np.zeros(nn_k)
            weights[0] = 1.0
        else:
            weights = np.exp(-d_nn / d_min)
        weights /= weights.sum() + 1e-10

        # Predict: weighted average of next-step values
        next_vals = lib_data[idx + 1, 0]  # first column = target
        predictions[i] = np.sum(weights * next_vals)

    return predictions


def simplex_predict_mve(lib_emb, lib_target, pred_emb, E, nn_k=None):
    """Simplex for MVE: find neighbors in embedding space, predict target.

    lib_emb: (T_lib, D) library embedding points
    lib_target: (T_lib,) library target values (next-step of target species)
    pred_emb: (T_pred, D) prediction embedding points
    """
    if nn_k is None:
        nn_k = E + 1
    T_pred = len(pred_emb)
    predictions = np.full(T_pred, np.nan)

    for i in range(T_pred):
        dists = np.sqrt(((lib_emb[:-1] - pred_emb[i]) ** 2).sum(axis=1))
        idx = np.argsort(dists)[:nn_k]
        d_nn = dists[idx]
        d_min = d_nn[0]
        if d_min < 1e-10:
            weights = np.zeros(nn_k)
            weights[0] = 1.0
        else:
            weights = np.exp(-d_nn / d_min)
        weights /= weights.sum() + 1e-10
        # Predict from target values at next step
        if idx.max() + 1 >= len(lib_target):
            continue
        next_vals = lib_target[idx + 1]
        predictions[i] = np.sum(weights * next_vals)

    return predictions


def build_embedding(series, E, tau=1):
    """Build delay embedding matrix from 1D series.
    Returns (T-E*tau, E) matrix.
    """
    T = len(series)
    n = T - (E - 1) * tau
    if n <= 0:
        return None
    emb = np.column_stack([series[(E-1)*tau - l*tau : T - l*tau] for l in range(E)])
    return emb


def univariate_edm_residual(visible, hidden, E_range=[2, 3, 4, 5], tau=1):
    """Univariate EDM: Simplex on each species, residual PCA."""
    T, N = visible.shape
    train_end = int(0.75 * T)
    best_pear = -1

    for E in E_range:
        all_resid = []
        valid = True
        for j in range(N):
            emb = build_embedding(visible[:, j], E, tau)
            if emb is None:
                valid = False; break
            offset = (E - 1) * tau
            lib = emb[:train_end - offset]
            pred = emb[:T - offset]
            if len(lib) < E + 2:
                valid = False; break
            preds = simplex_predict(lib, pred, E)
            actual = visible[offset:, j]
            L = min(len(preds), len(actual))
            resid = actual[:L] - preds[:L]
            resid = np.nan_to_num(resid, nan=0.0)
            all_resid.append(resid)
        if not valid or len(all_resid) == 0:
            continue
        min_len = min(len(r) for r in all_resid)
        resid_mat = np.column_stack([r[:min_len] for r in all_resid])
        if resid_mat.std() < 1e-8:
            continue
        pca = PCA(n_components=1)
        h_est = pca.fit_transform(resid_mat).flatten()
        offset_h = (E - 1) * tau
        p = pearson(h_est, hidden[offset_h:offset_h + min_len])
        if p > best_pear:
            best_pear = p
    return best_pear


def multiview_embedding(visible, hidden, E=3, tau=1, top_k=None):
    """Multiview Embedding (Ye & Sugihara 2016).

    Generate all E-dimensional embeddings from combinations of visible species
    (using lags 0 to E-1 of each selected variable).
    Rank by in-sample forecast skill, average top-k predictions.
    """
    T, N = visible.shape
    train_end = int(0.75 * T)

    # Generate candidate embeddings: choose E columns from N species x lags
    # Simplified MVE: each "view" uses E different species at lag 0
    # (as in Ye 2016: different variable combinations, not just lag combinations)
    if N < E:
        E = N

    if top_k is None:
        top_k = max(1, int(np.sqrt(N)))

    # All combinations of E species
    combos = list(combinations(range(N), E))
    if len(combos) > 200:
        # Subsample if too many
        rng = np.random.RandomState(42)
        idx = rng.choice(len(combos), 200, replace=False)
        combos = [combos[i] for i in idx]

    # For each target species, compute MVE prediction
    all_resid = []
    for target_j in range(N):
        view_preds = []
        view_skills = []

        for combo in combos:
            # Build embedding from selected species
            emb = np.column_stack([visible[:, c] for c in combo])
            # Add one lag for temporal structure
            emb_lagged = emb[1:]  # current step
            target_next = visible[1:, target_j]  # target at next step

            lib_emb = emb_lagged[:train_end - 1]
            lib_target = target_next[:train_end - 1]

            if len(lib_emb) < E + 2:
                continue

            # Simplex on this view
            pred_emb = emb_lagged[:T - 1]
            # Library: pair (emb, target_next)
            lib_data = np.column_stack([lib_target.reshape(-1, 1), lib_emb])
            pred_data = pred_emb

            try:
                # Distance on embedding dims only, predict from target column
                preds = simplex_predict_mve(lib_emb, lib_target, pred_data, E)
            except Exception:
                continue

            # In-sample skill
            valid_mask = ~np.isnan(preds[:train_end - 1])
            if valid_mask.sum() < 5:
                continue
            skill = np.corrcoef(
                preds[:train_end - 1][valid_mask],
                target_next[:train_end - 1][valid_mask]
            )[0, 1]
            if np.isnan(skill):
                continue

            view_preds.append(preds)
            view_skills.append(skill)

        if len(view_preds) == 0:
            all_resid.append(np.zeros(T - 1))
            continue

        # Select top-k views
        order = np.argsort(view_skills)[::-1][:top_k]
        skills_topk = np.array([view_skills[i] for i in order])
        preds_topk = np.array([view_preds[i] for i in order])

        # Weighted average (weight by skill)
        weights = np.maximum(skills_topk, 0)
        if weights.sum() < 1e-8:
            weights = np.ones(len(weights))
        weights /= weights.sum()

        avg_pred = np.zeros(T - 1)
        for w, p in zip(weights, preds_topk):
            p_clean = np.nan_to_num(p, nan=0.0)
            avg_pred += w * p_clean[:T - 1]

        resid = visible[1:, target_j] - avg_pred
        all_resid.append(resid)

    min_len = min(len(r) for r in all_resid)
    resid_mat = np.column_stack([r[:min_len] for r in all_resid])

    if resid_mat.std() < 1e-8:
        return 0.0
    pca = PCA(n_components=1)
    h_est = pca.fit_transform(resid_mat).flatten()
    return pearson(h_est, hidden[1:1 + min_len])


# ===== Data loaders =====
def load_huisman():
    d = np.load("runs/huisman1999_chaos/trajectories.npz")
    N_all = d["N_all"]; R_all = d["resources"]
    results = []
    for sp_idx in range(6):
        vis = np.concatenate([np.delete(N_all, sp_idx, axis=1), R_all], axis=1)
        vis = (vis + 0.01) / (vis.mean(axis=0, keepdims=True) + 1e-3)
        hid = N_all[:, sp_idx]
        hid = (hid + 0.01) / (hid.mean() + 1e-3)
        results.append((vis.astype(np.float32), hid.astype(np.float32)))
    return results


def load_beninca():
    from scripts.load_beninca import load_beninca as _load
    full, species, _ = _load(include_nutrients=True)
    species = [str(s) for s in species]
    SP = ["Cyclopoids", "Calanoids", "Rotifers", "Nanophyto",
          "Picophyto", "Filam_diatoms", "Ostracods", "Harpacticoids", "Bacteria"]
    results = []
    for h in SP:
        h_idx = species.index(h)
        vis = np.delete(full, h_idx, axis=1).astype(np.float32)
        hid = full[:, h_idx].astype(np.float32)
        results.append((vis, hid, h))
    return results


def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"runs/{ts}_edm_baseline")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("EDM BASELINES: Univariate Simplex + Multiview Embedding (Ye 2016)")
    print("=" * 80)

    # Huisman
    print("\nHuisman (6->1):")
    dh = load_huisman()
    edm_ps = []; mve_ps = []
    for i, (v, h) in enumerate(dh):
        p1 = univariate_edm_residual(v, h)
        p2 = multiview_embedding(v, h, E=3)
        edm_ps.append(p1); mve_ps.append(p2)
        print(f"  sp{i+1}: EDM={p1:+.3f}  MVE={p2:+.3f}")
    print(f"  Overall: EDM={np.mean(edm_ps):+.3f}  MVE={np.mean(mve_ps):+.3f}")

    # Beninca
    print("\nBeninca (9->1):")
    db = load_beninca()
    edm_ps_b = []; mve_ps_b = []
    for v, h, name in db:
        p1 = univariate_edm_residual(v, h)
        p2 = multiview_embedding(v, h, E=3)
        edm_ps_b.append(p1); mve_ps_b.append(p2)
        print(f"  {name:<16}: EDM={p1:+.3f}  MVE={p2:+.3f}")
    print(f"  Overall: EDM={np.mean(edm_ps_b):+.3f}  MVE={np.mean(mve_ps_b):+.3f}")

    # Full table
    print(f"\n{'='*80}")
    print("FULL BASELINE TABLE")
    print(f"{'='*80}")
    print(f"{'Method':<16} {'Huisman':>10} {'Beninca':>10}")
    print("-" * 40)
    print(f"{'VAR+PCA':<16} {'+0.027':>10} {'+0.022':>10}")
    print(f"{'MLP+PCA':<16} {'+0.042':>10} {'+0.030':>10}")
    print(f"{'EDM Simplex':<16} {np.mean(edm_ps):>+10.3f} {np.mean(edm_ps_b):>+10.3f}")
    print(f"{'MVE (Ye 2016)':<16} {np.mean(mve_ps):>+10.3f} {np.mean(mve_ps_b):>+10.3f}")
    print(f"{'Neural ODE':<16} {'+0.498':>10} {'+0.035':>10}")
    print(f"{'LSTM':<16} {'+0.535':>10} {'+0.108':>10}")
    print(f"{'Eco-GNRD':<16} {'+0.502':>10} {'+0.162':>10}")

    with open(out_dir / "summary.md", "w", encoding="utf-8") as f:
        f.write("# EDM Baselines: Univariate + MVE (Ye 2016)\n\n")
        f.write("| Method | Huisman | Beninca |\n|---|---|---|\n")
        f.write(f"| VAR+PCA | +0.027 | +0.022 |\n")
        f.write(f"| MLP+PCA | +0.042 | +0.030 |\n")
        f.write(f"| EDM Simplex | {np.mean(edm_ps):+.3f} | {np.mean(edm_ps_b):+.3f} |\n")
        f.write(f"| MVE (Ye 2016) | {np.mean(mve_ps):+.3f} | {np.mean(mve_ps_b):+.3f} |\n")
        f.write(f"| Neural ODE | +0.498 | +0.035 |\n")
        f.write(f"| LSTM | +0.535 | +0.108 |\n")
        f.write(f"| **Eco-GNRD** | **+0.502** | **+0.162** |\n")

    print(f"\n[OK] {out_dir}")


if __name__ == "__main__":
    main()
