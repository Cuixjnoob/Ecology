"""Partial correlation specificity: after removing shared signals
(temperature, seasonal, top competitor), is there residual species-specific signal?
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr


def residualize(x, confounds, train_end):
    """Remove confound effects via OLS on train, apply to full series."""
    T = len(x)
    if confounds.ndim == 1:
        confounds = confounds.reshape(-1, 1)
    C_tr = np.column_stack([confounds[:train_end], np.ones(train_end)])
    coef, _, _, _ = np.linalg.lstsq(C_tr, x[:train_end], rcond=None)
    C_full = np.column_stack([confounds[:T], np.ones(T)])
    return x - C_full @ coef


def pearson_val(a, b, train_end):
    """Simple val-only Pearson (no lstsq, just raw correlation on val)."""
    L = min(len(a), len(b))
    if L <= train_end + 5:
        return np.nan
    va, vb = a[train_end:L], b[train_end:L]
    if np.std(va) < 1e-10 or np.std(vb) < 1e-10:
        return np.nan
    r, _ = pearsonr(va, vb)
    return float(r)


def pearson_val_fitted(h_pred, target, train_end):
    """Train-fit lstsq, val-eval Pearson."""
    L = min(len(h_pred), len(target))
    if L <= train_end + 5:
        return np.nan
    te = train_end
    X_tr = np.column_stack([h_pred[:te], np.ones(te)])
    coef, _, _, _ = np.linalg.lstsq(X_tr, target[:te], rcond=None)
    X_val = np.column_stack([h_pred[te:L], np.ones(L - te)])
    h_fitted = X_val @ coef
    target_val = target[te:L]
    if np.std(h_fitted) < 1e-10 or np.std(target_val) < 1e-10:
        return np.nan
    r, _ = pearsonr(h_fitted, target_val)
    return float(r)


def load_h_pred_mean(base_dir, species_name):
    sp_dir = Path(base_dir) / species_name
    if not sp_dir.exists():
        return None
    h_means = []
    for seed_dir in sorted(sp_dir.iterdir()):
        traj_file = seed_dir / "trajectory.npz"
        if traj_file.exists():
            d = np.load(traj_file)
            h_means.append(d['h_mean'])
    if not h_means:
        return None
    min_len = min(len(h) for h in h_means)
    return np.mean([h[:min_len] for h in h_means], axis=0)


def analyze_species(h_name, h_pred, all_sp_df, confound_signals, train_end):
    """For one species: raw corr, partial corr (after removing confounds),
    and specificity among other species after residualization."""
    true_self = all_sp_df[h_name].values
    L = min(len(h_pred), len(true_self))

    # Raw correlation
    raw_self = pearson_val_fitted(h_pred, true_self, train_end)

    # Build confound matrix
    conf_list = []
    conf_names = []
    for name, sig in confound_signals.items():
        if len(sig) >= L:
            conf_list.append(sig[:L])
            conf_names.append(name)

    if not conf_list:
        return {
            'species': h_name, 'raw_self': raw_self,
            'partial_self': raw_self, 'confounds_removed': 'none',
            'raw_top_other': None, 'raw_top_other_corr': np.nan,
            'partial_top_other': None, 'partial_top_other_corr': np.nan,
            'partial_ratio': np.nan, 'verdict': 'no-confounds',
        }

    confounds = np.column_stack(conf_list)

    # Residualize h_pred and all species
    h_resid = residualize(h_pred[:L], confounds[:L], train_end)
    self_resid = residualize(true_self[:L], confounds[:L], train_end)

    partial_self = pearson_val_fitted(h_resid, self_resid, train_end)

    # Check against all other species (both raw and partial)
    raw_others = {}
    partial_others = {}
    for col in all_sp_df.columns:
        if col == h_name:
            continue
        other = all_sp_df[col].values[:L]
        raw_others[col] = abs(pearson_val_fitted(h_pred[:L], other, train_end))
        other_resid = residualize(other, confounds[:L], train_end)
        partial_others[col] = abs(pearson_val_fitted(h_resid, other_resid, train_end))

    # Raw top competitor
    raw_top = max(raw_others, key=lambda k: raw_others[k] if not np.isnan(raw_others[k]) else -1)
    raw_top_corr = raw_others[raw_top]

    # Partial top competitor
    partial_top = max(partial_others, key=lambda k: partial_others[k] if not np.isnan(partial_others[k]) else -1)
    partial_top_corr = partial_others[partial_top]

    # Partial specificity ratio
    abs_partial_self = abs(partial_self) if not np.isnan(partial_self) else 0
    if partial_top_corr > 1e-10 and not np.isnan(partial_top_corr):
        partial_ratio = abs_partial_self / partial_top_corr
    else:
        partial_ratio = float('inf') if abs_partial_self > 0 else np.nan

    if np.isnan(partial_ratio):
        verdict = 'degenerate'
    elif partial_ratio > 1.2:
        verdict = 'species-specific'
    elif partial_ratio >= 0.8:
        verdict = 'ambiguous'
    else:
        verdict = 'proxy-dominated'

    return {
        'species': h_name,
        'raw_self': raw_self,
        'partial_self': partial_self,
        'confounds_removed': ', '.join(conf_names),
        'raw_top_other': raw_top,
        'raw_top_other_corr': raw_top_corr,
        'partial_top_other': partial_top,
        'partial_top_other_corr': partial_top_corr,
        'partial_ratio': partial_ratio,
        'verdict': verdict,
    }


def main():
    base_main = Path("重要实验/results/main/eco_gnrd_alt5_hdyn")
    out_dir = Path("重要实验/results/specificity")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    # === HUISMAN ===
    print("=" * 70)
    print("HUISMAN (confound: dominant species sp2)")
    print("=" * 70)
    d = np.load("runs/huisman1999_chaos/trajectories.npz")
    N_all = d["N_all"]; R_all = d["resources"]
    full = np.concatenate([N_all, R_all], axis=1)
    full = (full + 0.01) / (full.mean(axis=0, keepdims=True) + 1e-3)
    species_h = [f"sp{i+1}" for i in range(6)]
    sp_df = pd.DataFrame(full[:, :6], columns=species_h)
    T = len(sp_df); train_end = int(0.75 * T)

    # Huisman: no temperature, use PCA-1 of all species as confound
    from sklearn.decomposition import PCA
    pca = PCA(n_components=1).fit(full[:train_end, :6])
    pc1 = pca.transform(full[:, :6]).ravel()
    confounds_h = {'PC1_species': pc1}

    results_h = []
    for sp in species_h:
        h_pred = load_h_pred_mean(base_main / "huisman", sp)
        if h_pred is None:
            continue
        r = analyze_species(sp, h_pred, sp_df, confounds_h, train_end)
        results_h.append(r)
    all_results['huisman'] = results_h

    # === BENINCA ===
    print("\n" + "=" * 70)
    print("BENINCA (confound: SRP + PC1)")
    print("=" * 70)
    from scripts.load_beninca import load_beninca
    full_b, species_b, _ = load_beninca(include_nutrients=True)
    species_b = [str(s) for s in species_b]
    SP_B = ["Cyclopoids", "Calanoids", "Rotifers", "Nanophyto", "Picophyto",
            "Filam_diatoms", "Ostracods", "Harpacticoids", "Bacteria"]
    sp_indices = [species_b.index(s) for s in SP_B]
    sp_df_b = pd.DataFrame(full_b[:, sp_indices], columns=SP_B)
    T_b = len(sp_df_b); train_end_b = int(0.75 * T_b)

    # Confounds: SRP + PC1 of species
    srp_idx = species_b.index('SRP') if 'SRP' in species_b else None
    confounds_b = {}
    if srp_idx is not None:
        confounds_b['SRP'] = full_b[:, srp_idx]
    pca_b = PCA(n_components=1).fit(full_b[:train_end_b, sp_indices])
    confounds_b['PC1_species'] = pca_b.transform(full_b[:, sp_indices]).ravel()

    results_b = []
    for sp in SP_B:
        h_pred = load_h_pred_mean(base_main / "beninca", sp)
        if h_pred is None:
            continue
        r = analyze_species(sp, h_pred, sp_df_b, confounds_b, train_end_b)
        results_b.append(r)
    all_results['beninca'] = results_b

    # === MAIZURU ===
    print("\n" + "=" * 70)
    print("MAIZURU (confound: Temperature + seasonal)")
    print("=" * 70)
    from scripts.load_maizuru import load_maizuru
    full_m_notemp, species_m_nt, _ = load_maizuru(include_temp=False)
    full_m_temp, species_m_t, _ = load_maizuru(include_temp=True)
    species_m_nt = [str(s) for s in species_m_nt]
    species_m_t = [str(s) for s in species_m_t]

    sp_df_m = pd.DataFrame(full_m_notemp, columns=species_m_nt)
    T_m = len(sp_df_m); train_end_m = int(0.75 * T_m)

    # Confounds: surface temperature + sinusoidal seasonal
    temp_col = full_m_temp[:, -1] if len(species_m_t) > len(species_m_nt) else None
    confounds_m = {}
    if temp_col is not None:
        confounds_m['Temperature'] = temp_col[:T_m]
    seasonal = np.sin(2 * np.pi * np.arange(T_m) / 26)  # ~annual
    confounds_m['seasonal_sin'] = seasonal
    confounds_m['seasonal_cos'] = np.cos(2 * np.pi * np.arange(T_m) / 26)

    results_m = []
    for sp in species_m_nt:
        h_pred = load_h_pred_mean(base_main / "maizuru", sp)
        if h_pred is None:
            continue
        r = analyze_species(sp, h_pred, sp_df_m, confounds_m, train_end_m)
        results_m.append(r)
    all_results['maizuru'] = results_m

    # === Print all results ===
    for ds_name, results in all_results.items():
        total = len(results)
        specific = sum(1 for r in results if r['verdict'] == 'species-specific')
        ambiguous = sum(1 for r in results if r['verdict'] == 'ambiguous')
        proxy = sum(1 for r in results if r['verdict'] == 'proxy-dominated')

        print(f"\nDATASET: {ds_name}")
        print("-" * 70)
        print(f"Species-specific (ratio > 1.2): {specific}/{total}")
        print(f"Ambiguous       (0.8--1.2):     {ambiguous}/{total}")
        print(f"Proxy-dominated (< 0.8):        {proxy}/{total}")
        print(f"\n{'Species':<30} {'Raw':>7} {'Partial':>9} {'Ratio':>7} {'PartialTop':>20} {'Verdict'}")
        print("-" * 90)
        for r in sorted(results, key=lambda x: -(x.get('partial_ratio') or 0)):
            raw_s = f"{r['raw_self']:+.3f}" if not np.isnan(r['raw_self']) else "NaN"
            par_s = f"{r['partial_self']:+.3f}" if not np.isnan(r['partial_self']) else "NaN"
            pr = f"{r['partial_ratio']:.2f}" if not np.isnan(r['partial_ratio']) else "NaN"
            pt = r['partial_top_other'] or 'none'
            ptc = f"{r['partial_top_other_corr']:.3f}" if not np.isnan(r['partial_top_other_corr']) else "N/A"
            print(f"  {r['species']:<28} {raw_s:>7} {par_s:>9} {pr:>7} {pt:>15}({ptc}) {r['verdict']}")

    # Save summary
    with open(out_dir / "specificity_partial_summary.md", 'w', encoding='utf-8') as f:
        f.write("# Partial Correlation Specificity Analysis\n\n")
        f.write("After removing shared confounds (temperature, SRP, PC1, seasonal),\n")
        f.write("is there residual species-specific signal?\n\n")
        for ds_name, results in all_results.items():
            total = len(results)
            specific = sum(1 for r in results if r['verdict'] == 'species-specific')
            ambiguous = sum(1 for r in results if r['verdict'] == 'ambiguous')
            proxy = sum(1 for r in results if r['verdict'] == 'proxy-dominated')
            f.write(f"## {ds_name}\n\n")
            f.write(f"Confounds removed: {results[0]['confounds_removed'] if results else 'N/A'}\n\n")
            f.write(f"- Species-specific: {specific}/{total}\n")
            f.write(f"- Ambiguous: {ambiguous}/{total}\n")
            f.write(f"- Proxy-dominated: {proxy}/{total}\n\n")
            f.write("| Species | Raw self | Partial self | Partial ratio | Top competitor | Verdict |\n")
            f.write("|---|---|---|---|---|---|\n")
            for r in sorted(results, key=lambda x: -(x.get('partial_ratio') or 0)):
                raw_s = f"{r['raw_self']:+.3f}" if not np.isnan(r['raw_self']) else "NaN"
                par_s = f"{r['partial_self']:+.3f}" if not np.isnan(r['partial_self']) else "NaN"
                pr = f"{r['partial_ratio']:.2f}" if not np.isnan(r['partial_ratio']) else "NaN"
                pt = r['partial_top_other'] or 'none'
                f.write(f"| {r['species']} | {raw_s} | {par_s} | {pr} | {pt} | {r['verdict']} |\n")
            f.write("\n")

    print(f"\n[OK] Saved to {out_dir}/specificity_partial_summary.md")


if __name__ == "__main__":
    main()
