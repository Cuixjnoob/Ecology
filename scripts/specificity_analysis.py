"""Cross-species specificity matrix for Eco-GNRD recovery.

For each hidden species S, checks whether h_pred_S correlates more with
true_S than with any other signal (other species, environment, seasonal).
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr


def pearson_val_fitted(h_pred, target, train_end):
    """Train-fit lstsq, val-eval Pearson. Same protocol as main results."""
    L = min(len(h_pred), len(target))
    if L <= train_end + 5:
        return np.nan
    te = train_end
    # lstsq fit on train
    X_tr = np.column_stack([h_pred[:te], np.ones(te)])
    coef, _, _, _ = np.linalg.lstsq(X_tr, target[:te], rcond=None)
    # eval on val
    X_val = np.column_stack([h_pred[te:L], np.ones(L - te)])
    h_fitted = X_val @ coef
    target_val = target[te:L]
    # handle degenerate cases
    if np.std(h_fitted) < 1e-10 or np.std(target_val) < 1e-10:
        return np.nan
    r, _ = pearsonr(h_fitted, target_val)
    return float(r)


def specificity_for_species(h_name, h_pred, all_species_df, env_df,
                            train_end, seasonal_signal=None):
    """Compute specificity metrics for one hidden species.

    Args:
        h_name: name of hidden species
        h_pred: (T,) predicted trajectory (raw h_mean, averaged over seeds)
        all_species_df: DataFrame with all species as columns (including h_name)
        env_df: DataFrame with environmental variables (can be empty)
        train_end: int, train/val split index
        seasonal_signal: (T,) optional seasonal baseline
    Returns:
        dict with corr_self, corr_others, specificity_ratio, etc.
    """
    result = {
        'species': h_name,
        'corr_self': np.nan,
        'corr_other_species': {},
        'corr_env': {},
        'corr_seasonal': np.nan,
        'top_competing_signal': None,
        'top_competing_corr': -np.inf,
        'specificity_ratio': np.nan,
        'verdict': 'error',
    }

    # Check for degenerate h_pred
    if np.std(h_pred) < 1e-10:
        result['verdict'] = 'degenerate'
        return result

    # Self correlation
    true_self = all_species_df[h_name].values
    result['corr_self'] = pearson_val_fitted(h_pred, true_self, train_end)

    # Other species
    all_competing = {}
    for col in all_species_df.columns:
        if col == h_name:
            continue
        r = pearson_val_fitted(h_pred, all_species_df[col].values, train_end)
        result['corr_other_species'][col] = r
        if not np.isnan(r):
            all_competing[col] = abs(r)

    # Environmental variables
    if env_df is not None and len(env_df.columns) > 0:
        for col in env_df.columns:
            vals = env_df[col].values
            # pairwise complete: skip NaN
            mask = ~np.isnan(vals[:len(h_pred)])
            if mask.sum() > train_end + 5:
                r = pearson_val_fitted(h_pred[mask], vals[mask], min(train_end, mask[:train_end].sum()))
            else:
                r = np.nan
            result['corr_env'][col] = r
            if not np.isnan(r):
                all_competing[f"env:{col}"] = abs(r)

    # Seasonal signal
    if seasonal_signal is not None:
        r = pearson_val_fitted(h_pred, seasonal_signal, train_end)
        result['corr_seasonal'] = r
        if not np.isnan(r):
            all_competing['seasonal'] = abs(r)

    # Find top competitor
    if all_competing:
        top_name = max(all_competing, key=all_competing.get)
        top_corr = all_competing[top_name]
        result['top_competing_signal'] = top_name
        result['top_competing_corr'] = top_corr

        # Specificity ratio
        abs_self = abs(result['corr_self']) if not np.isnan(result['corr_self']) else 0
        if top_corr > 1e-10:
            result['specificity_ratio'] = abs_self / top_corr
        else:
            result['specificity_ratio'] = float('inf') if abs_self > 0 else np.nan

        # Verdict
        sr = result['specificity_ratio']
        if np.isnan(sr):
            result['verdict'] = 'degenerate'
        elif sr > 1.2:
            result['verdict'] = 'species-specific'
        elif sr >= 0.8:
            result['verdict'] = 'ambiguous'
        else:
            result['verdict'] = 'proxy-dominated'
    else:
        result['verdict'] = 'no-competitors'

    return result


def load_h_pred_mean(base_dir, species_name):
    """Load h_mean averaged over all seeds for a species."""
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
    # Average over seeds (align lengths)
    min_len = min(len(h) for h in h_means)
    h_means = [h[:min_len] for h in h_means]
    return np.mean(h_means, axis=0)


def make_seasonal(T, period_steps):
    """Simple sinusoidal seasonal signal."""
    t = np.arange(T)
    return np.sin(2 * np.pi * t / period_steps)


def run_dataset(ds_name, base_dir, all_species_df, species_list,
                env_df=None, seasonal_period=None):
    """Run specificity analysis for one dataset."""
    T = len(all_species_df)
    train_end = int(0.75 * T)

    seasonal = make_seasonal(T, seasonal_period) if seasonal_period else None

    results = []
    for sp in species_list:
        h_pred = load_h_pred_mean(base_dir, sp)
        if h_pred is None:
            print(f"  {sp}: no trajectory found, skipping")
            continue

        r = specificity_for_species(sp, h_pred, all_species_df, env_df,
                                    train_end, seasonal)
        results.append(r)

    return results


def print_summary(ds_name, results):
    """Print formatted summary."""
    total = len(results)
    specific = sum(1 for r in results if r['verdict'] == 'species-specific')
    ambiguous = sum(1 for r in results if r['verdict'] == 'ambiguous')
    proxy = sum(1 for r in results if r['verdict'] == 'proxy-dominated')
    degen = sum(1 for r in results if r['verdict'] in ('degenerate', 'error'))

    print(f"\nDATASET: {ds_name}")
    print("-" * 50)
    print(f"Species-specific (ratio > 1.2): {specific}/{total}")
    print(f"Ambiguous       (0.8--1.2):     {ambiguous}/{total}")
    print(f"Proxy-dominated (< 0.8):        {proxy}/{total}")
    if degen:
        print(f"Degenerate/error:               {degen}/{total}")
    print(f"\nPer-species breakdown:")
    for r in sorted(results, key=lambda x: -(x.get('specificity_ratio') or 0)):
        sr = r['specificity_ratio']
        sr_str = f"{sr:.2f}" if not np.isnan(sr) else "NaN"
        self_str = f"{r['corr_self']:+.3f}" if not np.isnan(r['corr_self']) else "NaN"
        top = r['top_competing_signal'] or 'none'
        tc = r['top_competing_corr']
        tc_str = f"{tc:+.3f}" if tc != -np.inf else "N/A"
        print(f"  {r['species']:<35s}: ratio={sr_str:>6s}, self={self_str}, "
              f"top_competitor={top} ({tc_str}), verdict={r['verdict']}")


def save_results(ds_name, results, out_dir):
    """Save CSV and contribute to summary."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for r in results:
        rows.append({
            'dataset': ds_name,
            'species': r['species'],
            'corr_self': r['corr_self'],
            'top_competing_signal': r['top_competing_signal'],
            'top_competing_corr': r['top_competing_corr'] if r['top_competing_corr'] != -np.inf else np.nan,
            'specificity_ratio': r['specificity_ratio'],
            'verdict': r['verdict'],
        })
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / f"specificity_matrix_{ds_name}.csv", index=False)
    return df


def main():
    out_dir = Path("重要实验/results/specificity")
    base_main = Path("重要实验/results/main/eco_gnrd_alt5_hdyn")
    all_dfs = []

    # === HUISMAN ===
    print("\n" + "=" * 60)
    print("Loading Huisman...")
    d = np.load("runs/huisman1999_chaos/trajectories.npz")
    N_all = d["N_all"]; R_all = d["resources"]
    # Normalize same as experiment
    full = np.concatenate([N_all, R_all], axis=1)
    full = (full + 0.01) / (full.mean(axis=0, keepdims=True) + 1e-3)
    species_h = [f"sp{i+1}" for i in range(6)]
    # For specificity, only compare against the 6 species (not resources)
    sp_df = pd.DataFrame(full[:, :6], columns=species_h)
    # Resources as "env"
    res_names = [f"resource_{i+1}" for i in range(R_all.shape[1])]
    env_df = pd.DataFrame(full[:, 6:], columns=res_names)

    results_h = run_dataset("huisman", base_main / "huisman", sp_df, species_h,
                            env_df=env_df, seasonal_period=None)
    print_summary("huisman", results_h)
    all_dfs.append(save_results("huisman", results_h, out_dir))

    # === BENINCA ===
    print("\n" + "=" * 60)
    print("Loading Beninca...")
    from scripts.load_beninca import load_beninca
    full_b, species_b, _ = load_beninca(include_nutrients=True)
    species_b = [str(s) for s in species_b]
    SP_B = ["Cyclopoids", "Calanoids", "Rotifers", "Nanophyto", "Picophyto",
            "Filam_diatoms", "Ostracods", "Harpacticoids", "Bacteria"]
    # Species columns
    sp_indices = [species_b.index(s) for s in SP_B]
    sp_df_b = pd.DataFrame(full_b[:, sp_indices], columns=SP_B)
    # Nutrients as env
    nutrient_names = [s for s in species_b if s not in SP_B]
    if nutrient_names:
        nut_indices = [species_b.index(s) for s in nutrient_names]
        env_df_b = pd.DataFrame(full_b[:, nut_indices], columns=nutrient_names)
    else:
        env_df_b = pd.DataFrame()

    results_b = run_dataset("beninca", base_main / "beninca", sp_df_b, SP_B,
                            env_df=env_df_b, seasonal_period=None)
    print_summary("beninca", results_b)
    all_dfs.append(save_results("beninca", results_b, out_dir))

    # === MAIZURU ===
    print("\n" + "=" * 60)
    print("Loading Maizuru...")
    from scripts.load_maizuru import load_maizuru
    full_m, species_m, _ = load_maizuru(include_temp=True)
    species_m = [str(s) for s in species_m]
    # Last column is temperature if include_temp=True
    sp_names_m = species_m[:-1]  # exclude Temperature
    sp_df_m = pd.DataFrame(full_m[:, :-1], columns=sp_names_m)
    env_df_m = pd.DataFrame(full_m[:, -1:], columns=["Temperature"])
    # Maizuru: biweekly, ~26 samples/year
    seasonal_period_m = 26

    results_m = run_dataset("maizuru", base_main / "maizuru", sp_df_m, sp_names_m,
                            env_df=env_df_m, seasonal_period=seasonal_period_m)
    print_summary("maizuru", results_m)
    all_dfs.append(save_results("maizuru", results_m, out_dir))

    # === Grand summary ===
    all_df = pd.concat(all_dfs, ignore_index=True)
    with open(out_dir / "specificity_summary.md", 'w', encoding='utf-8') as f:
        f.write("# Eco-GNRD Cross-Species Specificity Analysis\n\n")
        for ds in ["huisman", "beninca", "maizuru"]:
            sub = all_df[all_df['dataset'] == ds]
            total = len(sub)
            spec = (sub['verdict'] == 'species-specific').sum()
            amb = (sub['verdict'] == 'ambiguous').sum()
            proxy = (sub['verdict'] == 'proxy-dominated').sum()
            f.write(f"## {ds}\n\n")
            f.write(f"- Species-specific: {spec}/{total}\n")
            f.write(f"- Ambiguous: {amb}/{total}\n")
            f.write(f"- Proxy-dominated: {proxy}/{total}\n\n")
            f.write("| Species | Self corr | Top competitor | Comp corr | Ratio | Verdict |\n")
            f.write("|---|---|---|---|---|---|\n")
            for _, row in sub.iterrows():
                sr = f"{row['specificity_ratio']:.2f}" if not pd.isna(row['specificity_ratio']) else "NaN"
                sc = f"{row['corr_self']:+.3f}" if not pd.isna(row['corr_self']) else "NaN"
                tc = f"{row['top_competing_corr']:+.3f}" if not pd.isna(row['top_competing_corr']) else "N/A"
                f.write(f"| {row['species']} | {sc} | {row['top_competing_signal']} | {tc} | {sr} | {row['verdict']} |\n")
            f.write("\n")

    print(f"\n[OK] Results saved to {out_dir}")


if __name__ == "__main__":
    main()
