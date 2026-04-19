"""Loader for Blasius et al. 2020 chemostat predator-prey data.

10 experiments, 3 nodes: algae, rotifers (adults), eggs.
Source: https://figshare.com/articles/dataset/10045976
"""
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/real_datasets/blasius")
SPECIES = ["algae", "rotifers", "eggs"]


def load_blasius_experiment(exp_id, min_T=100):
    """Load one experiment (C1-C10).

    Returns:
        data: (T, 3) normalized array [algae, rotifers, eggs]
        species: ['algae', 'rotifers', 'eggs']
        meta: dict with experiment info
    """
    df = pd.read_csv(DATA_DIR / f"C{exp_id}.csv")
    # Columns: time, algae, rotifers, egg-ratio, eggs, dead animals, external medium
    cols = df.columns.tolist()
    data = df.iloc[:, [1, 2, 4]].values.astype(np.float32)  # algae, rotifers, eggs

    # Drop rows with NaN
    mask = ~np.isnan(data).any(axis=1)
    data = data[mask]

    if len(data) < min_T:
        return None, None, None

    # Normalize: log1p then per-channel mean=1
    data = np.log1p(np.clip(data, 0, None))
    data = data / (data.mean(axis=0, keepdims=True) + 1e-8)

    return data, SPECIES.copy(), {"exp_id": exp_id, "T": len(data)}


def load_all_blasius(min_T=100):
    """Load all experiments that are long enough.

    Returns list of (data, species, meta) tuples.
    """
    experiments = []
    for i in range(1, 11):
        data, sp, meta = load_blasius_experiment(i, min_T=min_T)
        if data is not None:
            experiments.append((data, sp, meta))
            print(f"  C{i}: T={len(data)}, {sp}")
        else:
            print(f"  C{i}: skipped (too short or too many NaN)")
    return experiments
