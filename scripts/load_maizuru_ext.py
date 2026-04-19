"""Loader for extended Maizuru dataset (2002-2024, Ohi/Ushio/Masuda).

Source: https://github.com/taka-ohi/DynaResp_Maizuru_Fish
540 semi-monthly samples, 113 fish species.
"""
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/real_datasets/maizuru_extended")

# Original 15 species mapping (Aurelia excluded - jellyfish, not in fish dataset)
SPECIES_MAP = {
    'Fish006': 'Engraulis.japonicus',
    'Fish002': 'Plotosus.japonicus',
    'Fish014': 'Sebastes.inermis',      # now S. thompsoni in new taxonomy
    'Fish031': 'Trachurus.japonicus',
    'Fish054': 'Girella.punctata',
    'Fish058': 'Pseudolabrus.sieboldi',
    'Fish061': 'Parajulis.poecilepterus',
    'Fish062': 'Halichoeres.tenuispinis',
    'Fish094': 'Chaenogobius.gulosus',
    'Fish085': 'Pterogobius.zonoleucus',
    'Fish088': 'Tridentiger.trigonocephalus',
    'Fish099': 'Siganus.fuscescens',
    'Fish100': 'Sphyraena.pinguis',
    'Fish105': 'Rudarius.ercodes',
}


def load_maizuru_ext(include_temp=False, min_presence=0.05):
    """Load extended Maizuru dataset.

    Returns:
        full: (T, N) normalized array
        species: list of species names
        meta: metadata dict
    """
    counts = pd.read_csv(DATA_DIR / "fishcount.txt", sep="\t", index_col=0)
    meta = pd.read_csv(DATA_DIR / "metadata.txt", sep="\t", index_col=0)

    # Use mapped species (original 14, excluding Aurelia)
    fish_ids = list(SPECIES_MAP.keys())
    species_names = [SPECIES_MAP[fid] for fid in fish_ids]

    # Extract count matrix
    data = counts[fish_ids].values.astype(np.float32)  # (540, 14)
    T, N = data.shape

    # Filter species with too few observations
    presence = (data > 0).sum(axis=0) / T
    keep = presence >= min_presence
    data = data[:, keep]
    species_names = [s for s, k in zip(species_names, keep) if k]
    fish_ids = [f for f, k in zip(fish_ids, keep) if k]

    # Normalize: log1p then per-channel mean=1
    data = np.log1p(data)
    data = data / (data.mean(axis=0, keepdims=True) + 1e-8)

    if include_temp:
        temp = meta["Water_temp_surface"].values.astype(np.float32)
        temp = (temp - temp.mean()) / (temp.std() + 1e-8)
        data = np.column_stack([data, temp[:, None]])
        species_names.append("Temperature")

    print(f"Maizuru-ext: {data.shape[0]} timesteps, {len(species_names)} species")
    print(f"Selected species (presence >= {min_presence}): {species_names}")

    return data, species_names, {"fish_ids": fish_ids, "T": T}
