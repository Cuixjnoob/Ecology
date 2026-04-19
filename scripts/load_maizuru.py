"""Maizuru Bay fish community data loader.

Ushio et al. 2018 Nature "Fluctuating interaction network and
time-varying stability of a natural fish community"

285 biweekly observations, 2002-2014, 15 dominant species + 2 temp.
We select species with <30% zero values to avoid sparse-event issues.
"""
import numpy as np
import pandas as pd
from pathlib import Path


SPECIES_ALL = [
    'Aurelia.sp', 'Engraulis.japonicus', 'Plotosus.japonicus',
    'Sebastes.inermis', 'Trachurus.japonicus', 'Girella.punctata',
    'Pseudolabrus.sieboldi', 'Parajulis.poecilepterus',
    'Halichoeres.tenuispinis', 'Chaenogobius.gulosus',
    'Pterogobius.zonoleucus', 'Tridentiger.trigonocephalus',
    'Siganus.fuscescens', 'Sphyraena.pinguis', 'Rudarius.ercodes',
]


def load_maizuru(
    csv_path="data/real_datasets/maizuru/Maizuru_dominant_sp.csv",
    max_zero_frac=1.0,  # keep all species by default
    include_temp=True,
    normalize=True,
):
    """Load Maizuru Bay data, filter sparse species.

    Returns:
        matrix: (T, N) array, species (+ temp if included)
        species: list of column names
        days: (T,) approximate day numbers
    """
    df = pd.read_csv(csv_path)

    # Select species with acceptable zero fraction
    sp_keep = []
    for sp in SPECIES_ALL:
        zero_frac = (df[sp] == 0).sum() / len(df)
        if zero_frac <= max_zero_frac:
            sp_keep.append(sp)

    matrix = df[sp_keep].values.astype(np.float32)
    species = list(sp_keep)

    if include_temp:
        temp = df[['surf.t', 'bot.t']].values.astype(np.float32)
        matrix = np.concatenate([matrix, temp], axis=1)
        species = species + ['surf_temp', 'bot_temp']

    # Approximate day numbers (biweekly ~ 14 days)
    days = np.arange(len(matrix)) * 14

    # Normalize
    matrix = matrix + 0.01
    if normalize:
        col_means = matrix.mean(axis=0, keepdims=True)
        col_means = np.maximum(col_means, 1e-3)
        matrix = matrix / col_means

    print(f"Maizuru: {matrix.shape[0]} timesteps, {len(sp_keep)} species"
          + (f" + 2 temp" if include_temp else ""))
    print(f"Selected species (zero_frac <= {max_zero_frac}): {sp_keep}")

    return matrix, species, days


if __name__ == "__main__":
    m, sp, d = load_maizuru()
    print(f"\nFinal: {m.shape}")
    print(f"Species: {sp}")
    for j, s in enumerate(sp):
        print(f"  {s:<30}: mean={m[:,j].mean():.3f} std={m[:,j].std():.3f}")
