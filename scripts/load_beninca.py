"""Beninca 2008 Baltic plankton mesocosm data loader.

803 采样点 × 10 物种组, 7.3 年封闭 mesocosm.
我们 drop Protozoa (10% 缺失), 用剩余 9 species.
"""
from __future__ import annotations

import numpy as np
from pathlib import Path


SPECIES_ALL = ["Cyclopoids", "Calanoids", "Rotifers", "Protozoa",
                "Nanophyto", "Picophyto", "Filam_diatoms", "Ostracods",
                "Harpacticoids", "Bacteria"]


def load_beninca(
    npz_path: str = "data/real_datasets/beninca_2008/parsed.npz",
    nutrients_npz: str = "data/real_datasets/beninca_2008/nutrients.npz",
    drop_species: tuple = ("Protozoa",),
    drop_nutrients: tuple = ("DIN",),     # DIN = NO2+NO3+NH4 冗余
    include_nutrients: bool = True,
    normalize_per_channel: bool = True,    # 每通道除以其 mean, 避免巨大 scale 差异
    interpolate_to_regular: bool = True,
    dt_days: int = 4,
    start_day: int = 30,
):
    """
    Returns:
      matrix: (T, N) interpolated log-abundance (or raw + softplus).
              N = len(SPECIES_ALL) - len(drop_species)
      species: list of N species names
      days: (T,) day numbers
    """
    d = np.load(npz_path, allow_pickle=True)
    raw = d["matrix"]              # (803, 10)
    day_num = d["day_num"]          # (803,)
    species_names = list(d["species"])

    # drop columns
    keep_idx = [i for i, s in enumerate(species_names) if s not in drop_species]
    raw = raw[:, keep_idx]
    species = [species_names[i] for i in keep_idx]

    # fill remaining sparse NaNs by forward-fill, then 0
    for j in range(raw.shape[1]):
        col = raw[:, j]
        mask = np.isnan(col)
        if mask.any():
            # forward fill
            last = np.nan
            for i in range(len(col)):
                if np.isnan(col[i]):
                    col[i] = last if not np.isnan(last) else 0.0
                else:
                    last = col[i]
            raw[:, j] = col

    # filter rows starting from start_day
    mask = day_num >= start_day
    raw = raw[mask]; day_num = day_num[mask]
    print(f"After start_day={start_day} filter: {raw.shape}")

    if interpolate_to_regular:
        d_min, d_max = int(day_num.min()), int(day_num.max())
        regular_days = np.arange(d_min, d_max + 1, dt_days)
        regular = np.zeros((len(regular_days), raw.shape[1]), dtype=np.float32)
        for j in range(raw.shape[1]):
            regular[:, j] = np.interp(regular_days, day_num, raw[:, j])
        matrix = regular
        days = regular_days
        print(f"Species interpolated to dt={dt_days} days: {matrix.shape}")
    else:
        matrix = raw
        days = day_num

    # 加入 nutrients (作 visible state, 不是外部 covariate — 它们是 mesocosm 内部状态)
    if include_nutrients:
        nutr = np.load(nutrients_npz, allow_pickle=True)
        nutr_vals_raw = nutr["values"]    # (348, 5)
        nutr_days = nutr["days"]
        nutr_names = list(nutr["names"])
        # drop redundant
        keep_n = [i for i, n in enumerate(nutr_names) if n not in drop_nutrients]
        nutr_vals = nutr_vals_raw[:, keep_n]
        nutr_keep_names = [nutr_names[i] for i in keep_n]
        # forward-fill NaN
        for j in range(nutr_vals.shape[1]):
            col = nutr_vals[:, j]; mask = np.isnan(col); last = np.nan
            for i in range(len(col)):
                if np.isnan(col[i]): col[i] = last if not np.isnan(last) else 0.0
                else: last = col[i]
            nutr_vals[:, j] = col
        # interpolate to species' time grid
        nutr_interp = np.zeros((len(days), nutr_vals.shape[1]), dtype=np.float32)
        for j in range(nutr_vals.shape[1]):
            nutr_interp[:, j] = np.interp(days, nutr_days, nutr_vals[:, j])
        # concatenate species + nutrients
        matrix = np.concatenate([matrix, nutr_interp], axis=1)
        species = list(species) + nutr_keep_names
        print(f"After adding nutrients ({nutr_keep_names}): {matrix.shape}")

    # Per-channel normalize (除以 channel mean, 不改变 log-ratio 但稳定 encoder 输入)
    matrix = matrix + 0.01   # avoid log(0)
    if normalize_per_channel:
        col_means = matrix.mean(axis=0, keepdims=True)
        col_means = np.maximum(col_means, 1e-3)
        matrix = matrix / col_means
        print(f"Per-channel normalized (each channel mean=1)")

    return matrix.astype(np.float32), species, days.astype(np.int64)


if __name__ == "__main__":
    matrix, species, days = load_beninca()
    print(f"\nFinal matrix: {matrix.shape}")
    print(f"Species: {species}")
    print(f"Days: {days[0]} → {days[-1]}")
    print(f"\nPer-species stats:")
    for j, s in enumerate(species):
        v = matrix[:, j]
        print(f"  {s:<18s}  mean={v.mean():.4f}  std={v.std():.4f}  max={v.max():.4f}  min={v.min():.4f}")
