"""Eco-GNRD on extended Maizuru dataset (2002-2024, 540 timesteps, 14 species).

Uses same config as original Maizuru (alt 5:1, no temp).
Saves to 重要实験/results/main/eco_gnrd_alt5_hdyn/maizuru_ext/
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
from pathlib import Path
import torch

from scripts.load_maizuru_ext import load_maizuru_ext
from scripts.run_main_experiment import (
    DATASET_CONFIGS, run_dataset, SEEDS, load_dataset,
)

# Register maizuru_ext as a dataset
DATASET_CONFIGS['maizuru_ext'] = dict(DATASET_CONFIGS['maizuru'])

# Monkey-patch load_dataset to support maizuru_ext
_orig_load = load_dataset

def load_dataset_ext(name):
    if name == 'maizuru_ext':
        full, species, _ = load_maizuru_ext(include_temp=False)
        species = [str(s) for s in species]
        tasks = []
        for h in species:
            h_idx = species.index(h)
            vis = np.delete(full, h_idx, axis=1).astype(np.float32)
            hid = full[:, h_idx].astype(np.float32)
            tasks.append((vis, hid, h, len(species) - 1))
        return tasks
    return _orig_load(name)

# Patch it in
import scripts.run_main_experiment as rme
rme.load_dataset = load_dataset_ext

device = "cuda" if torch.cuda.is_available() else "cpu"
out_root = Path("重要实验/results/main/eco_gnrd_alt5_hdyn")

print("Eco-GNRD on extended Maizuru (2002-2024)")
run_dataset('maizuru_ext', out_root, device, seeds=SEEDS[:5])
print("DONE")
