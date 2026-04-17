"""Smoke test for Stage 1e — 1 seed × 1 species (Bacteria hidden)."""
from __future__ import annotations

import sys
from datetime import datetime

import numpy as np
import torch

from scripts.load_beninca import load_beninca
from scripts.cvhi_beninca_stage1e_mte_mlp import train_one, SPECIES_ORDER

full, species, days = load_beninca()
h_name = "Bacteria"
h_idx = species.index(h_name)
visible = np.delete(full, h_idx, axis=1)
hidden = full[:, h_idx]
visible_species = [s for s in species if s != h_name]

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Smoke test: hidden={h_name}, N_visible={len(visible_species)}, device={device}")
print(f"Visible species: {visible_species}")

t0 = datetime.now()
# Shorter epochs for smoke
r = train_one(visible, hidden, seed=42, visible_species=visible_species,
              device=device, epochs=100)
dt = (datetime.now() - t0).total_seconds()
print(f"\n[DONE] {dt:.1f}s")
print(f"Pearson: {r['pearson']:+.4f}")
print(f"d_ratio: {r['d_ratio']:.3f}")
print(f"val_recon: {r['val_recon']:.4f}")
print(f"Learned B_0 taxa: {[f'{x:+.3f}' for x in r['learned_B0_taxa']]}")
print(f"Learned targets: {[f'{x:+.3f}' for x in r['learned_targets']]}")
