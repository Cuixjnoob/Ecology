"""Smoke test for Stage 1f - 1 seed Bacteria 100 epochs."""
from __future__ import annotations
from datetime import datetime
import numpy as np
import torch

from scripts.load_beninca import load_beninca
from scripts.cvhi_beninca_stage1f_conserve import train_one

full, species, days = load_beninca()
h_name = "Bacteria"
h_idx = species.index(h_name)
visible = np.delete(full, h_idx, axis=1)
hidden = full[:, h_idx]

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Smoke: hidden={h_name}, device={device}")

t0 = datetime.now()
r = train_one(visible, hidden, seed=42, device=device, epochs=100)
dt = (datetime.now() - t0).total_seconds()
print(f"\n[DONE] {dt:.1f}s")
print(f"Pearson: {r['pearson']:+.4f}  (Stage 1b baseline: +0.197)")
print(f"d_ratio: {r['d_ratio']:.3f}")
print(f"val_recon: {r['val_recon']:.4f}")
print(f"learned c: {r['learned_c']:.3f}")
