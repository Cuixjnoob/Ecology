"""补跑 20 个新 seeds 用于 40-seed bootstrap 评测."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import torch

from models.cvhi_residual import CVHI_Residual
from scripts.cvhi_residual_L1L3_diagnostics import (
    load_portal, evaluate, _configure_matplotlib,
)
from scripts.cvhi_mode_diagnostic import make_portal_model, train_one


SEEDS_EXTRA_20 = [101, 202, 303, 404, 505, 606, 707, 808, 909, 1001,
                   1234, 5678, 91011, 121314, 151617, 181920,
                   212223, 242526, 272829, 303132]


def main():
    _configure_matplotlib()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"runs/{ts}_mode_diagnostic_extra")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_dir}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vis, hid = load_portal("OT")
    trajectories, pearsons, val_recons = [], [], []
    for i, s in enumerate(SEEDS_EXTRA_20):
        t0 = datetime.now()
        h_mean, pear, vr = train_one(vis, hid, device, s)
        dt = (datetime.now() - t0).total_seconds()
        trajectories.append(h_mean); pearsons.append(pear); val_recons.append(vr)
        print(f"  [{i+1}/20] seed={s}  P={pear:+.3f}  val_recon={vr:.4f}  ({dt:.1f}s)")

    H = np.array(trajectories)
    pearsons = np.array(pearsons); val_recons = np.array(val_recons)
    Hc = H - H.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(Hc, axis=1, keepdims=True)
    pair_corr = (Hc @ Hc.T) / (norms @ norms.T + 1e-12)
    np.savez(out_dir / "results.npz",
             H=H, pearsons=pearsons, val_recons=val_recons, pair_corr=pair_corr,
             seeds=SEEDS_EXTRA_20)
    print(f"\nmean P={pearsons.mean():+.3f} std={pearsons.std():.3f} max={pearsons.max():+.3f}")
    print(f"[OK] {out_dir}")


if __name__ == "__main__":
    main()
