"""A: 简化 CVHI-NCD 在 Portal 上跑.

砍一半: 无 temporal_attn, 无 free_nn, d_species 16, top_k 3.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn.functional import mse_loss

from models.cvhi_ncd import CVHI_NCD
from scripts.cvhi_ncd_portal import (
    TOP12, aggregate_portal, compute_h_coarse, evaluate_hidden, _configure_matplotlib,
)


def train_one_simplified(visible, hidden, device="cpu", epochs=300, lr=0.0008, seed=42, verbose=True):
    T, N = visible.shape
    h_coarse, p_coarse = compute_h_coarse(visible, hidden, lam=0.3)
    if verbose:
        print(f"    h_coarse Pearson = {p_coarse:.4f}")
    x = torch.tensor(visible, dtype=torch.float32, device=device).unsqueeze(0)
    h_anchor = torch.tensor(h_coarse, dtype=torch.float32, device=device).unsqueeze(0)

    torch.manual_seed(seed)
    model = CVHI_NCD(
        num_visible=N, num_hidden=1,
        encoder_d=48, encoder_blocks=1, encoder_heads=4,
        takens_lags=(1, 2, 4, 8, 12), dropout=0.2,
        d_species=16, top_k=3,
        use_free_nn=False,
        use_temporal_attn=False,
        num_gnn_layers=2,          # NEW: multi-layer
        anchor_scale=0.3,           # NEW: bound delta_mu drift
        prior_std=0.5,              # tighter
    ).to(device)
    if verbose:
        print(f"    params: {sum(p.numel() for p in model.parameters()):,}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    def lr_lambda(step):
        if step < 60: return step / 60
        return 0.5 * (1 + np.cos(np.pi * (step - 60) / max(1, epochs - 60)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    train_end = int(0.75 * T)
    best_val = float("inf"); best_state = None; best_epoch = -1

    for epoch in range(epochs):
        beta = 0.02 * min(1.0, epoch / 150)
        model.train()
        opt.zero_grad()
        out = model(x, n_samples=2, h_anchor=h_anchor)
        losses = model.elbo_loss(out, beta=beta,
                                  lam_gates=0.1, lam_coefs=0.02,
                                  lam_attn=0.01, lam_smooth=0.02, free_bits=0.02,
                                  lam_anti_bypass=8.0, min_h2v_mass=0.15)
        pred = out["predicted_log_ratio_visible"]
        actual = out["actual_log_ratio_visible"]
        train_recon = mse_loss(pred[:, :, :train_end-1],
                                actual[:, :train_end-1].unsqueeze(0).expand(pred.shape[0], -1, -1, -1))
        total = train_recon + (losses["total"] - losses["recon"])
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()

        with torch.no_grad():
            val_recon = mse_loss(pred[:, :, train_end-1:],
                                  actual[:, train_end-1:].unsqueeze(0).expand(pred.shape[0], -1, -1, -1)).item()
        if val_recon < best_val:
            best_val = val_recon
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1
        if verbose and (epoch + 1) % 50 == 0:
            print(f"      ep {epoch+1}: train={train_recon.item():.4f} val={val_recon:.4f} "
                  f"KL={losses['kl'].item():.3f} σ={losses['sigma_mean'].item():.3f} "
                  f"h2v={losses['h2v_mass'].item():.3f}")

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        out_eval = model(x, n_samples=20, h_anchor=h_anchor)
        h_mean = out_eval["H_samples"].mean(dim=0)[0, :, 0].cpu().numpy()
    eval_mu = evaluate_hidden(h_mean, hidden)
    return {
        "h_mean": h_mean, "h_coarse": h_coarse, "coarse_pearson": p_coarse,
        "eval_mu": eval_mu, "best_epoch": best_epoch,
        "num_params": sum(p.numel() for p in model.parameters()),
    }


def main():
    _configure_matplotlib()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"runs/{timestamp}_cvhi_ncd_simplified_portal")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_dir}\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    matrix, months = aggregate_portal("data/real_datasets/portal_rodent.csv", TOP12)
    from scipy.ndimage import uniform_filter1d
    def smooth(x, w=3):
        pad = np.pad(x, ((w//2, w//2), (0, 0)), mode="edge")
        return uniform_filter1d(pad, size=w, axis=0)[w//2:w//2+x.shape[0]]
    matrix_s = smooth(matrix, w=3)
    valid = matrix_s.sum(axis=1) > 10
    matrix_s = matrix_s[valid]
    T = len(matrix_s)
    print(f"T={T} months\n")

    h_idx = TOP12.index("OT")
    keep = [i for i in range(len(TOP12)) if i != h_idx]
    visible = matrix_s[:, keep] + 0.5
    hidden = matrix_s[:, h_idx] + 0.5

    results = []
    for seed in [42, 123, 456]:
        print(f"\n=== seed {seed} ===")
        r = train_one_simplified(visible, hidden, device=device, seed=seed, epochs=300)
        r["seed"] = seed
        results.append(r)
        p = r["eval_mu"]["pearson_scaled"]
        print(f"  Pearson = {p:+.4f}  best_ep={r['best_epoch']}")

    pearsons = np.array([r["eval_mu"]["pearson_scaled"] for r in results])
    print(f"\n{'='*60}")
    print(f"SIMPLIFIED CVHI-NCD on Portal OT (3 seeds)")
    print(f"{'='*60}")
    print(f"Pearsons: {[f'{p:+.4f}' for p in pearsons]}")
    print(f"Mean = {pearsons.mean():+.4f} ± {pearsons.std():.4f}")
    print(f"Params: {results[0]['num_params']:,}")
    print(f"\nCompare:")
    print(f"  Linear baseline:     0.353")
    print(f"  CVHI-NCD full:       0.107 ± 0.052")
    print(f"  CVHI-NCD simplified: {pearsons.mean():+.3f} ± {pearsons.std():.3f}")

    with open(out_dir / "summary.md", "w", encoding="utf-8") as f:
        f.write(f"# Simplified CVHI-NCD on Portal OT\n\n")
        f.write(f"Params: {results[0]['num_params']:,}\n\n")
        for r in results:
            f.write(f"- seed {r['seed']}: Pearson = {r['eval_mu']['pearson_scaled']:+.4f}\n")
        f.write(f"\nMean = {pearsons.mean():+.4f} ± {pearsons.std():.4f}\n")
    np.savez(out_dir / "results.npz", pearsons=pearsons,
              seeds=np.array([r["seed"] for r in results]))
    print(f"\n[OK] saved to: {out_dir}")


if __name__ == "__main__":
    main()
