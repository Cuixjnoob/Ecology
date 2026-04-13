"""B: 完整 CVHI-NCD 在合成 LV 上跑.

验证架构本身是否可行 (合成数据信号干净).
- 合成 5 visible + 1 hidden, 820 步
- 使用完整架构 (temporal_attn + soft_forms + free_nn)
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
from scripts.cvhi_ncd_portal import compute_h_coarse, evaluate_hidden, _configure_matplotlib


def train_one_lv(visible, hidden, device="cpu", epochs=500, lr=0.0008, seed=42,
                  config="full", verbose=True):
    T, N = visible.shape
    h_coarse, p_coarse = compute_h_coarse(visible, hidden, lam=0.5)
    if verbose:
        print(f"    h_coarse Pearson = {p_coarse:.4f}")
    x = torch.tensor(visible, dtype=torch.float32, device=device).unsqueeze(0)
    h_anchor = torch.tensor(h_coarse, dtype=torch.float32, device=device).unsqueeze(0)

    torch.manual_seed(seed)
    if config == "full":
        model = CVHI_NCD(
            num_visible=N, num_hidden=1,
            encoder_d=96, encoder_blocks=3, encoder_heads=4,
            takens_lags=(1, 2, 4, 8), dropout=0.1,
            d_species=32, top_k=4,
            use_free_nn=True, free_nn_hidden=32,
            use_temporal_attn=True,
            num_gnn_layers=2, anchor_scale=0.3,
            prior_std=0.5,
        ).to(device)
    else:  # simple
        model = CVHI_NCD(
            num_visible=N, num_hidden=1,
            encoder_d=64, encoder_blocks=2, encoder_heads=4,
            takens_lags=(1, 2, 4, 8), dropout=0.1,
            d_species=16, top_k=3,
            use_free_nn=False, use_temporal_attn=False,
            num_gnn_layers=2, anchor_scale=0.3,
            prior_std=0.5,
        ).to(device)
    if verbose:
        print(f"    params: {sum(p.numel() for p in model.parameters()):,}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    def lr_lambda(step):
        if step < 100: return step / 100
        return 0.5 * (1 + np.cos(np.pi * (step - 100) / max(1, epochs - 100)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    train_end = int(0.75 * T)
    best_val = float("inf"); best_state = None; best_epoch = -1

    for epoch in range(epochs):
        beta = 0.05 * min(1.0, epoch / 200)
        model.train()
        opt.zero_grad()
        out = model(x, n_samples=2, h_anchor=h_anchor)
        losses = model.elbo_loss(out, beta=beta,
                                  lam_gates=0.05, lam_coefs=0.01,
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
        if verbose and (epoch + 1) % 100 == 0:
            print(f"      ep {epoch+1}: train={train_recon.item():.5f} val={val_recon:.5f} "
                  f"KL={losses['kl'].item():.3f} σ={losses['sigma_mean'].item():.3f} "
                  f"gates={losses['l1_gates'].item():.3f} h2v={losses['h2v_mass'].item():.3f}")

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
    out_dir = Path(f"runs/{timestamp}_cvhi_ncd_synthetic_lv")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_dir}\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    d = np.load("runs/analysis_5vs6_species/trajectories.npz")
    visible = d["states_B_5species"].astype(np.float32) + 0.01
    hidden = d["hidden_B"].astype(np.float32) + 0.01
    print(f"Synthetic LV: T={visible.shape[0]}, N_visible=5, 1 hidden\n")

    results = []
    for seed in [42, 123, 456]:
        print(f"\n=== seed {seed} ===")
        r = train_one_lv(visible, hidden, device=device, seed=seed, epochs=500, config="full")
        r["seed"] = seed
        results.append(r)
        p = r["eval_mu"]["pearson_scaled"]
        print(f"  Pearson = {p:+.4f}  RMSE = {r['eval_mu']['rmse_scaled']:.4f}  best_ep={r['best_epoch']}")

    pearsons = np.array([r["eval_mu"]["pearson_scaled"] for r in results])
    print(f"\n{'='*60}")
    print(f"CVHI-NCD on Synthetic LV (3 seeds)")
    print(f"{'='*60}")
    print(f"Pearsons: {[f'{p:+.4f}' for p in pearsons]}")
    print(f"Mean = {pearsons.mean():+.4f} ± {pearsons.std():.4f}")
    print(f"\nReference (LV synthetic data):")
    print(f"  Linear Sparse+EM: 0.977")
    print(f"  CVHI original:    0.88")
    print(f"  CVHI-NCD:         {pearsons.mean():+.3f} ± {pearsons.std():.3f}")

    # Plots
    def save_single(title, plot_fn, path, figsize=(13, 5)):
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        plot_fn(ax)
        fig.suptitle(title, fontsize=13, fontweight="bold")
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)

    def plot_recovery(ax):
        ht = hidden
        t_axis = np.arange(len(ht))
        ax.plot(t_axis, ht, color="black", linewidth=2.0, label="真实 hidden", zorder=10)
        colors = plt.cm.viridis(np.linspace(0, 1, len(results)))
        for i, r in enumerate(results):
            h_s = r["eval_mu"]["h_scaled"]
            L = min(len(h_s), len(t_axis))
            ax.plot(t_axis[:L], h_s[:L], color=colors[i], linewidth=1.0, alpha=0.85,
                    label=f"seed {r['seed']} (P={r['eval_mu']['pearson_scaled']:.3f})")
        ax.set_xlabel("时间步"); ax.set_ylabel("Hidden abundance")
        ax.legend(fontsize=10); ax.grid(alpha=0.25)
    save_single(f"CVHI-NCD on Synthetic LV (mean P={pearsons.mean():.3f})",
                 plot_recovery, out_dir / "fig_01_recovery.png", figsize=(14, 5))

    with open(out_dir / "summary.md", "w", encoding="utf-8") as f:
        f.write("# CVHI-NCD on Synthetic LV\n\n")
        f.write(f"Params: {results[0]['num_params']:,}\n\n")
        for r in results:
            f.write(f"- seed {r['seed']}: Pearson = {r['eval_mu']['pearson_scaled']:+.4f}\n")
        f.write(f"\nMean = {pearsons.mean():+.4f} ± {pearsons.std():.4f}\n")

    np.savez(out_dir / "results.npz", pearsons=pearsons,
              seeds=np.array([r["seed"] for r in results]))
    print(f"\n[OK] saved to: {out_dir}")


if __name__ == "__main__":
    main()
