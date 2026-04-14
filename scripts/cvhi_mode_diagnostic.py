"""Step 0 诊断: 20-seed Portal h 轨迹 pairwise 相关性分析.

问题: 20-seed Portal Pearson 方差大 (0.02~0.38) 是因为
  H_a: 数据支持多条合理 h 轨迹 (真多峰)  →  MoG 有意义
  H_b: 只有一个峰 + 训练噪声邻域 (h 形状相似)  →  MoG 白做

判据: 20 条 h_mean 轨迹两两 Pearson 相关的分布
  - 单峰高相关 (≥0.5) 团块      → H_b
  - 双峰分布 (一簇高 + 一簇低/负) → H_a
  - 低相关且分散                  → 训练噪声主导, 两者都不完全
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from models.cvhi_residual import CVHI_Residual
from scripts.cvhi_residual_L1L3_diagnostics import (
    load_portal, evaluate, _configure_matplotlib,
)


SEEDS_20 = [42, 123, 456, 789, 2024, 31415, 27182, 65537,
             7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]


def make_portal_model(N, device):
    return CVHI_Residual(
        num_visible=N,
        encoder_d=48, encoder_blocks=2, encoder_heads=4,
        takens_lags=(1, 2, 4, 8, 12), encoder_dropout=0.2,
        d_species_f=20, f_visible_layers=2, f_visible_top_k=4,
        d_species_G=12, G_field_layers=1, G_field_top_k=3,
        prior_std=1.0, gnn_backbone="mlp",
        use_formula_hints=True, use_G_field=True,
        num_mixture_components=1,
    ).to(device)


def train_one(visible, hidden_eval, device, seed, epochs=300):
    torch.manual_seed(seed)
    T, N = visible.shape
    x = torch.tensor(visible, dtype=torch.float32, device=device).unsqueeze(0)
    model = make_portal_model(N, device)
    opt = torch.optim.AdamW(model.parameters(), lr=0.0008, weight_decay=1e-4)
    warmup = int(0.2 * epochs); ramp = max(1, int(0.2 * epochs))

    def lr_lambda(step):
        if step < 50: return step / 50
        p = (step - 50) / max(1, epochs - 50)
        return 0.5 * (1 + np.cos(np.pi * p))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    train_end = int(0.75 * T)
    best_val = float("inf"); best_state = None
    m_null, m_shuf, min_e = 0.002, 0.001, 0.05

    for epoch in range(epochs):
        if epoch < warmup:
            h_w, K_r, lam_r = 0.0, 0, 0.0
        else:
            post = epoch - warmup
            h_w = min(1.0, post / ramp)
            k_ramp = min(1.0, post / (epochs - warmup) * 2)
            K_r = max(1 if h_w > 0 else 0, int(round(k_ramp * 3)))
            lam_r = 0.5 * h_w
        model.train(); opt.zero_grad()
        out = model(x, n_samples=2, rollout_K=K_r)
        tr_out = model.slice_out(out, 0, train_end)
        losses = model.loss(
            tr_out, beta_kl=0.03, free_bits=0.02,
            margin_null=m_null, margin_shuf=m_shuf,
            lam_necessary=5.0, lam_shuffle=3.0,
            lam_energy=2.0, min_energy=min_e,
            lam_smooth=0.02, lam_sparse=0.02,
            h_weight=h_w, lam_rollout=lam_r,
            rollout_weights=(1.0, 0.5, 0.25),
            lam_hf=0.0, lowpass_sigma=6.0,
        )
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        with torch.no_grad():
            val_out = model.slice_out(out, train_end, T)
            val_losses = model.loss(val_out, h_weight=1.0,
                margin_null=m_null, margin_shuf=m_shuf,
                lam_necessary=5.0, lam_shuffle=3.0,
                lam_energy=2.0, min_energy=min_e,
                lam_rollout=lam_r, rollout_weights=(1.0, 0.5, 0.25),
                lam_hf=0.0, lowpass_sigma=6.0,
            )
            val_recon = val_losses["recon_full"].item()
        if epoch > warmup + 15 and val_recon < best_val:
            best_val = val_recon
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        out_eval = model(x, n_samples=30, rollout_K=3)
        h_mean = out_eval["h_samples"].mean(dim=0)[0].cpu().numpy()
    pear, _ = evaluate(h_mean, hidden_eval)
    return h_mean, pear, best_val


def main():
    _configure_matplotlib()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"runs/{ts}_mode_diagnostic")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_dir}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vis, hid = load_portal("OT")
    T = vis.shape[0]

    trajectories = []
    pearsons = []
    val_recons = []
    for i, s in enumerate(SEEDS_20):
        t0 = datetime.now()
        h_mean, pear, vr = train_one(vis, hid, device, s)
        dt = (datetime.now() - t0).total_seconds()
        trajectories.append(h_mean)
        pearsons.append(pear); val_recons.append(vr)
        print(f"  [{i+1}/20] seed={s}  P={pear:+.3f}  val_recon={vr:.4f}  ({dt:.1f}s)")

    H = np.array(trajectories)   # (20, T)
    pearsons = np.array(pearsons); val_recons = np.array(val_recons)

    # 中心化 per-seed
    Hc = H - H.mean(axis=1, keepdims=True)
    # pairwise Pearson correlation (20 × 20)
    norms = np.linalg.norm(Hc, axis=1, keepdims=True)
    pair_corr = (Hc @ Hc.T) / (norms @ norms.T + 1e-12)

    # upper-triangle values only (no self-corr)
    iu = np.triu_indices(20, k=1)
    pair_vals = pair_corr[iu]

    # 对比: 每 seed 与 hidden_true 的 Pearson (= 已计算的 pearsons 数组)
    print(f"\n{'='*70}")
    print(f"Diagnostic Summary")
    print(f"{'='*70}")
    print(f"Pearson(h_seed, hidden_true):  {pearsons.mean():+.3f} ± {pearsons.std():.3f}  "
           f"min={pearsons.min():+.3f}  max={pearsons.max():+.3f}")
    print(f"Pairwise Pearson(h_i, h_j):   {pair_vals.mean():+.3f} ± {pair_vals.std():.3f}  "
           f"min={pair_vals.min():+.3f}  max={pair_vals.max():+.3f}")
    print(f"  median = {np.median(pair_vals):+.3f}   n_pairs = {len(pair_vals)}")
    # 分位
    q = np.quantile(pair_vals, [0.1, 0.25, 0.5, 0.75, 0.9])
    print(f"  q[10,25,50,75,90] = {[f'{x:+.3f}' for x in q]}")
    print()
    # 判据
    low_pair = (pair_vals < 0.2).sum()
    hi_pair  = (pair_vals > 0.5).sum()
    print(f"  pairs with corr < 0.2: {low_pair}/{len(pair_vals)}  ({100*low_pair/len(pair_vals):.0f}%)")
    print(f"  pairs with corr > 0.5: {hi_pair}/{len(pair_vals)}   ({100*hi_pair/len(pair_vals):.0f}%)")
    print()

    # Clustering: 用 (1 - |corr|) 作为距离, hierarchical
    from scipy.cluster.hierarchy import linkage, fcluster
    D = 1 - np.abs(pair_corr)
    np.fill_diagonal(D, 0)
    # condensed form
    from scipy.spatial.distance import squareform
    Dc = squareform(D, checks=False)
    Z = linkage(Dc, method="average")
    for nc in [2, 3, 4]:
        labels = fcluster(Z, t=nc, criterion="maxclust")
        sizes = [int((labels == k).sum()) for k in range(1, nc+1)]
        # 每 cluster 平均 Pearson
        avg_p = [float(pearsons[labels == k].mean()) if (labels == k).sum() else np.nan
                 for k in range(1, nc+1)]
        print(f"  {nc}-cluster: sizes={sizes}  cluster mean Pearson={[f'{p:+.3f}' for p in avg_p]}")

    # 保存
    np.savez(out_dir / "results.npz",
             H=H, pearsons=pearsons, val_recons=val_recons, pair_corr=pair_corr, pair_vals=pair_vals)

    # 画图: 分布 + 热图 + 几条轨迹
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    ax = axes[0, 0]
    ax.hist(pair_vals, bins=30, color="#1565c0", alpha=0.7, edgecolor="black")
    ax.axvline(0, color="gray", ls="--", alpha=0.5)
    ax.set_xlabel("pairwise Pearson(h_i, h_j)"); ax.set_ylabel("count")
    ax.set_title(f"Distribution of 190 pairwise corrs\nmean={pair_vals.mean():+.3f}, median={np.median(pair_vals):+.3f}")
    ax.grid(alpha=0.25)

    ax = axes[0, 1]
    # 按 pearsons 排序重排
    order = np.argsort(-pearsons)
    im = ax.imshow(pair_corr[order][:, order], cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_title("pairwise corr heatmap (seeds sorted by Pearson)")
    plt.colorbar(im, ax=ax, fraction=0.04)

    ax = axes[1, 0]
    # 选 Pearson 最高 3 个 + 最低 3 个
    top3 = np.argsort(-pearsons)[:3]
    bot3 = np.argsort(pearsons)[:3]
    for i in top3:
        ax.plot(H[i] - H[i].mean(), label=f"seed={SEEDS_20[i]} P={pearsons[i]:+.2f}", alpha=0.8)
    for i in bot3:
        ax.plot(H[i] - H[i].mean(), "--", label=f"seed={SEEDS_20[i]} P={pearsons[i]:+.2f}", alpha=0.6)
    ax.set_title("top-3 and bottom-3 seed trajectories (centered)")
    ax.legend(fontsize=8); ax.grid(alpha=0.25)
    ax.set_xlabel("t")

    ax = axes[1, 1]
    # Pearson vs val_recon + cluster labels
    labels_2 = fcluster(Z, t=2, criterion="maxclust")
    for c in [1, 2]:
        sel = labels_2 == c
        ax.scatter(val_recons[sel], pearsons[sel], s=60, alpha=0.7, label=f"cluster {c} (n={sel.sum()})")
    ax.set_xlabel("val_recon"); ax.set_ylabel("Pearson vs hidden_true")
    ax.set_title("seeds in (val_recon, Pearson) colored by 2-cluster label")
    ax.legend(); ax.grid(alpha=0.25)

    fig.savefig(out_dir / "fig_diagnostic.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # 写文字汇总
    with open(out_dir / "summary.md", "w", encoding="utf-8") as f:
        f.write("# Mode Diagnostic (Step 0)\n\n")
        f.write(f"- 20 seeds × Portal K=1 full config × 300 epochs\n\n")
        f.write("## Per-seed\n\n")
        f.write(f"Pearson vs hidden_true: mean={pearsons.mean():+.3f} std={pearsons.std():.3f} "
                 f"min={pearsons.min():+.3f} max={pearsons.max():+.3f}\n\n")
        f.write("## Pairwise Pearson across 190 seed pairs\n\n")
        f.write(f"mean={pair_vals.mean():+.3f} std={pair_vals.std():.3f} median={np.median(pair_vals):+.3f}\n\n")
        f.write(f"- pairs < 0.2: {low_pair}/{len(pair_vals)} ({100*low_pair/len(pair_vals):.0f}%)\n")
        f.write(f"- pairs > 0.5: {hi_pair}/{len(pair_vals)}  ({100*hi_pair/len(pair_vals):.0f}%)\n\n")
        f.write("## Clustering (hierarchical on 1−|corr|)\n\n")
        for nc in [2, 3]:
            labels = fcluster(Z, t=nc, criterion="maxclust")
            sizes = [int((labels == k).sum()) for k in range(1, nc+1)]
            avg_p = [float(pearsons[labels == k].mean()) if (labels == k).sum() else np.nan
                     for k in range(1, nc+1)]
            f.write(f"- {nc}-cluster sizes={sizes} cluster mean Pearson={[f'{p:+.3f}' for p in avg_p]}\n")
    print(f"\n[OK] {out_dir}")


if __name__ == "__main__":
    main()
