"""Portal OT 多 seed 验证: 跑 5 个 seeds，评估 CVHI 稳定性。

前置实验: cvhi_portal.py 显示 OT Pearson = 0.47 (1 seed)
本次目标: 5 seeds 验证趋势稳定性（Pearson 分布）
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from scripts.cvhi_portal import (
    TOP12, aggregate_portal, train_cvhi_portal,
    _configure_matplotlib,
)


def main():
    _configure_matplotlib()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"runs/{timestamp}_cvhi_portal_ot_multiseed")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_dir}\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    # Load
    matrix, months = aggregate_portal("data/real_datasets/portal_rodent.csv", TOP12)
    from scipy.ndimage import uniform_filter1d
    def smooth(x, w=3):
        pad = np.pad(x, ((w // 2, w // 2), (0, 0)), mode="edge")
        return uniform_filter1d(pad, size=w, axis=0)[w // 2 : w // 2 + x.shape[0]]
    matrix_s = smooth(matrix, w=3)
    valid = matrix_s.sum(axis=1) > 10
    matrix_s = matrix_s[valid]
    months_valid = [m for m, v in zip(months, valid) if v]
    T_final = len(months_valid)
    time_axis = np.array([y + m/12 for (y, m) in months_valid])
    print(f"T={T_final} months\n")

    h_idx = TOP12.index("OT")
    keep = [i for i in range(len(TOP12)) if i != h_idx]
    visible = matrix_s[:, keep] + 0.5
    hidden = matrix_s[:, h_idx] + 0.5

    seeds = [42, 123, 456, 789, 2024]
    results = []
    for s in seeds:
        print(f"\n===== Seed {s} =====")
        r = train_cvhi_portal(visible, hidden, device=device, seed=s)
        r["seed"] = s
        r["hidden_true"] = hidden
        print(f"  Pearson = {r['eval_mu']['pearson_scaled']:+.4f}  "
              f"RMSE = {r['eval_mu']['rmse_scaled']:.3f}")
        results.append(r)

    pearsons = np.array([r["eval_mu"]["pearson_scaled"] for r in results])
    rmses = np.array([r["eval_mu"]["rmse_scaled"] for r in results])

    print(f"\n{'='*60}")
    print(f"Multi-seed summary for OT (CVHI on Portal)")
    print(f"{'='*60}")
    print(f"  Seeds: {seeds}")
    print(f"  Pearsons: {[f'{p:+.4f}' for p in pearsons]}")
    print(f"  Mean Pearson = {pearsons.mean():+.4f} ± {pearsons.std():.4f}")
    print(f"  Median Pearson = {np.median(pearsons):+.4f}")
    print(f"  Pairwise pearson of h between seeds:")
    pair_mat = np.zeros((len(seeds), len(seeds)))
    L = len(hidden) - 1  # typical output length
    for i in range(len(seeds)):
        for j in range(len(seeds)):
            hi = results[i]["eval_mu"]["h_scaled"]
            hj = results[j]["eval_mu"]["h_scaled"]
            Lm = min(len(hi), len(hj))
            pair_mat[i, j] = np.corrcoef(hi[:Lm], hj[:Lm])[0, 1]
    # avg off-diagonal
    off = pair_mat[~np.eye(len(seeds), dtype=bool)]
    print(f"  Cross-seed pearson (stability): {off.mean():+.4f} ± {off.std():.4f}")

    # ==== Plots ====
    def save_single(title, plot_fn, path, figsize=(13, 5)):
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        plot_fn(ax)
        fig.suptitle(title, fontsize=13, fontweight="bold")
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)

    # Fig 1: overlay 5 seeds vs truth (focus: trend)
    def plot_overlay(ax):
        ht = hidden
        t_axis = time_axis[:len(ht)-1]
        ax.plot(t_axis, ht[:-1], color="black", linewidth=2.2, label=f"真实 OT", zorder=10)
        colors = plt.cm.viridis(np.linspace(0, 1, len(seeds)))
        for i, r in enumerate(results):
            h = r["eval_mu"]["h_scaled"]
            L = min(len(h), len(t_axis))
            ax.plot(t_axis[:L], h[:L], color=colors[i], linewidth=1.0, alpha=0.75,
                    label=f"seed {r['seed']} (P={r['eval_mu']['pearson_scaled']:.3f})")
        ax.set_xlabel("Year"); ax.set_ylabel("OT abundance")
        ax.legend(fontsize=10, ncol=2); ax.grid(alpha=0.25)
    save_single("Portal OT — CVHI (5 seeds) 趋势稳定性",
                 plot_overlay, out_dir / "fig_01_overlay.png", figsize=(14, 6))

    # Fig 2: pearson bar across seeds
    def plot_bars(ax):
        x = np.arange(len(seeds))
        ax.bar(x, pearsons, color="#1565c0")
        ax.axhline(pearsons.mean(), color="red", linestyle="--", linewidth=1.5,
                   label=f"mean = {pearsons.mean():.3f}")
        ax.axhline(0.3534, color="orange", linestyle=":", linewidth=1.5,
                   label=f"Linear baseline = 0.353")
        ax.set_xticks(x); ax.set_xticklabels([f"seed {s}" for s in seeds])
        ax.set_ylabel("Pearson")
        for i, p in enumerate(pearsons):
            ax.text(i, p + 0.005, f"{p:.3f}", ha="center", fontsize=10)
        ax.legend(fontsize=11); ax.grid(alpha=0.25, axis="y")
    save_single(f"Portal OT Pearson across 5 seeds (mean={pearsons.mean():.3f})",
                 plot_bars, out_dir / "fig_02_pearson_bars.png", figsize=(12, 5))

    # Fig 3: pairwise cross-seed pearson matrix
    def plot_matrix(ax):
        im = ax.imshow(pair_mat, cmap="RdYlGn", vmin=0, vmax=1)
        ax.set_xticks(range(len(seeds))); ax.set_xticklabels([f"s{s}" for s in seeds])
        ax.set_yticks(range(len(seeds))); ax.set_yticklabels([f"s{s}" for s in seeds])
        for i in range(len(seeds)):
            for j in range(len(seeds)):
                ax.text(j, i, f"{pair_mat[i, j]:.2f}", ha="center", va="center",
                        color="white" if pair_mat[i, j] < 0.5 else "black", fontsize=11)
        plt.colorbar(im, ax=ax)
    save_single(f"Cross-seed OT recovery Pearson (稳定性) mean_off_diag={off.mean():.3f}",
                 plot_matrix, out_dir / "fig_03_pairwise.png", figsize=(7, 6))

    # Summary md
    with open(out_dir / "summary.md", "w", encoding="utf-8") as f:
        f.write("# Portal OT — CVHI 5-seed 验证\n\n")
        f.write(f"数据: Portal Project, T={T_final} months, hidden = OT (Onychomys torridus)\n\n")
        f.write(f"Visible: 11 物种 (top-12 minus OT)\n\n")
        f.write("## 结果（5 seeds）\n\n")
        f.write("| Seed | Pearson | RMSE |\n|---|---|---|\n")
        for r in results:
            f.write(f"| {r['seed']} | {r['eval_mu']['pearson_scaled']:+.4f} | "
                    f"{r['eval_mu']['rmse_scaled']:.3f} |\n")
        f.write(f"\n- **Mean Pearson = {pearsons.mean():+.4f} ± {pearsons.std():.4f}**\n")
        f.write(f"- Median Pearson = {np.median(pearsons):+.4f}\n")
        f.write(f"- Cross-seed stability (pairwise Pearson) = {off.mean():+.4f} ± {off.std():.4f}\n\n")
        f.write("## 与 Linear Baseline 对比\n\n")
        f.write(f"- Linear Sparse+EM (OT as hidden): Pearson = +0.3534\n")
        f.write(f"- CVHI mean: Pearson = {pearsons.mean():+.4f}\n")
        f.write(f"- 提升 Δ = {pearsons.mean() - 0.3534:+.4f}\n")

    np.savez(out_dir / "results.npz",
              seeds=np.array(seeds), pearsons=pearsons, rmses=rmses,
              pair_mat=pair_mat, off_diag_mean=off.mean(), off_diag_std=off.std())
    print(f"\n[OK] saved to: {out_dir}")


if __name__ == "__main__":
    main()
