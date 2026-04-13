"""深度对比学到的 A 矩阵和真实生态结构（LV 和 Holling 双数据）。

目标:
  1. Heatmap 对比 (A_learned vs A_true_5x5 vs A_true_6x6)
  2. 捕食链边的识别准确率
  3. Hidden-related edges 的影响分析（虽然不在 5x5）
  4. 生成完整的结构对比报告

这是 论文级别 的 analysis，证明方法可以 recover 生态学意义的结构。
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def _configure_matplotlib() -> None:
    plt.rcParams["font.sans-serif"] = [
        "PingFang SC", "Hiragino Sans GB", "Heiti TC", "STHeiti",
        "Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def fit_sparse_baseline_batch(states, log_ratios, lam, n_iter=1500, lr=0.015, seed=42):
    torch.manual_seed(seed)
    r = torch.zeros(5, requires_grad=True)
    A = torch.zeros(5, 5, requires_grad=True)
    with torch.no_grad(): A.fill_diagonal_(-0.2)
    opt = torch.optim.Adam([r, A], lr=lr)
    x = torch.tensor(states[:-1], dtype=torch.float32)
    y = torch.tensor(log_ratios, dtype=torch.float32)
    for _ in range(n_iter):
        opt.zero_grad()
        pred = r.view(1, -1) + x @ A.T
        fit = ((pred - y) ** 2).mean()
        A_off = A - torch.diag(torch.diag(A))
        loss = fit + lam * A_off.abs().mean()
        loss.backward()
        opt.step()
    return r.detach().numpy(), A.detach().numpy()


def main():
    _configure_matplotlib()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"runs/{timestamp}_ecological_structure_analysis")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_dir}\n")

    datasets = []
    d_lv = np.load("runs/analysis_5vs6_species/trajectories.npz")
    datasets.append(("LV", d_lv, 0.5))   # best lam from previous sweep
    holling_dirs = sorted(glob.glob("runs/*_5vs6_holling/trajectories.npz"))
    d_holling = np.load(holling_dirs[-1])
    datasets.append(("Holling", d_holling, 2.0))  # best lam from previous sweep

    fig = plt.figure(figsize=(18, 14), constrained_layout=True)
    gs = fig.add_gridspec(4, 4)

    all_summary = {}
    for row_idx, (label, d, best_lam) in enumerate(datasets):
        states = d["states_B_5species"]
        A_true_full = d["interaction_matrix_full"]  # 6x6
        A_true_5x5 = A_true_full[:5, :5]

        safe = np.clip(states, 1e-6, None)
        log_ratios = np.log(safe[1:] / safe[:-1])
        log_ratios = np.clip(log_ratios, -1.12, 0.92)

        r_hat, A_learned = fit_sparse_baseline_batch(states, log_ratios, best_lam)

        # Analysis
        mask = ~np.eye(5, dtype=bool)
        meaningful = mask & (np.abs(A_true_5x5) > 0.05)

        # Renormalized A: includes indirect effect through hidden
        # A_true_eff = A_true_5x5 + outer(hidden_effect_on_vis, vis_effect_on_hidden) / self_limit_of_hidden
        # 简化: outer(A[:5, 5], A[5, :5])
        indirect = np.outer(A_true_full[:5, 5], A_true_full[5, :5]) / max(abs(A_true_full[5, 5]), 0.2)
        A_true_eff = A_true_5x5 + indirect

        sign_acc_5x5 = (np.sign(A_learned[meaningful]) == np.sign(A_true_5x5[meaningful])).mean()
        sign_acc_eff = (np.sign(A_learned[meaningful]) == np.sign(A_true_eff[meaningful])).mean()
        pearson_5x5 = np.corrcoef(A_learned[mask], A_true_5x5[mask])[0, 1]
        pearson_eff = np.corrcoef(A_learned[mask], A_true_eff[mask])[0, 1]

        all_summary[label] = {
            "sign_acc_5x5": float(sign_acc_5x5),
            "sign_acc_eff": float(sign_acc_eff),
            "pearson_5x5": float(pearson_5x5),
            "pearson_eff": float(pearson_eff),
            "A_learned": A_learned,
            "A_true_5x5": A_true_5x5,
            "A_true_eff": A_true_eff,
            "A_true_full": A_true_full,
        }

        # ===== Plots =====
        species_labels_5 = ["v1", "v2", "v3", "v4", "v5"]
        species_labels_6 = ["v1", "v2", "v3", "v4", "v5", "h"]

        # Col 1: A_learned
        ax = fig.add_subplot(gs[row_idx, 0])
        vmax = max(np.abs(A_learned).max(), np.abs(A_true_5x5).max(), np.abs(A_true_eff).max())
        im = ax.imshow(A_learned, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal")
        ax.set_xticks(range(5)); ax.set_yticks(range(5))
        ax.set_xticklabels(species_labels_5); ax.set_yticklabels(species_labels_5)
        ax.set_title(f"{label}: A_learned (λ={best_lam})", fontsize=11)
        for i in range(5):
            for j in range(5):
                v = A_learned[i, j]
                color = "white" if abs(v) > vmax * 0.5 else "black"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color=color)
        plt.colorbar(im, ax=ax, fraction=0.046)

        # Col 2: A_true_5x5 (真实 5x5 子集)
        ax = fig.add_subplot(gs[row_idx, 1])
        im = ax.imshow(A_true_5x5, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal")
        ax.set_xticks(range(5)); ax.set_yticks(range(5))
        ax.set_xticklabels(species_labels_5); ax.set_yticklabels(species_labels_5)
        ax.set_title(f"{label}: A_true (5x5 原始)", fontsize=11)
        for i in range(5):
            for j in range(5):
                v = A_true_5x5[i, j]
                color = "white" if abs(v) > vmax * 0.5 else "black"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color=color)
        plt.colorbar(im, ax=ax, fraction=0.046)

        # Col 3: A_true_eff (包含 hidden 间接影响)
        ax = fig.add_subplot(gs[row_idx, 2])
        im = ax.imshow(A_true_eff, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal")
        ax.set_xticks(range(5)); ax.set_yticks(range(5))
        ax.set_xticklabels(species_labels_5); ax.set_yticklabels(species_labels_5)
        ax.set_title(f"{label}: A_true_eff (含 hidden 间接)", fontsize=11)
        for i in range(5):
            for j in range(5):
                v = A_true_eff[i, j]
                color = "white" if abs(v) > vmax * 0.5 else "black"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color=color)
        plt.colorbar(im, ax=ax, fraction=0.046)

        # Col 4: 6x6 真实
        ax = fig.add_subplot(gs[row_idx, 3])
        vmax_6 = np.abs(A_true_full).max()
        im = ax.imshow(A_true_full, cmap="RdBu_r", vmin=-vmax_6, vmax=vmax_6, aspect="equal")
        ax.set_xticks(range(6)); ax.set_yticks(range(6))
        ax.set_xticklabels(species_labels_6); ax.set_yticklabels(species_labels_6)
        ax.set_title(f"{label}: A_true (完整 6x6, 含 hidden)", fontsize=11)
        for i in range(6):
            for j in range(6):
                v = A_true_full[i, j]
                color = "white" if abs(v) > vmax_6 * 0.5 else "black"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6, color=color)
        # Highlight hidden row/col
        for i in range(6):
            if i != 5:
                ax.add_patch(plt.Rectangle((i - 0.5, 4.5), 1, 1, fill=False, edgecolor="yellow", linewidth=2))
                ax.add_patch(plt.Rectangle((4.5, i - 0.5), 1, 1, fill=False, edgecolor="yellow", linewidth=2))
        plt.colorbar(im, ax=ax, fraction=0.046)

    # Scatter panels at bottom
    for idx, (label, d, best_lam) in enumerate(datasets):
        summary = all_summary[label]
        mask = ~np.eye(5, dtype=bool)
        ax = fig.add_subplot(gs[2, idx * 2])
        A_l = summary["A_learned"][mask]
        A_5x5 = summary["A_true_5x5"][mask]
        A_eff = summary["A_true_eff"][mask]
        ax.scatter(A_5x5, A_l, alpha=0.7, s=50, color="#1565c0", edgecolor="black", label="vs 5x5")
        ax.scatter(A_eff, A_l, alpha=0.7, s=50, color="#c62828", edgecolor="black", marker="^", label="vs eff")
        vmin = min(A_5x5.min(), A_eff.min(), A_l.min())
        vmax = max(A_5x5.max(), A_eff.max(), A_l.max())
        ax.plot([vmin, vmax], [vmin, vmax], "k--", linewidth=0.8, alpha=0.5)
        ax.axhline(0, color="grey", linewidth=0.5, alpha=0.3)
        ax.axvline(0, color="grey", linewidth=0.5, alpha=0.3)
        ax.set_xlabel("A_true[i,j]")
        ax.set_ylabel("A_learned[i,j]")
        ax.set_title(f"{label}: sign_5x5={summary['sign_acc_5x5']:.2f}, sign_eff={summary['sign_acc_eff']:.2f}")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.25)

    # Hidden edge analysis
    ax = fig.add_subplot(gs[3, :2])
    for idx, (label, d, _) in enumerate(datasets):
        A_full = all_summary[label]["A_true_full"]
        # Hidden edges: row 5 and column 5 of A_true_full
        h_out = A_full[5, :5]  # 5 edges: hidden -> vj
        h_in = A_full[:5, 5]   # 5 edges: vj -> hidden
        x_pos = np.arange(5) + idx * 0.35
        ax.bar(x_pos - 0.17, h_out, 0.15, label=f"{label}: hidden→vj", alpha=0.8,
               color="#1565c0" if label == "LV" else "#43a047")
        ax.bar(x_pos, h_in, 0.15, label=f"{label}: vj→hidden", alpha=0.8,
               color="#e53935" if label == "LV" else "#fb8c00")
    ax.set_xticks(np.arange(5) + 0.175)
    ax.set_xticklabels(["v1", "v2", "v3", "v4", "v5"])
    ax.set_ylabel("Edge strength")
    ax.set_title("Hidden 的对外交互强度（真实值）")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25, axis="y")

    # Summary text
    ax = fig.add_subplot(gs[3, 2:])
    ax.axis("off")
    txt = "生态结构恢复总结\n" + "=" * 40 + "\n"
    for label, s in all_summary.items():
        txt += f"\n{label}:\n"
        txt += f"  Sign accuracy (vs raw 5x5):        {s['sign_acc_5x5']:.3f}\n"
        txt += f"  Sign accuracy (vs eff 5x5):        {s['sign_acc_eff']:.3f}\n"
        txt += f"  Pearson (vs raw 5x5):              {s['pearson_5x5']:+.3f}\n"
        txt += f"  Pearson (vs eff 5x5):              {s['pearson_eff']:+.3f}\n"
    txt += "\n注释：\n"
    txt += "  - 'eff 5x5' = 真实 5x5 + hidden 间接影响\n"
    txt += "  - 高 Pearson vs eff 说明学到的 A 自然\n"
    txt += "    吸收了 hidden 的间接效应（无监督恢复的副产品）\n"
    ax.text(0.05, 0.95, txt, transform=ax.transAxes, va="top", ha="left", fontsize=10, family="monospace")

    fig.suptitle("深度生态结构分析: 学到的 A 矩阵 vs 真实 LV 系统结构", fontsize=14, fontweight="bold", y=1.01)
    fig.savefig(out_dir / "fig_ecological_structure.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] saved: {out_dir / 'fig_ecological_structure.png'}")

    # Summary markdown
    with open(out_dir / "summary.md", "w", encoding="utf-8") as f:
        f.write("# 生态结构恢复分析\n\n")
        for label, s in all_summary.items():
            f.write(f"## {label}\n\n")
            f.write(f"- Sign accuracy (vs raw 5x5 subset): {s['sign_acc_5x5']:.3f}\n")
            f.write(f"- Sign accuracy (vs eff 5x5 with hidden indirect): {s['sign_acc_eff']:.3f}\n")
            f.write(f"- Pearson (vs raw 5x5): {s['pearson_5x5']:+.3f}\n")
            f.write(f"- Pearson (vs eff 5x5): {s['pearson_eff']:+.3f}\n\n")
    print(f"[OK] saved: {out_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
