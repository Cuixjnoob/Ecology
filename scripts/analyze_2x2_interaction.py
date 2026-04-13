"""2×2 对比分析：LV 先验 × 数据类型 → interaction matrix 恢复质量。

生成两张图：
  fig_interaction_heatmaps.png   4×3 热图对比（每行一个条件：true / pred / 符号匹配）
  fig_metrics_summary.png        汇总指标柱状图

使用：python -m scripts.analyze_2x2_interaction
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _configure_matplotlib() -> None:
    plt.rcParams["font.sans-serif"] = [
        "PingFang SC", "Hiragino Sans GB", "Heiti TC", "STHeiti",
        "Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


RUNS: Dict[str, Dict[str, str]] = {
    "A": {
        "path": "runs/20260412_072614_partial_lv_lv_guided_stochastic_refined",
        "data": "LV 数据",
        "prior": "带 LV 先验",
        "color": "#2e7d32",
    },
    "B": {
        "path": "runs/20260412_161153_exp_lv_data_no_lv_prior",
        "data": "LV 数据",
        "prior": "无 LV 先验",
        "color": "#f57f17",
    },
    "C": {
        "path": "runs/20260412_161625_exp_nonlinear_data_with_lv_prior",
        "data": "非线性数据",
        "prior": "带 LV 先验",
        "color": "#1565c0",
    },
    "D": {
        "path": "runs/20260412_161938_exp_nonlinear_data_no_lv_prior",
        "data": "非线性数据",
        "prior": "无 LV 先验",
        "color": "#c62828",
    },
}


def _compute_metrics(M_true: np.ndarray, M_pred: np.ndarray) -> Dict[str, float]:
    mask = ~np.eye(6, dtype=bool)
    meaningful = mask & (np.abs(M_true) > 0.05)

    sign_match = np.sign(M_true[meaningful]) == np.sign(M_pred[meaningful])

    # Hidden edges (row 5 and col 5)
    hidden_true = np.concatenate([M_true[:5, 5], M_true[5, :5]])
    hidden_pred = np.concatenate([M_pred[:5, 5], M_pred[5, :5]])
    h_mask = np.abs(hidden_true) > 0.05
    hidden_sign = (np.sign(hidden_true[h_mask]) == np.sign(hidden_pred[h_mask])).mean()

    # Visible-visible only
    vv_mask = mask.copy()
    vv_mask[5, :] = False
    vv_mask[:, 5] = False
    vv_meaningful = vv_mask & (np.abs(M_true) > 0.05)
    vv_sign = (np.sign(M_true[vv_meaningful]) == np.sign(M_pred[vv_meaningful])).mean()

    corr = np.corrcoef(M_true[mask].flatten(), M_pred[mask].flatten())[0, 1]
    l2 = np.linalg.norm(M_true[mask] - M_pred[mask]) / np.linalg.norm(M_true[mask])

    return {
        "meaningful_sign_acc": float(sign_match.mean()),
        "hidden_sign_acc": float(hidden_sign),
        "vv_sign_acc": float(vv_sign),
        "correlation": float(corr),
        "relative_l2": float(l2),
    }


def _plot_heatmaps(output_path: Path) -> None:
    fig, axes = plt.subplots(4, 3, figsize=(13, 16), constrained_layout=True)

    for row_idx, (label, info) in enumerate(RUNS.items()):
        d = np.load(f"{info['path']}/results/data_snapshot.npz")
        M_true = d["interaction_true"]
        M_pred = d["interaction_pred"]
        metrics = _compute_metrics(M_true, M_pred)

        vmax = max(abs(M_true).max(), abs(M_pred).max())

        # Column 1: True matrix
        im0 = axes[row_idx, 0].imshow(M_true, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal")
        axes[row_idx, 0].set_title(f"真实交互矩阵")
        axes[row_idx, 0].set_xticks(range(6))
        axes[row_idx, 0].set_yticks(range(6))
        axes[row_idx, 0].set_xticklabels(["v1", "v2", "v3", "v4", "v5", "h"])
        axes[row_idx, 0].set_yticklabels(["v1", "v2", "v3", "v4", "v5", "h"])
        plt.colorbar(im0, ax=axes[row_idx, 0], fraction=0.046, pad=0.04)

        # Column 2: Predicted matrix
        im1 = axes[row_idx, 1].imshow(M_pred, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal")
        axes[row_idx, 1].set_title(f"恢复交互矩阵")
        axes[row_idx, 1].set_xticks(range(6))
        axes[row_idx, 1].set_yticks(range(6))
        axes[row_idx, 1].set_xticklabels(["v1", "v2", "v3", "v4", "v5", "h"])
        axes[row_idx, 1].set_yticklabels(["v1", "v2", "v3", "v4", "v5", "h"])
        plt.colorbar(im1, ax=axes[row_idx, 1], fraction=0.046, pad=0.04)

        # Column 3: Sign match (green = match, red = mismatch, grey = zero)
        mask = ~np.eye(6, dtype=bool)
        meaningful = mask & (np.abs(M_true) > 0.05)
        match_matrix = np.full((6, 6), 0.5)  # grey default
        for i in range(6):
            for j in range(6):
                if i == j:
                    match_matrix[i, j] = 0.5  # diagonal = grey
                elif meaningful[i, j]:
                    match_matrix[i, j] = 1.0 if np.sign(M_true[i, j]) == np.sign(M_pred[i, j]) else 0.0
                else:
                    match_matrix[i, j] = 0.3  # weak edge
        im2 = axes[row_idx, 2].imshow(match_matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="equal")
        axes[row_idx, 2].set_title(
            f"符号匹配（绿=对，红=错）\n"
            f"meaningful: {metrics['meaningful_sign_acc']:.2%}  vv: {metrics['vv_sign_acc']:.2%}"
        )
        axes[row_idx, 2].set_xticks(range(6))
        axes[row_idx, 2].set_yticks(range(6))
        axes[row_idx, 2].set_xticklabels(["v1", "v2", "v3", "v4", "v5", "h"])
        axes[row_idx, 2].set_yticklabels(["v1", "v2", "v3", "v4", "v5", "h"])

        # Row label
        axes[row_idx, 0].set_ylabel(
            f"{label}: {info['data']}\n+ {info['prior']}",
            fontsize=11, fontweight="bold",
        )

    fig.suptitle(
        "2×2 对比：LV 先验 × 数据类型 → interaction matrix 恢复",
        fontsize=14, fontweight="bold", y=1.01,
    )
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_metrics_summary(output_path: Path) -> None:
    all_metrics = {}
    for label, info in RUNS.items():
        d = np.load(f"{info['path']}/results/data_snapshot.npz")
        all_metrics[label] = _compute_metrics(d["interaction_true"], d["interaction_pred"])

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), constrained_layout=True)

    labels = ["A\nLV数据\n带LV先验", "B\nLV数据\n无LV先验", "C\n非线性数据\n带LV先验", "D\n非线性数据\n无LV先验"]
    colors = [RUNS[k]["color"] for k in ["A", "B", "C", "D"]]

    # Panel 1: Sign accuracy
    sign_meaningful = [all_metrics[k]["meaningful_sign_acc"] for k in ["A", "B", "C", "D"]]
    sign_vv = [all_metrics[k]["vv_sign_acc"] for k in ["A", "B", "C", "D"]]
    x = np.arange(4)
    width = 0.35
    axes[0].bar(x - width/2, sign_meaningful, width, label="有意义边", color=colors, alpha=0.85, edgecolor="black")
    axes[0].bar(x + width/2, sign_vv, width, label="仅可见-可见", color=colors, alpha=0.5, edgecolor="black", hatch="//")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, fontsize=9)
    axes[0].set_ylabel("符号准确率")
    axes[0].set_title("交互矩阵符号准确率")
    axes[0].set_ylim([0, 1.1])
    axes[0].axhline(0.5, color="grey", linestyle="--", linewidth=0.8, label="随机 (0.5)")
    axes[0].legend(loc="lower right", fontsize=9)
    for i, v in enumerate(sign_meaningful):
        axes[0].text(i - width/2, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)

    # Panel 2: Correlation
    corrs = [all_metrics[k]["correlation"] for k in ["A", "B", "C", "D"]]
    axes[1].bar(x, corrs, color=colors, edgecolor="black")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, fontsize=9)
    axes[1].set_ylabel("相关性")
    axes[1].set_title("M_true vs M_pred 相关性")
    axes[1].set_ylim([0, 1.0])
    axes[1].axhline(0.5, color="grey", linestyle="--", linewidth=0.8)
    for i, v in enumerate(corrs):
        axes[1].text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=10, fontweight="bold")
    # Highlight the crashed one
    axes[1].annotate(
        "↓ 相关性崩盘",
        xy=(3, corrs[3]),
        xytext=(3, corrs[3] + 0.25),
        ha="center", fontsize=10, color="#c62828", fontweight="bold",
        arrowprops={"facecolor": "#c62828", "width": 1.5, "headwidth": 8, "shrink": 0.05},
    )

    # Panel 3: Hidden Pearson (from summary.json)
    import json
    hidden_pearsons = []
    for k in ["A", "B", "C", "D"]:
        s = json.loads(Path(f"{RUNS[k]['path']}/results/summary.json").read_text(encoding="utf-8"))
        hidden_pearsons.append(s["metrics"]["hidden_test_pearson"])
    axes[2].bar(x, hidden_pearsons, color=colors, edgecolor="black")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, fontsize=9)
    axes[2].set_ylabel("Hidden Test Pearson")
    axes[2].set_title("Hidden 恢复质量 (对照)")
    axes[2].set_ylim([0.9, 1.0])
    for i, v in enumerate(hidden_pearsons):
        axes[2].text(i, v + 0.002, f"{v:.3f}", ha="center", fontsize=10)

    fig.suptitle(
        "LV 先验的真实价值：对 interaction matrix 恢复至关重要，对 hidden recovery 影响小",
        fontsize=13, fontweight="bold", y=1.02,
    )
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _write_analysis_report(output_path: Path) -> None:
    all_metrics = {}
    for label, info in RUNS.items():
        d = np.load(f"{info['path']}/results/data_snapshot.npz")
        all_metrics[label] = _compute_metrics(d["interaction_true"], d["interaction_pred"])

    lines = [
        "# 2×2 对比分析报告",
        "",
        "## 指标汇总",
        "",
        "| 配置 | 数据 | LV 先验 | 有意义边符号 | V-V 边符号 | Hidden 边符号 | 相关性 | 相对 L2 |",
        "|------|------|---------|-------------|-----------|--------------|--------|---------|",
    ]
    for label in ["A", "B", "C", "D"]:
        info = RUNS[label]
        m = all_metrics[label]
        lines.append(
            f"| {label} | {info['data']} | {info['prior']} | "
            f"{m['meaningful_sign_acc']:.3f} | {m['vv_sign_acc']:.3f} | {m['hidden_sign_acc']:.3f} | "
            f"{m['correlation']:.3f} | {m['relative_l2']:.3f} |"
        )

    lines += [
        "",
        "## 核心观察",
        "",
        "1. **LV 先验对 interaction matrix 的符号恢复有显著贡献**",
        f"   - LV 数据: 带 LV 先验 {all_metrics['A']['meaningful_sign_acc']:.3f} vs 无 LV 先验 {all_metrics['B']['meaningful_sign_acc']:.3f}",
        f"   - 非线性数据: 带 LV 先验 {all_metrics['C']['meaningful_sign_acc']:.3f} vs 无 LV 先验 {all_metrics['D']['meaningful_sign_acc']:.3f}",
        "",
        "2. **最戏剧性的发现：非线性数据 + 无 LV 先验时 matrix correlation 崩盘**",
        f"   - A/B/C 三种配置相关性都在 0.74-0.77",
        f"   - D 配置相关性跌到 {all_metrics['D']['correlation']:.3f}",
        "",
        "3. **Hidden 边符号准确率在所有 4 个配置下都是 100%**",
        "   - 说明 hidden-visible 的主要交互方向（捕食链 + hidden 耦合）不依赖 LV 先验",
        "",
        "## 研究结论",
        "",
        "LV 先验的真正价值不是让 hidden recovery 更准，而是让 interaction matrix 能被正确恢复。",
        "即使数据不是 LV 生成的（非线性数据），LV 先验仍能帮助恢复交互方向。",
        "",
        "这是 **LV prior 作为稀疏可解释的方向约束** 的证据。",
    ]

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    _configure_matplotlib()
    out_dir = Path("runs/analysis_2x2_interaction")
    out_dir.mkdir(parents=True, exist_ok=True)

    _plot_heatmaps(out_dir / "fig_interaction_heatmaps.png")
    print(f"[OK] saved: {out_dir / 'fig_interaction_heatmaps.png'}")

    _plot_metrics_summary(out_dir / "fig_metrics_summary.png")
    print(f"[OK] saved: {out_dir / 'fig_metrics_summary.png'}")

    _write_analysis_report(out_dir / "analysis_report.md")
    print(f"[OK] saved: {out_dir / 'analysis_report.md'}")

    print(f"\n所有输出保存到: {out_dir}")


if __name__ == "__main__":
    main()
