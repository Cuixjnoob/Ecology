"""重新绘制中期报告图 2：三数据集主结果 vs 监督基线.

关键改进:
- 并列柱形 (supervised vs unsupervised), 不再用折线连三个数据集
- 误差棒清晰
- Holling 上 "反超" 用箭头+标注强调
- Y 轴范围合理, 留标注空间
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def configure_font():
    plt.rcParams["font.sans-serif"] = [
        "PingFang SC", "Microsoft YaHei", "SimHei", "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.size"] = 12


def main():
    configure_font()
    out_path = Path("生态中期-fig2.png")

    datasets = ["Synthetic LV", "Synthetic Holling", "Portal OT (real)"]
    unsup_mean = [0.727, 0.738, 0.140]
    unsup_std  = [0.216, 0.190, 0.087]
    unsup_max  = [0.919, 0.955, 0.307]
    supervised = [0.977, 0.620, 0.353]

    x = np.arange(len(datasets))
    w = 0.32  # bar width

    fig, ax = plt.subplots(figsize=(10, 6.2), constrained_layout=True)

    # 柱形 1: 监督基线（浅灰色，表示参考线性质）
    bars_sup = ax.bar(
        x - w/2, supervised, w,
        color="#9e9e9e", edgecolor="black", linewidth=0.8,
        label="监督基线 (Linear Sparse + EM)",
        zorder=3,
    )

    # 柱形 2: 无监督方法 mean ± std（主色蓝）
    bars_unsup = ax.bar(
        x + w/2, unsup_mean, w,
        yerr=unsup_std, capsize=6,
        color="#1565c0", edgecolor="black", linewidth=0.8,
        error_kw=dict(lw=1.5, ecolor="black"),
        label="本方法 (无监督, 20 seeds, mean ± std)",
        zorder=3,
    )

    # 叠加一层 max 点（橙色菱形，表示每个数据集最佳 seed）
    ax.scatter(
        x + w/2, unsup_max,
        marker="D", s=90,
        color="#ff7f0e", edgecolor="black", linewidth=0.8,
        zorder=5, label="本方法最大值 (best seed)",
    )

    # 文字标注
    for i in range(len(datasets)):
        # supervised 数值
        ax.text(x[i] - w/2, supervised[i] + 0.025, f"{supervised[i]:.3f}",
                ha="center", va="bottom", fontsize=10.5,
                color="#424242", zorder=6)
        # unsup mean
        ax.text(x[i] + w/2, unsup_mean[i] + unsup_std[i] + 0.025,
                f"{unsup_mean[i]:.3f}",
                ha="center", va="bottom", fontsize=10.5,
                color="#0d47a1", fontweight="bold", zorder=6)
        # unsup max (橙色菱形右上方)
        ax.text(x[i] + w/2 + 0.04, unsup_max[i] + 0.005,
                f"max={unsup_max[i]:.3f}",
                ha="left", va="bottom", fontsize=9,
                color="#e65100", zorder=6)

    # Holling 上 "反超" 标注
    holling_idx = 1
    arrow_y_start = supervised[holling_idx] + 0.005
    arrow_y_end = unsup_mean[holling_idx] + 0.002
    ax.annotate(
        "",
        xy=(x[holling_idx] + w/2 - 0.05, arrow_y_end),
        xytext=(x[holling_idx] - w/2 + 0.05, arrow_y_start),
        arrowprops=dict(
            arrowstyle="->", color="#c62828",
            lw=2.2, connectionstyle="arc3,rad=-0.3",
        ),
        zorder=7,
    )
    ax.text(
        x[holling_idx], 0.83,
        "无监督反超监督基线\n(+0.118, +19%)",
        ha="center", va="bottom",
        fontsize=11, color="#c62828", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.35",
                  facecolor="#ffebee", edgecolor="#c62828", linewidth=1.2),
        zorder=8,
    )

    # 美化坐标轴
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=12)
    ax.set_ylabel("Pearson (scale-invariant)", fontsize=12)
    ax.set_ylim(0, 1.15)
    ax.set_yticks(np.arange(0, 1.21, 0.2))

    # 网格
    ax.grid(axis="y", linestyle="--", alpha=0.35, zorder=1)
    ax.set_axisbelow(True)

    # 去除顶部与右侧边框
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    # 标题
    ax.set_title(
        "三类数据集上无监督方法与监督基线对比 (20 seeds)",
        fontsize=13, fontweight="bold", pad=12,
    )

    # 图例（放到右上，避免挡柱子）
    ax.legend(loc="upper right", fontsize=10, framealpha=0.95,
              edgecolor="#bdbdbd")

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] 图已保存到: {out_path.resolve()}")


if __name__ == "__main__":
    main()
