"""把 final overview 拆分成 5 张独立的图，每张都可以单独展示。

图表列表：
  fig1_hidden_recovery.png       Hidden species 时间序列恢复
  fig2_env_recovery.png          Environment driver EM 恢复
  fig3_growth_rates.png          Growth rates 对比
  fig4_interaction_heatmap.png   Interaction matrix 热图对比
  fig5_interaction_scatter.png   Interaction 边散点
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from scripts.stage2_full_parameter_recovery import _fit_ricker_regression, _regenerate_ground_truth
from scripts.stage2_em_decomposition import em_decompose_residuals


def _configure_matplotlib() -> None:
    plt.rcParams["font.sans-serif"] = [
        "PingFang SC", "Hiragino Sans GB", "Heiti TC", "STHeiti",
        "Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.size"] = 11


TARGET_RUN = "runs/20260412_165241_partial_lv_lv_guided_stochastic_refined"


def main() -> None:
    _configure_matplotlib()
    out_dir = Path("runs/analysis_split_figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 加载数据
    d = np.load(f"{TARGET_RUN}/results/data_snapshot.npz")
    visible_true = d["visible_true_full"]
    hidden_true = d["hidden_true_full"]
    hidden_pred = d["hidden_pred_full"]
    growth_true = d["growth_rates_true"]
    interaction_true = d["interaction_true"]
    growth_e2e = d["growth_rates_pred"]
    interaction_e2e = d["interaction_pred"]

    _, _, env_true_full, pulse_true_full, _, _, _ = _regenerate_ground_truth(seed=42)

    valid_mask = ~np.isnan(hidden_pred[:, 0])
    visible_masked = visible_true[valid_mask]
    hidden_masked = hidden_pred[valid_mask]
    env_masked = env_true_full[valid_mask, 0]
    full_pipeline = np.concatenate([visible_masked, hidden_masked], axis=1)

    # Stage 2: EM
    _, _, _, residuals, _, _ = _fit_ricker_regression(full_pipeline, covariates=None)
    e_hat, p_hat, b_hat, c_hat, history = em_decompose_residuals(
        residuals, num_iter=100, smooth_lambda=20.0, sparse_lambda=0.005, verbose=False,
    )
    covar_series = np.stack([e_hat, p_hat], axis=1)
    growth_hat, interaction_hat, _, _, _, _ = _fit_ricker_regression(full_pipeline, covariates=covar_series)

    # 指标
    hidden_pearson = float(np.corrcoef(hidden_true[valid_mask, 0], hidden_pred[valid_mask, 0])[0, 1])
    hidden_rmse = float(np.sqrt(((hidden_true[valid_mask, 0] - hidden_pred[valid_mask, 0]) ** 2).mean()))
    env_ref = env_masked[:len(e_hat)]
    env_abs_corr = float(np.abs(np.corrcoef(e_hat, env_ref)[0, 1]))
    env_sign = np.sign(np.corrcoef(e_hat, env_ref)[0, 1])

    g_sp, _ = spearmanr(growth_true, growth_hat)
    g_sp = float(g_sp)
    g_sign = float((np.sign(growth_true) == np.sign(growth_hat)).mean())
    g_scale = float(np.linalg.norm(growth_hat) / np.linalg.norm(growth_true))

    mask = ~np.eye(6, dtype=bool)
    off_true = interaction_true[mask]
    off_hat = interaction_hat[mask]
    off_e2e = interaction_e2e[mask]
    meaningful = np.abs(off_true) > 0.05
    i_sign = float((np.sign(off_true[meaningful]) == np.sign(off_hat[meaningful])).mean())
    i_pearson = float(np.corrcoef(off_true, off_hat)[0, 1])
    i_scale = float(np.linalg.norm(off_hat) / np.linalg.norm(off_true))

    species_labels = ["v1", "v2", "v3", "v4", "v5", "h"]

    # =========================================================================
    # Fig 1: Hidden species recovery
    # =========================================================================
    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    t_hidden = np.arange(len(hidden_true))
    valid_idx = np.where(valid_mask)[0]
    ax.plot(t_hidden, hidden_true[:, 0], color="black", linewidth=1.6, label="真实 hidden 物种")
    ax.plot(valid_idx, hidden_pred[valid_idx, 0], color="#ff7f0e", linewidth=1.2, alpha=0.85, label="模型恢复的 hidden 物种")

    train_end, val_end = 492, 656
    for x in (train_end, val_end):
        ax.axvline(x - 0.5, color="#aaa", linestyle="--", linewidth=0.9)
    ymax = ax.get_ylim()[1]
    ax.text(train_end / 2, ymax * 0.93, "train", ha="center", fontsize=10, color="#666")
    ax.text((train_end + val_end) / 2, ymax * 0.93, "val", ha="center", fontsize=10, color="#666")
    ax.text((val_end + len(t_hidden)) / 2, ymax * 0.93, "test", ha="center", fontsize=10, color="#666")

    ax.set_title(f"图 1：Hidden 物种恢复（Stage 1，编码器映射）\nPearson = {hidden_pearson:.3f}    RMSE = {hidden_rmse:.3f}",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("时间步")
    ax.set_ylabel("丰度")
    ax.legend(loc="upper right", fontsize=11)
    ax.grid(alpha=0.25)

    fig.savefig(out_dir / "fig1_hidden_recovery.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] fig1_hidden_recovery.png")

    # =========================================================================
    # Fig 2: Environment recovery (time series + scatter)
    # =========================================================================
    def _norm(x): return (x - x.mean()) / (x.std() + 1e-8)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), constrained_layout=True,
                              gridspec_kw={"width_ratios": [2, 1]})
    t_env = np.arange(len(e_hat))
    axes[0].plot(t_env, _norm(env_ref), color="#2e7d32", linewidth=1.5, label="真实 environment")
    axes[0].plot(t_env, env_sign * _norm(e_hat), color="#1565c0", linewidth=1.1, alpha=0.85,
                 label="EM 从残差中恢复的 environment")
    axes[0].set_title("时序对比（标准化）", fontsize=12)
    axes[0].set_xlabel("时间步")
    axes[0].set_ylabel("环境值 (标准化)")
    axes[0].legend(loc="upper right", fontsize=10)
    axes[0].grid(alpha=0.25)

    axes[1].scatter(_norm(env_ref), env_sign * _norm(e_hat), alpha=0.4, s=12, color="#1565c0")
    axes[1].plot([-3, 3], [-3, 3], "k--", linewidth=0.8, alpha=0.5)
    axes[1].set_xlabel("真实 environment (标准化)")
    axes[1].set_ylabel("恢复 environment (标准化)")
    axes[1].set_title("散点对齐", fontsize=12)
    axes[1].grid(alpha=0.25)

    fig.suptitle(f"图 2：Environment Driver 恢复（Stage 2 EM 分解）    |corr| = {env_abs_corr:.3f}",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.savefig(out_dir / "fig2_env_recovery.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] fig2_env_recovery.png")

    # =========================================================================
    # Fig 3: Growth rates
    # =========================================================================
    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    x = np.arange(6)
    width = 0.28
    ax.bar(x - width, growth_true, width, label="真实", color="#2b2b2b", edgecolor="black")
    ax.bar(x, growth_hat, width, label="Pipeline (Stage1+Stage2+EM)", color="#1565c0", edgecolor="black", alpha=0.92)
    ax.bar(x + width, growth_e2e, width, label="端到端 Baseline", color="#c62828", edgecolor="black", alpha=0.85)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(species_labels, fontsize=11)
    ax.set_ylabel("Growth rate $r_i$")
    ax.set_title(f"图 3：Growth Rates 恢复\nPipeline: Spearman = {g_sp:.2f}, Sign = {g_sign:.2f}, Scale = {g_scale:.2f}",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="best", fontsize=10)
    ax.grid(alpha=0.25, axis="y")

    # 数值标注
    for i, (t, h, b) in enumerate(zip(growth_true, growth_hat, growth_e2e)):
        ax.text(i - width, t + 0.012, f"{t:.2f}", ha="center", fontsize=8, color="#2b2b2b")
        ax.text(i, h + 0.012, f"{h:.2f}", ha="center", fontsize=8, color="#1565c0")
        ax.text(i + width, b + 0.012 if b >= 0 else b - 0.025, f"{b:.2f}", ha="center", fontsize=8, color="#c62828")

    fig.savefig(out_dir / "fig3_growth_rates.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] fig3_growth_rates.png")

    # =========================================================================
    # Fig 4: Interaction matrix heatmaps (真实 + 恢复 + 端到端)
    # =========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    vmax = max(np.abs(interaction_true).max(), np.abs(interaction_hat).max(), np.abs(interaction_e2e).max())
    for ax_i, M, title in zip(
        axes,
        [interaction_true, interaction_hat, interaction_e2e],
        ["真实 A", "Pipeline 恢复 A", "端到端 Baseline A"],
    ):
        im = ax_i.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal")
        ax_i.set_xticks(range(6))
        ax_i.set_yticks(range(6))
        ax_i.set_xticklabels(species_labels)
        ax_i.set_yticklabels(species_labels)
        ax_i.set_title(title, fontsize=12)
        # 给每个格子写数值
        for i in range(6):
            for j in range(6):
                val = M[i, j]
                color = "white" if abs(val) > vmax * 0.5 else "black"
                ax_i.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color=color)
        plt.colorbar(im, ax=ax_i, fraction=0.046, pad=0.04)

    fig.suptitle(
        f"图 4：Interaction Matrix 恢复对比\n"
        f"Pipeline Pearson = {i_pearson:.2f}, Sign 准确率 = {i_sign:.2f}    vs    端到端 Baseline Pearson ≈ 0.77",
        fontsize=13, fontweight="bold", y=1.02,
    )
    fig.savefig(out_dir / "fig4_interaction_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] fig4_interaction_heatmap.png")

    # =========================================================================
    # Fig 5: Interaction scatter
    # =========================================================================
    fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)
    ax.scatter(off_true, off_hat, alpha=0.75, s=80, color="#1565c0", edgecolor="black",
               label=f"Pipeline (Pearson={i_pearson:.2f})")
    ax.scatter(off_true, off_e2e, alpha=0.75, s=80, color="#c62828", edgecolor="black",
               marker="^", label=f"端到端 Baseline (Pearson≈0.77)")
    vmin = min(off_true.min(), off_hat.min(), off_e2e.min()) - 0.05
    vmax_s = max(off_true.max(), off_hat.max(), off_e2e.max()) + 0.05
    ax.plot([vmin, vmax_s], [vmin, vmax_s], "k--", linewidth=1.0, alpha=0.5, label="y = x 理想线")
    ax.axhline(0, color="grey", linewidth=0.6, alpha=0.4)
    ax.axvline(0, color="grey", linewidth=0.6, alpha=0.4)
    ax.set_xlabel("真实 $A_{i,j}$", fontsize=12)
    ax.set_ylabel("恢复 $A_{i,j}$", fontsize=12)
    ax.set_title(f"图 5：每条交互边的恢复质量\nPipeline Scale = {i_scale:.2f}  (1.0 为完美)",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_aspect("equal")
    ax.set_xlim([vmin, vmax_s])
    ax.set_ylim([vmin, vmax_s])

    fig.savefig(out_dir / "fig5_interaction_scatter.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] fig5_interaction_scatter.png")

    print(f"\n所有图已保存到: {out_dir}")


if __name__ == "__main__":
    main()
