"""方法 3 增强版：Stage 2 恢复所有参数（r, A, environment, pulse）。

关键思路：
  Stage 2 first pass 拟合 log(x_{t+1}/x_t) ≈ r + A·x_t
  → 残差矩阵 R[t, :] ≈ env_loadings·env(t) + pulse_loadings·pulse(t) + noise
  → 对 R 做 SVD，前 k 个左奇异向量 = 恢复的协变量时间序列
  → Stage 2 second pass: 用 (1, x_t, covar_1(t), covar_2(t)) 重新回归

这样完全数据驱动地恢复所有参数，不需要改模型。

对比 4 种 Stage 2 变体:
  1. 无协变量：只用 (1, x)
  2. SVD 恢复：用 (1, x, covar_svd_1, covar_svd_2)
  3. Oracle 协变量：用 (1, x, true_env, true_pulse)
  4. 端到端 baseline：直接读模型参数
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Tuple
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from data.partial_lv_mvp import generate_partial_lv_mvp_system


def _configure_matplotlib() -> None:
    plt.rcParams["font.sans-serif"] = [
        "PingFang SC", "Hiragino Sans GB", "Heiti TC", "STHeiti",
        "Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


@dataclass
class Stage2Result:
    """Stage 2 拟合结果。"""
    growth_rates: np.ndarray          # (num_species,)
    interaction_matrix: np.ndarray    # (num_species, num_species)
    covariate_loadings: np.ndarray    # (num_species, num_covariates), 0 if no covariates
    covariate_series: np.ndarray       # (T-1, num_covariates), 0 if no covariates
    r_squared_mean: float
    residual_std_mean: float
    method: str


def _fit_ricker_regression(
    full_states: np.ndarray,
    covariates: np.ndarray | None = None,
    clamp_min: float = -1.12,
    clamp_max: float = 0.92,
    ridge_lambda: float = 1e-4,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """统一的线性回归拟合器。

    y[t, i] = r_i + sum_j A[i,j] x[t,j] + sum_c loadings[i,c] covar[t,c] + noise

    Returns:
        growth_rates, interaction_matrix, covariate_loadings, residuals, r2_mean, resid_std_mean
    """
    T, num_species = full_states.shape
    safe = np.clip(full_states, 1e-6, None)
    log_ratios = np.log(safe[1:] / safe[:-1])
    log_ratios = np.clip(log_ratios, clamp_min, clamp_max)

    # Build design matrix: [1, x_t, (covariates_t)]
    parts = [np.ones((T - 1, 1)), full_states[:-1]]
    num_covars = 0
    if covariates is not None and covariates.shape[1] > 0:
        # Covariates aligned to t=0..T-2 (same as x[:-1])
        assert covariates.shape[0] == T - 1, f"covariates length {covariates.shape[0]} != T-1 {T-1}"
        parts.append(covariates)
        num_covars = covariates.shape[1]
    Z = np.concatenate(parts, axis=1)  # (T-1, 1 + num_species + num_covars)

    # Ridge regression
    ZtZ = Z.T @ Z
    ZtZ_reg = ZtZ + ridge_lambda * np.eye(ZtZ.shape[0])
    beta = np.linalg.solve(ZtZ_reg, Z.T @ log_ratios)  # (1+num_species+num_covars, num_species)

    growth_rates = beta[0, :]
    interaction_matrix = beta[1:1 + num_species, :].T      # (num_species, num_species)
    if num_covars > 0:
        covariate_loadings = beta[1 + num_species:, :].T   # (num_species, num_covars)
    else:
        covariate_loadings = np.zeros((num_species, 0))

    pred = Z @ beta
    residuals = log_ratios - pred
    ss_res = (residuals ** 2).sum(axis=0)
    ss_tot = ((log_ratios - log_ratios.mean(axis=0, keepdims=True)) ** 2).sum(axis=0)
    r2 = 1.0 - ss_res / np.clip(ss_tot, 1e-8, None)

    return (
        growth_rates, interaction_matrix, covariate_loadings, residuals,
        float(r2.mean()), float(residuals.std(axis=0).mean()),
    )


def fit_stage2_no_covariates(full_states: np.ndarray) -> Stage2Result:
    """Stage 2 变体 1：无协变量。"""
    g, A, loadings, _, r2, std = _fit_ricker_regression(full_states, covariates=None)
    return Stage2Result(g, A, loadings, np.zeros((full_states.shape[0] - 1, 0)), r2, std, "no_covariates")


def fit_stage2_svd_covariates(full_states: np.ndarray, num_components: int = 2) -> Stage2Result:
    """Stage 2 变体 2：用 SVD 从残差恢复协变量。

    流程:
      1. Pass 1: 无协变量拟合 → 得到残差 R
      2. SVD(R) → 取前 num_components 个左奇异向量作为恢复的协变量
      3. Pass 2: 用恢复的协变量重新拟合
    """
    # Pass 1
    _, _, _, residuals, _, _ = _fit_ricker_regression(full_states, covariates=None)

    # SVD: residuals ∈ R^{T-1, num_species}
    # residuals = U @ diag(S) @ V^T
    # U 的前 num_components 列 = 时间序列上的协变量（标准化到单位范数）
    # 对应 S * V[:, :k]^T = 各物种的 loadings
    U, S, Vt = np.linalg.svd(residuals, full_matrices=False)
    # 把奇异值吸收到时间序列里，让 loadings 是单位范数（反过来也可以）
    covar_series = U[:, :num_components] * S[:num_components]  # (T-1, num_components)

    # Pass 2
    g, A, loadings, _, r2, std = _fit_ricker_regression(full_states, covariates=covar_series)
    return Stage2Result(g, A, loadings, covar_series, r2, std, "svd_covariates")


def fit_stage2_oracle_covariates(
    full_states: np.ndarray, env_driver: np.ndarray, pulse_driver: np.ndarray,
) -> Stage2Result:
    """Stage 2 变体 3：直接用真实 env 和 pulse 作为协变量（参数恢复上限）。"""
    # env_driver / pulse_driver shape: (T, 1)，取 t=0..T-2 对齐
    covar_series = np.concatenate([env_driver[:-1], pulse_driver[:-1]], axis=1)  # (T-1, 2)
    g, A, loadings, _, r2, std = _fit_ricker_regression(full_states, covariates=covar_series)
    return Stage2Result(g, A, loadings, covar_series, r2, std, "oracle_covariates")


def _compute_metrics(true_growth, pred_growth, true_interaction, pred_interaction):
    g_sign = float((np.sign(true_growth) == np.sign(pred_growth)).mean())
    try:
        g_spearman, _ = spearmanr(true_growth, pred_growth)
    except Exception:
        g_spearman = float("nan")
    g_pearson = float(np.corrcoef(true_growth, pred_growth)[0, 1])
    g_scale = float(np.linalg.norm(pred_growth) / max(np.linalg.norm(true_growth), 1e-8))
    g_l2_rel = float(np.linalg.norm(true_growth - pred_growth) / np.linalg.norm(true_growth))

    mask = ~np.eye(6, dtype=bool)
    off_true = true_interaction[mask]
    off_pred = pred_interaction[mask]
    meaningful = np.abs(off_true) > 0.05
    i_sign = float((np.sign(off_true[meaningful]) == np.sign(off_pred[meaningful])).mean())
    i_pearson = float(np.corrcoef(off_true, off_pred)[0, 1])
    i_scale = float(np.linalg.norm(off_pred) / max(np.linalg.norm(off_true), 1e-8))
    i_l2_rel = float(np.linalg.norm(off_true - off_pred) / np.linalg.norm(off_true))

    diag_true = true_interaction.diagonal()
    diag_pred = pred_interaction.diagonal()
    d_sign = float((np.sign(diag_true) == np.sign(diag_pred)).mean())
    try:
        d_spearman, _ = spearmanr(diag_true, diag_pred)
    except Exception:
        d_spearman = float("nan")
    d_scale = float(np.linalg.norm(diag_pred) / max(np.linalg.norm(diag_true), 1e-8))

    return {
        "growth_sign": g_sign,
        "growth_spearman": float(g_spearman) if not np.isnan(g_spearman) else 0.0,
        "growth_pearson": g_pearson,
        "growth_scale": g_scale,
        "growth_l2_rel": g_l2_rel,
        "interaction_sign": i_sign,
        "interaction_pearson": i_pearson,
        "interaction_scale": i_scale,
        "interaction_l2_rel": i_l2_rel,
        "diagonal_sign": d_sign,
        "diagonal_spearman": float(d_spearman) if not np.isnan(d_spearman) else 0.0,
        "diagonal_scale": d_scale,
    }


def _regenerate_ground_truth(seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """用同 seed 重新生成数据，获得真实 env/pulse。"""
    system = generate_partial_lv_mvp_system(seed=seed)
    return (
        system.visible_states.numpy(),       # (T, 5)
        system.hidden_states.numpy(),         # (T, 1)
        system.environment_driver.numpy(),    # (T, 1)
        system.pulse_driver.numpy(),          # (T, 1)
        system.growth_rates.numpy(),          # (6,)
        system.interaction_matrix.numpy(),    # (6, 6)
        system.environment_loadings.numpy(),  # (6,)
    )


def run_full_analysis(run_path: str, label: str, seed: int = 42) -> Dict[str, Any]:
    """执行完整分析：4 种 Stage 2 变体 + 比较 + 协变量质量验证。"""
    d = np.load(f"{run_path}/results/data_snapshot.npz")
    visible_true = d["visible_true_full"]
    hidden_true = d["hidden_true_full"]
    hidden_pred = d["hidden_pred_full"]
    growth_true = d["growth_rates_true"]
    interaction_true = d["interaction_true"]
    growth_e2e = d["growth_rates_pred"]
    interaction_e2e = d["interaction_pred"]

    # 重新生成获得 env/pulse ground truth
    _, _, env_driver_true, pulse_driver_true, _, _, env_loadings_true = _regenerate_ground_truth(seed)

    # Pipeline: 用恢复的 hidden
    valid_mask = ~np.isnan(hidden_pred[:, 0])
    visible_masked = visible_true[valid_mask]
    hidden_masked = hidden_pred[valid_mask]
    env_masked = env_driver_true[valid_mask]
    pulse_masked = pulse_driver_true[valid_mask]
    full_pipeline = np.concatenate([visible_masked, hidden_masked], axis=1)

    # Oracle: 用真 hidden
    full_oracle = np.concatenate([visible_true, hidden_true], axis=1)

    variants = {}

    # --- 4 种 Stage 2 变体 on Oracle 和 Pipeline ---
    for context_name, full, env_c, pulse_c in [
        ("Oracle",    full_oracle,   env_driver_true,  pulse_driver_true),
        ("Pipeline",  full_pipeline, env_masked,       pulse_masked),
    ]:
        variants[f"{context_name}|S2-nocovar"] = fit_stage2_no_covariates(full)
        variants[f"{context_name}|S2-svd"] = fit_stage2_svd_covariates(full, num_components=2)
        variants[f"{context_name}|S2-oracle"] = fit_stage2_oracle_covariates(full, env_c, pulse_c)

    # --- Baseline: 端到端模型参数 ---
    baseline_result = Stage2Result(
        growth_e2e, interaction_e2e, np.zeros((6, 0)),
        np.zeros((0, 0)), 0.0, 0.0, "baseline_e2e",
    )
    variants["Baseline|E2E"] = baseline_result

    # 计算每个变体的指标
    all_metrics = {
        name: _compute_metrics(growth_true, v.growth_rates, interaction_true, v.interaction_matrix)
        for name, v in variants.items()
    }

    # --- SVD 恢复的协变量 vs 真实 env/pulse 对比 ---
    svd_pipeline = variants["Pipeline|S2-svd"]
    svd_covars = svd_pipeline.covariate_series  # (T-1, 2)
    true_env_aligned = env_masked[:-1, 0]    # pipeline 用 masked 数据
    true_pulse_aligned = pulse_masked[:-1, 0]

    # 每个 SVD 成分和真 env/pulse 的相关性（取绝对值，SVD 方向可能翻转）
    covar_quality = {
        "svd_1_vs_env": float(np.abs(np.corrcoef(svd_covars[:, 0], true_env_aligned)[0, 1])),
        "svd_1_vs_pulse": float(np.abs(np.corrcoef(svd_covars[:, 0], true_pulse_aligned)[0, 1])),
        "svd_2_vs_env": float(np.abs(np.corrcoef(svd_covars[:, 1], true_env_aligned)[0, 1])),
        "svd_2_vs_pulse": float(np.abs(np.corrcoef(svd_covars[:, 1], true_pulse_aligned)[0, 1])),
    }

    return {
        "label": label,
        "run_path": run_path,
        "truth": {
            "growth": growth_true,
            "interaction": interaction_true,
            "env_driver": env_driver_true,
            "pulse_driver": pulse_driver_true,
            "env_loadings": env_loadings_true,
        },
        "variants": variants,
        "metrics": all_metrics,
        "covar_quality": covar_quality,
        "pipeline_data": {
            "svd_covars": svd_covars,
            "true_env": true_env_aligned,
            "true_pulse": true_pulse_aligned,
        },
    }


def _plot_metrics_heatmap(result: Dict[str, Any], output_path: Path) -> None:
    """画 4 个变体 × 多个指标的热图对比。"""
    metrics_to_show = [
        ("growth_spearman", "Growth\nSpearman"),
        ("growth_sign", "Growth\nSign Acc"),
        ("growth_pearson", "Growth\nPearson"),
        ("growth_scale", "Growth\nScale"),
        ("diagonal_sign", "Diagonal\nSign Acc"),
        ("diagonal_spearman", "Diagonal\nSpearman"),
        ("diagonal_scale", "Diagonal\nScale"),
        ("interaction_sign", "Interaction\nSign Acc"),
        ("interaction_pearson", "Interaction\nPearson"),
        ("interaction_scale", "Interaction\nScale"),
    ]

    variant_names = [
        "Oracle|S2-nocovar", "Oracle|S2-svd", "Oracle|S2-oracle",
        "Pipeline|S2-nocovar", "Pipeline|S2-svd", "Pipeline|S2-oracle",
        "Baseline|E2E",
    ]
    variant_labels = [
        "Oracle\nNo covar",
        "Oracle\nSVD covar",
        "Oracle\nTrue covar",
        "Pipeline\nNo covar",
        "Pipeline\nSVD covar",
        "Pipeline\nTrue covar",
        "Baseline\nE2E",
    ]

    matrix = np.zeros((len(variant_names), len(metrics_to_show)))
    for i, vname in enumerate(variant_names):
        for j, (mkey, _) in enumerate(metrics_to_show):
            matrix[i, j] = result["metrics"][vname][mkey]

    fig, ax = plt.subplots(figsize=(14, 6), constrained_layout=True)
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1.2, aspect="auto")

    ax.set_xticks(range(len(metrics_to_show)))
    ax.set_xticklabels([m[1] for m in metrics_to_show], fontsize=9)
    ax.set_yticks(range(len(variant_names)))
    ax.set_yticklabels(variant_labels, fontsize=9)

    for i in range(len(variant_names)):
        for j in range(len(metrics_to_show)):
            val = matrix[i, j]
            ax.text(
                j, i, f"{val:.2f}",
                ha="center", va="center",
                color="white" if val < 0.4 or val > 1.0 else "black",
                fontsize=8,
            )

    # 分隔 Oracle / Pipeline / Baseline
    ax.axhline(2.5, color="black", linewidth=2)
    ax.axhline(5.5, color="black", linewidth=2)

    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(f"{result['label']}: 参数恢复全景（越绿越好，scale 接近 1 最好）", fontsize=12)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_covar_recovery(result: Dict[str, Any], output_path: Path) -> None:
    """画 SVD 恢复的协变量 vs 真实 env/pulse。"""
    pd = result["pipeline_data"]
    svd_1 = pd["svd_covars"][:, 0]
    svd_2 = pd["svd_covars"][:, 1]
    true_env = pd["true_env"]
    true_pulse = pd["true_pulse"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)

    # 归一化后画时序
    def _norm(x):
        return (x - x.mean()) / (x.std() + 1e-8)

    t = np.arange(len(svd_1))

    axes[0, 0].plot(t, _norm(true_env), label="真实 env", color="#2e7d32", linewidth=1.5)
    axes[0, 0].plot(t, _norm(svd_1), label="SVD comp 1", color="#1565c0", linewidth=1.2, alpha=0.8)
    axes[0, 0].plot(t, _norm(svd_2), label="SVD comp 2", color="#c62828", linewidth=1.0, alpha=0.7, linestyle="--")
    axes[0, 0].set_title("归一化时序：真 env vs SVD 恢复成分")
    axes[0, 0].legend(fontsize=9)
    axes[0, 0].set_xlabel("时间步")

    axes[0, 1].plot(t, _norm(true_pulse), label="真实 pulse", color="#2e7d32", linewidth=1.5)
    axes[0, 1].plot(t, _norm(svd_1), label="SVD comp 1", color="#1565c0", linewidth=1.2, alpha=0.8)
    axes[0, 1].plot(t, _norm(svd_2), label="SVD comp 2", color="#c62828", linewidth=1.0, alpha=0.7, linestyle="--")
    axes[0, 1].set_title("归一化时序：真 pulse vs SVD 恢复成分")
    axes[0, 1].legend(fontsize=9)
    axes[0, 1].set_xlabel("时间步")

    # 散点图：真 vs SVD
    q = result["covar_quality"]
    axes[1, 0].scatter(_norm(true_env), _norm(svd_1), alpha=0.4, s=10, color="#1565c0")
    axes[1, 0].set_xlabel("真实 env (标准化)")
    axes[1, 0].set_ylabel("SVD comp 1 (标准化)")
    axes[1, 0].set_title(f"env vs SVD-1, |corr| = {q['svd_1_vs_env']:.3f}")

    axes[1, 1].scatter(_norm(true_pulse), _norm(svd_2), alpha=0.4, s=10, color="#c62828")
    axes[1, 1].set_xlabel("真实 pulse (标准化)")
    axes[1, 1].set_ylabel("SVD comp 2 (标准化)")
    axes[1, 1].set_title(f"pulse vs SVD-2, |corr| = {q['svd_2_vs_pulse']:.3f}")

    fig.suptitle(
        f"{result['label']}: SVD 协变量恢复质量",
        fontsize=13, fontweight="bold", y=1.02,
    )
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_growth_bars(result: Dict[str, Any], output_path: Path) -> None:
    """画 growth rates 对比柱状图：真实 + 3 种 Pipeline 变体 + Baseline。"""
    truth = result["truth"]["growth"]
    variants = result["variants"]
    fig, ax = plt.subplots(figsize=(12, 5.5), constrained_layout=True)
    x = np.arange(6)
    width = 0.15
    names_colors = [
        ("真实", truth, "#333"),
        ("Pipeline\nNo covar", variants["Pipeline|S2-nocovar"].growth_rates, "#f57f17"),
        ("Pipeline\nSVD covar", variants["Pipeline|S2-svd"].growth_rates, "#1565c0"),
        ("Pipeline\nTrue covar", variants["Pipeline|S2-oracle"].growth_rates, "#2e7d32"),
        ("Baseline\nE2E", variants["Baseline|E2E"].growth_rates, "#c62828"),
    ]
    for i, (name, vals, color) in enumerate(names_colors):
        ax.bar(x + (i - 2) * width, vals, width, label=name, color=color, edgecolor="black", alpha=0.88)
    ax.set_xticks(x)
    ax.set_xticklabels(["v1", "v2", "v3", "v4", "v5", "h"])
    ax.set_ylabel("增长率 r")
    ax.set_title(f"{result['label']}: Growth Rates 恢复对比")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.legend(fontsize=9, loc="best", ncol=5)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _write_report(result: Dict[str, Any], output_path: Path) -> None:
    lines = [
        "# Stage 2 全参数恢复分析报告",
        "",
        "## 方法",
        "",
        "对数线性回归：`log(x_{t+1}/x_t)[i] = r_i + A[i,:]·x_t + loadings[i,:]·covariates_t + noise`",
        "",
        "四种 Stage 2 变体对比：",
        "1. **No covar**：只用 (1, x)，不考虑 env/pulse",
        "2. **SVD covar**：从 Stage 2 残差做 SVD，用前 2 个成分作为恢复的协变量",
        "3. **True covar**：直接用真实 env 和 pulse（参数恢复上限）",
        "4. **Baseline**：直接读端到端模型参数",
        "",
        "## SVD 协变量恢复质量",
        "",
    ]
    q = result["covar_quality"]
    lines.append("| SVD 成分 | vs 真实 env \\|corr\\| | vs 真实 pulse \\|corr\\| |")
    lines.append("|---------|------------------|-------------------|")
    lines.append(f"| SVD-1 | {q['svd_1_vs_env']:.3f} | {q['svd_1_vs_pulse']:.3f} |")
    lines.append(f"| SVD-2 | {q['svd_2_vs_env']:.3f} | {q['svd_2_vs_pulse']:.3f} |")
    lines.append("")

    variant_order = [
        ("Oracle|S2-nocovar", "Oracle + 无协变量"),
        ("Oracle|S2-svd", "Oracle + SVD 协变量"),
        ("Oracle|S2-oracle", "Oracle + 真协变量"),
        ("Pipeline|S2-nocovar", "Pipeline + 无协变量"),
        ("Pipeline|S2-svd", "Pipeline + SVD 协变量"),
        ("Pipeline|S2-oracle", "Pipeline + 真协变量"),
        ("Baseline|E2E", "端到端 Baseline"),
    ]

    lines.append("## 参数恢复指标")
    lines.append("")
    header = "| 变体 | Growth Spearman | Growth Scale | Int Sign | Int Pearson | Int Scale | Diag Scale |"
    sep =    "|------|-----------------|--------------|----------|-------------|-----------|------------|"
    lines.append(header)
    lines.append(sep)
    for key, name in variant_order:
        m = result["metrics"][key]
        lines.append(
            f"| {name} | {m['growth_spearman']:.3f} | {m['growth_scale']:.3f} | "
            f"{m['interaction_sign']:.3f} | {m['interaction_pearson']:.3f} | "
            f"{m['interaction_scale']:.3f} | {m['diagonal_scale']:.3f} |"
        )

    lines += [
        "",
        "## 解读",
        "",
        "- 最好的方法应接近 Oracle + 真协变量（所有协变量都已知）",
        "- 实际可做的是 Pipeline + SVD 协变量（完全从 visible 数据出发）",
        "- Growth Scale 接近 1.0 说明尺度恢复了，远离 1.0 说明系统性偏差",
        "",
    ]

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    _configure_matplotlib()
    out_dir = Path("runs/analysis_stage2_full_recovery")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 目标 run：LV 数据 + LV 先验 + tanh（最佳 hidden recovery）
    target_run = "runs/20260412_165241_partial_lv_lv_guided_stochastic_refined"
    result = run_full_analysis(target_run, "LV数据+LV先验+tanh", seed=42)

    _plot_metrics_heatmap(result, out_dir / "fig_metrics_heatmap.png")
    print(f"[OK] saved: {out_dir / 'fig_metrics_heatmap.png'}")

    _plot_covar_recovery(result, out_dir / "fig_covar_recovery.png")
    print(f"[OK] saved: {out_dir / 'fig_covar_recovery.png'}")

    _plot_growth_bars(result, out_dir / "fig_growth_bars.png")
    print(f"[OK] saved: {out_dir / 'fig_growth_bars.png'}")

    _write_report(result, out_dir / "full_recovery_report.md")
    print(f"[OK] saved: {out_dir / 'full_recovery_report.md'}")

    # Print key metrics
    print("\n" + "=" * 80)
    print("  核心指标对比 (Growth Spearman / Growth Scale / Int Pearson / Int Scale)")
    print("=" * 80)
    variant_order = [
        "Oracle|S2-nocovar", "Oracle|S2-svd", "Oracle|S2-oracle",
        "Pipeline|S2-nocovar", "Pipeline|S2-svd", "Pipeline|S2-oracle",
        "Baseline|E2E",
    ]
    for v in variant_order:
        m = result["metrics"][v]
        print(
            f"  {v:28s}: growth(sp={m['growth_spearman']:+.2f}, scale={m['growth_scale']:.2f})  "
            f"int(pearson={m['interaction_pearson']:+.2f}, scale={m['interaction_scale']:.2f})"
        )

    q = result["covar_quality"]
    print(f"\n  SVD covar quality:")
    print(f"    SVD-1: |corr_env|={q['svd_1_vs_env']:.3f}  |corr_pulse|={q['svd_1_vs_pulse']:.3f}")
    print(f"    SVD-2: |corr_env|={q['svd_2_vs_env']:.3f}  |corr_pulse|={q['svd_2_vs_pulse']:.3f}")


if __name__ == "__main__":
    main()
