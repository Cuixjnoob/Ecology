"""方法 3 / Stage 2：给定完整状态（visible + hidden），拟合 Ricker 参数。

数学基础:
  数据生成: x_{t+1}[i] = x_t[i] * exp(r_i + A[i,:] @ x_t + noise)
  取对数:  log(x_{t+1}/x_t)[i] = r_i + A[i,:] @ x_t + noise
  这是线性回归！设 z_t = [1, x_t] ∈ R^7，β_i = [r_i, A[i,:]] ∈ R^7
  β_i = (Z^T Z)^{-1} Z^T y_i   (对每个物种独立)

Stage 2 与 Stage 1 完全解耦:
  - 输入: visible[T, 5] + hidden[T, 1] (不管 hidden 是真实还是恢复的)
  - 输出: r_hat (6,), A_hat (6, 6)
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr


def _configure_matplotlib() -> None:
    plt.rcParams["font.sans-serif"] = [
        "PingFang SC", "Hiragino Sans GB", "Heiti TC", "STHeiti",
        "Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def fit_ricker_parameters(
    full_states: np.ndarray,
    clamp_min: float = -1.12,
    clamp_max: float = 0.92,
    ridge_lambda: float = 1e-4,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Args:
        full_states: (T, num_species) 完整状态轨迹
        clamp_min/max: 和数据生成器一致的 log-ratio clamp（数据生成器做了这个 clamp）
        ridge_lambda: L2 正则化强度（防止 Z^T Z 病态）

    Returns:
        growth_rates_hat: (num_species,)
        interaction_matrix_hat: (num_species, num_species), A[i,j] = j 对 i 的效应
        diagnostics: dict with R^2, residual std per species
    """
    T, num_species = full_states.shape
    assert T >= num_species + 2, "Need at least num_species+2 time steps"

    # log-ratio: log(x_{t+1}/x_t) ∈ R^{T-1, num_species}
    # 用 clip 防止 log(0) 或 log(负)
    safe = np.clip(full_states, a_min=1e-6, a_max=None)
    log_ratios = np.log(safe[1:] / safe[:-1])

    # 数据生成器在 Ricker 里对 drive 做了 clamp，所以实际观测的 log_ratio 也被截断。
    # 拟合时也 clamp 输入，避免极端值主导回归。
    log_ratios_clamped = np.clip(log_ratios, clamp_min, clamp_max)

    # 设计矩阵: Z = [1, x_t]
    Z = np.concatenate(
        [np.ones((T - 1, 1)), full_states[:-1]], axis=1
    )  # (T-1, num_species+1)

    # Ridge regression for numerical stability: β = (Z^T Z + λI)^{-1} Z^T y
    ZtZ = Z.T @ Z
    ZtZ_reg = ZtZ + ridge_lambda * np.eye(ZtZ.shape[0])
    beta = np.linalg.solve(ZtZ_reg, Z.T @ log_ratios_clamped)  # (num_species+1, num_species)

    growth_rates_hat = beta[0, :]             # (num_species,)
    interaction_matrix_hat = beta[1:, :].T    # (num_species, num_species), i,j = effect of j on i

    # Diagnostics
    pred = Z @ beta
    residuals = log_ratios_clamped - pred
    # R^2 per species
    ss_res = (residuals ** 2).sum(axis=0)
    ss_tot = ((log_ratios_clamped - log_ratios_clamped.mean(axis=0, keepdims=True)) ** 2).sum(axis=0)
    r_squared = 1.0 - ss_res / np.clip(ss_tot, 1e-8, None)
    resid_std = residuals.std(axis=0)

    diagnostics = {
        "r_squared_per_species": r_squared.tolist(),
        "r_squared_mean": float(r_squared.mean()),
        "residual_std_per_species": resid_std.tolist(),
        "residual_std_mean": float(resid_std.mean()),
        "n_samples": int(T - 1),
    }
    return growth_rates_hat, interaction_matrix_hat, diagnostics


def _load_run_data(run_path: str) -> Dict[str, np.ndarray]:
    d = np.load(f"{run_path}/results/data_snapshot.npz")
    return {
        "visible_true": d["visible_true_full"],      # (T, 5)
        "hidden_true": d["hidden_true_full"],        # (T, 1)
        "hidden_pred": d["hidden_pred_full"],        # (T, 1) with NaN in first history_length-1 steps
        "growth_true": d["growth_rates_true"],       # (6,)
        "interaction_true": d["interaction_true"],   # (6, 6)
        "growth_e2e": d["growth_rates_pred"],        # (6,) end-to-end model
        "interaction_e2e": d["interaction_pred"],    # (6, 6)
    }


def _compute_recovery_metrics(
    true_growth: np.ndarray, pred_growth: np.ndarray,
    true_interaction: np.ndarray, pred_interaction: np.ndarray,
) -> Dict[str, float]:
    # growth rates
    g_sign = float((np.sign(true_growth) == np.sign(pred_growth)).mean())
    try:
        g_spearman, _ = spearmanr(true_growth, pred_growth)
    except Exception:
        g_spearman = float("nan")
    g_corr = float(np.corrcoef(true_growth, pred_growth)[0, 1])
    g_l2_rel = float(np.linalg.norm(true_growth - pred_growth) / np.linalg.norm(true_growth))
    g_scale = float(np.linalg.norm(pred_growth) / max(np.linalg.norm(true_growth), 1e-8))

    # interaction matrix
    mask = ~np.eye(6, dtype=bool)
    off_true = true_interaction[mask]
    off_pred = pred_interaction[mask]
    meaningful = np.abs(off_true) > 0.05
    i_sign = float((np.sign(off_true[meaningful]) == np.sign(off_pred[meaningful])).mean())
    i_corr = float(np.corrcoef(off_true, off_pred)[0, 1])
    try:
        i_spearman, _ = spearmanr(off_true, off_pred)
    except Exception:
        i_spearman = float("nan")
    i_l2_rel = float(np.linalg.norm(off_true - off_pred) / np.linalg.norm(off_true))
    i_scale = float(np.linalg.norm(off_pred) / max(np.linalg.norm(off_true), 1e-8))

    # diagonal
    diag_true = true_interaction.diagonal()
    diag_pred = pred_interaction.diagonal()
    d_sign = float((np.sign(diag_true) == np.sign(diag_pred)).mean())
    try:
        d_spearman, _ = spearmanr(diag_true, diag_pred)
    except Exception:
        d_spearman = float("nan")
    d_l2_rel = float(np.linalg.norm(diag_true - diag_pred) / np.linalg.norm(diag_true))
    d_scale = float(np.linalg.norm(diag_pred) / max(np.linalg.norm(diag_true), 1e-8))

    return {
        "growth_sign": g_sign,
        "growth_spearman": float(g_spearman) if not np.isnan(g_spearman) else float("nan"),
        "growth_pearson": g_corr,
        "growth_l2_rel": g_l2_rel,
        "growth_scale": g_scale,
        "interaction_sign_meaningful": i_sign,
        "interaction_pearson": i_corr,
        "interaction_spearman": float(i_spearman) if not np.isnan(i_spearman) else float("nan"),
        "interaction_l2_rel": i_l2_rel,
        "interaction_scale": i_scale,
        "diagonal_sign": d_sign,
        "diagonal_spearman": float(d_spearman) if not np.isnan(d_spearman) else float("nan"),
        "diagonal_l2_rel": d_l2_rel,
        "diagonal_scale": d_scale,
    }


def run_stage2_pipeline(run_path: str, label: str) -> Dict[str, Any]:
    """对一个 run 做 Stage 2 完整流水线：Oracle / Pipeline / Baseline 三种对比。"""
    data = _load_run_data(run_path)

    # --- Oracle: 用真实 hidden 做 Stage 2 ---
    full_oracle = np.concatenate([data["visible_true"], data["hidden_true"]], axis=1)
    g_oracle, I_oracle, diag_oracle = fit_ricker_parameters(full_oracle)

    # --- Pipeline: 用 Stage 1 恢复的 hidden 做 Stage 2 ---
    # hidden_pred 前 history_length-1 步是 NaN，需要排除
    hidden_pred = data["hidden_pred"]
    valid_mask = ~np.isnan(hidden_pred[:, 0])
    # 对应时间点的 visible 也要对齐
    visible_masked = data["visible_true"][valid_mask]
    hidden_masked = hidden_pred[valid_mask]
    full_pipeline = np.concatenate([visible_masked, hidden_masked], axis=1)
    g_pipeline, I_pipeline, diag_pipeline = fit_ricker_parameters(full_pipeline)

    # --- Baseline: 直接用端到端模型的参数 ---
    g_baseline = data["growth_e2e"]
    I_baseline = data["interaction_e2e"]

    # 指标
    metrics_oracle = _compute_recovery_metrics(
        data["growth_true"], g_oracle, data["interaction_true"], I_oracle
    )
    metrics_pipeline = _compute_recovery_metrics(
        data["growth_true"], g_pipeline, data["interaction_true"], I_pipeline
    )
    metrics_baseline = _compute_recovery_metrics(
        data["growth_true"], g_baseline, data["interaction_true"], I_baseline
    )

    return {
        "label": label,
        "run_path": run_path,
        "data": data,
        "oracle": {
            "growth_fit": g_oracle, "interaction_fit": I_oracle,
            "diagnostics": diag_oracle, "metrics": metrics_oracle,
        },
        "pipeline": {
            "growth_fit": g_pipeline, "interaction_fit": I_pipeline,
            "diagnostics": diag_pipeline, "metrics": metrics_pipeline,
        },
        "baseline": {
            "growth_fit": g_baseline, "interaction_fit": I_baseline,
            "diagnostics": None, "metrics": metrics_baseline,
        },
    }


def _plot_three_way_comparison(results: list, output_path: Path) -> None:
    """画每个 run 的 Oracle/Pipeline/Baseline 参数对比。"""
    num_runs = len(results)
    fig, axes = plt.subplots(num_runs, 3, figsize=(15, 4.5 * num_runs), constrained_layout=True)
    if num_runs == 1:
        axes = axes.reshape(1, 3)

    for row_idx, result in enumerate(results):
        data = result["data"]
        # Growth rates
        species_labels = ["v1", "v2", "v3", "v4", "v5", "h"]
        x = np.arange(6)
        width = 0.22
        ax = axes[row_idx, 0]
        ax.bar(x - 1.5*width, data["growth_true"], width, label="真实", color="#555", edgecolor="black")
        ax.bar(x - 0.5*width, result["oracle"]["growth_fit"], width, label="Oracle (真hidden)", color="#2e7d32", edgecolor="black", alpha=0.85)
        ax.bar(x + 0.5*width, result["pipeline"]["growth_fit"], width, label="Pipeline (恢复hidden)", color="#1565c0", edgecolor="black", alpha=0.85)
        ax.bar(x + 1.5*width, result["baseline"]["growth_fit"], width, label="Baseline (端到端)", color="#c62828", edgecolor="black", alpha=0.85)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(species_labels)
        ax.set_ylabel("增长率 r")
        ax.set_title(f"{result['label']}: Growth Rates")
        ax.legend(fontsize=8, loc="best")

        # Diagonal (death rate)
        diag_true = data["interaction_true"].diagonal()
        diag_oracle = result["oracle"]["interaction_fit"].diagonal()
        diag_pipeline = result["pipeline"]["interaction_fit"].diagonal()
        diag_baseline = result["baseline"]["interaction_fit"].diagonal()
        ax = axes[row_idx, 1]
        ax.bar(x - 1.5*width, diag_true, width, label="真实", color="#555", edgecolor="black")
        ax.bar(x - 0.5*width, diag_oracle, width, label="Oracle", color="#2e7d32", edgecolor="black", alpha=0.85)
        ax.bar(x + 0.5*width, diag_pipeline, width, label="Pipeline", color="#1565c0", edgecolor="black", alpha=0.85)
        ax.bar(x + 1.5*width, diag_baseline, width, label="Baseline", color="#c62828", edgecolor="black", alpha=0.85)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(species_labels)
        ax.set_ylabel("对角线 A[i,i]")
        ax.set_title(f"{result['label']}: Self-limitation / Death rate")
        ax.legend(fontsize=8, loc="best")

        # Interaction (off-diagonal scatter: true vs fit)
        ax = axes[row_idx, 2]
        mask = ~np.eye(6, dtype=bool)
        off_true = data["interaction_true"][mask]
        ax.scatter(off_true, result["oracle"]["interaction_fit"][mask], alpha=0.7, color="#2e7d32", s=60, edgecolor="black", label="Oracle")
        ax.scatter(off_true, result["pipeline"]["interaction_fit"][mask], alpha=0.7, color="#1565c0", s=60, edgecolor="black", label="Pipeline")
        ax.scatter(off_true, result["baseline"]["interaction_fit"][mask], alpha=0.7, color="#c62828", s=60, edgecolor="black", label="Baseline", marker="^")
        vmin = min(off_true.min(), result["oracle"]["interaction_fit"][mask].min(), result["pipeline"]["interaction_fit"][mask].min())
        vmax = max(off_true.max(), result["oracle"]["interaction_fit"][mask].max(), result["pipeline"]["interaction_fit"][mask].max())
        ax.plot([vmin, vmax], [vmin, vmax], "k--", linewidth=0.8, alpha=0.5)
        ax.axhline(0, color="grey", linewidth=0.5, alpha=0.3)
        ax.axvline(0, color="grey", linewidth=0.5, alpha=0.3)
        ax.set_xlabel("真实 A[i,j]")
        ax.set_ylabel("拟合 A[i,j]")
        ax.set_title(f"{result['label']}: Off-diagonal")
        ax.legend(fontsize=9)

    fig.suptitle(
        "方法 3: 两阶段参数恢复对比 (Oracle / Pipeline / Baseline)",
        fontsize=14, fontweight="bold", y=1.01,
    )
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_metrics_summary(results: list, output_path: Path) -> None:
    """画三种方法的核心指标对比柱状图。"""
    num_runs = len(results)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)

    methods = ["oracle", "pipeline", "baseline"]
    method_labels = ["Oracle\n(真hidden)", "Pipeline\n(恢复hidden)", "Baseline\n(端到端)"]
    method_colors = ["#2e7d32", "#1565c0", "#c62828"]

    # For each metric group, compute average across runs
    metric_panels = [
        ("growth_spearman", "Growth Rates\nSpearman 秩相关", -0.2, 1.0),
        ("interaction_sign_meaningful", "Interaction\n有意义边符号准确率", 0.0, 1.05),
        ("interaction_pearson", "Interaction\nPearson 相关", -0.1, 1.0),
    ]

    for panel_idx, (metric_key, title, ymin, ymax) in enumerate(metric_panels):
        ax = axes[panel_idx]
        x = np.arange(num_runs)
        width = 0.25
        for i, (method, color, mlabel) in enumerate(zip(methods, method_colors, method_labels)):
            vals = [r[method]["metrics"][metric_key] for r in results]
            vals = [0 if np.isnan(v) else v for v in vals]
            ax.bar(x + (i - 1) * width, vals, width, label=mlabel, color=color, edgecolor="black", alpha=0.85)
            for j, v in enumerate(vals):
                ax.text(j + (i - 1) * width, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels([r["label"] for r in results], fontsize=9)
        ax.set_ylabel("指标值")
        ax.set_title(title, fontsize=11)
        ax.set_ylim([ymin, ymax])
        ax.axhline(0, color="grey", linewidth=0.5)
        if panel_idx == 0:
            ax.legend(fontsize=9, loc="lower right")

    fig.suptitle(
        "方法 3 vs 端到端：参数恢复质量对比",
        fontsize=13, fontweight="bold", y=1.02,
    )
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _write_report(results: list, output_path: Path) -> None:
    lines = [
        "# 方法 3 (两阶段) 参数恢复分析报告",
        "",
        "## 方法",
        "",
        "- **Stage 1**: visible → encoder → hidden 恢复（复用已训模型）",
        "- **Stage 2**: (visible + hidden) → Ricker 对数线性回归 → 参数",
        "",
        "Ricker 关系: `log(x_{t+1}/x_t) = r + A·x_t + noise`，对每个物种独立做最小二乘。",
        "",
        "## 三种方法",
        "",
        "- **Oracle**: 用真实 hidden 做 Stage 2（参数恢复的上界）",
        "- **Pipeline**: 用 Stage 1 恢复的 hidden 做 Stage 2（实际方法 3 流水线）",
        "- **Baseline**: 直接用端到端模型的参数（之前的做法）",
        "",
    ]
    for result in results:
        lines.append(f"## {result['label']}")
        lines.append("")
        lines.append(f"Run: `{result['run_path']}`")
        lines.append("")
        lines.append(f"Stage 2 拟合质量 (Oracle): R^2 = {result['oracle']['diagnostics']['r_squared_mean']:.3f}, residual std = {result['oracle']['diagnostics']['residual_std_mean']:.4f}")
        lines.append(f"Stage 2 拟合质量 (Pipeline): R^2 = {result['pipeline']['diagnostics']['r_squared_mean']:.3f}, residual std = {result['pipeline']['diagnostics']['residual_std_mean']:.4f}")
        lines.append("")
        lines.append("| 指标 | Oracle | Pipeline | Baseline |")
        lines.append("|------|--------|----------|----------|")
        for key, name in [
            ("growth_sign", "Growth 符号准确率"),
            ("growth_spearman", "Growth Spearman"),
            ("growth_pearson", "Growth Pearson"),
            ("growth_l2_rel", "Growth 相对 L2"),
            ("growth_scale", "Growth Scale 比"),
            ("diagonal_sign", "Diagonal 符号"),
            ("diagonal_spearman", "Diagonal Spearman"),
            ("diagonal_scale", "Diagonal Scale 比"),
            ("interaction_sign_meaningful", "Interaction 有意义边符号"),
            ("interaction_pearson", "Interaction Pearson"),
            ("interaction_spearman", "Interaction Spearman"),
            ("interaction_scale", "Interaction Scale 比"),
        ]:
            o = result["oracle"]["metrics"][key]
            p = result["pipeline"]["metrics"][key]
            b = result["baseline"]["metrics"][key]
            lines.append(f"| {name} | {o:.3f} | {p:.3f} | {b:.3f} |")
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    _configure_matplotlib()
    out_dir = Path("runs/analysis_stage2_parameter_fit")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 对几个典型 run 做方法 3 分析
    target_runs = [
        {
            "path": "runs/20260412_165241_partial_lv_lv_guided_stochastic_refined",
            "label": "LV数据+LV先验+tanh",
        },
        {
            "path": "runs/20260412_165541_exp_lv_data_no_lv_prior",
            "label": "LV数据+无LV先验",
        },
        {
            "path": "runs/20260412_165809_exp_nonlinear_data_with_lv_prior",
            "label": "非线性+LV先验+tanh",
        },
    ]
    # 找最新的 ricker run
    import glob
    ricker_runs = sorted(glob.glob("runs/*exp_lv_data_ricker_form"))
    if ricker_runs:
        target_runs.append({"path": ricker_runs[-1], "label": "LV数据+LV先验+Ricker"})

    results = []
    for info in target_runs:
        if not Path(info["path"]).exists():
            print(f"[SKIP] run not found: {info['path']}")
            continue
        result = run_stage2_pipeline(info["path"], info["label"])
        results.append(result)
        m_o = result["oracle"]["metrics"]
        m_p = result["pipeline"]["metrics"]
        m_b = result["baseline"]["metrics"]
        print(f"\n=== {info['label']} ===")
        print(f"  Oracle   growth spearman: {m_o['growth_spearman']:.3f}, int. sign: {m_o['interaction_sign_meaningful']:.3f}, int. pearson: {m_o['interaction_pearson']:.3f}")
        print(f"  Pipeline growth spearman: {m_p['growth_spearman']:.3f}, int. sign: {m_p['interaction_sign_meaningful']:.3f}, int. pearson: {m_p['interaction_pearson']:.3f}")
        print(f"  Baseline growth spearman: {m_b['growth_spearman']:.3f}, int. sign: {m_b['interaction_sign_meaningful']:.3f}, int. pearson: {m_b['interaction_pearson']:.3f}")
        print(f"  Stage 2 R^2 (Oracle/Pipeline): {result['oracle']['diagnostics']['r_squared_mean']:.3f} / {result['pipeline']['diagnostics']['r_squared_mean']:.3f}")

    _plot_three_way_comparison(results, out_dir / "fig_three_way_comparison.png")
    print(f"\n[OK] saved: {out_dir / 'fig_three_way_comparison.png'}")
    _plot_metrics_summary(results, out_dir / "fig_metrics_summary.png")
    print(f"[OK] saved: {out_dir / 'fig_metrics_summary.png'}")
    _write_report(results, out_dir / "stage2_report.md")
    print(f"[OK] saved: {out_dir / 'stage2_report.md'}")


if __name__ == "__main__":
    main()
