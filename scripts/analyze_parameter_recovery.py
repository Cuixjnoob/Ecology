"""参数恢复分析：对比真实参数和模型学到的参数。

注意：模型用 tanh(r + Ax) * lv_drift_scale 形式，数据用 exp(r + Ax) 形式。
直接比较 growth_rates 或 diagonal 的数值不公平（参数化不同）。
因此我们做三种对比：

  1. 符号准确率：sign(true) vs sign(pred)
  2. 相对幅值（归一化到 [0,1] 后比较）
  3. 秩相关性 (Spearman)：保持排序结构

生成：
  fig_parameter_recovery.png
  parameter_recovery_report.md
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

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


RUNS: Dict[str, Dict[str, str]] = {
    "A": {
        "path": "runs/20260412_165241_partial_lv_lv_guided_stochastic_refined",
        "data": "LV 数据",
        "prior": "带 LV 先验",
        "color": "#2e7d32",
    },
    "B": {
        "path": "runs/20260412_165541_exp_lv_data_no_lv_prior",
        "data": "LV 数据",
        "prior": "无 LV 先验",
        "color": "#f57f17",
    },
    "C": {
        "path": "runs/20260412_165809_exp_nonlinear_data_with_lv_prior",
        "data": "非线性数据",
        "prior": "带 LV 先验",
        "color": "#1565c0",
    },
    "D": {
        "path": "runs/20260412_170106_exp_nonlinear_data_no_lv_prior",
        "data": "非线性数据",
        "prior": "无 LV 先验",
        "color": "#c62828",
    },
}


def _compute_param_metrics(true_vec: np.ndarray, pred_vec: np.ndarray) -> Dict[str, float]:
    """计算多种参数对比指标。"""
    # 1. 符号准确率（只看非零项）
    mask = np.abs(true_vec) > 1e-3
    if mask.sum() > 0:
        sign_acc = float((np.sign(true_vec[mask]) == np.sign(pred_vec[mask])).mean())
    else:
        sign_acc = float("nan")

    # 2. 秩相关（Spearman）— 保持排序结构，不受参数化差异影响
    try:
        rho, _ = spearmanr(true_vec, pred_vec)
    except Exception:
        rho = float("nan")

    # 3. Pearson 相关（经过 z-score 归一化后）
    true_centered = true_vec - true_vec.mean()
    pred_centered = pred_vec - pred_vec.mean()
    denom = np.sqrt(true_centered @ true_centered) * np.sqrt(pred_centered @ pred_centered)
    pearson = float(true_centered @ pred_centered / denom) if denom > 1e-8 else float("nan")

    # 4. 相对尺度（pred 与 true 的 L2 范数比，1.0 = 完美匹配）
    scale_ratio = float(np.linalg.norm(pred_vec) / (np.linalg.norm(true_vec) + 1e-8))

    # 5. 归一化 RMSE（到 true 的范围）
    true_range = true_vec.max() - true_vec.min()
    if true_range > 1e-6:
        nrmse = float(np.sqrt(((true_vec - pred_vec) ** 2).mean()) / true_range)
    else:
        nrmse = float("nan")

    return {
        "sign_accuracy": sign_acc,
        "spearman": float(rho) if not np.isnan(rho) else float("nan"),
        "pearson": pearson,
        "scale_ratio": scale_ratio,
        "normalized_rmse": nrmse,
    }


def _analyze_run(label: str, info: Dict[str, str]) -> Dict[str, Any]:
    d = np.load(f"{info['path']}/results/data_snapshot.npz")
    growth_true = d["growth_rates_true"]
    growth_pred = d["growth_rates_pred"]
    M_true = d["interaction_true"]
    M_pred = d["interaction_pred"]

    # 对角线 = 自限制/死亡率
    diag_true = M_true.diagonal()
    diag_pred = M_pred.diagonal()

    # 非对角（交互） — 按行展平
    mask = ~np.eye(6, dtype=bool)
    offdiag_true = M_true[mask]
    offdiag_pred = M_pred[mask]

    return {
        "label": label,
        "info": info,
        "raw": {
            "growth_true": growth_true, "growth_pred": growth_pred,
            "diag_true": diag_true, "diag_pred": diag_pred,
            "offdiag_true": offdiag_true, "offdiag_pred": offdiag_pred,
            "alpha_lv": float(d["alpha_lv_pred"][0]),
            "alpha_res": float(d["alpha_res_pred"][0]),
            "lv_drift_scale": float(d["lv_drift_scale_pred"][0]),
        },
        "metrics": {
            "growth_rates": _compute_param_metrics(growth_true, growth_pred),
            "diagonal_self_limit": _compute_param_metrics(diag_true, diag_pred),
            "offdiag_interaction": _compute_param_metrics(offdiag_true, offdiag_pred),
        },
    }


def _plot_parameter_comparison(results: Dict[str, Dict], output_path: Path) -> None:
    """画三组参数的 true vs pred 散点 + 指标柱状图。"""
    fig = plt.figure(figsize=(16, 12), constrained_layout=True)
    gs = fig.add_gridspec(3, 5, width_ratios=[1, 1, 1, 1, 1.3])

    param_groups = [
        ("growth_rates", "增长率 r_i", "Growth rates"),
        ("diagonal_self_limit", "对角线（自限制/死亡率）", "Diagonal (self-limitation)"),
        ("offdiag_interaction", "非对角线（种间交互）", "Off-diagonal (interactions)"),
    ]

    for row_idx, (key, title_zh, title_en) in enumerate(param_groups):
        # 4 个 scatter plot (A/B/C/D)
        for col_idx, label in enumerate(["A", "B", "C", "D"]):
            r = results[label]
            info = r["info"]
            raw = r["raw"]
            metrics = r["metrics"][key]

            if key == "growth_rates":
                x_true = raw["growth_true"]
                x_pred = raw["growth_pred"]
            elif key == "diagonal_self_limit":
                x_true = raw["diag_true"]
                x_pred = raw["diag_pred"]
            else:
                x_true = raw["offdiag_true"]
                x_pred = raw["offdiag_pred"]

            ax = fig.add_subplot(gs[row_idx, col_idx])
            ax.scatter(x_true, x_pred, alpha=0.7, color=info["color"], s=50, edgecolor="black")

            # 对角线参考（y=x）
            vmin = min(x_true.min(), x_pred.min())
            vmax = max(x_true.max(), x_pred.max())
            ax.plot([vmin, vmax], [vmin, vmax], "k--", linewidth=0.8, alpha=0.5, label="y=x")

            # 水平/垂直零线
            ax.axhline(0, color="grey", linewidth=0.5, alpha=0.3)
            ax.axvline(0, color="grey", linewidth=0.5, alpha=0.3)

            ax.set_xlabel("真实值")
            ax.set_ylabel("模型值")
            ax.set_title(
                f"{label}: {info['data']}\n{info['prior']}",
                fontsize=10,
            )

            # 指标文字
            metric_text = (
                f"sign: {metrics['sign_accuracy']:.2f}\n"
                f"spearman: {metrics['spearman']:.2f}\n"
                f"scale: {metrics['scale_ratio']:.2f}"
            )
            ax.text(
                0.03, 0.97, metric_text,
                transform=ax.transAxes, va="top", ha="left", fontsize=8,
                bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85, "edgecolor": "#cccccc"},
            )

        # 最后一列：柱状图汇总
        ax_bar = fig.add_subplot(gs[row_idx, 4])
        metric_types = ["sign_accuracy", "spearman", "pearson"]
        metric_labels = ["符号\n准确率", "Spearman\n秩相关", "Pearson\n相关"]
        x = np.arange(len(metric_types))
        width = 0.2
        for i, label in enumerate(["A", "B", "C", "D"]):
            vals = [results[label]["metrics"][key][m] for m in metric_types]
            # handle nan
            vals = [0 if np.isnan(v) else v for v in vals]
            ax_bar.bar(
                x + i * width - 1.5 * width, vals, width,
                label=label, color=RUNS[label]["color"], edgecolor="black", alpha=0.85,
            )
        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(metric_labels, fontsize=8)
        ax_bar.set_ylabel("指标值")
        ax_bar.set_title(f"{title_zh}指标汇总", fontsize=10)
        ax_bar.axhline(0.5, color="grey", linestyle="--", linewidth=0.8, alpha=0.5)
        ax_bar.set_ylim([-0.3, 1.1])
        ax_bar.legend(loc="lower right", fontsize=8)

    fig.suptitle(
        "参数恢复分析：真实参数 vs 模型参数",
        fontsize=14, fontweight="bold", y=1.01,
    )
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_growth_rate_bars(results: Dict[str, Dict], output_path: Path) -> None:
    """专门展示 growth rates：6 个物种 × 4 个配置的柱状对比。"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharey=True, constrained_layout=True)
    species_labels = ["v1", "v2", "v3", "v4", "v5", "h"]
    x = np.arange(6)
    width = 0.35

    for idx, (label, ax) in enumerate(zip(["A", "B", "C", "D"], axes.flat)):
        r = results[label]
        info = r["info"]
        raw = r["raw"]

        ax.bar(x - width/2, raw["growth_true"], width, label="真实", color="#555", edgecolor="black")
        ax.bar(x + width/2, raw["growth_pred"], width, label="模型恢复", color=info["color"], edgecolor="black", alpha=0.85)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(species_labels)
        ax.set_ylabel("增长率 r")
        ax.set_title(f"{label}: {info['data']} + {info['prior']}", fontsize=11)
        ax.legend(fontsize=9, loc="upper right")

        # 标注 spearman
        metrics = r["metrics"]["growth_rates"]
        ax.text(
            0.02, 0.97,
            f"Spearman: {metrics['spearman']:.2f}\nSign: {metrics['sign_accuracy']:.2f}",
            transform=ax.transAxes, va="top", ha="left", fontsize=9,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85, "edgecolor": "#cccccc"},
        )

    fig.suptitle("Growth Rates 恢复对比：每个物种的真实 vs 模型值", fontsize=13, fontweight="bold")
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _write_report(results: Dict[str, Dict], output_path: Path) -> None:
    lines = [
        "# 参数恢复分析报告",
        "",
        "## 方法说明",
        "",
        "- **参数化差异**：数据用 Ricker 形式 `x_{t+1} = x_t * exp(r + A·x)`，",
        "  模型用 `x_{t+1} = x_t + lv_scale · x_t · tanh(r + A·x) + residual`。",
        "  参数数值不可直接对比，因此使用**符号准确率 + 秩相关**作为核心指标。",
        "",
        "## 指标汇总",
        "",
    ]
    for param_key, param_name in [
        ("growth_rates", "增长率 (growth rates)"),
        ("diagonal_self_limit", "对角线 (自限制/死亡率)"),
        ("offdiag_interaction", "非对角线 (种间交互)"),
    ]:
        lines += [
            f"### {param_name}",
            "",
            "| 配置 | 符号准确率 | Spearman 秩相关 | Pearson 相关 | 尺度比 | 归一化 RMSE |",
            "|------|-----------|----------------|--------------|--------|-------------|",
        ]
        for label in ["A", "B", "C", "D"]:
            info = RUNS[label]
            m = results[label]["metrics"][param_key]
            lines.append(
                f"| {label} ({info['data']}+{info['prior']}) | "
                f"{m['sign_accuracy']:.3f} | {m['spearman']:.3f} | {m['pearson']:.3f} | "
                f"{m['scale_ratio']:.3f} | {m['normalized_rmse']:.3f} |"
            )
        lines.append("")

    # 模型专有参数
    lines += [
        "## 模型特有的混合权重（无对应真实值）",
        "",
        "| 配置 | alpha_lv | alpha_res | lv_drift_scale |",
        "|------|----------|-----------|----------------|",
    ]
    for label in ["A", "B", "C", "D"]:
        raw = results[label]["raw"]
        lines.append(
            f"| {label} | {raw['alpha_lv']:.3f} | {raw['alpha_res']:.3f} | {raw['lv_drift_scale']:.3f} |"
        )

    lines += [
        "",
        "## 关键观察",
        "",
        "1. **Growth rates 恢复**：",
    ]
    for label in ["A", "B", "C", "D"]:
        m = results[label]["metrics"]["growth_rates"]
        lines.append(f"   - {label}: spearman={m['spearman']:.2f}, sign={m['sign_accuracy']:.2f}")

    lines += [
        "",
        "2. **对角线（死亡率）恢复**：",
    ]
    for label in ["A", "B", "C", "D"]:
        m = results[label]["metrics"]["diagonal_self_limit"]
        lines.append(f"   - {label}: spearman={m['spearman']:.2f}, sign={m['sign_accuracy']:.2f}")

    lines += [
        "",
        "3. **种间交互恢复**（之前 2×2 分析的补充）：",
    ]
    for label in ["A", "B", "C", "D"]:
        m = results[label]["metrics"]["offdiag_interaction"]
        lines.append(f"   - {label}: spearman={m['spearman']:.2f}, sign={m['sign_accuracy']:.2f}")

    lines += [
        "",
        "## 初步结论",
        "",
        "- 数值层面的参数恢复受参数化差异限制，不能期望 true ≈ pred",
        "- 符号准确率和秩相关是更合理的评估方式",
        "- 需要进一步分析：哪些参数被可靠恢复？哪些是不可恢复的？",
    ]

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    _configure_matplotlib()
    out_dir = Path("runs/analysis_parameter_recovery")
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for label, info in RUNS.items():
        results[label] = _analyze_run(label, info)

    _plot_parameter_comparison(results, out_dir / "fig_parameter_recovery.png")
    print(f"[OK] saved: {out_dir / 'fig_parameter_recovery.png'}")

    _plot_growth_rate_bars(results, out_dir / "fig_growth_rate_bars.png")
    print(f"[OK] saved: {out_dir / 'fig_growth_rate_bars.png'}")

    _write_report(results, out_dir / "parameter_recovery_report.md")
    print(f"[OK] saved: {out_dir / 'parameter_recovery_report.md'}")

    # 控制台输出核心指标
    print("\n" + "=" * 80)
    print("  Parameter Recovery Summary")
    print("=" * 80)
    for label in ["A", "B", "C", "D"]:
        info = RUNS[label]
        print(f"\n--- {label}: {info['data']} + {info['prior']} ---")
        for key, name in [("growth_rates", "Growth"), ("diagonal_self_limit", "Diagonal"), ("offdiag_interaction", "OffDiag")]:
            m = results[label]["metrics"][key]
            print(f"  {name:10s}: sign={m['sign_accuracy']:.3f}  spearman={m['spearman']:.3f}  scale_ratio={m['scale_ratio']:.3f}")


if __name__ == "__main__":
    main()
