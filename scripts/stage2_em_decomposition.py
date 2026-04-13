"""Stage 2 增强：迭代 EM 分解残差为 (smooth env, sparse pulse)。

核心思想（结构化 rank-2 因子分解）：
  residual R[t, d] ≈ b_d * e[t] + c_d * p[t] + noise
  其中:
    e 平滑（低频, env driver 特征）
    p 稀疏（尖峰事件, pulse driver 特征）
    b, c 是 loadings

EM-style 交替优化:
  1. 初始化 (b, c, e, p) 用 SVD
  2. 循环:
     a. 给定 (e, p)，用线性回归更新 (b, c)，并归一化 ||b||=||c||=1
     b. 给定 (b, c)，对每个 t 解 2 维线性系统得到 (e_raw[t], p_raw[t])
     c. e <- 平滑 e_raw (TV or Tikhonov)
     d. p <- 软阈值 p_raw (L1 proximal)
  3. 收敛后重新用 (e, p) 作为协变量做 Stage 2

对比 SVD-only，EM 应该能分离"平滑"和"稀疏"两种不同结构的信号。
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from data.partial_lv_mvp import generate_partial_lv_mvp_system
from scripts.stage2_full_parameter_recovery import (
    _fit_ricker_regression,
    _compute_metrics,
    _regenerate_ground_truth,
)


def _configure_matplotlib() -> None:
    plt.rcParams["font.sans-serif"] = [
        "PingFang SC", "Hiragino Sans GB", "Heiti TC", "STHeiti",
        "Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def smooth_tikhonov(y: np.ndarray, lam: float) -> np.ndarray:
    """Tikhonov 平滑: argmin ||x-y||² + lam * ||Dx||²
    解为 (I + lam*D^T D) x = y，其中 D 是一阶差分矩阵。
    闭式解，通过构造 tridiagonal 系统求解。
    """
    T = len(y)
    # (I + lam * D^T D) 是一个 tridiagonal: 对角 (1+2*lam, ..., 1+2*lam) 边界 (1+lam)
    # 次对角 -lam
    main_diag = np.full(T, 1.0 + 2.0 * lam)
    main_diag[0] = 1.0 + lam
    main_diag[-1] = 1.0 + lam
    off_diag = np.full(T - 1, -lam)
    # 构造稀疏的 tridiagonal 然后用 numpy 解
    # 使用 Thomas algorithm for tridiagonal systems
    return _thomas_tridiag(main_diag.copy(), off_diag.copy(), off_diag.copy(), y.copy())


def _thomas_tridiag(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Solve Ax=d for tridiagonal A with main a, sub b, sup c. All modified in place."""
    n = len(a)
    # Forward elimination
    for i in range(1, n):
        m = b[i - 1] / a[i - 1]
        a[i] = a[i] - m * c[i - 1]
        d[i] = d[i] - m * d[i - 1]
    # Back substitution
    x = np.zeros(n)
    x[-1] = d[-1] / a[-1]
    for i in range(n - 2, -1, -1):
        x[i] = (d[i] - c[i] * x[i + 1]) / a[i]
    return x


def soft_threshold(y: np.ndarray, tau: float) -> np.ndarray:
    """Soft thresholding: sign(y) * max(|y| - tau, 0)"""
    return np.sign(y) * np.maximum(np.abs(y) - tau, 0.0)


def em_decompose_residuals(
    residuals: np.ndarray,
    num_iter: int = 100,
    smooth_lambda: float = 10.0,
    sparse_lambda: float = 0.005,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """残差 R ≈ outer(e, b) + outer(p, c), e 平滑, p 稀疏.

    Returns:
        e: (T,) smooth component (env-like)
        p: (T,) sparse component (pulse-like)
        b: (D,) env loadings
        c: (D,) pulse loadings
        diagnostics: dict
    """
    T, D = residuals.shape

    # Init: SVD 前两个成分
    U, S, Vt = np.linalg.svd(residuals, full_matrices=False)
    e = U[:, 0] * S[0]
    p = U[:, 1] * S[1]
    b = Vt[0, :].copy()
    c = Vt[1, :].copy()

    history = {"fit_error": [], "e_smoothness": [], "p_sparsity": []}

    for it in range(num_iter):
        # Step 1: 更新 loadings (b, c) given (e, p)
        X = np.stack([e, p], axis=1)  # (T, 2)
        # Solve X @ [b; c] = residuals  i.e. loadings = pinv(X) @ residuals
        XtX = X.T @ X + 1e-8 * np.eye(2)
        BC = np.linalg.solve(XtX, X.T @ residuals)  # (2, D)
        b = BC[0]
        c = BC[1]

        # 归一化：让 b 的能量吸收到 e, c 的能量吸收到 p
        b_norm = np.linalg.norm(b) + 1e-8
        e = e * b_norm
        b = b / b_norm
        c_norm = np.linalg.norm(c) + 1e-8
        p = p * c_norm
        c = c / c_norm

        # Step 2: 更新 (e[t], p[t]) given (b, c)
        # For each t: [b c] @ [e[t]; p[t]] = residuals[t, :]
        # Least squares: (M^T M) @ [e[t]; p[t]] = M^T residuals[t, :]
        M = np.stack([b, c], axis=1)  # (D, 2)
        MtM = M.T @ M + 1e-8 * np.eye(2)
        MtR = residuals @ M  # (T, 2)
        # Solve element-wise: coef[t] = (MtM)^-1 @ MtR[t]
        coefs = np.linalg.solve(MtM, MtR.T).T  # (T, 2)
        e_raw = coefs[:, 0]
        p_raw = coefs[:, 1]

        # Step 3: 应用正则化
        # e_raw: 平滑（Tikhonov）
        e = smooth_tikhonov(e_raw, smooth_lambda)
        # p_raw: 稀疏（soft threshold）
        p = soft_threshold(p_raw, sparse_lambda)

        # 诊断
        fit = np.outer(e, b) + np.outer(p, c)
        err = np.linalg.norm(residuals - fit) / np.linalg.norm(residuals)
        smoothness = np.sum((np.diff(e)) ** 2)
        sparsity = np.sum(np.abs(p))
        history["fit_error"].append(err)
        history["e_smoothness"].append(smoothness)
        history["p_sparsity"].append(sparsity)

        if verbose and (it % 20 == 0 or it == num_iter - 1):
            nonzero_p = int(np.sum(np.abs(p) > 1e-8))
            print(f"  iter {it}: fit_err={err:.4f}, e_smooth={smoothness:.4f}, p_nonzero={nonzero_p}/{T}")

    return e, p, b, c, history


def fit_stage2_em(full_states: np.ndarray, em_iter: int = 100) -> Dict[str, Any]:
    """Stage 2 变体 4: 迭代 EM 分解协变量。"""
    # Pass 1: 无协变量拟合残差
    _, _, _, residuals, _, _ = _fit_ricker_regression(full_states, covariates=None)

    # EM 分解
    e_hat, p_hat, b_hat, c_hat, history = em_decompose_residuals(
        residuals,
        num_iter=em_iter,
        smooth_lambda=20.0,
        sparse_lambda=0.005,
        verbose=True,
    )

    # Pass 2: 用 (e_hat, p_hat) 作为协变量
    covar_series = np.stack([e_hat, p_hat], axis=1)  # (T-1, 2)
    g, A, loadings, _, r2, std = _fit_ricker_regression(full_states, covariates=covar_series)

    return {
        "growth_rates": g,
        "interaction_matrix": A,
        "covariate_loadings": loadings,
        "covariate_series": covar_series,
        "em_history": history,
        "em_components": {"e": e_hat, "p": p_hat, "b": b_hat, "c": c_hat},
        "r2_mean": r2,
        "residual_std_mean": std,
    }


def main() -> None:
    _configure_matplotlib()
    out_dir = Path("runs/analysis_stage2_em")
    out_dir.mkdir(parents=True, exist_ok=True)

    target_run = "runs/20260412_165241_partial_lv_lv_guided_stochastic_refined"

    # Load
    d = np.load(f"{target_run}/results/data_snapshot.npz")
    visible_true = d["visible_true_full"]
    hidden_true = d["hidden_true_full"]
    hidden_pred = d["hidden_pred_full"]
    growth_true = d["growth_rates_true"]
    interaction_true = d["interaction_true"]

    # Ground truth env / pulse
    _, _, env_true, pulse_true, _, _, env_loadings_true = _regenerate_ground_truth(seed=42)

    # Pipeline: 恢复的 hidden
    valid_mask = ~np.isnan(hidden_pred[:, 0])
    visible_masked = visible_true[valid_mask]
    hidden_masked = hidden_pred[valid_mask]
    env_masked = env_true[valid_mask]
    pulse_masked = pulse_true[valid_mask]
    full_pipeline = np.concatenate([visible_masked, hidden_masked], axis=1)

    # Oracle: 真 hidden
    full_oracle = np.concatenate([visible_true, hidden_true], axis=1)

    # Run EM 分解 on Pipeline 和 Oracle
    print("\n=== Oracle EM ===")
    oracle_em = fit_stage2_em(full_oracle)
    print("\n=== Pipeline EM ===")
    pipeline_em = fit_stage2_em(full_pipeline)

    # 对比恢复的协变量和真实 env/pulse
    def covar_quality(e_hat, p_hat, env_ref, pulse_ref):
        # 取对齐后的长度
        L = min(len(e_hat), len(env_ref) - 1)
        return {
            "e_vs_env_abs_corr": float(np.abs(np.corrcoef(e_hat[:L], env_ref[:L, 0])[0, 1])),
            "e_vs_pulse_abs_corr": float(np.abs(np.corrcoef(e_hat[:L], pulse_ref[:L, 0])[0, 1])),
            "p_vs_env_abs_corr": float(np.abs(np.corrcoef(p_hat[:L], env_ref[:L, 0])[0, 1])),
            "p_vs_pulse_abs_corr": float(np.abs(np.corrcoef(p_hat[:L], pulse_ref[:L, 0])[0, 1])),
        }

    oracle_cq = covar_quality(oracle_em["em_components"]["e"], oracle_em["em_components"]["p"], env_true, pulse_true)
    pipeline_cq = covar_quality(pipeline_em["em_components"]["e"], pipeline_em["em_components"]["p"], env_masked, pulse_masked)

    # 参数恢复指标
    oracle_metrics = _compute_metrics(growth_true, oracle_em["growth_rates"], interaction_true, oracle_em["interaction_matrix"])
    pipeline_metrics = _compute_metrics(growth_true, pipeline_em["growth_rates"], interaction_true, pipeline_em["interaction_matrix"])

    # 打印
    print("\n" + "=" * 80)
    print("  EM 分解后的 Stage 2 参数恢复")
    print("=" * 80)
    for name, m in [("Oracle+EM", oracle_metrics), ("Pipeline+EM", pipeline_metrics)]:
        print(f"\n  {name}:")
        print(f"    Growth:  spearman={m['growth_spearman']:+.3f}  scale={m['growth_scale']:.3f}")
        print(f"    Interaction:  sign={m['interaction_sign']:.3f}  pearson={m['interaction_pearson']:+.3f}  scale={m['interaction_scale']:.3f}")
        print(f"    Diagonal:  sign={m['diagonal_sign']:.3f}  spearman={m['diagonal_spearman']:+.3f}  scale={m['diagonal_scale']:.3f}")

    print("\n  协变量恢复质量 (|corr|):")
    for name, cq in [("Oracle", oracle_cq), ("Pipeline", pipeline_cq)]:
        print(f"    {name}:")
        print(f"      e_hat vs env:  {cq['e_vs_env_abs_corr']:.3f}   e_hat vs pulse: {cq['e_vs_pulse_abs_corr']:.3f}")
        print(f"      p_hat vs env:  {cq['p_vs_env_abs_corr']:.3f}   p_hat vs pulse: {cq['p_vs_pulse_abs_corr']:.3f}")

    # 画图
    fig, axes = plt.subplots(3, 2, figsize=(14, 11), constrained_layout=True)

    # Row 1: Pipeline 恢复的 e, p vs 真实 env, pulse
    e_hat = pipeline_em["em_components"]["e"]
    p_hat = pipeline_em["em_components"]["p"]
    L = min(len(e_hat), len(env_masked) - 1)
    t_axis = np.arange(L)

    def _norm(x):
        return (x - x.mean()) / (x.std() + 1e-8)

    axes[0, 0].plot(t_axis, _norm(env_masked[:L, 0]), color="#2e7d32", linewidth=1.4, label="真实 env")
    axes[0, 0].plot(t_axis, _norm(e_hat[:L]), color="#1565c0", linewidth=1.1, alpha=0.85, label="EM e_hat")
    axes[0, 0].set_title(f"Env 恢复: |corr|={pipeline_cq['e_vs_env_abs_corr']:.3f}")
    axes[0, 0].legend(fontsize=9)
    axes[0, 0].set_xlabel("时间步")

    axes[0, 1].plot(t_axis, _norm(pulse_masked[:L, 0]), color="#2e7d32", linewidth=1.4, label="真实 pulse")
    axes[0, 1].plot(t_axis, _norm(p_hat[:L]), color="#c62828", linewidth=1.1, alpha=0.85, label="EM p_hat")
    axes[0, 1].set_title(f"Pulse 恢复: |corr|={pipeline_cq['p_vs_pulse_abs_corr']:.3f}")
    axes[0, 1].legend(fontsize=9)
    axes[0, 1].set_xlabel("时间步")

    # Row 2: Scatter 对比
    axes[1, 0].scatter(_norm(env_masked[:L, 0]), _norm(e_hat[:L]), alpha=0.4, s=10, color="#1565c0")
    axes[1, 0].set_xlabel("真实 env (标准化)")
    axes[1, 0].set_ylabel("EM e_hat (标准化)")
    axes[1, 0].set_title("env vs e_hat")

    axes[1, 1].scatter(_norm(pulse_masked[:L, 0]), _norm(p_hat[:L]), alpha=0.4, s=10, color="#c62828")
    axes[1, 1].set_xlabel("真实 pulse (标准化)")
    axes[1, 1].set_ylabel("EM p_hat (标准化)")
    axes[1, 1].set_title("pulse vs p_hat")

    # Row 3: Growth rates 对比
    species_labels = ["v1", "v2", "v3", "v4", "v5", "h"]
    x = np.arange(6)
    width = 0.25
    axes[2, 0].bar(x - width, growth_true, width, label="真实", color="#555", edgecolor="black")
    axes[2, 0].bar(x, pipeline_em["growth_rates"], width, label="EM", color="#1565c0", edgecolor="black", alpha=0.85)
    # 也画 baseline (E2E)
    axes[2, 0].bar(x + width, d["growth_rates_pred"], width, label="E2E Baseline", color="#c62828", edgecolor="black", alpha=0.85)
    axes[2, 0].set_xticks(x)
    axes[2, 0].set_xticklabels(species_labels)
    axes[2, 0].set_ylabel("r")
    axes[2, 0].set_title(f"Growth Rates (Pipeline+EM spearman={pipeline_metrics['growth_spearman']:.3f})")
    axes[2, 0].axhline(0, color="black", linewidth=0.5)
    axes[2, 0].legend(fontsize=9)

    # Interaction off-diagonal scatter
    mask = ~np.eye(6, dtype=bool)
    off_true = interaction_true[mask]
    off_em = pipeline_em["interaction_matrix"][mask]
    off_e2e = d["interaction_pred"][mask]
    axes[2, 1].scatter(off_true, off_em, alpha=0.6, s=50, color="#1565c0", edgecolor="black", label="Pipeline+EM")
    axes[2, 1].scatter(off_true, off_e2e, alpha=0.6, s=50, color="#c62828", edgecolor="black", label="E2E Baseline", marker="^")
    vmin = min(off_true.min(), off_em.min(), off_e2e.min())
    vmax = max(off_true.max(), off_em.max(), off_e2e.max())
    axes[2, 1].plot([vmin, vmax], [vmin, vmax], "k--", linewidth=0.8, alpha=0.5)
    axes[2, 1].axhline(0, color="grey", linewidth=0.5, alpha=0.3)
    axes[2, 1].axvline(0, color="grey", linewidth=0.5, alpha=0.3)
    axes[2, 1].set_xlabel("真实 A[i,j]")
    axes[2, 1].set_ylabel("恢复 A[i,j]")
    axes[2, 1].set_title(f"Off-diagonal (Pipeline+EM pearson={pipeline_metrics['interaction_pearson']:.3f})")
    axes[2, 1].legend(fontsize=9)

    fig.suptitle("Stage 2 迭代 EM：恢复 env + pulse + 全参数", fontsize=13, fontweight="bold", y=1.01)
    fig.savefig(out_dir / "fig_em_recovery.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[OK] saved: {out_dir / 'fig_em_recovery.png'}")


if __name__ == "__main__":
    main()
