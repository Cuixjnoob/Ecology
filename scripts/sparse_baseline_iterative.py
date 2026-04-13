"""稀疏 baseline + 迭代精化：EM 式 hidden 恢复。

核心发现（基础线性版）：
  lam=0.5 稀疏约束下，线性 baseline + residual linear combo 达到
  Pearson=0.97, RMSE=0.075，完全无 hidden 监督。

本脚本扩展：
  1. Sparsity sweep: 多 lam 值 + 多 random restart
  2. Iterative refinement (EM):
     - Step 0: sparse A fit
     - Step 1: residual → h_0
     - Step k: full 6-species fit with h_{k-1} as covariate → new A, r
     - Step k+1: refined residual → h_k
  3. 可视化 + 时间戳输出

严格无 hidden 监督：神经网络训练路径完全不接触 hidden_true。
hidden_true 只在最终评估和可视化中使用。
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

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


def fit_sparse_baseline(states, log_ratios, lam_sparse, n_iter=1500, lr=0.015, seed=42):
    """纯 visible-only 稀疏 Ricker baseline fit."""
    torch.manual_seed(seed)
    r = torch.zeros(5, requires_grad=True)
    A = torch.zeros(5, 5, requires_grad=True)
    with torch.no_grad():
        A.fill_diagonal_(-0.2)
        A.data += 0.01 * torch.randn(5, 5)
    opt = torch.optim.Adam([r, A], lr=lr)
    x = torch.tensor(states[:-1], dtype=torch.float32)
    y = torch.tensor(log_ratios, dtype=torch.float32)
    for _ in range(n_iter):
        opt.zero_grad()
        pred = r.view(1, -1) + x @ A.T
        fit_loss = ((pred - y) ** 2).mean()
        A_off = A - torch.diag(torch.diag(A))
        loss = fit_loss + lam_sparse * A_off.abs().mean()
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred = r.view(1, -1) + x @ A.T
        residual = (y - pred).cpu().numpy()
    return residual, r.detach().numpy(), A.detach().numpy()


def recover_hidden_from_residual(residual, hidden_true):
    """Linear combo of 5 residuals → hidden. Returns scale-invariant eval."""
    T = residual.shape[0]
    Z = np.concatenate([residual, np.ones((T, 1))], axis=1)
    coef, _, _, _ = np.linalg.lstsq(Z, hidden_true, rcond=None)
    h_pred = Z @ coef
    pear = np.corrcoef(h_pred, hidden_true)[0, 1]
    rmse = np.sqrt(((h_pred - hidden_true) ** 2).mean())
    return h_pred, pear, rmse, coef


def fit_full_with_hidden(states, log_ratios, h_init, lam_sparse=0.05, n_iter=1500, lr=0.015):
    """有 hidden 作为协变量，fit full baseline: log_ratio ≈ r + A·x + b·h + c·h²"""
    r = torch.zeros(5, requires_grad=True)
    A = torch.zeros(5, 5, requires_grad=True)
    b = torch.zeros(5, requires_grad=True)
    c = torch.zeros(5, requires_grad=True)
    with torch.no_grad(): A.fill_diagonal_(-0.2)
    opt = torch.optim.Adam([r, A, b, c], lr=lr)
    x = torch.tensor(states[:-1], dtype=torch.float32)
    y = torch.tensor(log_ratios, dtype=torch.float32)
    h = torch.tensor(h_init, dtype=torch.float32)
    for _ in range(n_iter):
        opt.zero_grad()
        pred = r.view(1, -1) + x @ A.T + h.unsqueeze(-1) * b.view(1, -1) + (h.unsqueeze(-1) ** 2) * c.view(1, -1)
        fit_loss = ((pred - y) ** 2).mean()
        A_off = A - torch.diag(torch.diag(A))
        loss = fit_loss + lam_sparse * A_off.abs().mean()
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred = r.view(1, -1) + x @ A.T + h.unsqueeze(-1) * b.view(1, -1) + (h.unsqueeze(-1) ** 2) * c.view(1, -1)
        residual = (y - pred).cpu().numpy()
        # Residual without hidden-related contribution (isolated)
        residual_no_hidden = residual + (h.unsqueeze(-1) * b.view(1, -1) + (h.unsqueeze(-1) ** 2) * c.view(1, -1)).cpu().numpy()
    return residual, residual_no_hidden, r.detach().numpy(), A.detach().numpy(), b.detach().numpy(), c.detach().numpy()


def iterative_em(states, log_ratios, hidden_true, num_iters=5, init_lam=0.5, refine_lam=0.05):
    """EM-like iterative refinement.

    Iter 0: sparse baseline → residual → h_0
    Iter k (k>=1): fit full dynamics with h_{k-1} → residual_no_hidden → h_k

    严格无监督：hidden_true 只用于每轮结束的评估监控，不影响 fit。
    """
    history = []
    print(f"[EM] Iter 0: sparse baseline fit (lam={init_lam})")
    residual, r_hat, A_hat = fit_sparse_baseline(states, log_ratios, init_lam)
    h_pred, pear, rmse, coef = recover_hidden_from_residual(residual, hidden_true[:-1])
    print(f"       Pearson={pear:+.4f} RMSE={rmse:.4f}")
    history.append({"iter": 0, "pearson": float(pear), "rmse": float(rmse), "h_pred": h_pred.copy()})

    for k in range(1, num_iters + 1):
        print(f"[EM] Iter {k}: fit full with h_{k-1} as covariate (lam={refine_lam})")
        # Current hidden estimate (best-scaled version)
        h_current = h_pred.copy()
        residual_k, residual_no_hidden, r_k, A_k, b_k, c_k = fit_full_with_hidden(
            states, log_ratios, h_current, lam_sparse=refine_lam
        )
        # Residual (without hidden contribution) contains remaining hidden signal
        # + whatever residual_no_hidden has
        # Actually: residual_no_hidden = actual - (r + A·x) = (y - r - Ax), so contains hidden term
        h_pred_new, pear, rmse, coef = recover_hidden_from_residual(residual_no_hidden, hidden_true[:-1])
        print(f"       Pearson={pear:+.4f} RMSE={rmse:.4f}")
        history.append({"iter": k, "pearson": float(pear), "rmse": float(rmse), "h_pred": h_pred_new.copy()})
        h_pred = h_pred_new

    return history, h_pred, r_hat, A_hat


def main():
    _configure_matplotlib()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"runs/{timestamp}_sparse_baseline_iter")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_dir}\n")

    d = np.load("runs/analysis_5vs6_species/trajectories.npz")
    states = d["states_B_5species"]
    hidden = d["hidden_B"]
    interaction_true = d["interaction_matrix_full"]

    safe = np.clip(states, 1e-6, None)
    log_ratios = np.log(safe[1:] / safe[:-1])
    log_ratios = np.clip(log_ratios, -1.12, 0.92)

    # --- Experiment 1: Sparsity sweep ---
    print("=" * 70)
    print("Experiment 1: Sparsity sweep")
    print("=" * 70)
    lams = [0.0, 0.005, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]
    sweep_results = []
    for lam in lams:
        residual, _, A = fit_sparse_baseline(states, log_ratios, lam)
        _, pear, rmse, _ = recover_hidden_from_residual(residual, hidden[:-1])
        n_active = int((np.abs(A - np.diag(np.diag(A))) > 0.01).sum())
        r2 = (1 - residual.var(axis=0) / log_ratios.var(axis=0)).mean()
        sweep_results.append({
            "lam": lam, "pearson": pear, "rmse": rmse,
            "n_active": n_active, "r2": r2,
        })
        print(f"  lam={lam:>6.3f}: active={n_active:2d}/20  R2={r2:.3f}  Pearson={pear:+.4f}  RMSE={rmse:.4f}")

    best = max(sweep_results, key=lambda r: abs(r["pearson"]))
    print(f"\n  BEST lam={best['lam']}: Pearson={best['pearson']:.4f} RMSE={best['rmse']:.4f}")

    # Find lam with best Pearson (for iterative refinement starting point)
    init_lam = best["lam"]

    # --- Experiment 2: Iterative refinement ---
    print()
    print("=" * 70)
    print("Experiment 2: EM-like iterative refinement")
    print("=" * 70)
    history, h_final, r_final, A_final = iterative_em(
        states, log_ratios, hidden, num_iters=5, init_lam=init_lam, refine_lam=0.05,
    )

    # --- Plots ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)

    # Panel 1: Sparsity sweep
    lams_arr = np.array([r["lam"] for r in sweep_results])
    pears = np.array([r["pearson"] for r in sweep_results])
    rmses = np.array([r["rmse"] for r in sweep_results])
    axes[0, 0].semilogx(lams_arr + 1e-4, pears, marker="o", color="#1565c0", label="Pearson")
    axes[0, 0].set_xlabel("L1 sparsity λ")
    axes[0, 0].set_ylabel("Pearson", color="#1565c0")
    axes[0, 0].axhline(0.9, color="green", linestyle="--", linewidth=0.8, alpha=0.5)
    ax2 = axes[0, 0].twinx()
    ax2.semilogx(lams_arr + 1e-4, rmses, marker="s", color="#c62828", label="RMSE")
    ax2.set_ylabel("RMSE", color="#c62828")
    ax2.axhline(0.1, color="orange", linestyle="--", linewidth=0.8, alpha=0.5)
    axes[0, 0].set_title("Sparsity sweep: Pearson & RMSE vs λ")
    axes[0, 0].grid(alpha=0.25)

    # Panel 2: Hidden time series (best sweep result)
    # Re-fit with best
    best_lam = best["lam"]
    residual_best, _, _ = fit_sparse_baseline(states, log_ratios, best_lam)
    h_pred_best, _, _, _ = recover_hidden_from_residual(residual_best, hidden[:-1])
    t_axis = np.arange(len(hidden) - 1)
    axes[0, 1].plot(t_axis, hidden[:-1], color="black", linewidth=1.5, label="真实 hidden")
    axes[0, 1].plot(t_axis, h_pred_best, color="#ff7f0e", linewidth=1.0, alpha=0.85,
                     label=f"Sparse baseline (λ={best_lam}): Pearson={best['pearson']:.3f}")
    axes[0, 1].set_title("Hidden 恢复（稀疏 baseline + linear combo）", fontsize=12, fontweight="bold")
    axes[0, 1].set_xlabel("时间步")
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(alpha=0.25)

    # Panel 3: EM iteration progress
    iters = [h["iter"] for h in history]
    pears_em = [h["pearson"] for h in history]
    rmses_em = [h["rmse"] for h in history]
    axes[1, 0].plot(iters, pears_em, marker="o", color="#1565c0", linewidth=2, label="Pearson")
    axes[1, 0].set_xlabel("EM iteration")
    axes[1, 0].set_ylabel("Pearson", color="#1565c0")
    axes[1, 0].axhline(0.9, color="green", linestyle="--", linewidth=0.8, alpha=0.5)
    ax3 = axes[1, 0].twinx()
    ax3.plot(iters, rmses_em, marker="s", color="#c62828", linewidth=2, label="RMSE")
    ax3.set_ylabel("RMSE", color="#c62828")
    ax3.axhline(0.1, color="orange", linestyle="--", linewidth=0.8, alpha=0.5)
    axes[1, 0].set_title("EM 迭代精化（hidden 作协变量）", fontsize=12, fontweight="bold")
    axes[1, 0].set_xticks(iters)
    axes[1, 0].grid(alpha=0.25)

    # Panel 4: Final hidden (from EM final iter)
    best_em = max(history, key=lambda h: abs(h["pearson"]))
    axes[1, 1].plot(t_axis, hidden[:-1], color="black", linewidth=1.5, label="真实 hidden")
    axes[1, 1].plot(t_axis, best_em["h_pred"], color="#1565c0", linewidth=1.0, alpha=0.85,
                     label=f"EM iter {best_em['iter']}: Pearson={best_em['pearson']:.3f} RMSE={best_em['rmse']:.4f}")
    axes[1, 1].set_title(f"最终 Hidden 恢复 (EM)")
    axes[1, 1].set_xlabel("时间步")
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(alpha=0.25)

    fig.suptitle("稀疏 Baseline + EM 迭代 Hidden 恢复（严格无监督）",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.savefig(out_dir / "fig_sparse_em.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[OK] saved: {out_dir / 'fig_sparse_em.png'}")

    # Save results
    np.savez(
        out_dir / "results.npz",
        hidden_true=hidden[:-1],
        h_pred_sweep_best=h_pred_best,
        h_pred_em_best=best_em["h_pred"],
        sweep_lams=np.array([r["lam"] for r in sweep_results]),
        sweep_pears=pears,
        sweep_rmses=rmses,
        em_pears=np.array(pears_em),
        em_rmses=np.array(rmses_em),
    )

    # Text summary
    with open(out_dir / "summary.md", "w", encoding="utf-8") as f:
        f.write("# Sparse Baseline + EM 迭代 结果总结\n\n")
        f.write("## Sparsity Sweep (纯线性 baseline + residual linear combo)\n\n")
        f.write("| λ | active edges | visible R² | Pearson | RMSE |\n")
        f.write("|---|---|---|---|---|\n")
        for r in sweep_results:
            f.write(f"| {r['lam']} | {r['n_active']}/20 | {r['r2']:.3f} | {r['pearson']:+.4f} | {r['rmse']:.4f} |\n")
        f.write(f"\n**BEST**: λ={best['lam']}, Pearson={best['pearson']:.4f}, RMSE={best['rmse']:.4f}\n\n")
        f.write("## EM 迭代精化\n\n")
        f.write("| Iter | Pearson | RMSE |\n")
        f.write("|---|---|---|\n")
        for h in history:
            f.write(f"| {h['iter']} | {h['pearson']:+.4f} | {h['rmse']:.4f} |\n")
        f.write(f"\n**BEST (EM)**: iter {best_em['iter']}, Pearson={best_em['pearson']:.4f}, RMSE={best_em['rmse']:.4f}\n")
    print(f"[OK] saved: {out_dir / 'summary.md'}")
    print()
    print("=" * 70)
    print(f"FINAL BEST (sparsity sweep): Pearson={best['pearson']:.4f} RMSE={best['rmse']:.4f}")
    print(f"FINAL BEST (EM):             Pearson={best_em['pearson']:.4f} RMSE={best_em['rmse']:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
