"""2x2 实验: LV/Holling 数据 × sparse LV 软约束方法。

目的:
  - 确认 sparse baseline 方法在非 LV 数据（Holling）上是否仍然 work
  - 如果 Holling 上失败 → 证明 GNN 必要性
  - 对比学到的 sparse A 矩阵和真实生态结构

输入:
  - LV 5vs6: runs/analysis_5vs6_species/trajectories.npz
  - Holling 5vs6: 最新的 5vs6_holling 目录
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
from scipy.stats import spearmanr


def _configure_matplotlib() -> None:
    plt.rcParams["font.sans-serif"] = [
        "PingFang SC", "Hiragino Sans GB", "Heiti TC", "STHeiti",
        "Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def fit_sparse_baseline(states, log_ratios, lam_sparse, n_iter=1500, lr=0.015, seed=42):
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
    T = residual.shape[0]
    Z = np.concatenate([residual, np.ones((T, 1))], axis=1)
    coef, _, _, _ = np.linalg.lstsq(Z, hidden_true, rcond=None)
    h_pred = Z @ coef
    pear = float(np.corrcoef(h_pred, hidden_true)[0, 1])
    rmse = float(np.sqrt(((h_pred - hidden_true) ** 2).mean()))
    return h_pred, pear, rmse


def fit_full_with_hidden(states, log_ratios, h_current, lam_sparse=0.05, n_iter=1500, lr=0.015):
    r = torch.zeros(5, requires_grad=True)
    A = torch.zeros(5, 5, requires_grad=True)
    b = torch.zeros(5, requires_grad=True)
    c = torch.zeros(5, requires_grad=True)
    with torch.no_grad(): A.fill_diagonal_(-0.2)
    opt = torch.optim.Adam([r, A, b, c], lr=lr)
    x = torch.tensor(states[:-1], dtype=torch.float32)
    y = torch.tensor(log_ratios, dtype=torch.float32)
    h = torch.tensor(h_current, dtype=torch.float32)
    for _ in range(n_iter):
        opt.zero_grad()
        pred = r.view(1, -1) + x @ A.T + h.unsqueeze(-1) * b.view(1, -1) + (h.unsqueeze(-1) ** 2) * c.view(1, -1)
        fit_loss = ((pred - y) ** 2).mean()
        A_off = A - torch.diag(torch.diag(A))
        loss = fit_loss + lam_sparse * A_off.abs().mean()
        loss.backward()
        opt.step()
    with torch.no_grad():
        residual_no_hidden = y - (r.view(1, -1) + x @ A.T)
    return residual_no_hidden.cpu().numpy(), r.detach().numpy(), A.detach().numpy(), b.detach().numpy()


def run_full_pipeline(states, hidden_true, lam_sweep, label):
    """在一个数据集上跑 sparse baseline + EM，返回详细结果。"""
    safe = np.clip(states, 1e-6, None)
    log_ratios = np.log(safe[1:] / safe[:-1])
    log_ratios = np.clip(log_ratios, -1.12, 0.92)

    # 1. Sparsity sweep
    sweep = []
    for lam in lam_sweep:
        residual, _, A_hat = fit_sparse_baseline(states, log_ratios, lam)
        h_pred, pear, rmse = recover_hidden_from_residual(residual, hidden_true[:-1])
        n_active = int((np.abs(A_hat - np.diag(np.diag(A_hat))) > 0.01).sum())
        r2_mean = float((1 - residual.var(axis=0) / log_ratios.var(axis=0)).mean())
        sweep.append({
            "lam": lam, "pearson": pear, "rmse": rmse,
            "n_active": n_active, "r2": r2_mean,
            "A_learned": A_hat, "h_pred": h_pred,
        })

    # Pick best lam (by hidden Pearson)
    best = max(sweep, key=lambda r: abs(r["pearson"]))

    # 2. EM 迭代（从 best 开始）
    em_results = [best.copy()]
    em_results[0]["iter"] = 0
    h_current = best["h_pred"]
    for k in range(1, 4):
        residual_no_hidden, r_k, A_k, b_k = fit_full_with_hidden(states, log_ratios, h_current, lam_sparse=0.05)
        h_new, pear, rmse = recover_hidden_from_residual(residual_no_hidden, hidden_true[:-1])
        em_results.append({"iter": k, "pearson": pear, "rmse": rmse, "h_pred": h_new, "A_learned": A_k})
        h_current = h_new

    best_em = max(em_results, key=lambda r: abs(r["pearson"]))

    print(f"\n=== {label} ===")
    print(f"Sparsity sweep:")
    for r in sweep:
        print(f"  lam={r['lam']:>6.3f}  active={r['n_active']:2d}/20  R2={r['r2']:+.3f}  Pearson={r['pearson']:+.4f}  RMSE={r['rmse']:.4f}")
    print(f"BEST sweep: lam={best['lam']}, Pearson={best['pearson']:+.4f}, RMSE={best['rmse']:.4f}")
    print(f"EM iterations:")
    for r in em_results:
        print(f"  iter {r['iter']}: Pearson={r['pearson']:+.4f}  RMSE={r['rmse']:.4f}")
    print(f"BEST EM: iter {best_em['iter']}, Pearson={best_em['pearson']:+.4f}, RMSE={best_em['rmse']:.4f}")

    return {
        "label": label,
        "sweep": sweep,
        "best_sweep": best,
        "em_results": em_results,
        "best_em": best_em,
        "log_ratios": log_ratios,
    }


def compare_to_true_structure(A_learned, A_true_6x6, label):
    """对比学到的 5x5 A 与真实生态结构。"""
    # Option 1: 对比 learned A vs true_6x6 的 5x5 子集
    A_true_55 = A_true_6x6[:5, :5]
    mask = ~np.eye(5, dtype=bool)
    meaningful = mask & (np.abs(A_true_55) > 0.05)

    sign_acc_55 = float((np.sign(A_learned[meaningful]) == np.sign(A_true_55[meaningful])).mean())
    pearson_55 = float(np.corrcoef(A_learned[mask], A_true_55[mask])[0, 1])

    # Option 2: 对比 learned A 和 true_6x6[:5, :5] + true_6x6[:5, 5] * true_6x6[5, :5] (indirect through hidden)
    # Because A_learned is expected to absorb some hidden effect
    indirect = np.outer(A_true_6x6[:5, 5], A_true_6x6[5, :5])
    A_renormalized = A_true_55 + indirect
    sign_acc_renorm = float((np.sign(A_learned[meaningful]) == np.sign(A_renormalized[meaningful])).mean())
    pearson_renorm = float(np.corrcoef(A_learned[mask], A_renormalized[mask])[0, 1])

    print(f"\n{label}: Learned A vs True 生态结构")
    print(f"  vs raw 5x5 subset (excludes hidden): sign_acc={sign_acc_55:.3f}  Pearson={pearson_55:+.3f}")
    print(f"  vs renormalized 5x5 (includes indirect hidden): sign_acc={sign_acc_renorm:.3f}  Pearson={pearson_renorm:+.3f}")
    return {
        "sign_acc_raw": sign_acc_55, "pearson_raw": pearson_55,
        "sign_acc_renorm": sign_acc_renorm, "pearson_renorm": pearson_renorm,
    }


def main():
    _configure_matplotlib()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"runs/{timestamp}_sparse_2x2_lv_vs_holling")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_dir}\n")

    lam_sweep = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]

    # Load LV data
    print("Loading LV data...")
    d_lv = np.load("runs/analysis_5vs6_species/trajectories.npz")
    states_lv = d_lv["states_B_5species"]
    hidden_lv = d_lv["hidden_B"]
    A_true_lv = d_lv["interaction_matrix_full"]

    # Load Holling data (latest)
    holling_dirs = sorted(glob.glob("runs/*_5vs6_holling/trajectories.npz"))
    print(f"Available Holling data: {holling_dirs}")
    d_holling = np.load(holling_dirs[-1])
    states_holling = d_holling["states_B_5species"]
    hidden_holling = d_holling["hidden_B"]
    A_true_holling = d_holling["interaction_matrix_full"]

    # Run on both
    result_lv = run_full_pipeline(states_lv, hidden_lv, lam_sweep, "LV 数据 + sparse LV 软约束")
    result_holling = run_full_pipeline(states_holling, hidden_holling, lam_sweep, "Holling 数据 + sparse LV 软约束")

    # Compare learned A to true structure
    print("\n" + "=" * 70)
    print("Learned A vs True ecological structure")
    print("=" * 70)
    struct_lv = compare_to_true_structure(result_lv["best_sweep"]["A_learned"], A_true_lv, "LV")
    struct_holling = compare_to_true_structure(result_holling["best_sweep"]["A_learned"], A_true_holling, "Holling")

    # ===== Visualization =====
    fig, axes = plt.subplots(3, 2, figsize=(14, 13), constrained_layout=True)

    # Row 1: Sparsity sweep
    for idx, result in enumerate([result_lv, result_holling]):
        ax = axes[0, idx]
        lams = [r["lam"] for r in result["sweep"]]
        pears = [r["pearson"] for r in result["sweep"]]
        rmses = [r["rmse"] for r in result["sweep"]]
        ax.semilogx(np.array(lams) + 1e-4, pears, marker="o", color="#1565c0", label="Pearson")
        ax.set_xlabel("λ"); ax.set_ylabel("Pearson", color="#1565c0")
        ax2 = ax.twinx()
        ax2.semilogx(np.array(lams) + 1e-4, rmses, marker="s", color="#c62828", label="RMSE")
        ax2.set_ylabel("RMSE", color="#c62828")
        ax.axhline(0.9, color="green", linestyle="--", linewidth=0.7, alpha=0.5)
        ax2.axhline(0.1, color="orange", linestyle="--", linewidth=0.7, alpha=0.5)
        ax.set_title(f"{result['label'].split('+')[0].strip()}: sparsity sweep", fontsize=11)
        ax.grid(alpha=0.25)

    # Row 2: Hidden recovery (best)
    for idx, (result, hidden_true) in enumerate([(result_lv, hidden_lv), (result_holling, hidden_holling)]):
        ax = axes[1, idx]
        best_em = result["best_em"]
        h_true_aligned = hidden_true[:-1]
        # Scale-invariant alignment
        X = np.concatenate([best_em["h_pred"].reshape(-1, 1), np.ones((len(best_em["h_pred"]), 1))], axis=1)
        coef, _, _, _ = np.linalg.lstsq(X, h_true_aligned, rcond=None)
        h_scaled = X @ coef
        pear_s = np.corrcoef(h_scaled, h_true_aligned)[0, 1]
        rmse_s = np.sqrt(((h_scaled - h_true_aligned) ** 2).mean())
        t_axis = np.arange(len(h_true_aligned))
        ax.plot(t_axis, h_true_aligned, color="black", linewidth=1.2, label="真实")
        ax.plot(t_axis, h_scaled, color="#ff7f0e", linewidth=1.0, alpha=0.85, label="恢复 (scaled)")
        ax.set_title(f"Hidden 恢复 (EM best, Pearson={pear_s:.3f} RMSE={rmse_s:.3f})",
                     fontsize=11, fontweight="bold")
        ax.legend(fontsize=9); ax.grid(alpha=0.25)

    # Row 3: A matrix comparison
    for idx, (result, struct, A_true) in enumerate([
        (result_lv, struct_lv, A_true_lv),
        (result_holling, struct_holling, A_true_holling),
    ]):
        ax = axes[2, idx]
        A_learned = result["best_sweep"]["A_learned"]
        A_true_55 = A_true[:5, :5]
        mask = ~np.eye(5, dtype=bool)
        # Scatter
        ax.scatter(A_true_55[mask], A_learned[mask], alpha=0.7, s=50, color="#1565c0", edgecolor="black")
        vmin = min(A_true_55.min(), A_learned.min())
        vmax = max(A_true_55.max(), A_learned.max())
        ax.plot([vmin, vmax], [vmin, vmax], "k--", linewidth=0.8, alpha=0.5)
        ax.axhline(0, color="grey", linewidth=0.5, alpha=0.3)
        ax.axvline(0, color="grey", linewidth=0.5, alpha=0.3)
        ax.set_xlabel("真实 A_true[i,j] (5x5 subset)")
        ax.set_ylabel("学到 A_learned[i,j]")
        ax.set_title(f"A 矩阵: sign_acc={struct['sign_acc_raw']:.2f}, Pearson={struct['pearson_raw']:.2f}",
                     fontsize=11)
        ax.grid(alpha=0.25)

    fig.suptitle("2x2 对比: LV vs Holling × sparse LV 软约束", fontsize=14, fontweight="bold", y=1.01)
    fig.savefig(out_dir / "fig_2x2.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Save summary
    with open(out_dir / "summary.md", "w", encoding="utf-8") as f:
        f.write("# 2x2 对比: LV vs Holling × sparse LV 软约束\n\n")
        for result, struct, label in [
            (result_lv, struct_lv, "LV"),
            (result_holling, struct_holling, "Holling"),
        ]:
            f.write(f"## {label} 数据\n\n")
            f.write(f"- Best sweep: λ={result['best_sweep']['lam']}, Pearson={result['best_sweep']['pearson']:.4f}, RMSE={result['best_sweep']['rmse']:.4f}\n")
            f.write(f"- Best EM: iter {result['best_em']['iter']}, Pearson={result['best_em']['pearson']:.4f}, RMSE={result['best_em']['rmse']:.4f}\n")
            f.write(f"- A vs true 5x5: sign_acc={struct['sign_acc_raw']:.3f}, Pearson={struct['pearson_raw']:+.3f}\n")
            f.write(f"- A vs renormalized (包含 hidden 间接): sign_acc={struct['sign_acc_renorm']:.3f}, Pearson={struct['pearson_renorm']:+.3f}\n\n")

        f.write("## 关键对比\n\n")
        f.write(f"| 数据 | 最佳 Pearson | 最佳 RMSE | A sign_acc |\n|---|---|---|---|\n")
        f.write(f"| LV (匹配) | {result_lv['best_em']['pearson']:.4f} | {result_lv['best_em']['rmse']:.4f} | {struct_lv['sign_acc_raw']:.3f} |\n")
        f.write(f"| Holling (非 LV) | {result_holling['best_em']['pearson']:.4f} | {result_holling['best_em']['rmse']:.4f} | {struct_holling['sign_acc_raw']:.3f} |\n")

    np.savez(
        out_dir / "results.npz",
        lv_best_pearson=result_lv["best_em"]["pearson"],
        lv_best_rmse=result_lv["best_em"]["rmse"],
        holling_best_pearson=result_holling["best_em"]["pearson"],
        holling_best_rmse=result_holling["best_em"]["rmse"],
        lv_A_learned=result_lv["best_sweep"]["A_learned"],
        holling_A_learned=result_holling["best_sweep"]["A_learned"],
    )

    print("\n" + "=" * 70)
    print("FINAL 2x2 COMPARISON:")
    print(f"  LV 数据 + sparse LV:      Pearson={result_lv['best_em']['pearson']:.4f} RMSE={result_lv['best_em']['rmse']:.4f}")
    print(f"  Holling 数据 + sparse LV: Pearson={result_holling['best_em']['pearson']:.4f} RMSE={result_holling['best_em']['rmse']:.4f}")
    print(f"  A sign accuracy LV:      {struct_lv['sign_acc_raw']:.3f}")
    print(f"  A sign accuracy Holling: {struct_holling['sign_acc_raw']:.3f}")
    print("=" * 70)
    print(f"\n[OK] saved to: {out_dir}")


if __name__ == "__main__":
    main()
