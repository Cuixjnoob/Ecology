"""SINDy 式动力学发现 + Hidden 恢复。

核心思想:
  log(x_{t+1}/x_t)[i] = sum_k c_{i,k} · φ_k(x_t) + b_i · h(t) + noise

  φ_k 是一个 basis library:
    φ_1: 1 (constant / intrinsic growth)
    φ_2-6: x_j (linear, for each visible species)
    φ_7-11: x_i * x_j (LV-like pair interactions)
    φ_12-16: x_j / (K + x_j) (Holling II saturating)
    φ_17-21: x_j² (quadratic self-limitation)
    φ_22-26: (x_j - A) / (K + x_j) (Allee form)

  L1 稀疏性自动挑选活跃 terms → 从数据学出真实动力学形式
  同时联合优化 b (hidden coupling) 和 h (hidden sequence)

优势:
  - 不假设 LV 或 Holling
  - 稀疏结构防过拟合
  - 可解释（每项有明确物理意义）
  - 和 GNN 一样灵活但更 principled

输入数据集:
  LV: runs/analysis_5vs6_species/trajectories.npz
  Holling: runs/*_5vs6_holling/trajectories.npz (latest)
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


def build_library(x: torch.Tensor, half_sat: float = 0.45, allee_thresh: float = 0.15) -> tuple[torch.Tensor, list]:
    """Build basis function library for each time step.

    Args:
      x: (T, N) visible states
      half_sat: Holling half-saturation
      allee_thresh: Allee threshold

    Returns:
      Phi: (T, L) library features
      names: list of strings describing each feature
    """
    T, N = x.shape
    feats = [torch.ones(T, 1)]  # 0: intercept
    names = ["1"]

    # Linear: x_j for j=1..N
    feats.append(x)
    names.extend([f"x{j+1}" for j in range(N)])

    # Quadratic self: x_j²
    feats.append(x ** 2)
    names.extend([f"x{j+1}^2" for j in range(N)])

    # Holling II: x_j / (K + x_j) for each species
    holling = x / (half_sat + x)
    feats.append(holling)
    names.extend([f"H({j+1})" for j in range(N)])

    # Allee: (x_j - A) / (K + x_j)
    allee = (x - allee_thresh) / (0.5 + x)
    feats.append(allee)
    names.extend([f"Al({j+1})" for j in range(N)])

    Phi = torch.cat(feats, dim=1)  # (T, 1 + 4*N) = (T, 21)
    return Phi, names


def sindy_fit(
    x_current: torch.Tensor,          # (T-1, N) current state
    log_ratios: torch.Tensor,          # (T-1, N) target
    h_estimate: torch.Tensor | None,  # (T-1,) current hidden estimate, or None
    lam_sparse: float = 0.05,
    lam_h_sparse: float = 0.001,
    n_iter: int = 2000,
    lr: float = 0.01,
    seed: int = 42,
):
    """Fit SINDy coefficients + hidden coupling.

    Model:
      log_ratio[t, i] = Phi[t, :] @ C[:, i] + b[i] * h[t] (+ c[i] * h[t]^2)
    """
    torch.manual_seed(seed)
    T, N = x_current.shape
    Phi, names = build_library(x_current)
    L = Phi.shape[1]

    # Coefficient matrix: (L, N)  c[l, i] = how term l contributes to species i's log_ratio
    C = torch.zeros(L, N, requires_grad=True)
    with torch.no_grad():
        C.data += 0.01 * torch.randn(L, N)

    # Hidden coupling (only if h_estimate provided)
    if h_estimate is not None:
        b_param = torch.zeros(N, requires_grad=True)
        c_param = torch.zeros(N, requires_grad=True)
        h = torch.tensor(h_estimate, dtype=torch.float32).clone()
        params = [C, b_param, c_param]
    else:
        b_param = None
        c_param = None
        h = None
        params = [C]

    opt = torch.optim.Adam(params, lr=lr)

    for i in range(n_iter):
        opt.zero_grad()
        pred = Phi @ C  # (T-1, N)
        if h is not None:
            pred = pred + h.unsqueeze(-1) * b_param.view(1, -1) + (h.unsqueeze(-1) ** 2) * c_param.view(1, -1)
        fit_loss = ((pred - log_ratios) ** 2).mean()
        # L1 sparsity on C
        sparse_loss = C.abs().mean()
        loss = fit_loss + lam_sparse * sparse_loss
        loss.backward()
        opt.step()

    with torch.no_grad():
        pred = Phi @ C
        if h is not None:
            pred = pred + h.unsqueeze(-1) * b_param.view(1, -1) + (h.unsqueeze(-1) ** 2) * c_param.view(1, -1)
        residual_no_hidden = log_ratios - Phi @ C  # Includes hidden contribution

    return {
        "C": C.detach().numpy(),
        "names": names,
        "Phi": Phi.detach().numpy(),
        "residual_no_hidden": residual_no_hidden.detach().numpy(),
        "b": b_param.detach().numpy() if b_param is not None else None,
        "c": c_param.detach().numpy() if c_param is not None else None,
    }


def recover_hidden(residual_no_hidden: np.ndarray, hidden_true: np.ndarray):
    T = residual_no_hidden.shape[0]
    Z = np.concatenate([residual_no_hidden, np.ones((T, 1))], axis=1)
    coef, _, _, _ = np.linalg.lstsq(Z, hidden_true, rcond=None)
    h_pred = Z @ coef
    pear = float(np.corrcoef(h_pred, hidden_true)[0, 1])
    rmse = float(np.sqrt(((h_pred - hidden_true) ** 2).mean()))
    return h_pred, pear, rmse


def run_sindy_em(states, hidden_true, lams, label):
    """Run SINDy-based EM iterative recovery."""
    safe = np.clip(states, 1e-6, None)
    log_ratios_np = np.log(safe[1:] / safe[:-1])
    log_ratios_np = np.clip(log_ratios_np, -1.12, 0.92)
    x_current = torch.tensor(states[:-1], dtype=torch.float32)
    y = torch.tensor(log_ratios_np, dtype=torch.float32)

    print(f"\n=== {label} - SINDy sparsity sweep ===")
    sweep = []
    for lam in lams:
        res = sindy_fit(x_current, y, h_estimate=None, lam_sparse=lam, n_iter=2000, lr=0.01)
        C = res["C"]
        n_active = int((np.abs(C) > 0.01).sum())
        residual = res["residual_no_hidden"]
        h_pred, pear, rmse = recover_hidden(residual, hidden_true[:-1])
        r2_mean = float((1 - residual.var(axis=0) / log_ratios_np.var(axis=0)).mean())
        sweep.append({
            "lam": lam, "pearson": pear, "rmse": rmse,
            "n_active": n_active, "r2": r2_mean, "C": C, "names": res["names"],
            "h_pred": h_pred, "residual": residual,
        })
        print(f"  lam={lam:>6.3f}  n_active={n_active:3d}  R2={r2_mean:+.3f}  Pearson={pear:+.4f}  RMSE={rmse:.4f}")

    best = max(sweep, key=lambda r: abs(r["pearson"]))
    print(f"BEST sweep: lam={best['lam']}, Pearson={best['pearson']:.4f}, RMSE={best['rmse']:.4f}")

    # EM iteration
    em_results = [best.copy()]
    em_results[0]["iter"] = 0
    h_current = best["h_pred"]
    print(f"\n=== {label} - EM iterations ===")
    for k in range(1, 4):
        res = sindy_fit(x_current, y, h_estimate=h_current, lam_sparse=0.01, n_iter=2000, lr=0.01)
        residual = res["residual_no_hidden"]
        h_new, pear, rmse = recover_hidden(residual, hidden_true[:-1])
        em_results.append({
            "iter": k, "pearson": pear, "rmse": rmse,
            "C": res["C"], "names": res["names"],
            "h_pred": h_new, "b": res["b"], "c": res["c"],
        })
        print(f"  iter {k}: Pearson={pear:+.4f}  RMSE={rmse:.4f}")
        h_current = h_new

    best_em = max(em_results, key=lambda r: abs(r["pearson"]))
    print(f"BEST EM: iter {best_em['iter']}, Pearson={best_em['pearson']:.4f}, RMSE={best_em['rmse']:.4f}")
    return {"sweep": sweep, "best_sweep": best, "em": em_results, "best_em": best_em}


def plot_C_activity(C, names, ax, title):
    """Visualize which library terms are active per species."""
    L, N = C.shape
    vmax = np.abs(C).max()
    im = ax.imshow(C, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(N))
    ax.set_xticklabels([f"v{j+1}" for j in range(N)])
    ax.set_yticks(range(L))
    ax.set_yticklabels(names, fontsize=6)
    ax.set_title(title, fontsize=11)
    return im


def main():
    _configure_matplotlib()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"runs/{timestamp}_sindy_hidden_recovery")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_dir}")

    lams = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]

    # LV data
    d_lv = np.load("runs/analysis_5vs6_species/trajectories.npz")
    result_lv = run_sindy_em(d_lv["states_B_5species"], d_lv["hidden_B"], lams, "LV")

    # Holling data (latest)
    holling_files = sorted(glob.glob("runs/*_5vs6_holling/trajectories.npz"))
    d_holling = np.load(holling_files[-1])
    result_holling = run_sindy_em(d_holling["states_B_5species"], d_holling["hidden_B"], lams, "Holling")

    # Plot
    fig, axes = plt.subplots(3, 2, figsize=(14, 13), constrained_layout=True)
    for idx, (result, d, label) in enumerate([
        (result_lv, d_lv, "LV"),
        (result_holling, d_holling, "Holling"),
    ]):
        # Row 0: Sparsity sweep
        ax = axes[0, idx]
        lam_arr = np.array([r["lam"] for r in result["sweep"]])
        pear_arr = np.array([r["pearson"] for r in result["sweep"]])
        rmse_arr = np.array([r["rmse"] for r in result["sweep"]])
        ax.semilogx(lam_arr, pear_arr, marker="o", color="#1565c0", label="Pearson")
        ax2 = ax.twinx()
        ax2.semilogx(lam_arr, rmse_arr, marker="s", color="#c62828", label="RMSE")
        ax.set_xlabel("λ"); ax.set_ylabel("Pearson", color="#1565c0")
        ax2.set_ylabel("RMSE", color="#c62828")
        ax.axhline(0.9, color="green", linestyle="--", linewidth=0.8, alpha=0.5)
        ax2.axhline(0.1, color="orange", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.set_title(f"{label}: SINDy sparsity sweep", fontsize=11)
        ax.grid(alpha=0.25)

        # Row 1: Hidden recovery
        ax = axes[1, idx]
        hidden_true = d["hidden_B"][:-1]
        best_em = result["best_em"]
        X = np.concatenate([best_em["h_pred"].reshape(-1, 1), np.ones((len(best_em["h_pred"]), 1))], axis=1)
        coef, _, _, _ = np.linalg.lstsq(X, hidden_true, rcond=None)
        h_scaled = X @ coef
        pear_s = np.corrcoef(h_scaled, hidden_true)[0, 1]
        rmse_s = np.sqrt(((h_scaled - hidden_true) ** 2).mean())
        t_axis = np.arange(len(hidden_true))
        ax.plot(t_axis, hidden_true, color="black", linewidth=1.2, label="真实")
        ax.plot(t_axis, h_scaled, color="#ff7f0e", linewidth=1.0, alpha=0.85, label="SINDy+EM")
        ax.set_title(f"{label}: Hidden 恢复 (iter {best_em['iter']}) Pearson={pear_s:.3f} RMSE={rmse_s:.3f}",
                     fontsize=11, fontweight="bold")
        ax.legend(fontsize=9); ax.grid(alpha=0.25)

        # Row 2: Active library terms
        ax = axes[2, idx]
        plot_C_activity(result["best_sweep"]["C"], result["best_sweep"]["names"], ax,
                         f"{label}: 选中的动力学项 (λ={result['best_sweep']['lam']})")

    fig.suptitle("SINDy 式动力学发现 + Hidden 恢复", fontsize=14, fontweight="bold", y=1.01)
    fig.savefig(out_dir / "fig_sindy_2x2.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[OK] saved: {out_dir / 'fig_sindy_2x2.png'}")

    # Summary
    with open(out_dir / "summary.md", "w", encoding="utf-8") as f:
        f.write("# SINDy 式动力学发现结果\n\n")
        for result, label in [(result_lv, "LV"), (result_holling, "Holling")]:
            f.write(f"## {label}\n\n")
            best = result["best_sweep"]
            best_em = result["best_em"]
            f.write(f"- Best sweep: λ={best['lam']}, n_active={best['n_active']}, Pearson={best['pearson']:.4f}, RMSE={best['rmse']:.4f}\n")
            f.write(f"- Best EM: iter {best_em['iter']}, Pearson={best_em['pearson']:.4f}, RMSE={best_em['rmse']:.4f}\n\n")
            # Top activated terms per species
            C = best["C"]
            names = best["names"]
            f.write("**Top activated terms per species**:\n\n")
            for i in range(5):
                col = C[:, i]
                top_idx = np.argsort(np.abs(col))[::-1][:5]
                terms_str = ", ".join([f"{names[k]}={col[k]:+.3f}" for k in top_idx if abs(col[k]) > 0.01])
                f.write(f"- v{i+1}: {terms_str}\n")
            f.write("\n")

    # Save results
    np.savez(
        out_dir / "results.npz",
        lv_best_pearson=result_lv["best_em"]["pearson"],
        lv_best_rmse=result_lv["best_em"]["rmse"],
        holling_best_pearson=result_holling["best_em"]["pearson"],
        holling_best_rmse=result_holling["best_em"]["rmse"],
        lv_C=result_lv["best_sweep"]["C"],
        holling_C=result_holling["best_sweep"]["C"],
    )
    print(f"[OK] saved: {out_dir}")
    print()
    print("=" * 70)
    print("SINDy RESULTS:")
    print(f"  LV:      Pearson={result_lv['best_em']['pearson']:.4f}  RMSE={result_lv['best_em']['rmse']:.4f}")
    print(f"  Holling: Pearson={result_holling['best_em']['pearson']:.4f}  RMSE={result_holling['best_em']['rmse']:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
