"""在真实 Hudson Bay lynx-hare 数据上测试 hidden recovery 方法。

数据: 1847-1903 (57 年), hare + lynx 毛皮交易数量
方法:
  A. 直接把 hare 当 visible (N=1), lynx 当 hidden, Linear Sparse + EM
  B. 用 Takens 延迟嵌入构造 N=4 virtual species, lynx 当 hidden
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
    plt.rcParams["font.size"] = 12


def fit_sparse_linear(states, log_ratios, lam_sparse, n_iter=2000, lr=0.015, seed=42):
    torch.manual_seed(seed)
    N = states.shape[1]
    r = torch.zeros(N, requires_grad=True)
    A = torch.zeros(N, N, requires_grad=True)
    with torch.no_grad():
        A.fill_diagonal_(-0.2)
        A.data += 0.01 * torch.randn(N, N)
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


def recover_hidden_linear(residual, hidden_true):
    T = residual.shape[0]
    Z = np.concatenate([residual, np.ones((T, 1))], axis=1)
    coef, _, _, _ = np.linalg.lstsq(Z, hidden_true, rcond=None)
    h_pred = Z @ coef
    pear = float(np.corrcoef(h_pred, hidden_true)[0, 1])
    rmse = float(np.sqrt(((h_pred - hidden_true) ** 2).mean()))
    return h_pred, pear, rmse


def main():
    _configure_matplotlib()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"runs/{timestamp}_real_data_lynx_hare")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_dir}\n")

    # Load data
    data = np.genfromtxt("data/real_datasets/lynx_hare_long.csv", delimiter=",", skip_header=1)
    year = data[:, 0]
    hare = data[:, 1].astype(np.float32) / 1000.0  # 千计, rescale to ~units like synthetic
    lynx = data[:, 2].astype(np.float32) / 1000.0
    T = len(year)
    print(f"Data: {T} years ({int(year[0])}-{int(year[-1])})")
    print(f"  Hare: range=[{hare.min():.1f}, {hare.max():.1f}], mean={hare.mean():.1f}")
    print(f"  Lynx: range=[{lynx.min():.1f}, {lynx.max():.1f}], mean={lynx.mean():.1f}")

    # ===== Method A: Hare only (N=1) =====
    print("\n=== Method A: Hare as visible (N=1), Lynx as hidden ===")
    states_A = hare.reshape(-1, 1)  # (T, 1)
    safe_A = np.clip(states_A, 1e-3, None)
    log_ratios_A = np.log(safe_A[1:] / safe_A[:-1])
    log_ratios_A = np.clip(log_ratios_A, -2.0, 2.0)

    # Sparsity sweep
    best_A = None
    sweep_A = []
    for lam in [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]:
        residual, r, A = fit_sparse_linear(states_A, log_ratios_A, lam)
        h_pred, pear, rmse = recover_hidden_linear(residual, lynx[:-1])
        sweep_A.append({"lam": lam, "pearson": pear, "rmse": rmse, "h_pred": h_pred})
        print(f"  λ={lam:>4.2f}: Pearson={pear:+.3f}  RMSE={rmse:.2f}")
        if best_A is None or abs(pear) > abs(best_A["pearson"]):
            best_A = {"lam": lam, "pearson": pear, "rmse": rmse, "h_pred": h_pred}
    print(f"  BEST Method A: λ={best_A['lam']}, Pearson={best_A['pearson']:.4f}")

    # ===== Method B: Hare delay embedding (N=4 virtual species) =====
    print("\n=== Method B: Hare 延迟嵌入 (N=4 virtual species) ===")
    # Build delay embedding
    lag = 1
    embed_dim = 4
    start = (embed_dim - 1) * lag  # first valid t
    states_B = np.zeros((T - start, embed_dim), dtype=np.float32)
    for i in range(embed_dim):
        states_B[:, i] = hare[start - i * lag : T - i * lag] if i == 0 else hare[start - i * lag : T - i * lag]
    # Correct construction: virtual_i(t) = hare(t - i*lag) for i=0..3
    # indices: t in [start, T), virtual_i maps to hare[t - i*lag]
    for t_idx in range(T - start):
        t_global = t_idx + start
        for i in range(embed_dim):
            states_B[t_idx, i] = hare[t_global - i * lag]
    lynx_B = lynx[start:]
    print(f"  Embedded states shape: {states_B.shape}")
    print(f"  Aligned lynx shape: {lynx_B.shape}")

    safe_B = np.clip(states_B, 1e-3, None)
    log_ratios_B = np.log(safe_B[1:] / safe_B[:-1])
    log_ratios_B = np.clip(log_ratios_B, -2.0, 2.0)

    best_B = None
    sweep_B = []
    for lam in [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]:
        residual, r, A = fit_sparse_linear(states_B, log_ratios_B, lam)
        h_pred, pear, rmse = recover_hidden_linear(residual, lynx_B[:-1])
        sweep_B.append({"lam": lam, "pearson": pear, "rmse": rmse, "h_pred": h_pred})
        print(f"  λ={lam:>4.2f}: Pearson={pear:+.3f}  RMSE={rmse:.2f}")
        if best_B is None or abs(pear) > abs(best_B["pearson"]):
            best_B = {"lam": lam, "pearson": pear, "rmse": rmse, "h_pred": h_pred}
    print(f"  BEST Method B: λ={best_B['lam']}, Pearson={best_B['pearson']:.4f}")

    # ====== Plots (one per PNG) ======
    def save_single(title, plot_fn, path, figsize=(12, 6)):
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        plot_fn(ax)
        fig.suptitle(title, fontsize=13, fontweight="bold")
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)

    # Fig 1: raw data
    def plot_raw(ax):
        ax.plot(year, hare, marker="o", markersize=4, linewidth=1.5, color="#2e7d32", label="Hare (x1000)")
        ax.plot(year, lynx, marker="s", markersize=4, linewidth=1.5, color="#c62828", label="Lynx (x1000)")
        ax.set_xlabel("Year"); ax.set_ylabel("Population (x1000 pelts)")
        ax.legend(fontsize=11); ax.grid(alpha=0.25)
    save_single("Hudson Bay Company: Hare + Lynx 1847-1903 (真实数据)",
                 plot_raw, out_dir / "fig_01_raw_data.png")

    # Fig 2: Method A result
    def plot_A(ax):
        t = year[:-1]
        ax.plot(t, lynx[:-1], color="black", linewidth=1.8, label="真实 Lynx (hidden)", marker="s", markersize=3)
        # Scale-invariant
        h = best_A["h_pred"]
        X = np.concatenate([h.reshape(-1, 1), np.ones((len(h), 1))], axis=1)
        coef, _, _, _ = np.linalg.lstsq(X, lynx[:-1], rcond=None)
        h_scaled = X @ coef
        pear_s = np.corrcoef(h_scaled, lynx[:-1])[0, 1]
        ax.plot(t, h_scaled, color="#ff7f0e", linewidth=1.5, alpha=0.85,
                label=f"Method A 恢复 (P={pear_s:.3f})")
        ax.set_xlabel("Year"); ax.set_ylabel("Lynx (x1000)")
        ax.legend(fontsize=11); ax.grid(alpha=0.25)
    save_single(f"方法 A: Hare (N=1) → Recover Lynx  [λ={best_A['lam']}]",
                 plot_A, out_dir / "fig_02_method_A_hare_only.png")

    # Fig 3: Method B result
    def plot_B(ax):
        t = year[start:-1]
        ax.plot(t, lynx_B[:-1], color="black", linewidth=1.8, label="真实 Lynx (hidden)", marker="s", markersize=3)
        h = best_B["h_pred"]
        X = np.concatenate([h.reshape(-1, 1), np.ones((len(h), 1))], axis=1)
        coef, _, _, _ = np.linalg.lstsq(X, lynx_B[:-1], rcond=None)
        h_scaled = X @ coef
        pear_s = np.corrcoef(h_scaled, lynx_B[:-1])[0, 1]
        ax.plot(t, h_scaled, color="#1565c0", linewidth=1.5, alpha=0.85,
                label=f"Method B 恢复 (P={pear_s:.3f})")
        ax.set_xlabel("Year"); ax.set_ylabel("Lynx (x1000)")
        ax.legend(fontsize=11); ax.grid(alpha=0.25)
    save_single(f"方法 B: Hare Takens 嵌入 (N=4) → Recover Lynx  [λ={best_B['lam']}]",
                 plot_B, out_dir / "fig_03_method_B_takens.png")

    # Fig 4: sparsity sweep comparison
    def plot_sweep(ax):
        lams_A = [s["lam"] for s in sweep_A]
        pears_A = [s["pearson"] for s in sweep_A]
        lams_B = [s["lam"] for s in sweep_B]
        pears_B = [s["pearson"] for s in sweep_B]
        ax.semilogx(np.array(lams_A) + 1e-4, pears_A, marker="o", linewidth=2, color="#ff7f0e", label="Method A (N=1)")
        ax.semilogx(np.array(lams_B) + 1e-4, pears_B, marker="s", linewidth=2, color="#1565c0", label="Method B (N=4 Takens)")
        ax.set_xlabel("L1 sparsity λ")
        ax.set_ylabel("|Pearson| to true Lynx")
        ax.axhline(0, color="grey", linewidth=0.5)
        ax.legend(fontsize=11); ax.grid(alpha=0.25)
    save_single("Sparsity Sweep: 两种 Method 对比",
                 plot_sweep, out_dir / "fig_04_sparsity_sweep.png")

    # Summary
    with open(out_dir / "summary.md", "w", encoding="utf-8") as f:
        f.write("# 真实数据 (Hudson Bay lynx-hare) 初步结果\n\n")
        f.write(f"数据: {T} 年 ({int(year[0])}-{int(year[-1])}), 2 物种\n\n")
        f.write("## Method A: Hare (N=1) → Recover Lynx\n\n")
        f.write(f"- BEST: λ={best_A['lam']}, Pearson={best_A['pearson']:.4f}, RMSE={best_A['rmse']:.2f}\n\n")
        f.write("## Method B: Hare Takens 嵌入 (N=4) → Recover Lynx\n\n")
        f.write(f"- BEST: λ={best_B['lam']}, Pearson={best_B['pearson']:.4f}, RMSE={best_B['rmse']:.2f}\n\n")
        f.write("## 注意\n\n")
        f.write("- 这是 2 物种数据，不完全符合我们原框架的 5 visible + 1 hidden\n")
        f.write("- 两种 adapter: 单物种 or Takens 嵌入\n")
        f.write("- 结果作为 **proof-of-concept**，不是 main result\n")
    print(f"\n[OK] saved to: {out_dir}")


if __name__ == "__main__":
    main()
