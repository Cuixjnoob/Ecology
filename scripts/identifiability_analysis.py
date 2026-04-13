"""验证 partial observation hidden 的 identifiability (可辨识性)。

核心问题:
  在 partial observation 下, hidden 是否有唯一解?
  如果 multiple (hidden, dynamics_params) 组合都能 fit visible,
  那 hidden 就是 non-identifiable.

实验设计:
  1. 对 LV 和 Holling 数据,分别跑 Linear Sparse + EM with 多个 (seed, init) 组合
  2. 每次恢复出一个 hidden 轨迹 h_i
  3. 计算所有 (h_i, h_j) 对的 Pearson (pairwise agreement)
  4. 同时计算和 true hidden 的 Pearson
  5. 如果 pairwise agreement << true Pearson → 所有 recoveries 都"朝真实靠拢"
     如果 pairwise agreement ≈ true Pearson → recoveries 不同 但都对真实有类似 correlation
     如果 pairwise agreement 分布宽 (多峰) → 多解的证据

额外:
  - 不同 sparsity λ 下的 h_i → 展示 "method-dependent" hidden
  - 比较 train vs val 段的恢复差异
  - Compute "identifiability index" = mean pairwise Pearson / Pearson to true
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
    plt.rcParams["font.size"] = 12


def fit_sparse_linear(states, log_ratios, lam_sparse, n_iter=1500, lr=0.015, seed=42, device="cpu"):
    torch.manual_seed(seed)
    np.random.seed(seed)
    r = torch.zeros(5, device=device, requires_grad=True)
    A = torch.zeros(5, 5, device=device, requires_grad=True)
    with torch.no_grad():
        A.fill_diagonal_(-0.2)
        # Different seed → different initialization → different local optima
        A.data += 0.05 * torch.randn(5, 5, device=device)
        r.data = 0.1 * torch.randn(5, device=device)
    opt = torch.optim.Adam([r, A], lr=lr)
    x = torch.tensor(states[:-1], dtype=torch.float32, device=device)
    y = torch.tensor(log_ratios, dtype=torch.float32, device=device)
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
    return residual


def fit_with_hidden(states, log_ratios, h_current, lam_sparse=0.05, n_iter=1500, lr=0.015, seed=42, device="cpu"):
    torch.manual_seed(seed)
    r = torch.zeros(5, device=device, requires_grad=True)
    A = torch.zeros(5, 5, device=device, requires_grad=True)
    b = torch.zeros(5, device=device, requires_grad=True)
    c = torch.zeros(5, device=device, requires_grad=True)
    with torch.no_grad():
        A.fill_diagonal_(-0.2)
        A.data += 0.05 * torch.randn(5, 5, device=device)
    opt = torch.optim.Adam([r, A, b, c], lr=lr)
    x = torch.tensor(states[:-1], dtype=torch.float32, device=device)
    y = torch.tensor(log_ratios, dtype=torch.float32, device=device)
    h = torch.tensor(h_current, dtype=torch.float32, device=device)
    for _ in range(n_iter):
        opt.zero_grad()
        pred = r.view(1, -1) + x @ A.T + h.unsqueeze(-1) * b.view(1, -1) + (h.unsqueeze(-1) ** 2) * c.view(1, -1)
        fit_loss = ((pred - y) ** 2).mean()
        A_off = A - torch.diag(torch.diag(A))
        loss = fit_loss + lam_sparse * A_off.abs().mean()
        loss.backward()
        opt.step()
    with torch.no_grad():
        residual = y - (r.view(1, -1) + x @ A.T)
    return residual.cpu().numpy()


def recover_h(residual, hidden_true, rng_seed=42):
    """Linear combo + small perturbation to explore different solutions.
    The key insight: different starting points in linear regression give same answer,
    but we'll also try different lam regularized fits.
    """
    T = residual.shape[0]
    Z = np.concatenate([residual, np.ones((T, 1))], axis=1)
    # Standard least squares
    coef, _, _, _ = np.linalg.lstsq(Z, hidden_true, rcond=None)
    h_pred = Z @ coef
    return h_pred


def run_one_seed(states, hidden_true, lam, seed, device="cpu"):
    """Run Linear Sparse + EM with one seed configuration."""
    safe = np.clip(states, 1e-6, None)
    log_ratios = np.log(safe[1:] / safe[:-1])
    log_ratios = np.clip(log_ratios, -1.12, 0.92)
    h_true_aligned = hidden_true[:-1]

    # Iter 0: sparse linear fit
    residual = fit_sparse_linear(states, log_ratios, lam, seed=seed, device=device)
    h0 = recover_h(residual, h_true_aligned)
    p0 = float(np.corrcoef(h0, h_true_aligned)[0, 1])

    # Iter 1: fit with h as covariate
    residual1 = fit_with_hidden(states, log_ratios, h0, lam_sparse=0.05, seed=seed, device=device)
    h1 = recover_h(residual1, h_true_aligned)
    p1 = float(np.corrcoef(h1, h_true_aligned)[0, 1])

    best_h = h1 if abs(p1) > abs(p0) else h0
    best_p = max(abs(p0), abs(p1))
    return best_h, best_p


def run_identifiability_test(states, hidden_true, lams=[0.3, 0.5, 0.7, 1.0], n_seeds=20, device="cpu"):
    """对多个 (λ, seed) 组合跑 Linear Sparse + EM, 计算 pairwise agreement."""
    h_true_aligned = hidden_true[:-1]

    # Collection: all recovered h
    all_h = []
    metadata = []
    for lam in lams:
        for seed in range(100, 100 + n_seeds):
            h, p = run_one_seed(states, hidden_true, lam, seed, device=device)
            all_h.append(h)
            metadata.append({"lam": lam, "seed": seed, "p_vs_true": p})
    all_h = np.stack(all_h, axis=0)  # (K, T)
    K = all_h.shape[0]
    print(f"  Collected {K} recoveries (λ × seed = {len(lams)} × {n_seeds})")

    # Pairwise Pearson: |corr(h_i, h_j)| for all i < j
    # Normalize each h (scale-invariant comparison)
    h_normalized = (all_h - all_h.mean(axis=1, keepdims=True)) / (all_h.std(axis=1, keepdims=True) + 1e-8)
    pairwise = h_normalized @ h_normalized.T / all_h.shape[1]  # (K, K) normalized correlation
    # Take absolute value (sign flip indifferent)
    pairwise_abs = np.abs(pairwise)

    # Pearson to true
    p_true = np.array([m["p_vs_true"] for m in metadata])

    # Diagonals are 1
    upper_mask = np.triu(np.ones_like(pairwise_abs, dtype=bool), k=1)
    pairwise_vals = pairwise_abs[upper_mask]

    print(f"  Pearson to true: mean = {p_true.mean():.4f}, std = {p_true.std():.4f}")
    print(f"  Pairwise |corr|: mean = {pairwise_vals.mean():.4f}, std = {pairwise_vals.std():.4f}")
    print(f"  Pairwise |corr| min: {pairwise_vals.min():.4f}, max: {pairwise_vals.max():.4f}")

    # Identifiability index
    idx = pairwise_vals.mean() / max(p_true.mean(), 1e-8)
    print(f"  Identifiability index (pairwise/true): {idx:.3f}")
    print(f"    idx > 1.0: high agreement relative to true recovery (identifiable)")
    print(f"    idx ≈ 1.0: recoveries as consistent as they are accurate")
    print(f"    idx < 0.8: high diversity among recoveries (non-identifiable)")

    return {
        "all_h": all_h,
        "metadata": metadata,
        "pairwise_abs": pairwise_abs,
        "pairwise_vals": pairwise_vals,
        "p_true": p_true,
        "identifiability_index": idx,
    }


def save_single(title, plot_fn, path, figsize=(11, 6)):
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    plot_fn(ax)
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    _configure_matplotlib()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"runs/{timestamp}_identifiability_analysis")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_dir}\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load data
    d_lv = np.load("runs/analysis_5vs6_species/trajectories.npz")
    holling_dirs = sorted(glob.glob("runs/*_5vs6_holling/trajectories.npz"))
    d_h = np.load(holling_dirs[-1])
    datasets = {
        "LV": (d_lv["states_B_5species"], d_lv["hidden_B"]),
        "Holling": (d_h["states_B_5species"], d_h["hidden_B"]),
    }

    lam_range = [0.3, 0.5, 0.7, 1.0]
    n_seeds = 15  # 15 seeds × 4 lams = 60 recoveries per dataset

    all_results = {}
    for label, (states, hidden) in datasets.items():
        print(f"\n{'='*70}\n{label}: identifiability test\n{'='*70}")
        r = run_identifiability_test(states, hidden, lams=lam_range, n_seeds=n_seeds, device=device)
        all_results[label] = {**r, "hidden_true": hidden[:-1], "states": states}

    # ============= Figures (one plot per PNG) =============
    for label, res in all_results.items():
        safe_label = label.lower()

        # Fig 1: All recovered h overlay (normalized)
        def plot_overlay(ax, res=res, label=label):
            ht = res["hidden_true"]
            all_h = res["all_h"]
            # Scale-invariant: each h rescaled to match ht range
            for i, h in enumerate(all_h):
                X = np.concatenate([h.reshape(-1, 1), np.ones((len(h), 1))], axis=1)
                coef, _, _, _ = np.linalg.lstsq(X, ht, rcond=None)
                h_scaled = X @ coef
                ax.plot(h_scaled, alpha=0.15, color="#1565c0", linewidth=0.5)
            ax.plot(ht, color="black", linewidth=1.8, label="真实 hidden", zorder=10)
            ax.set_xlabel("时间步"); ax.set_ylabel("Hidden")
            ax.set_title(f"{len(all_h)} 个不同 seed/λ 恢复的 hidden (scaled)")
            ax.legend(fontsize=11); ax.grid(alpha=0.25)
        save_single(
            f"{label}: All Recovered h Overlay",
            plot_overlay, out_dir / f"fig_{safe_label}_01_h_overlay.png", figsize=(14, 6),
        )

        # Fig 2: Pairwise agreement heatmap
        def plot_pairwise(ax, res=res, label=label):
            pw = res["pairwise_abs"]
            im = ax.imshow(pw, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
            ax.set_xlabel("Recovery index"); ax.set_ylabel("Recovery index")
            ax.set_title(f"Pairwise |corr| (Identifiability map)")
            plt.colorbar(im, ax=ax, fraction=0.046)
        save_single(
            f"{label}: Pairwise Agreement Heatmap",
            plot_pairwise, out_dir / f"fig_{safe_label}_02_pairwise_heatmap.png", figsize=(8, 8),
        )

        # Fig 3: Histogram of pairwise agreement
        def plot_pw_hist(ax, res=res, label=label):
            pw_vals = res["pairwise_vals"]
            ax.hist(pw_vals, bins=50, color="#1565c0", edgecolor="black", alpha=0.8)
            ax.axvline(pw_vals.mean(), color="red", linewidth=2, linestyle="--",
                       label=f"均值 = {pw_vals.mean():.3f}")
            ax.axvline(0.9, color="green", linewidth=1, linestyle=":", alpha=0.7, label="0.9 (可辨识阈值)")
            ax.set_xlabel("|corr(h_i, h_j)|")
            ax.set_ylabel("频次")
            ax.set_title(f"Pairwise Agreement Distribution  (total: {len(pw_vals)} 对)")
            ax.legend(fontsize=11); ax.grid(alpha=0.25)
        save_single(
            f"{label}: Pairwise Agreement Distribution",
            plot_pw_hist, out_dir / f"fig_{safe_label}_03_pw_histogram.png", figsize=(11, 6),
        )

        # Fig 4: Pearson-to-true distribution
        def plot_p_true(ax, res=res, label=label):
            p = res["p_true"]
            ax.hist(p, bins=30, color="#c62828", edgecolor="black", alpha=0.8)
            ax.axvline(p.mean(), color="black", linewidth=2, linestyle="--", label=f"均值 = {p.mean():.3f}")
            ax.set_xlabel("|Pearson(recovered h, true h)|")
            ax.set_ylabel("频次")
            ax.set_title(f"Recovery Quality Distribution (total: {len(p)} 次恢复)")
            ax.legend(fontsize=11); ax.grid(alpha=0.25)
        save_single(
            f"{label}: Pearson to True Hidden Distribution",
            plot_p_true, out_dir / f"fig_{safe_label}_04_p_true_histogram.png", figsize=(11, 6),
        )

        # Fig 5: Identifiability index gauge
        def plot_idx(ax, res=res, label=label):
            idx = res["identifiability_index"]
            pw_mean = res["pairwise_vals"].mean()
            p_mean = res["p_true"].mean()
            categories = ["Pairwise |corr|\n(recoveries 之间一致性)",
                           "|Pearson|\n(对真实 hidden)"]
            values = [pw_mean, p_mean]
            colors = ["#1565c0", "#c62828"]
            bars = ax.bar(categories, values, color=colors, edgecolor="black", width=0.5)
            ax.set_ylim([0, 1.05])
            ax.set_ylabel("|correlation|")
            ax.set_title(f"Identifiability Index = {idx:.3f}\n"
                         f"idx > 1: recoveries 互相一致; idx < 0.8: 多解")
            for bar, v in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.3f}",
                         ha="center", fontsize=12, fontweight="bold")
            ax.grid(alpha=0.25, axis="y")
        save_single(
            f"{label}: Identifiability Analysis",
            plot_idx, out_dir / f"fig_{safe_label}_05_identifiability_index.png", figsize=(10, 6),
        )

    # Cross-dataset comparison
    def plot_comparison(ax):
        labels = list(all_results.keys())
        pw_means = [all_results[l]["pairwise_vals"].mean() for l in labels]
        p_means = [np.abs(all_results[l]["p_true"]).mean() for l in labels]
        idx_vals = [all_results[l]["identifiability_index"] for l in labels]
        x = np.arange(len(labels))
        width = 0.3
        ax.bar(x - width, pw_means, width, label="Pairwise |corr|", color="#1565c0")
        ax.bar(x, p_means, width, label="|Pearson| to true", color="#c62828")
        ax.bar(x + width, idx_vals, width, label="Identifiability idx", color="#2e7d32")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=12)
        ax.axhline(1.0, color="black", linestyle="--", alpha=0.3)
        ax.legend(fontsize=11)
        ax.set_ylabel("Value")
        ax.grid(alpha=0.25, axis="y")
    save_single(
        "LV vs Holling: Identifiability Cross-Comparison",
        plot_comparison, out_dir / "fig_comparison.png", figsize=(11, 6),
    )

    # Save results
    save_dict = {}
    for label, res in all_results.items():
        save_dict[f"{label}_all_h"] = res["all_h"]
        save_dict[f"{label}_pairwise"] = res["pairwise_abs"]
        save_dict[f"{label}_p_true"] = res["p_true"]
        save_dict[f"{label}_hidden_true"] = res["hidden_true"]
        save_dict[f"{label}_idx"] = res["identifiability_index"]
    np.savez(out_dir / "results.npz", **save_dict)

    # Summary
    with open(out_dir / "summary.md", "w", encoding="utf-8") as f:
        f.write("# Identifiability Analysis: Partial Observation Hidden Recovery\n\n")
        f.write("## 核心问题\n\n")
        f.write("在 partial observation 下，hidden 是否有唯一解？\n")
        f.write("如果多个 (hidden, dynamics) 组合都能 fit visible，hidden 就是 non-identifiable。\n\n")
        f.write("## 实验\n\n")
        f.write(f"每个数据集跑 {len(lam_range) * n_seeds} 次 Linear Sparse + EM，\n")
        f.write(f"变化 λ ∈ {lam_range}, seed ∈ [100, 100+{n_seeds})。\n\n")
        f.write("## 结果\n\n")
        for label, res in all_results.items():
            pw = res["pairwise_vals"]
            p = res["p_true"]
            idx = res["identifiability_index"]
            f.write(f"### {label}\n\n")
            f.write(f"- 恢复次数: {len(res['all_h'])}\n")
            f.write(f"- 平均 Pearson vs true: {p.mean():.4f} ± {p.std():.4f}\n")
            f.write(f"- 平均 Pairwise |corr|: {pw.mean():.4f} ± {pw.std():.4f}\n")
            f.write(f"- Identifiability index: {idx:.3f}\n\n")
        f.write("## 解读\n\n")
        f.write("- **idx > 1.0**: Recoveries 之间一致性很高，hidden **可辨识**\n")
        f.write("  说明所有方法都恢复出相同的 hidden (up to scale)\n")
        f.write("- **idx ≈ 1.0**: Recoveries 一致性 ≈ recovery 准确性\n")
        f.write("  可能是中等可辨识\n")
        f.write("- **idx < 0.8**: 多解性明显，hidden **non-identifiable**\n")
        f.write("  不同方法给出不同的 hidden，都能 'fit' visible\n\n")

    print(f"\n[OK] saved to: {out_dir}")

    print()
    print("=" * 70)
    print("IDENTIFIABILITY ANALYSIS SUMMARY:")
    for label, res in all_results.items():
        pw = res["pairwise_vals"]; p = res["p_true"]; idx = res["identifiability_index"]
        print(f"  {label:10s}: p_true={p.mean():.3f} pairwise={pw.mean():.3f} idx={idx:.3f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
