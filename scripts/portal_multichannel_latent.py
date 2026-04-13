"""Portal 多通道潜变量实验：把 hidden 从 R^1 扩展到 R^k.

研究问题:
  单一 hidden 无法同时捕捉 (1) 物种耦合 (2) 慢变环境驱动 (3) 季节循环。
  如果用 k=3 通道，是否自然分化？是否能让 target species 的 Pearson 显著提升？

Stage 1 (本脚本): 同质 L1 sparse，观察自然分化
  - 模型: log(x_{t+1}/x_t) = r + A·x + B·H, H ∈ R^k, B ∈ R^{N×k}
  - 对每个通道: 时间序列 + 频谱 + 自相关
  - 对 target 物种: 最佳通道的 Pearson

评估指标:
  - 各通道对 OT 的 Pearson (哪个通道最匹配)
  - 各通道 ACF(1) (自相关强度 = 慢变性)
  - 各通道 12-月周期成分能量
  - 多通道组合 (linear reg) 对 OT 的 Pearson
"""
from __future__ import annotations

import csv
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

TOP12 = ["PP", "DM", "PB", "DO", "OT", "RM", "PE", "DS", "PF", "NA", "OL", "PM"]


def _configure_matplotlib() -> None:
    plt.rcParams["font.sans-serif"] = [
        "PingFang SC", "Hiragino Sans GB", "Heiti TC", "STHeiti",
        "Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.size"] = 11


def aggregate_portal(csv_path: str, species_list):
    counts = defaultdict(lambda: defaultdict(int))
    with open(csv_path) as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                year = int(row['year']); month = int(row['month'])
            except (ValueError, KeyError):
                continue
            sp = row['species']
            if sp in species_list:
                counts[(year, month)][sp] += 1
    all_months = sorted(counts.keys())
    matrix = np.zeros((len(all_months), len(species_list)), dtype=np.float32)
    for t, (y, m) in enumerate(all_months):
        for j, sp in enumerate(species_list):
            matrix[t, j] = counts[(y, m)].get(sp, 0)
    return matrix, all_months


def fit_multichannel(states, log_ratios, k_latent, lam_A=0.3, lam_H=0.01, lam_B=0.01,
                     n_iter=3000, lr=0.01, seed=42):
    """Fit: y = r + A·x + B·H with learned H ∈ R^{T-1 × k}.

    - A: (N, N), sparse off-diagonal
    - B: (N, k), sparse (which visible species are driven by which latent)
    - H: (T-1, k), L1 sparse (different seeds → different structures)
    """
    torch.manual_seed(seed)
    N = states.shape[1]
    T_m1 = states.shape[0] - 1
    r = torch.zeros(N, requires_grad=True)
    A = torch.zeros(N, N, requires_grad=True)
    with torch.no_grad():
        A.fill_diagonal_(-0.2); A.data += 0.01 * torch.randn(N, N)
    B = torch.randn(N, k_latent, requires_grad=True) * 0.1
    H = torch.randn(T_m1, k_latent, requires_grad=True) * 0.1
    B.requires_grad_(True)
    H.requires_grad_(True)
    opt = torch.optim.Adam([r, A, B, H], lr=lr)
    x = torch.tensor(states[:-1], dtype=torch.float32)
    y = torch.tensor(log_ratios, dtype=torch.float32)
    for _ in range(n_iter):
        opt.zero_grad()
        pred = r.view(1, -1) + x @ A.T + H @ B.T
        fit_loss = ((pred - y) ** 2).mean()
        A_off = A - torch.diag(torch.diag(A))
        loss = fit_loss + lam_A * A_off.abs().mean() \
                        + lam_B * B.abs().mean() + lam_H * H.abs().mean()
        loss.backward()
        opt.step()
    with torch.no_grad():
        H_fit = H.cpu().numpy()
        B_fit = B.cpu().numpy()
        A_fit = A.cpu().numpy()
    return H_fit, B_fit, A_fit


def rotate_channels_by_variance(H):
    """Use SVD on H to orthogonalize + sort by variance (like PCA)."""
    U, S, Vt = np.linalg.svd(H - H.mean(axis=0, keepdims=True), full_matrices=False)
    H_rot = U * S[None, :]  # (T, k)
    return H_rot, S


def channel_stats(h_channel):
    """Compute ecological interpretability features of a channel."""
    h = h_channel - h_channel.mean()
    # ACF(1) — slow-variable indicator
    if len(h) < 2 or h.std() < 1e-8:
        acf1 = 0.0
    else:
        acf1 = float(np.corrcoef(h[:-1], h[1:])[0, 1])
    # Fourier 12-month energy (seasonality)
    freq = np.fft.rfftfreq(len(h), d=1.0)  # cycles per month
    power = np.abs(np.fft.rfft(h)) ** 2
    # 12-month = freq 1/12
    # window: [1/13, 1/11]
    mask_12 = (freq >= 1/13) & (freq <= 1/11)
    seas_energy = float(power[mask_12].sum() / (power.sum() + 1e-12))
    # "Fast" energy (periods < 4 months)
    mask_fast = freq > 1/4
    fast_energy = float(power[mask_fast].sum() / (power.sum() + 1e-12))
    # "Slow" energy (periods > 24 months)
    mask_slow = (freq < 1/24) & (freq > 0)
    slow_energy = float(power[mask_slow].sum() / (power.sum() + 1e-12))
    return {
        "acf1": acf1,
        "seasonal_12mo_energy": seas_energy,
        "fast_energy_le4mo": fast_energy,
        "slow_energy_ge24mo": slow_energy,
        "std": float(h.std()),
    }


def best_channel_pearson(H, target):
    """For each channel, compute Pearson with target (scale-invariant)."""
    pears = []
    for k in range(H.shape[1]):
        h = H[:, k]
        L = min(len(h), len(target))
        X = np.concatenate([h[:L].reshape(-1, 1), np.ones((L, 1))], axis=1)
        coef, _, _, _ = np.linalg.lstsq(X, target[:L], rcond=None)
        h_s = X @ coef
        pears.append(float(np.corrcoef(h_s, target[:L])[0, 1]))
    return pears


def combined_pearson(H, target):
    """Multi-channel combined: fit full H to target via linear regression."""
    L = min(H.shape[0], len(target))
    X = np.concatenate([H[:L], np.ones((L, 1))], axis=1)
    coef, _, _, _ = np.linalg.lstsq(X, target[:L], rcond=None)
    h_combined = X @ coef
    return float(np.corrcoef(h_combined, target[:L])[0, 1]), h_combined


def main():
    _configure_matplotlib()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"runs/{timestamp}_portal_multichannel_latent")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_dir}\n")

    # Data
    matrix, months = aggregate_portal("data/real_datasets/portal_rodent.csv", TOP12)
    from scipy.ndimage import uniform_filter1d
    def smooth(x, w=3):
        pad = np.pad(x, ((w//2, w//2), (0, 0)), mode="edge")
        return uniform_filter1d(pad, size=w, axis=0)[w//2:w//2+x.shape[0]]
    matrix_s = smooth(matrix, w=3)
    valid = matrix_s.sum(axis=1) > 10
    matrix_s = matrix_s[valid]
    months_valid = [m for m, v in zip(months, valid) if v]
    time_axis = np.array([y + m/12 for (y, m) in months_valid])
    T = len(months_valid)
    print(f"T={T} months\n")

    # Hidden = OT (best candidate)
    h_idx = TOP12.index("OT")
    keep = [i for i in range(len(TOP12)) if i != h_idx]
    visible = matrix_s[:, keep] + 0.5
    hidden = matrix_s[:, h_idx] + 0.5

    safe = np.clip(visible, 1e-6, None)
    log_ratios = np.log(safe[1:] / safe[:-1])
    log_ratios = np.clip(log_ratios, -2.5, 2.5)

    # Also need monthly seasonal (12-mo sine/cos) and temperature-like proxy for later
    # For now, construct synthetic "season" = 12-mo sine of month
    month_num = np.array([m for (y, m) in months_valid])
    seasonal_proxy = np.sin(2 * np.pi * month_num / 12)

    print("=" * 70)
    print("Stage 1: k=3 latent, homogeneous L1 sparsity (natural differentiation)")
    print("=" * 70)

    # Run for multiple seeds to check stability
    all_results = []
    for seed in [42, 123, 456]:
        print(f"\n--- seed {seed} ---")
        H_raw, B, A = fit_multichannel(visible, log_ratios, k_latent=3,
                                       lam_A=0.3, lam_H=0.02, lam_B=0.02, seed=seed)
        # Orthogonalize via SVD (stable interpretation)
        H, S = rotate_channels_by_variance(H_raw)
        print(f"  Channel singular values: {S}")

        # Per-channel analysis
        chs = []
        for k in range(3):
            st = channel_stats(H[:, k])
            st["channel"] = k
            chs.append(st)
            print(f"  Channel {k}: std={st['std']:.3f}  ACF(1)={st['acf1']:+.3f}  "
                  f"seasonal_12mo={st['seasonal_12mo_energy']:.3f}  "
                  f"fast={st['fast_energy_le4mo']:.3f}  slow={st['slow_energy_ge24mo']:.3f}")

        # Pearson to OT
        pears_ot = best_channel_pearson(H, hidden[:-1])
        print(f"  Pearson to OT per channel: "
              f"ch0={pears_ot[0]:+.3f}  ch1={pears_ot[1]:+.3f}  ch2={pears_ot[2]:+.3f}")

        # Combined (all 3 channels)
        p_comb, h_comb = combined_pearson(H, hidden[:-1])
        print(f"  Combined (k=3): Pearson = {p_comb:+.4f}")

        # Pearson to seasonal proxy (external validation)
        pears_seas = best_channel_pearson(H, seasonal_proxy[:-1])
        print(f"  Pearson to seasonal_12mo proxy per channel: "
              f"ch0={pears_seas[0]:+.3f}  ch1={pears_seas[1]:+.3f}  ch2={pears_seas[2]:+.3f}")

        all_results.append({
            "seed": seed, "H": H, "B": B, "S": S, "chs": chs,
            "pears_ot": pears_ot, "pears_seas": pears_seas,
            "p_comb": p_comb, "h_comb": h_comb,
        })

    # Pick best seed by combined pearson
    best = max(all_results, key=lambda r: abs(r["p_comb"]))
    print(f"\nBest seed by combined Pearson: seed={best['seed']}, P={best['p_comb']:+.4f}")

    # Compare with 1-channel baseline (rerun with k=1 for comparison)
    print("\n--- Baseline k=1 (for comparison) ---")
    H1, _, _ = fit_multichannel(visible, log_ratios, k_latent=1, lam_A=0.3, lam_H=0.02,
                                 lam_B=0.02, seed=42)
    p1, _ = combined_pearson(H1, hidden[:-1])
    print(f"  k=1 Pearson = {p1:+.4f}")
    p2_list = [r["p_comb"] for r in all_results]
    print(f"  k=3 combined Pearson per seed = {[f'{p:+.4f}' for p in p2_list]}")
    print(f"  Mean k=3 = {np.mean(p2_list):+.4f}")

    # ======= Plots =======
    def save_single(title, plot_fn, path, figsize=(13, 5)):
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        plot_fn(ax)
        fig.suptitle(title, fontsize=13, fontweight="bold")
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)

    # Fig 1: best combined H vs OT
    def plot_best_combined(ax):
        t_axis = time_axis[:len(best["h_comb"])]
        # scale-invariant OT alignment
        ht = hidden[:len(t_axis)]
        ax.plot(t_axis, ht, color="black", linewidth=1.8, label="真实 OT")
        ax.plot(t_axis, best["h_comb"], color="#1565c0", linewidth=1.3, alpha=0.85,
                label=f"k=3 multi-channel combined (P={best['p_comb']:.3f})")
        ax.set_xlabel("Year"); ax.set_ylabel("OT abundance")
        ax.legend(fontsize=11); ax.grid(alpha=0.25)
    save_single(f"OT recovery: k=3 multi-channel combined (seed {best['seed']})",
                 plot_best_combined, out_dir / "fig_01_combined_vs_OT.png")

    # Fig 2: three channels separately
    def plot_channels(ax):
        H = best["H"]
        t_axis = time_axis[:H.shape[0]]
        colors = ["#1565c0", "#2e7d32", "#e65100"]
        labels = ["Channel 0", "Channel 1", "Channel 2"]
        for k in range(3):
            h = H[:, k]
            ax.plot(t_axis, (h - h.mean()) / (h.std() + 1e-8), color=colors[k],
                    linewidth=1.2, alpha=0.85, label=f"{labels[k]} (z-score)")
        # Also show OT z-scored for comparison
        ot_z = (hidden[:len(t_axis)] - hidden.mean()) / hidden.std()
        ax.plot(t_axis, ot_z, color="black", linewidth=1.6, alpha=0.7,
                label="真实 OT (z-score)", linestyle="--")
        ax.set_xlabel("Year"); ax.set_ylabel("z-score")
        ax.legend(fontsize=10, ncol=2); ax.grid(alpha=0.25)
    save_single(f"三通道时间序列 (seed {best['seed']}) z-score 对比",
                 plot_channels, out_dir / "fig_02_channels.png", figsize=(14, 5))

    # Fig 3: channel spectra
    def plot_spectra(ax):
        H = best["H"]
        freq = np.fft.rfftfreq(H.shape[0], d=1.0)
        colors = ["#1565c0", "#2e7d32", "#e65100"]
        for k in range(3):
            h = H[:, k] - H[:, k].mean()
            power = np.abs(np.fft.rfft(h)) ** 2
            ax.loglog(freq[1:], power[1:], color=colors[k], linewidth=1.2, alpha=0.85,
                      label=f"Channel {k}")
        # Mark 12-month frequency
        ax.axvline(1/12, color="red", linestyle="--", alpha=0.5, label="12-mo period")
        ax.set_xlabel("Frequency (cycles/month)")
        ax.set_ylabel("Power")
        ax.legend(fontsize=11); ax.grid(alpha=0.25)
    save_single(f"三通道功率谱 (seed {best['seed']})",
                 plot_spectra, out_dir / "fig_03_spectra.png", figsize=(10, 6))

    # Fig 4: per-channel pearson bars across seeds
    def plot_pearson_bars(ax):
        x = np.arange(3)
        w = 0.25
        colors = ["#1565c0", "#2e7d32", "#e65100"]
        for i, r in enumerate(all_results):
            ax.bar(x + (i - 1) * w, [abs(p) for p in r["pears_ot"]], w,
                   color=colors[i], label=f"seed {r['seed']}")
        ax.set_xticks(x); ax.set_xticklabels(["ch 0", "ch 1", "ch 2"])
        ax.set_ylabel("|Pearson| to OT")
        ax.legend(fontsize=10); ax.grid(alpha=0.25, axis="y")
    save_single("每通道对 OT 的 |Pearson| (跨 seed)",
                 plot_pearson_bars, out_dir / "fig_04_pearson_per_channel.png", figsize=(10, 5))

    # Fig 5: per-channel ecological features (stacked comparison)
    def plot_features(ax):
        H = best["H"]
        features = [channel_stats(H[:, k]) for k in range(3)]
        x = np.arange(3)
        feats_to_show = ["acf1", "seasonal_12mo_energy", "fast_energy_le4mo", "slow_energy_ge24mo"]
        colors = ["#1565c0", "#2e7d32", "#e65100", "#c62828"]
        w = 0.2
        for i, key in enumerate(feats_to_show):
            vals = [abs(f[key]) for f in features]
            ax.bar(x + (i - 1.5) * w, vals, w, color=colors[i], label=key)
        ax.set_xticks(x); ax.set_xticklabels(["ch 0", "ch 1", "ch 2"])
        ax.set_ylabel("feature value")
        ax.legend(fontsize=10); ax.grid(alpha=0.25, axis="y")
    save_single(f"通道生态特征指纹 (seed {best['seed']})",
                 plot_features, out_dir / "fig_05_channel_signatures.png", figsize=(10, 5))

    # Summary
    with open(out_dir / "summary.md", "w", encoding="utf-8") as f:
        f.write("# Portal Multi-channel Latent (k=3) 实验\n\n")
        f.write(f"Target hidden: OT (Onychomys torridus)\n\n")
        f.write(f"## 结果概览\n\n")
        f.write(f"- k=1 baseline Pearson = {p1:+.4f}\n")
        f.write(f"- k=3 combined Pearson 跨 3 seeds: {[f'{r['p_comb']:+.4f}' for r in all_results]}\n")
        f.write(f"- k=3 mean combined = {np.mean(p2_list):+.4f}, best = {best['p_comb']:+.4f}\n\n")
        f.write(f"## Best seed ({best['seed']}) 通道分析\n\n")
        f.write(f"| Channel | std | ACF(1) | 12-mo seasonal | fast(<4mo) | slow(>24mo) | |P→OT| | |P→season| |\n")
        f.write(f"|---|---|---|---|---|---|---|---|\n")
        for k, st in enumerate(best["chs"]):
            f.write(f"| {k} | {st['std']:.3f} | {st['acf1']:+.3f} | "
                    f"{st['seasonal_12mo_energy']:.3f} | {st['fast_energy_le4mo']:.3f} | "
                    f"{st['slow_energy_ge24mo']:.3f} | "
                    f"{abs(best['pears_ot'][k]):.3f} | {abs(best['pears_seas'][k]):.3f} |\n")

    # Save
    np.savez(out_dir / "results.npz",
              H=best["H"], B=best["B"], S=best["S"],
              h_comb=best["h_comb"], p_comb=best["p_comb"],
              pears_ot=np.array(best["pears_ot"]),
              pears_seas=np.array(best["pears_seas"]),
              time_axis=time_axis, hidden_OT=hidden,
              p1_baseline=p1)
    print(f"\n[OK] saved to: {out_dir}")


if __name__ == "__main__":
    main()
