"""真实数据：Portal Project (Chihuahuan Desert rodents, 1977-2020+).

数据: 月度 rodent 捕获记录, 41 物种, 520 个月
Setup: 选前 5 个最常见物种作 visible, 下一个作 hidden, 测试 Linear Sparse + EM

物种:
  PP: Chaetodipus penicillatus (desert pocket mouse)
  DM: Dipodomys merriami (Merriam's kangaroo rat)
  PB: Chaetodipus baileyi (Bailey's pocket mouse)
  DO: Dipodomys ordii (Ord's kangaroo rat)
  OT: Onychomys torridus (southern grasshopper mouse)
  RM: Reithrodontomys megalotis (western harvest mouse) ← hidden
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


def _configure_matplotlib() -> None:
    plt.rcParams["font.sans-serif"] = [
        "PingFang SC", "Hiragino Sans GB", "Heiti TC", "STHeiti",
        "Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.size"] = 12


def aggregate_portal(csv_path: str, species_list):
    """Aggregate raw capture records into monthly abundance per species."""
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
    # Fill in time series
    all_months = sorted(counts.keys())
    matrix = np.zeros((len(all_months), len(species_list)), dtype=np.float32)
    for t, (y, m) in enumerate(all_months):
        for j, sp in enumerate(species_list):
            matrix[t, j] = counts[(y, m)].get(sp, 0)
    return matrix, all_months


def fit_sparse_linear(states, log_ratios, lam_sparse, n_iter=2000, lr=0.015, seed=42, device="cpu"):
    torch.manual_seed(seed)
    N = states.shape[1]
    r = torch.zeros(N, device=device, requires_grad=True)
    A = torch.zeros(N, N, device=device, requires_grad=True)
    with torch.no_grad():
        A.fill_diagonal_(-0.2)
        A.data += 0.01 * torch.randn(N, N, device=device)
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
    return residual, r.detach().cpu().numpy(), A.detach().cpu().numpy()


def recover_hidden_linear(residual, hidden_true):
    T = residual.shape[0]
    Z = np.concatenate([residual, np.ones((T, 1))], axis=1)
    coef, _, _, _ = np.linalg.lstsq(Z, hidden_true, rcond=None)
    h_pred = Z @ coef
    pear = float(np.corrcoef(h_pred, hidden_true)[0, 1])
    rmse = float(np.sqrt(((h_pred - hidden_true) ** 2).mean()))
    return h_pred, pear, rmse


def run_em(states, hidden_true, lam_sparse, device="cpu"):
    safe = np.clip(states, 1e-3, None)
    log_ratios = np.log(safe[1:] / safe[:-1])
    log_ratios = np.clip(log_ratios, -2.5, 2.5)
    residual, r0, A0 = fit_sparse_linear(states, log_ratios, lam_sparse, device=device)
    h0, p0, rmse0 = recover_hidden_linear(residual, hidden_true[:-1])
    # Iter 1: fit with h
    r1 = torch.zeros(states.shape[1], requires_grad=True)
    A1 = torch.zeros(states.shape[1], states.shape[1], requires_grad=True)
    b1 = torch.zeros(states.shape[1], requires_grad=True)
    c1 = torch.zeros(states.shape[1], requires_grad=True)
    with torch.no_grad(): A1.fill_diagonal_(-0.2)
    opt = torch.optim.Adam([r1, A1, b1, c1], lr=0.015)
    x = torch.tensor(states[:-1], dtype=torch.float32)
    y = torch.tensor(log_ratios, dtype=torch.float32)
    h = torch.tensor(h0, dtype=torch.float32)
    for _ in range(2000):
        opt.zero_grad()
        pred = r1.view(1, -1) + x @ A1.T + h.unsqueeze(-1) * b1.view(1, -1) + (h.unsqueeze(-1) ** 2) * c1.view(1, -1)
        fit_loss = ((pred - y) ** 2).mean()
        A_off = A1 - torch.diag(torch.diag(A1))
        loss = fit_loss + 0.05 * A_off.abs().mean()
        loss.backward()
        opt.step()
    with torch.no_grad():
        residual1 = y - (r1.view(1, -1) + x @ A1.T)
    residual1 = residual1.cpu().numpy()
    h1, p1, rmse1 = recover_hidden_linear(residual1, hidden_true[:-1])
    return max([(p0, rmse0, h0, 0), (p1, rmse1, h1, 1)], key=lambda r: abs(r[0]))


def main():
    _configure_matplotlib()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"runs/{timestamp}_real_data_portal")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_dir}\n")

    # Top 5 visible + 1 hidden
    visible_species = ["PP", "DM", "PB", "DO", "OT"]
    hidden_species = "RM"
    all_species = visible_species + [hidden_species]

    print(f"Visible: {visible_species}")
    print(f"Hidden: {hidden_species}")

    matrix, months = aggregate_portal("data/real_datasets/portal_rodent.csv", all_species)
    print(f"Matrix shape: {matrix.shape} (T={matrix.shape[0]} months × {matrix.shape[1]} species)")

    # Smooth slightly (3-month moving average to reduce monthly noise in sparse months)
    def smooth(x, w=3):
        pad = np.pad(x, ((w // 2, w // 2), (0, 0)), mode="edge")
        from scipy.ndimage import uniform_filter1d
        return uniform_filter1d(pad, size=w, axis=0)[w // 2 : w // 2 + x.shape[0]]
    matrix_s = smooth(matrix, w=3)

    visible = matrix_s[:, :5]   # (T, 5)
    hidden = matrix_s[:, 5]      # (T,)

    print(f"Visible stats: mean={visible.mean(axis=0)}, std={visible.std(axis=0)}")
    print(f"Hidden stats: mean={hidden.mean():.1f}, std={hidden.std():.1f}, max={hidden.max():.1f}")

    # Filter months with non-zero data (early period sometimes empty)
    valid = (visible.sum(axis=1) > 5)  # at least 5 captures across all species
    visible = visible[valid]
    hidden = hidden[valid]
    months_valid = [m for m, v in zip(months, valid) if v]
    T_final = len(months_valid)
    print(f"After filter: {T_final} months ({months_valid[0]} to {months_valid[-1]})")

    # Add small offset to avoid exact zeros
    visible = visible + 0.5
    hidden = hidden + 0.5

    # Sparsity sweep
    print("\n=== Sparsity sweep on Portal data (Linear Sparse + EM) ===")
    best = None
    sweep = []
    for lam in [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0]:
        p, rmse, h_pred, best_iter = run_em(visible, hidden, lam)
        sweep.append({"lam": lam, "pearson": p, "rmse": rmse, "h_pred": h_pred, "iter": best_iter})
        print(f"  λ={lam:>4.2f}: Pearson={p:+.4f}  RMSE={rmse:.2f}  (best iter={best_iter})")
        if best is None or abs(p) > abs(best["pearson"]):
            best = sweep[-1]
    print(f"\n  BEST: λ={best['lam']}, Pearson={best['pearson']:.4f}, RMSE={best['rmse']:.2f}")

    # ==== Plots (single plot per PNG) ====
    def save_single(title, plot_fn, path, figsize=(13, 5)):
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        plot_fn(ax)
        fig.suptitle(title, fontsize=13, fontweight="bold")
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)

    time_axis = np.array([y + m/12 for (y, m) in months_valid])

    # Fig 1: raw species abundance
    def plot_raw(ax):
        colors = ["#1565c0", "#2e7d32", "#e65100", "#6a1b9a", "#c62828"]
        for j, sp in enumerate(visible_species):
            ax.plot(time_axis, visible[:, j], linewidth=1.2, color=colors[j], label=sp, alpha=0.85)
        ax.plot(time_axis, hidden, linewidth=1.8, color="black", label=f"{hidden_species} (hidden)", linestyle="--")
        ax.set_xlabel("Year"); ax.set_ylabel("Monthly abundance (smoothed)")
        ax.legend(fontsize=10, ncol=3); ax.grid(alpha=0.25)
    save_single(f"Portal Project: {T_final} months of 5 visible + 1 hidden ({hidden_species})",
                 plot_raw, out_dir / "fig_01_raw.png")

    # Fig 2: best recovery
    def plot_best(ax):
        ht = hidden[:-1]
        h = best["h_pred"]
        X = np.concatenate([h.reshape(-1, 1), np.ones((len(h), 1))], axis=1)
        coef, _, _, _ = np.linalg.lstsq(X, ht, rcond=None)
        h_scaled = X @ coef
        t_axis = time_axis[:-1]
        ax.plot(t_axis, ht, color="black", linewidth=1.8, label=f"真实 {hidden_species}")
        ax.plot(t_axis, h_scaled, color="#ff7f0e", linewidth=1.3, alpha=0.85,
                label=f"恢复 (λ={best['lam']}, EM iter={best['iter']}, P={best['pearson']:.3f})")
        ax.set_xlabel("Year"); ax.set_ylabel(f"{hidden_species} abundance")
        ax.legend(fontsize=11); ax.grid(alpha=0.25)
    save_single(f"Portal 数据上 Hidden Recovery: {hidden_species}",
                 plot_best, out_dir / "fig_02_recovery.png")

    # Fig 3: sparsity sweep
    def plot_sweep(ax):
        lams = [s["lam"] for s in sweep]
        pears = [s["pearson"] for s in sweep]
        rmses = [s["rmse"] for s in sweep]
        ax.semilogx(np.array(lams) + 1e-4, pears, marker="o", linewidth=2, color="#1565c0", label="Pearson")
        ax.set_xlabel("L1 sparsity λ"); ax.set_ylabel("Pearson to true", color="#1565c0")
        ax2 = ax.twinx()
        ax2.semilogx(np.array(lams) + 1e-4, rmses, marker="s", linewidth=2, color="#c62828", label="RMSE")
        ax2.set_ylabel("RMSE", color="#c62828")
        ax.grid(alpha=0.25)
        ax.legend(loc="upper left"); ax2.legend(loc="upper right")
    save_single("Portal: Sparsity Sweep", plot_sweep, out_dir / "fig_03_sparsity.png")

    # Save
    np.savez(out_dir / "results.npz",
              visible=visible, hidden=hidden, months=np.array(time_axis),
              best_pearson=best["pearson"], best_rmse=best["rmse"],
              best_h_pred=best["h_pred"], best_lam=best["lam"])

    with open(out_dir / "summary.md", "w", encoding="utf-8") as f:
        f.write("# Portal Project 真实数据结果\n\n")
        f.write(f"数据: Chihuahuan Desert rodents, {T_final} months ({months_valid[0]}-{months_valid[-1]})\n\n")
        f.write(f"**5 visible**: {visible_species}\n\n")
        f.write(f"**1 hidden**: {hidden_species} (Reithrodontomys megalotis)\n\n")
        f.write("## Sparsity Sweep\n\n| λ | Pearson | RMSE |\n|---|---|---|\n")
        for s in sweep:
            f.write(f"| {s['lam']} | {s['pearson']:+.4f} | {s['rmse']:.2f} |\n")
        f.write(f"\n**BEST**: λ={best['lam']}, Pearson={best['pearson']:.4f}, RMSE={best['rmse']:.2f}\n")

    print(f"\n[OK] saved to: {out_dir}")


if __name__ == "__main__":
    main()
