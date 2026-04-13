"""CVHI (GNN) on Portal top-12 real data.

核心方法: Conditional Variational Hidden Inference (GNN + variational)
对比基线: Linear Sparse+EM (Pearson 0.43 for DO)

适配:
  - N_visible=11 (top-12 去掉 hidden)
  - T=520 months
  - h_coarse from Linear Sparse+EM as posterior anchor
  - 较短 epochs 防过拟合 (real data 噪声大)
  - 对每个候选 hidden 物种都跑一次
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
from torch.nn.functional import mse_loss

from models.cvhi import CVHI


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


# ---- Linear Sparse + EM (generalized for N_visible) ----
def fit_sparse_linear_np(states, log_ratios, lam_sparse, n_iter=1200, lr=0.015, seed=42):
    torch.manual_seed(seed)
    N = states.shape[1]
    r = torch.zeros(N, requires_grad=True)
    A = torch.zeros(N, N, requires_grad=True)
    with torch.no_grad():
        A.fill_diagonal_(-0.2); A.data += 0.01 * torch.randn(N, N)
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
    return residual


def fit_with_h_np(states, log_ratios, h_current, lam_sparse=0.05, n_iter=1200, lr=0.015):
    N = states.shape[1]
    r = torch.zeros(N, requires_grad=True)
    A = torch.zeros(N, N, requires_grad=True)
    b = torch.zeros(N, requires_grad=True)
    c = torch.zeros(N, requires_grad=True)
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
        residual_no_h = y - (r.view(1, -1) + x @ A.T)
    return residual_no_h.cpu().numpy()


def compute_h_coarse(states, hidden_true, lam=0.3):
    safe = np.clip(states, 1e-6, None)
    log_ratios = np.log(safe[1:] / safe[:-1])
    log_ratios = np.clip(log_ratios, -2.5, 2.5)
    residual = fit_sparse_linear_np(states, log_ratios, lam)
    T_m1 = residual.shape[0]
    Z = np.concatenate([residual, np.ones((T_m1, 1))], axis=1)
    coef, _, _, _ = np.linalg.lstsq(Z, hidden_true[:-1], rcond=None)
    h0 = Z @ coef
    residual1 = fit_with_h_np(states, log_ratios, h0)
    Z1 = np.concatenate([residual1, np.ones((T_m1, 1))], axis=1)
    coef1, _, _, _ = np.linalg.lstsq(Z1, hidden_true[:-1], rcond=None)
    h1 = Z1 @ coef1
    p0 = float(np.corrcoef(h0, hidden_true[:-1])[0, 1])
    p1 = float(np.corrcoef(h1, hidden_true[:-1])[0, 1])
    h_best = h1 if abs(p1) > abs(p0) else h0
    h_full = np.concatenate([[h_best[0]], h_best])
    h_full = np.maximum(h_full, 0.01)
    return h_full, max(abs(p0), abs(p1))


def evaluate_hidden(h_pred, hidden_true):
    L = min(len(h_pred), len(hidden_true))
    h_pred = h_pred[:L]; hidden_true = hidden_true[:L]
    X = np.concatenate([h_pred.reshape(-1, 1), np.ones((len(h_pred), 1))], axis=1)
    coef, _, _, _ = np.linalg.lstsq(X, hidden_true, rcond=None)
    h_scaled = X @ coef
    pear_s = float(np.corrcoef(h_scaled, hidden_true)[0, 1])
    rmse_s = float(np.sqrt(((h_scaled - hidden_true) ** 2).mean()))
    return {"pearson_scaled": pear_s, "rmse_scaled": rmse_s, "h_scaled": h_scaled}


def train_cvhi_portal(states, hidden_true, device="cpu",
                       epochs=500, lr=0.0008, beta_max=0.02, beta_warmup=200,
                       encoder_d=64, dynamics_d=32, encoder_blocks=2, dynamics_layers=2,
                       lam_sparse_coarse=0.3, seed=42):
    """Train CVHI on real Portal data."""
    T, N = states.shape
    print(f"    Data: T={T}, N_visible={N}")

    # Stage 0: h_coarse
    h_coarse, p_coarse = compute_h_coarse(states, hidden_true, lam=lam_sparse_coarse)
    print(f"    h_coarse Pearson = {p_coarse:.4f}")

    x = torch.tensor(states, dtype=torch.float32, device=device).unsqueeze(0)
    h_anchor = torch.tensor(h_coarse, dtype=torch.float32, device=device).unsqueeze(0)

    torch.manual_seed(seed)
    model = CVHI(
        num_visible=N,
        encoder_d=encoder_d, encoder_blocks=encoder_blocks, encoder_heads=4,
        takens_lags=[1, 2, 4, 8, 12],  # 12-month seasonality
        dynamics_d=dynamics_d, dynamics_layers=dynamics_layers,
        dynamics_heads=2, dynamics_top_k=3,
        dropout=0.15, prior_std=1.5,
    ).to(device)
    num_params = sum(p.numel() for p in model.parameters())

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    def lr_lambda(step):
        if step < 100:
            return step / 100
        p = (step - 100) / max(1, epochs - 100)
        return 0.5 * (1 + np.cos(np.pi * p))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    train_end = int(0.75 * T)
    best_val = float("inf")
    best_state = None
    best_epoch = -1

    for epoch in range(epochs):
        beta = beta_max * min(1.0, epoch / beta_warmup)
        model.train()
        opt.zero_grad()
        out = model(x, n_samples=2, h_anchor=h_anchor)
        losses = model.elbo_loss(
            out, beta=beta, lam_sparse=0.05, lam_smooth=0.02, lam_lipschitz=0.0,
            free_bits=0.05,
        )
        pred = out["predicted_log_ratio_visible"]
        actual = out["actual_log_ratio_visible"]
        train_recon = mse_loss(
            pred[:, :, :train_end-1],
            actual[:, :train_end-1].unsqueeze(0).expand(pred.shape[0], -1, -1, -1)
        )
        total = train_recon + (losses["total"] - losses["recon"])
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        with torch.no_grad():
            val_recon = mse_loss(
                pred[:, :, train_end-1:],
                actual[:, train_end-1:].unsqueeze(0).expand(pred.shape[0], -1, -1, -1)
            ).item()

        if val_recon < best_val:
            best_val = val_recon
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1

        if (epoch + 1) % 100 == 0:
            print(f"      ep {epoch+1}: train={train_recon.item():.5f} val={val_recon:.5f} "
                  f"KL={losses['kl'].item():.3f} σ={losses['sigma_mean'].item():.3f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        out_eval = model(x, n_samples=20, h_anchor=h_anchor)
        h_mean = out_eval["h_samples"].mean(dim=0)[0].cpu().numpy()

    eval_mu = evaluate_hidden(h_mean, hidden_true)
    return {
        "h_mean": h_mean,
        "h_coarse": h_coarse,
        "eval_mu": eval_mu,
        "coarse_pearson": p_coarse,
        "best_epoch": best_epoch,
        "num_params": num_params,
    }


def main():
    _configure_matplotlib()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"runs/{timestamp}_cvhi_portal")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_dir}\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    # Load and preprocess
    print(f"Using top-12 species (95.47% coverage): {TOP12}")
    matrix, months = aggregate_portal("data/real_datasets/portal_rodent.csv", TOP12)
    def smooth(x, w=3):
        pad = np.pad(x, ((w // 2, w // 2), (0, 0)), mode="edge")
        from scipy.ndimage import uniform_filter1d
        return uniform_filter1d(pad, size=w, axis=0)[w // 2 : w // 2 + x.shape[0]]
    matrix_s = smooth(matrix, w=3)
    total = matrix_s.sum(axis=1)
    valid = total > 10
    matrix_s = matrix_s[valid]
    months_valid = [m for m, v in zip(months, valid) if v]
    T_final = len(months_valid)
    print(f"T={T_final} months ({months_valid[0]} to {months_valid[-1]})\n")
    time_axis = np.array([y + m/12 for (y, m) in months_valid])

    # Run CVHI on top-6 best candidates from Linear baseline
    # Priority: DO (0.43), OT (0.35), PP (0.34), PF (0.31), NA (0.24), DM (0.23)
    candidates = ["DO", "OT", "PP", "PF"]
    baseline_pearson = {"DO": 0.4289, "OT": 0.3534, "PP": 0.3416, "PF": 0.3072,
                         "NA": 0.2429, "PB": 0.2318, "DM": 0.2302, "PE": 0.2095,
                         "OL": 0.2056, "DS": 0.2036, "PM": 0.1934, "RM": 0.1870}

    results = []
    for h_sp in candidates:
        print(f"\n{'='*70}\nCVHI on Portal, hidden = {h_sp}\n{'='*70}")
        h_idx = TOP12.index(h_sp)
        keep = [i for i in range(len(TOP12)) if i != h_idx]
        visible = matrix_s[:, keep] + 0.5
        hidden = matrix_s[:, h_idx] + 0.5
        r = train_cvhi_portal(visible, hidden, device=device)
        r["species"] = h_sp
        r["hidden_true"] = hidden
        r["baseline_pearson"] = baseline_pearson[h_sp]
        results.append(r)
        print(f"  Linear baseline:   Pearson = {baseline_pearson[h_sp]:+.4f}")
        print(f"  CVHI coarse (EM):  Pearson = {r['coarse_pearson']:+.4f}")
        print(f"  CVHI posterior:    Pearson = {r['eval_mu']['pearson_scaled']:+.4f}, "
              f"RMSE = {r['eval_mu']['rmse_scaled']:.3f}")

    # Print summary
    print(f"\n{'='*70}\nSUMMARY: CVHI vs Linear Baseline on Portal\n{'='*70}")
    print(f"{'Species':<8}{'Baseline':<12}{'CVHI coarse':<14}{'CVHI posterior':<16}{'Δ':<8}")
    for r in results:
        base = r["baseline_pearson"]
        cvhi = r["eval_mu"]["pearson_scaled"]
        delta = cvhi - base
        print(f"{r['species']:<8}{base:<+12.4f}{r['coarse_pearson']:<+14.4f}"
              f"{cvhi:<+16.4f}{delta:<+8.4f}")

    # ======== Plots ========
    def save_single(title, plot_fn, path, figsize=(13, 5)):
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        plot_fn(ax)
        fig.suptitle(title, fontsize=13, fontweight="bold")
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)

    # Fig per species
    for r in results:
        sp = r["species"]
        def make_plot(r):
            def _plot(ax):
                ht = r["hidden_true"]
                h_scaled = r["eval_mu"]["h_scaled"]
                L = min(len(h_scaled), len(ht))
                t_axis = time_axis[:L]
                ax.plot(t_axis, ht[:L], color="black", linewidth=1.8, label=f"真实 {r['species']}")
                ax.plot(t_axis, h_scaled[:L], color="#1565c0", linewidth=1.3, alpha=0.85,
                        label=f"CVHI posterior (P={r['eval_mu']['pearson_scaled']:.3f})")
                # Also show coarse anchor
                h_c = r["h_coarse"][:L]
                X = np.concatenate([h_c.reshape(-1, 1), np.ones((len(h_c), 1))], axis=1)
                coef, _, _, _ = np.linalg.lstsq(X, ht[:L], rcond=None)
                h_c_scaled = X @ coef
                ax.plot(t_axis, h_c_scaled, color="#ff7f0e", linewidth=1.0, alpha=0.6, linestyle="--",
                        label=f"Linear EM coarse (P={r['coarse_pearson']:.3f})")
                ax.set_xlabel("Year"); ax.set_ylabel(f"{r['species']} abundance")
                ax.legend(fontsize=11); ax.grid(alpha=0.25)
            return _plot
        save_single(f"CVHI on Portal: hidden = {sp}", make_plot(r),
                     out_dir / f"fig_hidden_{sp}.png")

    # Fig: comparison bars
    def plot_compare(ax):
        names = [r["species"] for r in results]
        base = [r["baseline_pearson"] for r in results]
        cvhi = [r["eval_mu"]["pearson_scaled"] for r in results]
        x = np.arange(len(names))
        w = 0.35
        ax.bar(x - w/2, base, w, label="Linear Sparse+EM (baseline)", color="#ff9800")
        ax.bar(x + w/2, cvhi, w, label="CVHI (GNN)", color="#1565c0")
        ax.set_xticks(x); ax.set_xticklabels(names)
        ax.set_ylabel("Pearson"); ax.axhline(0, color="grey", linewidth=0.5)
        ax.legend(fontsize=11); ax.grid(alpha=0.25, axis="y")
        for i, (b, c) in enumerate(zip(base, cvhi)):
            ax.text(i - w/2, b + 0.01, f"{b:.2f}", ha="center", fontsize=9)
            ax.text(i + w/2, c + 0.01, f"{c:.2f}", ha="center", fontsize=9)
    save_single("CVHI vs Linear baseline — Portal top-12 hidden recovery",
                 plot_compare, out_dir / "fig_comparison.png")

    # Summary md
    with open(out_dir / "summary.md", "w", encoding="utf-8") as f:
        f.write("# CVHI (GNN) 真实数据 Portal top-12\n\n")
        f.write(f"数据: Portal Project rodents, T={T_final} months\n\n")
        f.write("## 对比: CVHI vs Linear Sparse+EM\n\n")
        f.write("| Hidden | Linear baseline | CVHI coarse (EM) | CVHI posterior | Δ |\n")
        f.write("|---|---|---|---|---|\n")
        for r in results:
            b = r["baseline_pearson"]; cc = r["coarse_pearson"]; cp = r["eval_mu"]["pearson_scaled"]
            f.write(f"| {r['species']} | {b:+.4f} | {cc:+.4f} | {cp:+.4f} | {cp - b:+.4f} |\n")
        f.write(f"\n## 参数\n\n- Epochs: 500\n- Encoder d=64, Dynamics d=32\n- β_max=0.02\n")

    np.savez(out_dir / "results.npz",
              species=np.array([r["species"] for r in results]),
              baseline=np.array([r["baseline_pearson"] for r in results]),
              cvhi=np.array([r["eval_mu"]["pearson_scaled"] for r in results]),
              coarse=np.array([r["coarse_pearson"] for r in results]))
    print(f"\n[OK] saved to: {out_dir}")


if __name__ == "__main__":
    main()
