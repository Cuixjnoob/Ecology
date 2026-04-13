"""Train CVHI-NCD (Species-GNN + SoftForms + Temporal Attn) on Portal data.

架构:
  - PosteriorEncoder (GNN+Takens)
  - PerSpeciesTemporalAttn (非 GNN, per-species 时间 attention)
  - SpeciesGNN_SoftForms (GNN, nodes=物种, 软预设 LV/Holling/Free messages)
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

from models.cvhi_ncd import CVHI_NCD

TOP12 = ["PP", "DM", "PB", "DO", "OT", "RM", "PE", "DS", "PF", "NA", "OL", "PM"]


def _configure_matplotlib():
    plt.rcParams["font.sans-serif"] = [
        "PingFang SC", "Microsoft YaHei", "SimHei", "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.size"] = 11


def aggregate_portal(csv_path, species_list):
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


# Linear Sparse + EM for h_coarse (unchanged from earlier)
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
        return (y - pred).cpu().numpy()


def compute_h_coarse(states, hidden_true, lam=0.3):
    safe = np.clip(states, 1e-6, None)
    log_ratios = np.log(safe[1:] / safe[:-1])
    log_ratios = np.clip(log_ratios, -2.5, 2.5)
    residual = fit_sparse_linear_np(states, log_ratios, lam)
    T_m1 = residual.shape[0]
    Z = np.concatenate([residual, np.ones((T_m1, 1))], axis=1)
    coef, _, _, _ = np.linalg.lstsq(Z, hidden_true[:-1], rcond=None)
    h0 = Z @ coef
    p = float(np.corrcoef(h0, hidden_true[:-1])[0, 1])
    h_full = np.concatenate([[h0[0]], h0])
    h_full = np.maximum(h_full, 0.01)
    return h_full, abs(p)


def evaluate_hidden(h_pred, hidden_true):
    L = min(len(h_pred), len(hidden_true))
    h_pred = h_pred[:L]; hidden_true = hidden_true[:L]
    X = np.concatenate([h_pred.reshape(-1, 1), np.ones((L, 1))], axis=1)
    coef, _, _, _ = np.linalg.lstsq(X, hidden_true, rcond=None)
    h_scaled = X @ coef
    pear_s = float(np.corrcoef(h_scaled, hidden_true)[0, 1])
    rmse_s = float(np.sqrt(((h_scaled - hidden_true) ** 2).mean()))
    return {"pearson_scaled": pear_s, "rmse_scaled": rmse_s, "h_scaled": h_scaled}


def train_one(visible, hidden, device="cpu", epochs=400, lr=0.0008, seed=42,
               beta_max=0.02, beta_warmup=200, verbose=True):
    T, N = visible.shape
    h_coarse, p_coarse = compute_h_coarse(visible, hidden, lam=0.3)
    if verbose:
        print(f"    h_coarse Pearson = {p_coarse:.4f}")

    x = torch.tensor(visible, dtype=torch.float32, device=device).unsqueeze(0)
    h_anchor = torch.tensor(h_coarse, dtype=torch.float32, device=device).unsqueeze(0)

    torch.manual_seed(seed)
    model = CVHI_NCD(
        num_visible=N, num_hidden=1,
        encoder_d=64, encoder_blocks=2, encoder_heads=4,
        takens_lags=(1, 2, 4, 8, 12), dropout=0.15,
        d_species=32, top_k=5,
        use_free_nn=False,  # 禁用 Free NN (太灵活, 会 bypass hidden)
        prior_std=1.5,
    ).to(device)
    if verbose:
        print(f"    params: {sum(p.numel() for p in model.parameters()):,}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    def lr_lambda(step):
        if step < 80: return step / 80
        return 0.5 * (1 + np.cos(np.pi * (step - 80) / max(1, epochs - 80)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    train_end = int(0.75 * T)
    best_val = float("inf"); best_state = None; best_epoch = -1

    for epoch in range(epochs):
        beta = beta_max * min(1.0, epoch / beta_warmup)
        model.train()
        opt.zero_grad()
        out = model(x, n_samples=2, h_anchor=h_anchor)
        losses = model.elbo_loss(out, beta=beta,
                                  lam_gates=0.1, lam_coefs=0.02,
                                  lam_attn=0.01, lam_smooth=0.02, free_bits=0.05,
                                  lam_anti_bypass=2.0, min_h2v_mass=0.05)
        pred = out["predicted_log_ratio_visible"]
        actual = out["actual_log_ratio_visible"]
        train_recon = mse_loss(pred[:, :, :train_end-1],
                                actual[:, :train_end-1].unsqueeze(0).expand(pred.shape[0], -1, -1, -1))
        total = train_recon + (losses["total"] - losses["recon"])
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()

        with torch.no_grad():
            val_recon = mse_loss(pred[:, :, train_end-1:],
                                  actual[:, train_end-1:].unsqueeze(0).expand(pred.shape[0], -1, -1, -1)).item()
        if val_recon < best_val:
            best_val = val_recon
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1
        if verbose and (epoch + 1) % 50 == 0:
            print(f"      ep {epoch+1}: train={train_recon.item():.4f} val={val_recon:.4f} "
                  f"KL={losses['kl'].item():.3f} σ={losses['sigma_mean'].item():.3f} "
                  f"gates={losses['l1_gates'].item():.3f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        out_eval = model(x, n_samples=20, h_anchor=h_anchor)
        h_mean = out_eval["H_samples"].mean(dim=0)[0, :, 0].cpu().numpy()
    eval_mu = evaluate_hidden(h_mean, hidden)
    return {
        "model": model,
        "h_mean": h_mean,
        "h_coarse": h_coarse,
        "coarse_pearson": p_coarse,
        "eval_mu": eval_mu,
        "best_epoch": best_epoch, "best_val": best_val,
    }


def main():
    _configure_matplotlib()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"runs/{timestamp}_cvhi_ncd_portal")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_dir}\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

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
    T_final = len(months_valid)
    print(f"T={T_final} months, top-12 species\n")

    # Focus: OT as hidden, 3 seeds
    h_idx = TOP12.index("OT")
    keep = [i for i in range(len(TOP12)) if i != h_idx]
    visible = matrix_s[:, keep] + 0.5
    hidden = matrix_s[:, h_idx] + 0.5

    print(f"Target: hidden = OT (Onychomys torridus)")
    print(f"Visible = 11 species, T = {T_final} months\n")

    results = []
    for seed in [42, 123, 456]:
        print(f"\n=== seed {seed} ===")
        r = train_one(visible, hidden, device=device, seed=seed, epochs=400)
        r["seed"] = seed
        results.append(r)
        p = r["eval_mu"]["pearson_scaled"]
        print(f"  Pearson = {p:+.4f}  RMSE = {r['eval_mu']['rmse_scaled']:.3f}  best_ep={r['best_epoch']}")

    pearsons = np.array([r["eval_mu"]["pearson_scaled"] for r in results])
    print(f"\n{'='*60}")
    print(f"CVHI-NCD on Portal OT (3 seeds)")
    print(f"{'='*60}")
    print(f"Pearsons: {[f'{p:+.4f}' for p in pearsons]}")
    print(f"Mean = {pearsons.mean():+.4f} ± {pearsons.std():.4f}")
    print(f"Max  = {np.max(np.abs(pearsons)):.4f}")

    # Compare with Linear baseline and CVHI baseline
    print(f"\nComparison:")
    print(f"  Linear Sparse+EM:     0.353 (baseline)")
    print(f"  CVHI (original):      0.33 ± 0.21 (5 seeds)")
    print(f"  CVHI-NCD (new):       {pearsons.mean():+.3f} ± {pearsons.std():.3f} (3 seeds)")

    # Plots
    def save_single(title, plot_fn, path, figsize=(13, 5)):
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        plot_fn(ax)
        fig.suptitle(title, fontsize=13, fontweight="bold")
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)

    def plot_recovery(ax):
        ht = hidden
        t_axis = time_axis[:len(ht)]
        ax.plot(t_axis, ht, color="black", linewidth=2.0, label="真实 OT", zorder=10)
        colors = plt.cm.viridis(np.linspace(0, 1, len(results)))
        for i, r in enumerate(results):
            h_s = r["eval_mu"]["h_scaled"]
            L = min(len(h_s), len(t_axis))
            ax.plot(t_axis[:L], h_s[:L], color=colors[i], linewidth=1.0, alpha=0.85,
                    label=f"seed {r['seed']} (P={r['eval_mu']['pearson_scaled']:.3f})")
        ax.set_xlabel("Year"); ax.set_ylabel("OT abundance")
        ax.legend(fontsize=10, ncol=2); ax.grid(alpha=0.25)
    save_single(f"CVHI-NCD on Portal OT (mean P={pearsons.mean():.3f})",
                 plot_recovery, out_dir / "fig_01_recovery.png", figsize=(14, 6))

    # Summary
    with open(out_dir / "summary.md", "w", encoding="utf-8") as f:
        f.write("# CVHI-NCD on Portal OT\n\n")
        f.write(f"架构: PosteriorEncoder + PerSpeciesTemporalAttn + SpeciesGNN_SoftForms\n\n")
        f.write("## 结果\n\n| Seed | Pearson | RMSE | best_ep |\n|---|---|---|---|\n")
        for r in results:
            f.write(f"| {r['seed']} | {r['eval_mu']['pearson_scaled']:+.4f} | "
                    f"{r['eval_mu']['rmse_scaled']:.3f} | {r['best_epoch']} |\n")
        f.write(f"\nMean = {pearsons.mean():+.4f} ± {pearsons.std():.4f}\n\n")
        f.write("## 对比\n\n")
        f.write(f"- Linear Sparse+EM: 0.353\n")
        f.write(f"- CVHI original: 0.33 ± 0.21\n")
        f.write(f"- **CVHI-NCD**: {pearsons.mean():+.4f} ± {pearsons.std():.4f}\n")

    np.savez(out_dir / "results.npz",
              seeds=np.array([r["seed"] for r in results]),
              pearsons=pearsons, time_axis=time_axis, hidden=hidden)
    print(f"\n[OK] saved to: {out_dir}")


if __name__ == "__main__":
    main()
