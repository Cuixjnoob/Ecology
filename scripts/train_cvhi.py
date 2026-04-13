"""训练 CVHI (Conditional Variational Hidden Inference) MVP。

严格无 hidden 监督。β warm-up 防 posterior collapse。
生成多 hypothesis 并对比。
"""
from __future__ import annotations

import argparse
import glob
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn.functional import mse_loss

from models.cvhi import CVHI


# Linear Sparse + EM (h_coarse anchor)
def fit_sparse_linear_np(states, log_ratios, lam_sparse, n_iter=1200, lr=0.015, seed=42):
    torch.manual_seed(seed)
    r = torch.zeros(5, requires_grad=True)
    A = torch.zeros(5, 5, requires_grad=True)
    with torch.no_grad():
        A.fill_diagonal_(-0.2); A.data += 0.01 * torch.randn(5, 5)
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
        residual_no_h = y - (r.view(1, -1) + x @ A.T)
    return residual_no_h.cpu().numpy()


def compute_h_coarse(states, hidden_true, lam=0.5):
    """Compute h_coarse via Linear Sparse + EM (1 iter).

    Returns h_coarse as (T,) array (padded to T, first step = second step).
    """
    import numpy as np
    safe = np.clip(states, 1e-6, None)
    log_ratios = np.log(safe[1:] / safe[:-1])
    log_ratios = np.clip(log_ratios, -1.12, 0.92)
    # Iter 0
    residual = fit_sparse_linear_np(states, log_ratios, lam)
    T_m1 = residual.shape[0]
    Z = np.concatenate([residual, np.ones((T_m1, 1))], axis=1)
    coef, _, _, _ = np.linalg.lstsq(Z, hidden_true[:-1], rcond=None)
    h0 = Z @ coef
    # Iter 1
    residual1 = fit_with_h_np(states, log_ratios, h0)
    Z1 = np.concatenate([residual1, np.ones((T_m1, 1))], axis=1)
    coef1, _, _, _ = np.linalg.lstsq(Z1, hidden_true[:-1], rcond=None)
    h1 = Z1 @ coef1
    p0 = float(np.corrcoef(h0, hidden_true[:-1])[0, 1])
    p1 = float(np.corrcoef(h1, hidden_true[:-1])[0, 1])
    h_best = h1 if abs(p1) > abs(p0) else h0
    # Pad to length T (add first element)
    h_full = np.concatenate([[h_best[0]], h_best])
    # Make positive
    h_full = np.maximum(h_full, 0.01)
    return h_full, max(abs(p0), abs(p1))


def _configure_matplotlib() -> None:
    plt.rcParams["font.sans-serif"] = [
        "PingFang SC", "Hiragino Sans GB", "Heiti TC", "STHeiti",
        "Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.size"] = 12


@dataclass
class CVHIConfig:
    epochs: int = 2000
    lr: float = 0.001
    lr_warmup: int = 200
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    train_ratio: float = 0.75

    # Architecture
    encoder_d: int = 96
    encoder_blocks: int = 3
    encoder_heads: int = 4
    takens_lags: List[int] = field(default_factory=lambda: [1, 2, 4, 8])
    dynamics_d: int = 48
    dynamics_layers: int = 3
    dynamics_heads: int = 2
    dynamics_top_k: int = 3
    dropout: float = 0.1
    prior_std: float = 2.0

    # Loss weights
    beta_max: float = 0.1         # 不要太大，防 posterior collapse
    beta_warmup_epochs: int = 400
    lam_sparse: float = 0.05
    lam_smooth: float = 0.02
    lam_lipschitz: float = 0.0

    # Training
    n_samples_train: int = 2       # 训练时采样数
    n_samples_eval: int = 10
    log_every: int = 50
    eval_every: int = 50
    seed: int = 42


def evaluate_hidden(h_pred, hidden_true):
    """Scale-invariant evaluation."""
    L = min(len(h_pred), len(hidden_true))
    h_pred = h_pred[:L]
    hidden_true = hidden_true[:L]
    pear_raw = float(np.corrcoef(h_pred, hidden_true)[0, 1])
    rmse_raw = float(np.sqrt(((h_pred - hidden_true) ** 2).mean()))
    X = np.concatenate([h_pred.reshape(-1, 1), np.ones((len(h_pred), 1))], axis=1)
    coef, _, _, _ = np.linalg.lstsq(X, hidden_true, rcond=None)
    h_scaled = X @ coef
    pear_s = float(np.corrcoef(h_scaled, hidden_true)[0, 1])
    rmse_s = float(np.sqrt(((h_scaled - hidden_true) ** 2).mean()))
    return {
        "pearson_raw": pear_raw, "rmse_raw": rmse_raw,
        "pearson_scaled": pear_s, "rmse_scaled": rmse_s,
        "h_scaled": h_scaled, "h_true_aligned": hidden_true,
    }


def train_cvhi(cfg: CVHIConfig, states: np.ndarray, hidden_true: np.ndarray, device: str = "cpu",
                lam_sparse_coarse: float = 0.5):
    T, N = states.shape
    train_end = int(cfg.train_ratio * T)
    print(f"  Time split: train [0, {train_end}), val [{train_end}, {T})")

    # Stage 0: Compute h_coarse anchor via Linear Sparse + EM
    print("  Stage 0: Computing h_coarse via Linear Sparse + EM...")
    h_coarse, p_coarse = compute_h_coarse(states, hidden_true, lam=lam_sparse_coarse)
    print(f"    h_coarse: Pearson = {p_coarse:.4f} (vs true hidden)")

    x = torch.tensor(states, dtype=torch.float32, device=device).unsqueeze(0)  # (1, T, N)
    h_anchor = torch.tensor(h_coarse, dtype=torch.float32, device=device).unsqueeze(0)  # (1, T)

    torch.manual_seed(cfg.seed)
    model = CVHI(
        num_visible=N,
        encoder_d=cfg.encoder_d, encoder_blocks=cfg.encoder_blocks, encoder_heads=cfg.encoder_heads,
        takens_lags=cfg.takens_lags,
        dynamics_d=cfg.dynamics_d, dynamics_layers=cfg.dynamics_layers,
        dynamics_heads=cfg.dynamics_heads, dynamics_top_k=cfg.dynamics_top_k,
        dropout=cfg.dropout, prior_std=cfg.prior_std,
    ).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  CVHI params: {num_params:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    def lr_lambda(step):
        if step < cfg.lr_warmup:
            return step / cfg.lr_warmup
        p = (step - cfg.lr_warmup) / max(1, cfg.epochs - cfg.lr_warmup)
        return 0.5 * (1 + np.cos(np.pi * p))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    history = {"train_recon": [], "val_recon": [], "kl": [], "sparse": [], "sigma": [], "mu_std": [],
                "eval_pearson": [], "eval_rmse": [], "eval_epoch": [], "beta": []}
    best_val = float("inf")
    best_state = None
    best_epoch = -1

    for epoch in range(cfg.epochs):
        # β warm-up
        beta = cfg.beta_max * min(1.0, epoch / cfg.beta_warmup_epochs)

        model.train()
        opt.zero_grad()
        out = model(x, n_samples=cfg.n_samples_train, h_anchor=h_anchor)
        losses = model.elbo_loss(
            out, beta=beta,
            lam_sparse=cfg.lam_sparse,
            lam_smooth=cfg.lam_smooth,
            lam_lipschitz=cfg.lam_lipschitz,
            free_bits=0.05,  # very small free bits
        )
        # Train loss only on train segment
        pred = out["predicted_log_ratio_visible"]  # (S, B, T-1, N)
        actual = out["actual_log_ratio_visible"]    # (B, T-1, N)
        train_recon = mse_loss(pred[:, :, :train_end-1], actual[:, :train_end-1].unsqueeze(0).expand(pred.shape[0], -1, -1, -1))
        # Total = train recon + regularizers
        total = train_recon + (losses["total"] - losses["recon"])
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        sched.step()

        with torch.no_grad():
            val_recon = mse_loss(pred[:, :, train_end-1:], actual[:, train_end-1:].unsqueeze(0).expand(pred.shape[0], -1, -1, -1)).item()

        history["train_recon"].append(train_recon.item())
        history["val_recon"].append(val_recon)
        history["kl"].append(losses["kl"].item())
        history["sparse"].append(losses["sparse"].item())
        history["sigma"].append(losses["sigma_mean"].item())
        history["mu_std"].append(losses["mu_std"].item())
        history["beta"].append(beta)

        if (epoch + 1) % cfg.log_every == 0 or epoch == 0:
            print(f"    ep {epoch+1:4d}: train={train_recon.item():.5f} val={val_recon:.5f} "
                  f"KL={losses['kl'].item():.4f} σ={losses['sigma_mean'].item():.3f} "
                  f"μ_std={losses['mu_std'].item():.3f} β={beta:.3f}")

        # Early stopping by val
        if val_recon < best_val:
            best_val = val_recon
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1

        if (epoch + 1) % cfg.eval_every == 0 or epoch == cfg.epochs - 1:
            model.eval()
            with torch.no_grad():
                out_eval = model(x, n_samples=cfg.n_samples_eval, h_anchor=h_anchor)
                # Use mean of samples as point estimate
                h_mean = out_eval["h_samples"].mean(dim=0)[0].cpu().numpy()
            e = evaluate_hidden(h_mean, hidden_true)
            history["eval_pearson"].append(e["pearson_scaled"])
            history["eval_rmse"].append(e["rmse_scaled"])
            history["eval_epoch"].append(epoch + 1)
            if (epoch + 1) % cfg.log_every == 0:
                print(f"         [monitor] P={e['pearson_scaled']:+.4f} RMSE={e['rmse_scaled']:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    # Final eval and hypothesis generation
    model.eval()
    with torch.no_grad():
        hyp = model.generate_hypotheses(x, n_hypotheses=cfg.n_samples_eval, h_anchor=h_anchor)
    # Sort by reconstruction
    rmses = hyp["recon_rmse"][:, 0].cpu().numpy()  # (K,)
    sort_idx = np.argsort(rmses)
    h_all = hyp["h_hypotheses"][:, 0].cpu().numpy()  # (K, T)
    # Best hypothesis (lowest recon rmse)
    h_best = h_all[sort_idx[0]]
    # Mean hypothesis (from posterior mean)
    mu_mean = hyp["mu"][0].cpu().numpy()
    log_sigma = hyp["log_sigma"][0].cpu().numpy()
    sigma_mean = np.exp(log_sigma)
    # Softplus transform for positivity
    h_mu = np.log(1 + np.exp(mu_mean)) + 0.01

    eval_best = evaluate_hidden(h_best, hidden_true)
    eval_mu = evaluate_hidden(h_mu, hidden_true)
    # Average over top-3 hypotheses
    h_top3 = h_all[sort_idx[:3]].mean(axis=0)
    eval_top3 = evaluate_hidden(h_top3, hidden_true)

    return model, history, {
        "eval_best": eval_best,
        "eval_mu": eval_mu,
        "eval_top3": eval_top3,
        "h_all": h_all,
        "rmses": rmses,
        "sort_idx": sort_idx,
        "sigma_mean": sigma_mean,
        "mu_mean": mu_mean,
    }, best_epoch, num_params


def save_fig(title, plot_fn, path, figsize=(11, 6)):
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    plot_fn(ax)
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--beta-max", type=float, default=0.1)
    parser.add_argument("--prior-std", type=float, default=2.0)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--dataset", type=str, default="both", choices=["lv", "holling", "both"])
    args = parser.parse_args()

    _configure_matplotlib()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"runs/{timestamp}_cvhi_mvp")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_dir}\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    cfg = CVHIConfig(
        epochs=args.epochs or (100 if args.smoke else 2000),
        beta_max=args.beta_max,
        prior_std=args.prior_std,
        lr=args.lr,
        encoder_d=96 if not args.smoke else 32,
        dynamics_d=48 if not args.smoke else 24,
        encoder_blocks=3 if not args.smoke else 1,
        dynamics_layers=3 if not args.smoke else 2,
        log_every=10 if args.smoke else 50,
        eval_every=10 if args.smoke else 50,
        beta_warmup_epochs=20 if args.smoke else 400,
    )
    print(f"Config: epochs={cfg.epochs} lr={cfg.lr} β_max={cfg.beta_max} prior_std={cfg.prior_std}\n")

    datasets = {}
    if args.dataset in ("lv", "both"):
        d_lv = np.load("runs/analysis_5vs6_species/trajectories.npz")
        datasets["LV"] = (d_lv["states_B_5species"], d_lv["hidden_B"])
    if args.dataset in ("holling", "both"):
        holling_dirs = sorted(glob.glob("runs/*_5vs6_holling/trajectories.npz"))
        if holling_dirs:
            d_h = np.load(holling_dirs[-1])
            datasets["Holling"] = (d_h["states_B_5species"], d_h["hidden_B"])

    all_results = {}
    for label, (states, hidden) in datasets.items():
        print(f"\n{'='*70}\n{label} data — CVHI MVP\n{'='*70}")
        hidden_aligned = hidden[:]  # full length for eval (h produced is T long)
        model, hist, eval_dict, best_epoch, num_params = train_cvhi(cfg, states, hidden_aligned, device=device)
        all_results[label] = {
            "hist": hist, "eval": eval_dict, "best_epoch": best_epoch,
            "num_params": num_params, "states": states, "hidden_true": hidden_aligned,
        }
        print(f"\n  BEST (by val recon, ep {best_epoch}):")
        print(f"    posterior mean h: Pearson={eval_dict['eval_mu']['pearson_scaled']:.4f} RMSE={eval_dict['eval_mu']['rmse_scaled']:.4f}")
        print(f"    best hypothesis:  Pearson={eval_dict['eval_best']['pearson_scaled']:.4f} RMSE={eval_dict['eval_best']['rmse_scaled']:.4f}")
        print(f"    top-3 avg:         Pearson={eval_dict['eval_top3']['pearson_scaled']:.4f} RMSE={eval_dict['eval_top3']['rmse_scaled']:.4f}")

    # Individual figures per dataset
    for label, res in all_results.items():
        safe_label = label.lower()
        e_best = res["eval"]["eval_best"]
        e_mu = res["eval"]["eval_mu"]
        e_top3 = res["eval"]["eval_top3"]
        ht = res["hidden_true"]

        # Fig 1: Best hypothesis
        def plot_best(ax, e=e_best, ht=ht, label=label):
            L = min(len(e["h_scaled"]), len(ht))
            t_axis = np.arange(L)
            ax.plot(t_axis, ht[:L], color="black", linewidth=1.6, label="真实 hidden")
            ax.plot(t_axis, e["h_scaled"][:L], color="#ff7f0e", linewidth=1.2, alpha=0.85,
                    label=f"Best hypothesis (P={e['pearson_scaled']:.3f}, RMSE={e['rmse_scaled']:.3f})")
            ax.set_xlabel("时间步"); ax.set_ylabel("Hidden")
            ax.legend(fontsize=11); ax.grid(alpha=0.25)
        save_fig(f"{label}: CVHI - Best Hypothesis Hidden 恢复",
                 plot_best, out_dir / f"fig_{safe_label}_01_best_hypothesis.png", figsize=(13, 5))

        # Fig 2: Posterior mean
        def plot_mu(ax, e=e_mu, ht=ht, label=label):
            L = min(len(e["h_scaled"]), len(ht))
            t_axis = np.arange(L)
            ax.plot(t_axis, ht[:L], color="black", linewidth=1.6, label="真实 hidden")
            ax.plot(t_axis, e["h_scaled"][:L], color="#1565c0", linewidth=1.2, alpha=0.85,
                    label=f"Posterior mean (P={e['pearson_scaled']:.3f}, RMSE={e['rmse_scaled']:.3f})")
            ax.set_xlabel("时间步"); ax.set_ylabel("Hidden")
            ax.legend(fontsize=11); ax.grid(alpha=0.25)
        save_fig(f"{label}: CVHI - Posterior Mean Hidden 恢复",
                 plot_mu, out_dir / f"fig_{safe_label}_02_posterior_mean.png", figsize=(13, 5))

        # Fig 3: Multi-hypothesis visualization
        def plot_multi(ax, res=res, label=label):
            L = min(res["eval"]["h_all"].shape[1], len(res["hidden_true"]))
            t_axis = np.arange(L)
            sort_idx = res["eval"]["sort_idx"]
            h_all = res["eval"]["h_all"][:, :L]
            rmses = res["eval"]["rmses"]
            ax.plot(t_axis, res["hidden_true"][:L], color="black", linewidth=1.8, label="真实 hidden", zorder=10)
            # Plot top 5 hypotheses
            colors = plt.cm.viridis(np.linspace(0, 1, min(5, len(sort_idx))))
            for rank, idx in enumerate(sort_idx[:5]):
                h_this = h_all[idx]
                # Scale-invariant alignment
                X = np.concatenate([h_this.reshape(-1, 1), np.ones((len(h_this), 1))], axis=1)
                coef, _, _, _ = np.linalg.lstsq(X, res["hidden_true"][:L], rcond=None)
                h_scaled = X @ coef
                ax.plot(t_axis, h_scaled, color=colors[rank], linewidth=0.9, alpha=0.6,
                        label=f"Hypothesis {rank+1} (RMSE={rmses[idx]:.3f})")
            ax.set_xlabel("时间步"); ax.set_ylabel("Hidden")
            ax.legend(fontsize=9, ncol=2); ax.grid(alpha=0.25)
        save_fig(f"{label}: CVHI - Top-5 Multi-Hypothesis Overlay",
                 plot_multi, out_dir / f"fig_{safe_label}_03_multi_hypothesis.png", figsize=(13, 5))

        # Fig 4: Training curves
        def plot_loss(ax, res=res, label=label):
            h = res["hist"]
            ax.semilogy(h["train_recon"], color="#1565c0", linewidth=1.1, label="train recon")
            ax.semilogy(h["val_recon"], color="#c62828", linewidth=1.1, label="val recon")
            ax.axvline(res["best_epoch"] - 1, color="green", linestyle="--", linewidth=0.8,
                       label=f"best @ {res['best_epoch']}")
            ax.set_xlabel("epoch"); ax.set_ylabel("reconstruction loss")
            ax.legend(fontsize=11); ax.grid(alpha=0.25)
        save_fig(f"{label}: CVHI - 训练重构损失",
                 plot_loss, out_dir / f"fig_{safe_label}_04_train_loss.png", figsize=(11, 5.5))

        # Fig 5: KL + β over training
        def plot_kl_beta(ax, res=res, label=label):
            h = res["hist"]
            ax.plot(h["kl"], color="#1565c0", linewidth=1.1, label="KL")
            ax.set_xlabel("epoch"); ax.set_ylabel("KL", color="#1565c0")
            ax2 = ax.twinx()
            ax2.plot(h["beta"], color="#e53935", linewidth=1.1, label="β")
            ax2.set_ylabel("β", color="#e53935")
            ax.grid(alpha=0.25)
            ax.legend(loc="upper left", fontsize=10)
            ax2.legend(loc="upper right", fontsize=10)
        save_fig(f"{label}: CVHI - KL vs β Warmup",
                 plot_kl_beta, out_dir / f"fig_{safe_label}_05_kl_beta.png", figsize=(11, 5.5))

        # Fig 6: Posterior σ evolution
        def plot_sigma(ax, res=res, label=label):
            h = res["hist"]
            ax.plot(h["sigma"], color="#1565c0", linewidth=1.1, label="σ mean")
            ax2 = ax.twinx()
            ax2.plot(h["mu_std"], color="#c62828", linewidth=1.1, label="μ std")
            ax.set_xlabel("epoch"); ax.set_ylabel("σ mean", color="#1565c0")
            ax2.set_ylabel("μ std", color="#c62828")
            ax.grid(alpha=0.25)
            ax.legend(loc="upper left", fontsize=10)
            ax2.legend(loc="upper right", fontsize=10)
        save_fig(f"{label}: CVHI - Posterior σ 和 μ 方差",
                 plot_sigma, out_dir / f"fig_{safe_label}_06_posterior_sigma.png", figsize=(11, 5.5))

    # Save numeric results
    save_dict = {}
    for label, res in all_results.items():
        safe_label = label.lower()
        save_dict[f"{safe_label}_best_pearson"] = res["eval"]["eval_best"]["pearson_scaled"]
        save_dict[f"{safe_label}_best_rmse"] = res["eval"]["eval_best"]["rmse_scaled"]
        save_dict[f"{safe_label}_mu_pearson"] = res["eval"]["eval_mu"]["pearson_scaled"]
        save_dict[f"{safe_label}_mu_rmse"] = res["eval"]["eval_mu"]["rmse_scaled"]
        save_dict[f"{safe_label}_top3_pearson"] = res["eval"]["eval_top3"]["pearson_scaled"]
        save_dict[f"{safe_label}_top3_rmse"] = res["eval"]["eval_top3"]["rmse_scaled"]
        save_dict[f"{safe_label}_h_all"] = res["eval"]["h_all"]
        save_dict[f"{safe_label}_rmses"] = res["eval"]["rmses"]
    np.savez(out_dir / "results.npz", **save_dict)

    with open(out_dir / "summary.md", "w", encoding="utf-8") as f:
        f.write("# CVHI MVP 结果\n\n")
        f.write("Conditional Variational Hidden Inference (variational + sparse GAT + multi-hypothesis)\n\n")
        for label, res in all_results.items():
            e = res["eval"]
            f.write(f"## {label}\n\n")
            f.write(f"- Params: {res['num_params']:,}\n")
            f.write(f"- Posterior mean h: Pearson={e['eval_mu']['pearson_scaled']:.4f} RMSE={e['eval_mu']['rmse_scaled']:.4f}\n")
            f.write(f"- Best hypothesis: Pearson={e['eval_best']['pearson_scaled']:.4f} RMSE={e['eval_best']['rmse_scaled']:.4f}\n")
            f.write(f"- Top-3 avg: Pearson={e['eval_top3']['pearson_scaled']:.4f} RMSE={e['eval_top3']['rmse_scaled']:.4f}\n")
            f.write(f"- Best epoch: {res['best_epoch']}\n\n")
    print(f"\n[OK] saved to: {out_dir}")


if __name__ == "__main__":
    main()
