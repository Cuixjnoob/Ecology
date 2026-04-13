"""Run CVHI_Residual on Portal or synthetic LV.

NO anchor. NO hidden supervision. Pure unsupervised.
hidden_true ONLY used at final eval for Pearson computation.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from models.cvhi_residual import CVHI_Residual


TOP12 = ["PP", "DM", "PB", "DO", "OT", "RM", "PE", "DS", "PF", "NA", "OL", "PM"]


def _configure_matplotlib():
    plt.rcParams["font.sans-serif"] = [
        "PingFang SC", "Microsoft YaHei", "SimHei", "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.size"] = 11


def load_portal(hidden_species="OT"):
    counts = defaultdict(lambda: defaultdict(int))
    with open("data/real_datasets/portal_rodent.csv") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                year = int(row['year']); month = int(row['month'])
            except (ValueError, KeyError):
                continue
            sp = row['species']
            if sp in TOP12:
                counts[(year, month)][sp] += 1
    all_months = sorted(counts.keys())
    matrix = np.zeros((len(all_months), len(TOP12)), dtype=np.float32)
    for t, (y, m) in enumerate(all_months):
        for j, sp in enumerate(TOP12):
            matrix[t, j] = counts[(y, m)].get(sp, 0)
    from scipy.ndimage import uniform_filter1d
    def smooth(x, w=3):
        pad = np.pad(x, ((w//2, w//2), (0, 0)), mode="edge")
        return uniform_filter1d(pad, size=w, axis=0)[w//2:w//2+x.shape[0]]
    matrix_s = smooth(matrix, w=3)
    valid = matrix_s.sum(axis=1) > 10
    matrix_s = matrix_s[valid]
    months_valid = [m for m, v in zip(all_months, valid) if v]
    h_idx = TOP12.index(hidden_species)
    keep = [i for i in range(len(TOP12)) if i != h_idx]
    visible = matrix_s[:, keep] + 0.5
    hidden = matrix_s[:, h_idx] + 0.5
    time_axis = np.array([y + m/12 for (y, m) in months_valid])
    return visible, hidden, time_axis, f"Portal top-12 hidden={hidden_species}"


def load_lv():
    d = np.load("runs/analysis_5vs6_species/trajectories.npz")
    visible = d["states_B_5species"].astype(np.float32) + 0.01
    hidden = d["hidden_B"].astype(np.float32) + 0.01
    time_axis = np.arange(len(visible), dtype=np.float32)
    return visible, hidden, time_axis, "Synthetic LV (5+1)"


def evaluate(h_pred, hidden_true):
    L = min(len(h_pred), len(hidden_true))
    h_pred = h_pred[:L]; hidden_true = hidden_true[:L]
    X = np.concatenate([h_pred.reshape(-1, 1), np.ones((L, 1))], axis=1)
    coef, _, _, _ = np.linalg.lstsq(X, hidden_true, rcond=None)
    h_scaled = X @ coef
    pear_s = float(np.corrcoef(h_scaled, hidden_true)[0, 1])
    rmse = float(np.sqrt(((h_scaled - hidden_true) ** 2).mean()))
    return pear_s, rmse, h_scaled


def train_one(visible, hidden_eval, device, seed, epochs=500,
               warmup_frac=0.20, verbose=True,
               dataset_name="?"):
    """
    visible: (T, N) numpy
    hidden_eval: (T,) numpy — ONLY used at final eval, NOT in training.
    """
    torch.manual_seed(seed)
    T, N = visible.shape
    x = torch.tensor(visible, dtype=torch.float32, device=device).unsqueeze(0)

    is_portal = (N > 6)
    if is_portal:
        # Portal: smaller G_field (real data noise)
        model = CVHI_Residual(
            num_visible=N,
            encoder_d=48, encoder_blocks=2, encoder_heads=4,
            takens_lags=(1, 2, 4, 8, 12), encoder_dropout=0.2,
            d_species_f=20, f_visible_layers=2, f_visible_top_k=4,
            d_species_G=12, G_field_layers=1, G_field_top_k=3,
            prior_std=1.0,
        ).to(device)
    else:
        model = CVHI_Residual(
            num_visible=N,
            encoder_d=64, encoder_blocks=2, encoder_heads=4,
            takens_lags=(1, 2, 4, 8), encoder_dropout=0.1,
            d_species_f=24, f_visible_layers=2, f_visible_top_k=4,
            d_species_G=16, G_field_layers=1, G_field_top_k=3,
            prior_std=1.0,
        ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"    [{dataset_name}] seed={seed} params={num_params:,} T={T} N={N}")

    opt = torch.optim.AdamW(model.parameters(), lr=0.0008, weight_decay=1e-4)
    warmup_epochs = int(warmup_frac * epochs)

    def lr_lambda(step):
        if step < 50:
            return step / 50
        p = (step - 50) / max(1, epochs - 50)
        return 0.5 * (1 + np.cos(np.pi * p))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    train_end = int(0.75 * T)
    best_val = float("inf"); best_state = None; best_epoch = -1

    history = {"recon_full": [], "recon_null": [], "margin_null": [],
                "margin_shuf": [], "h_var": [], "sigma": [], "kl": []}

    # Margins scale with dataset noise level
    if is_portal:
        margin_null, margin_shuf = 0.002, 0.001
        min_energy = 0.05
    else:
        margin_null, margin_shuf = 0.003, 0.002
        min_energy = 0.02

    for epoch in range(epochs):
        # h_weight schedule: 0 during warmup, ramp up to 1 over 20% of epochs after warmup
        if epoch < warmup_epochs:
            h_weight = 0.0
        else:
            ramp_len = max(1, int(0.2 * epochs))
            h_weight = min(1.0, (epoch - warmup_epochs) / ramp_len)

        model.train()
        opt.zero_grad()
        out = model(x, n_samples=2)
        # Train only on first train_end steps
        train_out = model.slice_out(out, 0, train_end)
        losses = model.loss(
            train_out,
            beta_kl=0.03, free_bits=0.02,
            margin_null=margin_null, margin_shuf=margin_shuf,
            lam_necessary=5.0, lam_shuffle=3.0,
            lam_energy=2.0, min_energy=min_energy,
            lam_smooth=0.05, lam_sparse=0.02,
            h_weight=h_weight,
        )
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()

        with torch.no_grad():
            val_out = model.slice_out(out, train_end, T)
            val_losses = model.loss(val_out, h_weight=1.0)
            val_recon = val_losses["recon_full"].item()

        history["recon_full"].append(losses["recon_full"].item())
        history["recon_null"].append(losses["recon_null"].item())
        history["margin_null"].append(losses["margin_null_obs"].item())
        history["margin_shuf"].append(losses["margin_shuf_obs"].item())
        history["h_var"].append(losses["h_var"].item())
        history["sigma"].append(losses["sigma_mean"].item())
        history["kl"].append(losses["kl"].item())

        if epoch > warmup_epochs + 20 and val_recon < best_val:
            best_val = val_recon
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1

        if verbose and (epoch + 1) % 50 == 0:
            print(f"      ep {epoch+1:4d} w={h_weight:.2f} "
                  f"recon={losses['recon_full'].item():.4f} "
                  f"m_null={losses['margin_null_obs'].item():+.4f} "
                  f"m_shuf={losses['margin_shuf_obs'].item():+.4f} "
                  f"h_var={losses['h_var'].item():.3f} "
                  f"σ={losses['sigma_mean'].item():.3f} "
                  f"KL={losses['kl'].item():.3f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        out_eval = model(x, n_samples=30)
        h_mean = out_eval["h_samples"].mean(dim=0)[0].cpu().numpy()
        # Also check margins at eval
        eval_losses = model.loss(out_eval, h_weight=1.0)

    pear, rmse, h_scaled = evaluate(h_mean, hidden_eval)
    return {
        "pearson": pear,
        "rmse": rmse,
        "h_mean": h_mean,
        "h_scaled": h_scaled,
        "best_epoch": best_epoch,
        "final_margin_null": float(eval_losses["margin_null_obs"]),
        "final_margin_shuf": float(eval_losses["margin_shuf_obs"]),
        "final_h_var": float(eval_losses["h_var"]),
        "num_params": num_params,
        "history": history,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["portal", "lv", "both"], default="both")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    parser.add_argument("--hidden", type=str, default="OT",
                         help="Hidden species for Portal (default OT)")
    args = parser.parse_args()

    _configure_matplotlib()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"runs/{timestamp}_cvhi_residual")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_dir}\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    datasets = {}
    if args.dataset in ("portal", "both"):
        v, h, ta, name = load_portal(args.hidden)
        datasets["portal"] = {"visible": v, "hidden_eval": h, "time_axis": ta, "name": name}
    if args.dataset in ("lv", "both"):
        v, h, ta, name = load_lv()
        datasets["lv"] = {"visible": v, "hidden_eval": h, "time_axis": ta, "name": name}

    all_results = {}
    for ds_key, ds in datasets.items():
        print(f"\n{'='*72}\n{ds['name']}: T={ds['visible'].shape[0]}, N={ds['visible'].shape[1]}")
        print(f"{'='*72}")
        results = []
        for seed in args.seeds:
            print(f"\n--- seed {seed} ---")
            r = train_one(ds["visible"], ds["hidden_eval"], device, seed,
                           epochs=args.epochs, dataset_name=ds_key)
            r["seed"] = seed
            results.append(r)
            print(f"  Pearson = {r['pearson']:+.4f}  RMSE = {r['rmse']:.3f}  "
                  f"best_ep={r['best_epoch']}  m_null_final={r['final_margin_null']:+.4f} "
                  f"h_var_final={r['final_h_var']:.3f}")
        all_results[ds_key] = {"data": ds, "results": results}

        pearsons = np.array([r["pearson"] for r in results])
        print(f"\n  Mean = {pearsons.mean():+.4f} ± {pearsons.std():.4f}")
        print(f"  Max  = {np.max(np.abs(pearsons)):.4f}")

    # Plots
    def save_single(title, plot_fn, path, figsize=(13, 5)):
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        plot_fn(ax)
        fig.suptitle(title, fontsize=13, fontweight="bold")
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)

    for ds_key, rr in all_results.items():
        ds = rr["data"]
        results = rr["results"]
        def plot_recovery(ax, ds=ds, results=results):
            ht = ds["hidden_eval"]
            t_axis = ds["time_axis"][:len(ht)]
            ax.plot(t_axis, ht, color="black", linewidth=2.0, label="真实 hidden", zorder=10)
            colors = plt.cm.viridis(np.linspace(0, 1, len(results)))
            for i, r in enumerate(results):
                h = r["h_scaled"]
                L = min(len(h), len(t_axis))
                ax.plot(t_axis[:L], h[:L], color=colors[i], linewidth=1.0, alpha=0.85,
                        label=f"seed {r['seed']} (P={r['pearson']:.3f})")
            ax.set_xlabel("时间"); ax.set_ylabel("hidden abundance")
            ax.legend(fontsize=10, ncol=2); ax.grid(alpha=0.25)
        pearsons = np.array([r["pearson"] for r in results])
        save_single(f"{ds['name']} — CVHI_Residual (mean P={pearsons.mean():.3f} ± {pearsons.std():.3f})",
                     plot_recovery,
                     out_dir / f"fig_{ds_key}_recovery.png", figsize=(14, 6))

        # Training curves
        def plot_curves(ax, results=results):
            colors = plt.cm.viridis(np.linspace(0, 1, len(results)))
            for i, r in enumerate(results):
                h = r["history"]
                ax.plot(h["margin_null"], color=colors[i], linewidth=1.0, alpha=0.8,
                        label=f"seed {r['seed']} m_null")
            ax.axhline(0, color="grey", linewidth=0.5)
            ax.set_xlabel("epoch"); ax.set_ylabel("margin_null_obs")
            ax.legend(fontsize=10); ax.grid(alpha=0.25)
        save_single(f"{ds['name']} — Counterfactual null margin over training",
                     plot_curves, out_dir / f"fig_{ds_key}_margins.png", figsize=(12, 5))

    # Summary
    with open(out_dir / "summary.md", "w", encoding="utf-8") as f:
        f.write("# CVHI_Residual — 纯无监督 hidden recovery\n\n")
        f.write("架构: f_visible(x) + h·G(x) (残差分解) + 反事实必要性\n\n")
        f.write("训练: 无 anchor, 无 hidden 监督, 反事实 null/shuffle margins 强制 h 必要\n\n")
        for ds_key, rr in all_results.items():
            ds = rr["data"]; results = rr["results"]
            pearsons = np.array([r["pearson"] for r in results])
            f.write(f"## {ds['name']}\n\n")
            f.write(f"T = {ds['visible'].shape[0]}, N = {ds['visible'].shape[1]}\n\n")
            f.write(f"| Seed | Pearson | RMSE | m_null_final | h_var | best_ep |\n")
            f.write(f"|---|---|---|---|---|---|\n")
            for r in results:
                f.write(f"| {r['seed']} | {r['pearson']:+.4f} | {r['rmse']:.3f} | "
                        f"{r['final_margin_null']:+.4f} | {r['final_h_var']:.3f} | "
                        f"{r['best_epoch']} |\n")
            f.write(f"\n**Mean = {pearsons.mean():+.4f} ± {pearsons.std():.4f}**\n")
            f.write(f"Max  = {np.max(np.abs(pearsons)):.4f}\n\n")

        f.write("## 对照 (其他方法)\n\n")
        f.write("| 方法 | Portal OT | 合成 LV | 是否用 hidden 监督 |\n")
        f.write("|---|---|---|---|\n")
        f.write("| Linear Sparse + EM | 0.35 | 0.98 | 是 (投影步骤) |\n")
        f.write("| CVHI 原版 + anchor | 0.33 ± 0.21 | 0.88 | 间接 (anchor 来自 Linear) |\n")
        f.write("| CVHI-NCD + anchor (v4) | 0.23 ± 0.002 | 0.84 ± 0.0002 | 间接 |\n")
        f.write("| **CVHI_Residual (本次)** | ? | ? | **无 (纯无监督)** |\n")

    np.savez(out_dir / "results.npz",
              **{f"{ds_key}_pearsons": np.array([r["pearson"] for r in rr["results"]])
                 for ds_key, rr in all_results.items()})
    print(f"\n[OK] saved to: {out_dir}")


if __name__ == "__main__":
    main()
