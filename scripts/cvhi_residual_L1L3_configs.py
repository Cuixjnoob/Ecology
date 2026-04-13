"""3 config comparison: L1-only vs L1+weak-L3 vs L1-only+long-rollout.

所有 config 都严格无 hidden supervision.
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

from models.cvhi_residual import CVHI_Residual
from scripts.cvhi_residual_L1L3_diagnostics import (
    load_portal, load_lv, evaluate, pairwise_corr, make_model,
    hidden_true_substitution, _configure_matplotlib,
)


# Config menu
CONFIGS = {
    "L1only_K3": dict(lam_rollout=0.5, rollout_weights=(1.0, 0.5, 0.25),
                       max_rollout_K=3, lam_hf=0.0, lowpass_sigma_default=6.0),
    "L1_weakL3_K3": dict(lam_rollout=0.5, rollout_weights=(1.0, 0.5, 0.25),
                          max_rollout_K=3, lam_hf=0.1, lowpass_sigma_default=2.5),
    "L1only_K5": dict(lam_rollout=0.5, rollout_weights=(1.0, 0.5, 0.3, 0.2, 0.15),
                       max_rollout_K=5, lam_hf=0.0, lowpass_sigma_default=6.0),
}


def train_one_cfg(visible, hidden_for_eval, device, seed, cfg_name, cfg,
                   epochs=250, warmup_frac=0.2, is_portal=False):
    torch.manual_seed(seed)
    T, N = visible.shape
    x = torch.tensor(visible, dtype=torch.float32, device=device).unsqueeze(0)
    model = make_model(N, is_portal).to(device)

    # Dataset-specific lowpass sigma
    if cfg["lam_hf"] > 0:
        if is_portal:
            lowpass_sigma = cfg["lowpass_sigma_default"]
        else:
            lowpass_sigma = cfg["lowpass_sigma_default"] + 0.5
    else:
        lowpass_sigma = 6.0  # unused but must be valid

    opt = torch.optim.AdamW(model.parameters(), lr=0.0008, weight_decay=1e-4)
    warmup_epochs = int(warmup_frac * epochs)
    ramp_epochs = max(1, int(0.2 * epochs))

    def lr_lambda(step):
        if step < 40: return step / 40
        p = (step - 40) / max(1, epochs - 40)
        return 0.5 * (1 + np.cos(np.pi * p))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    train_end = int(0.75 * T)
    best_val = float("inf"); best_state = None; best_epoch = -1

    margin_null = 0.002 if is_portal else 0.003
    margin_shuf = 0.001 if is_portal else 0.002
    min_energy = 0.05 if is_portal else 0.02
    max_K = cfg["max_rollout_K"]

    for epoch in range(epochs):
        if epoch < warmup_epochs:
            h_weight = 0.0
            rollout_K = 0
            lam_rollout = 0.0
            lam_hf = 0.0
        else:
            post_warmup = epoch - warmup_epochs
            h_weight = min(1.0, post_warmup / ramp_epochs)
            k_ramp = min(1.0, post_warmup / (epochs - warmup_epochs) * 2)
            rollout_K = int(round(k_ramp * max_K))
            rollout_K = max(1 if h_weight > 0 else 0, rollout_K)
            lam_rollout = cfg["lam_rollout"] * h_weight
            lam_hf = cfg["lam_hf"] * h_weight

        model.train(); opt.zero_grad()
        out = model(x, n_samples=2, rollout_K=rollout_K)
        train_out = model.slice_out(out, 0, train_end)
        losses = model.loss(
            train_out, beta_kl=0.03, free_bits=0.02,
            margin_null=margin_null, margin_shuf=margin_shuf,
            lam_necessary=5.0, lam_shuffle=3.0,
            lam_energy=2.0, min_energy=min_energy,
            lam_smooth=0.02, lam_sparse=0.02, h_weight=h_weight,
            lam_rollout=lam_rollout,
            rollout_weights=cfg["rollout_weights"],
            lam_hf=lam_hf, lowpass_sigma=lowpass_sigma,
        )
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        with torch.no_grad():
            val_out = model.slice_out(out, train_end, T)
            val_losses = model.loss(
                val_out, h_weight=1.0,
                margin_null=margin_null, margin_shuf=margin_shuf,
                lam_energy=2.0, min_energy=min_energy,
                lam_rollout=cfg["lam_rollout"],
                rollout_weights=cfg["rollout_weights"],
                lam_hf=cfg["lam_hf"], lowpass_sigma=lowpass_sigma,
            )
            val_recon = val_losses["recon_full"].item()
        if epoch > warmup_epochs + 15 and val_recon < best_val:
            best_val = val_recon
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        out_eval = model(x, n_samples=30, rollout_K=max_K)
        h_mean = out_eval["h_samples"].mean(dim=0)[0].cpu().numpy()
        full = model.loss(
            out_eval, h_weight=1.0,
            margin_null=margin_null, margin_shuf=margin_shuf,
            lam_energy=2.0, min_energy=min_energy,
            lam_rollout=cfg["lam_rollout"],
            rollout_weights=cfg["rollout_weights"],
            lam_hf=cfg["lam_hf"], lowpass_sigma=lowpass_sigma,
        )

    pear, _ = evaluate(h_mean, hidden_for_eval)
    d = hidden_true_substitution(model, visible, hidden_for_eval, device)
    ratio = d["recon_true_scaled"] / d["recon_encoder"]
    return {
        "seed": seed, "cfg": cfg_name,
        "pearson": pear, "val_recon": float(full["recon_full"]),
        "m_null": float(full["margin_null_obs"]),
        "hf_frac": float(full["hf_frac"]),
        "h_var": float(full["h_var"]),
        "rollout_loss": float(full["rollout"]),
        "d_ratio": ratio,
        "recon_true": d["recon_true_scaled"],
        "recon_encoder": d["recon_encoder"],
        "h_mean": h_mean,
        "best_epoch": best_epoch,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_seeds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=250)
    args = parser.parse_args()

    _configure_matplotlib()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"runs/{ts}_cvhi_residual_L1L3_configs")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_dir}\n")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    seeds = [42, 123, 456, 789, 2024, 31415, 27182, 65537][:args.n_seeds]
    datasets = {
        "portal": {**dict(zip(("visible", "hidden"), load_portal("OT"))), "is_portal": True, "name": "Portal OT"},
        "lv": {**dict(zip(("visible", "hidden"), load_lv())), "is_portal": False, "name": "Synthetic LV"},
    }

    all_results = {cfg: {ds: [] for ds in datasets} for cfg in CONFIGS}

    total_runs = len(CONFIGS) * len(datasets) * len(seeds)
    run_idx = 0
    for cfg_name, cfg in CONFIGS.items():
        for ds_key, ds in datasets.items():
            for seed in seeds:
                run_idx += 1
                print(f"[{run_idx}/{total_runs}] {cfg_name} / {ds['name']} / seed={seed}")
                r = train_one_cfg(ds["visible"], ds["hidden"], device, seed,
                                   cfg_name, cfg, epochs=args.epochs,
                                   is_portal=ds["is_portal"])
                all_results[cfg_name][ds_key].append(r)
                print(f"    P={r['pearson']:+.4f}  d_ratio={r['d_ratio']:.3f}  "
                      f"hf_frac={r['hf_frac']:.3f}  val={r['val_recon']:.4f}")

    # Analysis: per config per dataset, aggregated stats
    print(f"\n{'='*96}")
    print(f"{'Config':<20}{'Dataset':<10}{'mean P':<10}{'max P':<10}{'std P':<10}"
          f"{'ρ(val,P)':<12}{'mean dR':<10}{'mean hf':<10}")
    print("="*96)
    summary = {}
    for cfg_name in CONFIGS:
        for ds_key in datasets:
            rs = all_results[cfg_name][ds_key]
            pears = np.array([r["pearson"] for r in rs])
            vals = np.array([r["val_recon"] for r in rs])
            ratios = np.array([r["d_ratio"] for r in rs])
            hfs = np.array([r["hf_frac"] for r in rs])
            rho_val, _ = spearmanr(vals, pears)
            summary[(cfg_name, ds_key)] = {
                "mean_p": float(pears.mean()), "max_p": float(pears.max()),
                "std_p": float(pears.std()), "rho_val": float(rho_val),
                "mean_d_ratio": float(ratios.mean()),
                "mean_hf": float(hfs.mean()),
                "pearsons": pears.tolist(),
                "val_recons": vals.tolist(),
                "d_ratios": ratios.tolist(),
                "hf_fracs": hfs.tolist(),
            }
            s = summary[(cfg_name, ds_key)]
            print(f"{cfg_name:<20}{ds_key:<10}{s['mean_p']:<+10.4f}{s['max_p']:<+10.4f}"
                  f"{s['std_p']:<10.4f}{s['rho_val']:<+12.3f}{s['mean_d_ratio']:<10.3f}"
                  f"{s['mean_hf']:<10.3f}")

    # Compute deltas vs L1only_K3 baseline
    print(f"\n{'Comparison vs L1only_K3 baseline':=^96}")
    base_cfg = "L1only_K3"
    for ds_key in datasets:
        print(f"\n{ds_key.upper()}:")
        base = summary[(base_cfg, ds_key)]
        for cfg_name in CONFIGS:
            if cfg_name == base_cfg: continue
            s = summary[(cfg_name, ds_key)]
            print(f"  {cfg_name} vs L1only_K3:")
            print(f"    Δ mean P = {s['mean_p'] - base['mean_p']:+.4f}  "
                  f"Δ max P = {s['max_p'] - base['max_p']:+.4f}  "
                  f"Δ d_ratio = {s['mean_d_ratio'] - base['mean_d_ratio']:+.3f}")

    # Verdict
    print(f"\n{'Verdict':=^96}")
    lv_means = {cfg_name: summary[(cfg_name, 'lv')]['mean_p'] for cfg_name in CONFIGS}
    best_lv_cfg = max(lv_means, key=lv_means.get)
    print(f"Best cfg on LV: {best_lv_cfg} (mean P = {lv_means[best_lv_cfg]:+.4f})")
    print(f"All cfgs on LV:  {[(k, f'{v:+.3f}') for k, v in lv_means.items()]}")

    portal_means = {cfg_name: summary[(cfg_name, 'portal')]['mean_p'] for cfg_name in CONFIGS}
    best_portal_cfg = max(portal_means, key=portal_means.get)
    print(f"Best cfg on Portal: {best_portal_cfg} (mean P = {portal_means[best_portal_cfg]:+.4f})")
    print(f"All cfgs on Portal: {[(k, f'{v:+.3f}') for k, v in portal_means.items()]}")

    # Write summary
    with open(out_dir / "summary.md", "w", encoding="utf-8") as f:
        f.write("# L1+L3 config 对比\n\n")
        f.write("seeds: " + ", ".join(str(s) for s in seeds) + f", epochs={args.epochs}\n\n")
        f.write("## 结果表\n\n")
        f.write("| Config | Dataset | mean P | max P | std P | ρ(val,P) | mean d_ratio | mean hf |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        for cfg_name in CONFIGS:
            for ds_key in datasets:
                s = summary[(cfg_name, ds_key)]
                f.write(f"| {cfg_name} | {ds_key} | {s['mean_p']:+.4f} | {s['max_p']:+.4f} | "
                        f"{s['std_p']:.4f} | {s['rho_val']:+.3f} | {s['mean_d_ratio']:.3f} | "
                        f"{s['mean_hf']:.3f} |\n")
        f.write(f"\n## 参考 pre-L1L3 baseline\n\n")
        f.write("| Dataset | mean P | max P | ρ(val,P) | d_ratio |\n|---|---|---|---|---|\n")
        f.write("| Portal | 0.146 | 0.277 | +0.738 | 1.10 |\n")
        f.write("| LV | 0.689 | 0.915 | +0.405 | 3.01 |\n")
        f.write(f"\n## 各 seed 详细\n\n")
        for cfg_name in CONFIGS:
            for ds_key in datasets:
                rs = all_results[cfg_name][ds_key]
                f.write(f"### {cfg_name} / {ds_key}\n\n")
                f.write("| seed | Pearson | d_ratio | hf_frac | val_recon |\n")
                f.write("|---|---|---|---|---|\n")
                for r in rs:
                    f.write(f"| {r['seed']} | {r['pearson']:+.4f} | {r['d_ratio']:.3f} | "
                            f"{r['hf_frac']:.3f} | {r['val_recon']:.4f} |\n")

    # Save numerical
    import json
    serializable = {}
    for (cfg_name, ds_key), s in summary.items():
        serializable[f"{cfg_name}__{ds_key}"] = {k: v for k, v in s.items()}
    with open(out_dir / "raw_summary.json", "w") as f:
        json.dump(serializable, f, indent=2, default=float)

    # Plot: 3 configs x 2 datasets = 6 bar groups
    def save_single(title, plot_fn, path, figsize=(13, 6)):
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        plot_fn(ax)
        fig.suptitle(title, fontsize=13, fontweight="bold")
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)

    def plot_compare_bars(ax, metric="mean_p"):
        configs = list(CONFIGS.keys())
        dss = list(datasets.keys())
        x = np.arange(len(configs))
        w = 0.35
        colors = ["#1565c0", "#c62828"]
        for i, ds_key in enumerate(dss):
            vals = [summary[(c, ds_key)][metric] for c in configs]
            ax.bar(x + (i - 0.5) * w, vals, w, color=colors[i], label=ds_key)
            for j, v in enumerate(vals):
                ax.text(x[j] + (i - 0.5) * w, v + 0.01, f"{v:+.3f}",
                        ha="center", fontsize=9)
        ax.set_xticks(x); ax.set_xticklabels(configs)
        ax.set_ylabel(metric)
        ax.legend(fontsize=11); ax.grid(alpha=0.25, axis="y")
    save_single("Mean Pearson across configs", lambda ax: plot_compare_bars(ax, "mean_p"),
                 out_dir / "fig_mean_pearson.png")
    save_single("Max Pearson across configs", lambda ax: plot_compare_bars(ax, "max_p"),
                 out_dir / "fig_max_pearson.png")
    save_single("Mean d_ratio across configs (越接近 1 越好)",
                 lambda ax: plot_compare_bars(ax, "mean_d_ratio"),
                 out_dir / "fig_d_ratio.png")

    print(f"\n[OK] saved to: {out_dir}")


if __name__ == "__main__":
    main()
