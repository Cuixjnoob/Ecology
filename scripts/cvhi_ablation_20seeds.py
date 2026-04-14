"""完整逐组件 ablation 实验 (20 seeds × 3 数据集 × 5 配置 = 300 训练).

5 configurations:
  full          基线: 完整方法 (MLP + hints + L1 rollout + 残差分解 + 反事实)
  no_hints      移除 formula hints 输入 (MLP 只看 [x_i, x_j, s_i, s_j])
  no_rollout    移除 L1 多步前推 (只保留 1-step 重构)
  no_residual   移除 G(x) 敏感度场 (h 均匀加, 无 per-species 调制)
  no_cf         移除两项反事实损失 (lam_necessary=lam_shuffle=0)
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

from models.cvhi_residual import CVHI_Residual
from scripts.cvhi_residual_L1L3_diagnostics import (
    load_portal, evaluate, _configure_matplotlib, hidden_true_substitution,
)
from scripts.cvhi_residual_synthetic_comparison import load_lv, load_holling


SEEDS_20 = [42, 123, 456, 789, 2024, 31415, 27182, 65537,
             7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]

ABLATIONS = {
    "full": dict(use_formula_hints=True, use_G_field=True,
                  use_rollout=True, use_cf=True),
    "no_hints": dict(use_formula_hints=False, use_G_field=True,
                      use_rollout=True, use_cf=True),
    "no_rollout": dict(use_formula_hints=True, use_G_field=True,
                        use_rollout=False, use_cf=True),
    "no_residual": dict(use_formula_hints=True, use_G_field=False,
                         use_rollout=True, use_cf=True),
    "no_cf": dict(use_formula_hints=True, use_G_field=True,
                   use_rollout=True, use_cf=False),
}


def make_model(N, ablation_cfg, is_portal, device):
    if is_portal:
        return CVHI_Residual(
            num_visible=N,
            encoder_d=48, encoder_blocks=2, encoder_heads=4,
            takens_lags=(1, 2, 4, 8, 12), encoder_dropout=0.2,
            d_species_f=20, f_visible_layers=2, f_visible_top_k=4,
            d_species_G=12, G_field_layers=1, G_field_top_k=3,
            prior_std=1.0,
            gnn_backbone="mlp",
            use_formula_hints=ablation_cfg["use_formula_hints"],
            use_G_field=ablation_cfg["use_G_field"],
        ).to(device)
    return CVHI_Residual(
        num_visible=N,
        encoder_d=64, encoder_blocks=2, encoder_heads=4,
        takens_lags=(1, 2, 4, 8), encoder_dropout=0.1,
        d_species_f=24, f_visible_layers=2, f_visible_top_k=4,
        d_species_G=16, G_field_layers=1, G_field_top_k=3,
        prior_std=1.0,
        gnn_backbone="mlp",
        use_formula_hints=ablation_cfg["use_formula_hints"],
        use_G_field=ablation_cfg["use_G_field"],
    ).to(device)


def train_one(visible, hidden_eval, device, seed, ablation_cfg, epochs=300, is_portal=False):
    torch.manual_seed(seed)
    T, N = visible.shape
    x = torch.tensor(visible, dtype=torch.float32, device=device).unsqueeze(0)
    model = make_model(N, ablation_cfg, is_portal, device)
    opt = torch.optim.AdamW(model.parameters(), lr=0.0008, weight_decay=1e-4)
    warmup_epochs = int(0.2 * epochs)
    ramp_epochs = max(1, int(0.2 * epochs))

    def lr_lambda(step):
        if step < 50: return step / 50
        p = (step - 50) / max(1, epochs - 50)
        return 0.5 * (1 + np.cos(np.pi * p))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    train_end = int(0.75 * T)
    best_val = float("inf"); best_state = None; best_epoch = -1

    if is_portal:
        margin_null, margin_shuf, min_energy = 0.002, 0.001, 0.05
    else:
        margin_null, margin_shuf, min_energy = 0.003, 0.002, 0.02

    use_rollout = ablation_cfg["use_rollout"]
    use_cf = ablation_cfg["use_cf"]

    for epoch in range(epochs):
        if epoch < warmup_epochs:
            h_weight, rollout_K, lam_rollout = 0.0, 0, 0.0
        else:
            post = epoch - warmup_epochs
            h_weight = min(1.0, post / ramp_epochs)
            if use_rollout:
                k_ramp = min(1.0, post / (epochs - warmup_epochs) * 2)
                rollout_K = max(1 if h_weight > 0 else 0, int(round(k_ramp * 3)))
                lam_rollout = 0.5 * h_weight
            else:
                rollout_K = 0
                lam_rollout = 0.0

        # CF 损失权重
        lam_necessary = 5.0 if use_cf else 0.0
        lam_shuffle = 3.0 if use_cf else 0.0

        model.train(); opt.zero_grad()
        out = model(x, n_samples=2, rollout_K=rollout_K)
        train_out = model.slice_out(out, 0, train_end)
        losses = model.loss(
            train_out, beta_kl=0.03, free_bits=0.02,
            margin_null=margin_null, margin_shuf=margin_shuf,
            lam_necessary=lam_necessary, lam_shuffle=lam_shuffle,
            lam_energy=2.0, min_energy=min_energy,
            lam_smooth=0.02, lam_sparse=0.02,
            h_weight=h_weight, lam_rollout=lam_rollout,
            rollout_weights=(1.0, 0.5, 0.25),
            lam_hf=0.0, lowpass_sigma=6.0,
        )
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()

        with torch.no_grad():
            val_out = model.slice_out(out, train_end, T)
            val_losses = model.loss(
                val_out, h_weight=1.0,
                margin_null=margin_null, margin_shuf=margin_shuf,
                lam_necessary=lam_necessary, lam_shuffle=lam_shuffle,
                lam_energy=2.0, min_energy=min_energy,
                lam_rollout=lam_rollout, rollout_weights=(1.0, 0.5, 0.25),
                lam_hf=0.0, lowpass_sigma=6.0,
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
        rollout_K_eval = 3 if use_rollout else 0
        out_eval = model(x, n_samples=30, rollout_K=rollout_K_eval)
        h_mean = out_eval["h_samples"].mean(dim=0)[0].cpu().numpy()

    pear, _ = evaluate(h_mean, hidden_eval)

    # d_ratio
    d = hidden_true_substitution(model, visible, hidden_eval, device)
    d_ratio = d["recon_true_scaled"] / d["recon_encoder"]

    return {
        "seed": seed,
        "pearson": pear,
        "val_recon": best_val,
        "d_ratio": d_ratio,
        "best_epoch": best_epoch,
    }


def aggregate(pearsons):
    arr = np.asarray(pearsons)
    n = len(arr)
    mean = float(arr.mean()); std = float(arr.std(ddof=1))
    from scipy import stats as scipy_stats
    se = std / np.sqrt(n)
    t_crit = scipy_stats.t.ppf(0.975, n - 1)
    return {
        "n": n, "mean": mean, "std": std,
        "ci95_lo": mean - t_crit * se, "ci95_hi": mean + t_crit * se,
        "median": float(np.median(arr)), "min": float(arr.min()), "max": float(arr.max()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_seeds", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--datasets", nargs="+", default=["LV", "Holling", "Portal"])
    parser.add_argument("--ablations", nargs="+",
                         default=["full", "no_hints", "no_rollout", "no_residual", "no_cf"])
    args = parser.parse_args()

    _configure_matplotlib()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"runs/{ts}_ablation_20seeds")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_dir}\n")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    seeds = SEEDS_20[:args.n_seeds]
    print(f"Seeds: {len(seeds)}, epochs: {args.epochs}")
    print(f"Ablations: {args.ablations}")
    print(f"Datasets: {args.datasets}\n")

    loaders = {
        "LV": (load_lv, False),
        "Holling": (load_holling, False),
        "Portal": (lambda: load_portal("OT"), True),
    }

    # 运行所有组合
    all_results = {ab: {ds: [] for ds in args.datasets} for ab in args.ablations}

    total = len(args.ablations) * len(args.datasets) * len(seeds)
    run_i = 0
    for ablation in args.ablations:
        cfg = ABLATIONS[ablation]
        print(f"\n{'='*72}")
        print(f"Ablation: {ablation}  cfg={cfg}")
        print(f"{'='*72}")
        for ds_key in args.datasets:
            loader, is_portal = loaders[ds_key]
            visible, hidden = loader()
            for seed in seeds:
                run_i += 1
                t0 = datetime.now()
                r = train_one(visible, hidden, device, seed, cfg,
                               epochs=args.epochs, is_portal=is_portal)
                dt = (datetime.now() - t0).total_seconds()
                all_results[ablation][ds_key].append(r)
                if (run_i % 10 == 0) or (run_i == total):
                    print(f"  [{run_i}/{total}] {ablation}/{ds_key}/seed={seed}  "
                          f"P={r['pearson']:+.3f}  d_ratio={r['d_ratio']:.2f}  ({dt:.1f}s)")

    # 聚合统计
    summary = {}
    for ablation in args.ablations:
        summary[ablation] = {}
        for ds_key in args.datasets:
            rs = all_results[ablation][ds_key]
            pearsons = [r["pearson"] for r in rs]
            d_ratios = [r["d_ratio"] for r in rs]
            summary[ablation][ds_key] = {
                "pearson": aggregate(pearsons),
                "d_ratio": aggregate(d_ratios),
            }

    # 打印 summary
    print(f"\n{'='*100}")
    print(f"ABLATION SUMMARY ({len(seeds)} seeds, epochs={args.epochs})")
    print(f"{'='*100}")
    print(f"{'Ablation':<14s}{'Dataset':<10s}{'mean P':<12s}{'std':<8s}{'95% CI':<22s}{'max':<8s}{'mean d_ratio':<12s}")
    for ablation in args.ablations:
        for ds_key in args.datasets:
            s = summary[ablation][ds_key]
            p = s["pearson"]; d = s["d_ratio"]
            print(f"{ablation:<14s}{ds_key:<10s}"
                  f"{p['mean']:<+12.4f}{p['std']:<8.4f}"
                  f"[{p['ci95_lo']:+.3f}, {p['ci95_hi']:+.3f}]    "
                  f"{p['max']:<+8.3f}{d['mean']:<12.3f}")
        print()

    # Δ vs full
    print(f"\n{'Δ vs full (mean P)':=^60}")
    for ablation in args.ablations:
        if ablation == "full":
            continue
        for ds_key in args.datasets:
            full_p = summary["full"][ds_key]["pearson"]["mean"]
            this_p = summary[ablation][ds_key]["pearson"]["mean"]
            print(f"  {ablation:<14s}{ds_key:<10s}: Δ = {this_p - full_p:+.4f}")
        print()

    # 写 summary.md
    with open(out_dir / "summary.md", "w", encoding="utf-8") as f:
        f.write(f"# 完整逐组件 Ablation 实验报告\n\n")
        f.write(f"- seeds: {len(seeds)}\n")
        f.write(f"- epochs: {args.epochs}\n")
        f.write(f"- 总训练次数: {total}\n\n")
        f.write(f"## 主结果表\n\n")
        f.write(f"| Ablation | Dataset | mean P | std | 95% CI | max | d_ratio |\n")
        f.write(f"|---|---|---|---|---|---|---|\n")
        for ablation in args.ablations:
            for ds_key in args.datasets:
                s = summary[ablation][ds_key]
                p = s["pearson"]; d = s["d_ratio"]
                f.write(f"| {ablation} | {ds_key} | {p['mean']:+.4f} | {p['std']:.4f} | "
                        f"[{p['ci95_lo']:+.3f}, {p['ci95_hi']:+.3f}] | {p['max']:+.3f} | "
                        f"{d['mean']:.3f} |\n")
        f.write(f"\n## Δ vs full baseline\n\n")
        f.write(f"| Ablation | Dataset | Δ mean P | Δ d_ratio |\n|---|---|---|---|\n")
        for ablation in args.ablations:
            if ablation == "full":
                continue
            for ds_key in args.datasets:
                full_p = summary["full"][ds_key]["pearson"]["mean"]
                full_d = summary["full"][ds_key]["d_ratio"]["mean"]
                this_p = summary[ablation][ds_key]["pearson"]["mean"]
                this_d = summary[ablation][ds_key]["d_ratio"]["mean"]
                f.write(f"| {ablation} | {ds_key} | {this_p - full_p:+.4f} | {this_d - full_d:+.3f} |\n")
        f.write("\n## 配置说明\n\n")
        for ablation, cfg in ABLATIONS.items():
            f.write(f"- **{ablation}**: {cfg}\n")

    # 保存 raw json
    serializable = {ab: {ds: [{k: v for k, v in r.items()} for r in rs]
                          for ds, rs in d.items()}
                    for ab, d in all_results.items()}
    with open(out_dir / "raw_results.json", "w") as f:
        json.dump({"summary": summary, "raw": serializable}, f, indent=2, default=float)

    # 主图: 条形对比
    fig, axes = plt.subplots(1, len(args.datasets), figsize=(5 * len(args.datasets), 5),
                              constrained_layout=True)
    if len(args.datasets) == 1:
        axes = [axes]
    for ax, ds_key in zip(axes, args.datasets):
        means = [summary[ab][ds_key]["pearson"]["mean"] for ab in args.ablations]
        stds = [summary[ab][ds_key]["pearson"]["std"] for ab in args.ablations]
        x_pos = np.arange(len(args.ablations))
        colors = ["#1565c0" if ab == "full" else "#e65100" for ab in args.ablations]
        ax.bar(x_pos, means, yerr=stds, capsize=5, color=colors)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(args.ablations, rotation=20, ha="right", fontsize=10)
        ax.set_ylabel("mean Pearson")
        ax.set_title(ds_key, fontweight="bold")
        ax.grid(alpha=0.25, axis="y")
        for i, (m, s) in enumerate(zip(means, stds)):
            ax.text(i, m + s + 0.02, f"{m:.3f}", ha="center", fontsize=9)
    fig.savefig(out_dir / "fig_ablation_bars.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"\n[OK] saved to: {out_dir}")


if __name__ == "__main__":
    main()
