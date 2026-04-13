"""20-seeds 正式统计实验 (paper-level).

方法: CVHI_Residual + MLP backbone + formula hints + L1 rollout
数据: Synthetic LV, Synthetic Holling, Portal OT (hidden=OT)
输出:
  - mean / std / 95% CI / max / min / median
  - top-K (by val_recon) ensemble
  - 全部 seeds 轨迹图
  - 对比基线的 bar chart
  - paper 级 summary.md + json
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats as scipy_stats
from scipy.stats import spearmanr

from models.cvhi_residual import CVHI_Residual
from scripts.cvhi_residual_L1L3_diagnostics import (
    load_portal, evaluate, _configure_matplotlib, hidden_true_substitution,
)
from scripts.cvhi_residual_synthetic_comparison import load_lv, load_holling


SEEDS_20 = [42, 123, 456, 789, 2024, 31415, 27182, 65537,
             7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]


def make_model(N, device, is_portal):
    """最终方法的统一超参数配置."""
    if is_portal:
        return CVHI_Residual(
            num_visible=N,
            encoder_d=48, encoder_blocks=2, encoder_heads=4,
            takens_lags=(1, 2, 4, 8, 12), encoder_dropout=0.2,
            d_species_f=20, f_visible_layers=2, f_visible_top_k=4,
            d_species_G=12, G_field_layers=1, G_field_top_k=3,
            prior_std=1.0,
            gnn_backbone="mlp",
        ).to(device)
    return CVHI_Residual(
        num_visible=N,
        encoder_d=64, encoder_blocks=2, encoder_heads=4,
        takens_lags=(1, 2, 4, 8), encoder_dropout=0.1,
        d_species_f=24, f_visible_layers=2, f_visible_top_k=4,
        d_species_G=16, G_field_layers=1, G_field_top_k=3,
        prior_std=1.0,
        gnn_backbone="mlp",
    ).to(device)


def train_one(visible, hidden_eval, device, seed, epochs=300, is_portal=False):
    torch.manual_seed(seed)
    T, N = visible.shape
    x = torch.tensor(visible, dtype=torch.float32, device=device).unsqueeze(0)
    model = make_model(N, device, is_portal)
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

    for epoch in range(epochs):
        if epoch < warmup_epochs:
            h_weight, rollout_K, lam_rollout = 0.0, 0, 0.0
        else:
            post = epoch - warmup_epochs
            h_weight = min(1.0, post / ramp_epochs)
            k_ramp = min(1.0, post / (epochs - warmup_epochs) * 2)
            rollout_K = max(1 if h_weight > 0 else 0, int(round(k_ramp * 3)))
            lam_rollout = 0.5 * h_weight

        model.train(); opt.zero_grad()
        out = model(x, n_samples=2, rollout_K=rollout_K)
        train_out = model.slice_out(out, 0, train_end)
        losses = model.loss(
            train_out, beta_kl=0.03, free_bits=0.02,
            margin_null=margin_null, margin_shuf=margin_shuf,
            lam_necessary=5.0, lam_shuffle=3.0,
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
                lam_energy=2.0, min_energy=min_energy,
                lam_rollout=0.5, rollout_weights=(1.0, 0.5, 0.25),
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
        out_eval = model(x, n_samples=30, rollout_K=3)
        h_mean = out_eval["h_samples"].mean(dim=0)[0].cpu().numpy()
        full = model.loss(
            out_eval, h_weight=1.0,
            margin_null=margin_null, margin_shuf=margin_shuf,
            lam_energy=2.0, min_energy=min_energy,
            lam_rollout=0.5, rollout_weights=(1.0, 0.5, 0.25),
            lam_hf=0.0, lowpass_sigma=6.0,
        )
    pear, h_scaled = evaluate(h_mean, hidden_eval)
    d = hidden_true_substitution(model, visible, hidden_eval, device)
    return {
        "seed": seed,
        "pearson": pear,
        "h_mean": h_mean,
        "h_scaled": h_scaled,
        "val_recon": float(full["recon_full"]),
        "m_null": float(full["margin_null_obs"]),
        "h_var": float(full["h_var"]),
        "d_ratio": d["recon_true_scaled"] / d["recon_encoder"],
        "recon_null": d["recon_null"],
        "recon_encoder": d["recon_encoder"],
        "recon_true": d["recon_true_scaled"],
        "num_params": sum(p.numel() for p in model.parameters()),
        "best_epoch": best_epoch,
    }


def aggregate_stats(pearsons):
    """Compute mean / std / 95% CI / median / min / max."""
    arr = np.asarray(pearsons)
    n = len(arr)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1))
    # 95% CI 使用 t 分布
    se = std / np.sqrt(n)
    t_crit = scipy_stats.t.ppf(0.975, n - 1)
    ci_half = t_crit * se
    return {
        "n": n, "mean": mean, "std": std,
        "ci95_lo": mean - ci_half, "ci95_hi": mean + ci_half,
        "median": float(np.median(arr)),
        "min": float(arr.min()), "max": float(arr.max()),
    }


def ensemble_top_k(results, k):
    """Select top-k seeds by val_recon (ascending), ensemble their h_mean."""
    sorted_r = sorted(results, key=lambda r: r["val_recon"])
    top_k = sorted_r[:k]
    h_ens = np.mean([r["h_mean"] for r in top_k], axis=0)
    return h_ens, top_k


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_seeds", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--datasets", nargs="+", default=["LV", "Holling", "Portal"])
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    _configure_matplotlib()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"runs/{ts}_cvhi_residual_20seeds_formal")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_dir}\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    seeds = SEEDS_20[:args.n_seeds]
    print(f"Seeds ({len(seeds)}): {seeds}\n")

    datasets_loaders = {
        "LV": (load_lv, False, "Synthetic LV"),
        "Holling": (load_holling, False, "Synthetic Holling II + Allee"),
        "Portal": (lambda: load_portal("OT"), True, "Portal OT"),
    }

    all_results = {}
    total_runs = len(args.datasets) * len(seeds)
    run_i = 0
    for ds_key in args.datasets:
        loader, is_portal, display_name = datasets_loaders[ds_key]
        visible, hidden = loader()
        print(f"\n{'='*80}\n{display_name}  (T={visible.shape[0]}, N={visible.shape[1]})")
        print(f"{'='*80}")
        ds_results = []
        for seed in seeds:
            run_i += 1
            t0 = datetime.now()
            r = train_one(visible, hidden, device, seed, args.epochs, is_portal)
            dt = (datetime.now() - t0).total_seconds()
            ds_results.append(r)
            print(f"[{run_i}/{total_runs}] seed={seed:5d}  "
                  f"P={r['pearson']:+.4f}  val={r['val_recon']:.4f}  "
                  f"d_ratio={r['d_ratio']:.3f}  ({dt:.1f}s)")
        all_results[ds_key] = {"results": ds_results, "visible": visible, "hidden": hidden,
                                "display_name": display_name}

    # === 聚合统计 ===
    stats_table = {}
    for ds_key in args.datasets:
        rs = all_results[ds_key]["results"]
        pearsons = [r["pearson"] for r in rs]
        val_recons = [r["val_recon"] for r in rs]
        d_ratios = [r["d_ratio"] for r in rs]
        h_vars = [r["h_var"] for r in rs]
        # Ensemble by val-recon
        h_ens, top_k_rs = ensemble_top_k(rs, args.top_k)
        hidden = all_results[ds_key]["hidden"]
        ens_pearson, _ = evaluate(h_ens, hidden)
        top_k_pearsons = [r["pearson"] for r in top_k_rs]
        stats_table[ds_key] = {
            "pearson": aggregate_stats(pearsons),
            "val_recon": aggregate_stats(val_recons),
            "d_ratio": aggregate_stats(d_ratios),
            "h_var": aggregate_stats(h_vars),
            "ensemble_top_k": {
                "k": args.top_k,
                "top_k_seeds": [r["seed"] for r in top_k_rs],
                "top_k_pearsons": top_k_pearsons,
                "top_k_mean_pearson": float(np.mean(top_k_pearsons)),
                "ensemble_pearson": ens_pearson,
            },
            "rho_val_pearson": float(spearmanr(val_recons, pearsons)[0]),
        }

    # 打印 paper-level 表
    print(f"\n{'='*100}")
    print(f"FORMAL STATS ({args.n_seeds} seeds, epochs={args.epochs}, method: MLP+hints+L1)")
    print(f"{'='*100}\n")
    for ds_key in args.datasets:
        ds = all_results[ds_key]
        s = stats_table[ds_key]
        p = s["pearson"]
        print(f"{ds['display_name']}:")
        print(f"  Pearson  mean={p['mean']:+.4f}  std={p['std']:.4f}  "
              f"95% CI=[{p['ci95_lo']:+.4f}, {p['ci95_hi']:+.4f}]")
        print(f"           median={p['median']:+.4f}  min={p['min']:+.4f}  max={p['max']:+.4f}")
        print(f"  d_ratio  mean={s['d_ratio']['mean']:.3f}  median={s['d_ratio']['median']:.3f}")
        print(f"  ρ(val,P) {s['rho_val_pearson']:+.3f}")
        print(f"  ensemble_top_{args.top_k}: P={s['ensemble_top_k']['ensemble_pearson']:+.4f}  "
              f"(top-K mean: {s['ensemble_top_k']['top_k_mean_pearson']:+.4f})")
        print()

    # === Paper-quality plots ===

    # Fig 1: per-dataset overlay
    for ds_key in args.datasets:
        ds = all_results[ds_key]
        rs = ds["results"]
        hidden = ds["hidden"]
        fig, ax = plt.subplots(figsize=(14, 6), constrained_layout=True)
        t_axis = np.arange(len(hidden))
        ax.plot(t_axis, hidden, color="black", linewidth=2.0,
                 label=f"真实 hidden", zorder=10)
        # plot all seeds translucent
        for r in rs:
            h = r["h_scaled"]; L = min(len(h), len(t_axis))
            ax.plot(t_axis[:L], h[:L], color="#c62828", linewidth=0.5, alpha=0.25)
        # median seed highlighted
        sorted_rs = sorted(rs, key=lambda r: r["pearson"])
        median_r = sorted_rs[len(sorted_rs) // 2]
        ax.plot(t_axis[:len(median_r["h_scaled"])], median_r["h_scaled"],
                 color="#1565c0", linewidth=1.8, alpha=0.9,
                 label=f"median seed (P={median_r['pearson']:.3f})")
        # max seed highlighted
        max_r = max(rs, key=lambda r: r["pearson"])
        ax.plot(t_axis[:len(max_r["h_scaled"])], max_r["h_scaled"],
                 color="#2e7d32", linewidth=1.8, alpha=0.9, linestyle="--",
                 label=f"best seed (P={max_r['pearson']:.3f})")
        p = stats_table[ds_key]["pearson"]
        ax.set_title(f"{ds['display_name']} — {len(seeds)} seeds, "
                     f"mean P={p['mean']:+.3f} ± {p['std']:.3f}",
                     fontsize=13, fontweight="bold")
        ax.set_xlabel("时间步"); ax.set_ylabel("hidden abundance")
        ax.legend(fontsize=10); ax.grid(alpha=0.25)
        fig.savefig(out_dir / f"fig_{ds_key}_overlay.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

    # Fig 2: all datasets summary bar
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    xs = np.arange(len(args.datasets))
    means = [stats_table[k]["pearson"]["mean"] for k in args.datasets]
    stds = [stats_table[k]["pearson"]["std"] for k in args.datasets]
    maxes = [stats_table[k]["pearson"]["max"] for k in args.datasets]
    ax.bar(xs - 0.25, means, 0.5, yerr=stds, capsize=6, color="#1565c0",
           label="mean ± std")
    ax.bar(xs + 0.25, maxes, 0.5, color="#ff7f0e", alpha=0.8, label="max")
    # reference supervised baselines
    supervised_baselines = {"LV": 0.977, "Holling": 0.620, "Portal": 0.353}
    for i, k in enumerate(args.datasets):
        if k in supervised_baselines:
            ax.plot([i - 0.4, i + 0.4], [supervised_baselines[k]] * 2,
                     color="red", linewidth=2, linestyle="--")
            ax.text(i, supervised_baselines[k] + 0.02,
                    f"supervised baseline = {supervised_baselines[k]:.3f}",
                    ha="center", fontsize=9, color="red")
    ax.set_xticks(xs); ax.set_xticklabels([all_results[k]["display_name"] for k in args.datasets])
    ax.set_ylabel("Pearson to hidden_true")
    ax.legend(fontsize=11); ax.grid(alpha=0.25, axis="y")
    ax.set_title(f"CVHI_Residual 最终方法 — {len(seeds)} seeds",
                 fontsize=13, fontweight="bold")
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i - 0.25, m + s + 0.02, f"{m:+.3f}±{s:.3f}",
                ha="center", fontsize=10)
    for i, mx in enumerate(maxes):
        ax.text(i + 0.25, mx + 0.02, f"{mx:+.3f}", ha="center", fontsize=10)
    fig.savefig(out_dir / "fig_summary_bars.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Fig 3: val_recon vs Pearson scatter per dataset
    for ds_key in args.datasets:
        ds = all_results[ds_key]
        rs = ds["results"]
        fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
        vals = [r["val_recon"] for r in rs]
        pears = [r["pearson"] for r in rs]
        ax.scatter(vals, pears, s=70, c="#1565c0", alpha=0.7)
        # annotate top-K
        top_k_seeds = stats_table[ds_key]["ensemble_top_k"]["top_k_seeds"]
        for r in rs:
            if r["seed"] in top_k_seeds:
                ax.scatter([r["val_recon"]], [r["pearson"]], s=140,
                            facecolors="none", edgecolors="#c62828", linewidth=2)
        rho = stats_table[ds_key]["rho_val_pearson"]
        ax.set_xlabel("val_recon (lower = better)")
        ax.set_ylabel("Pearson to hidden_true")
        ax.set_title(f"{ds['display_name']}: ρ = {rho:+.3f}",
                     fontsize=12, fontweight="bold")
        ax.grid(alpha=0.3)
        fig.savefig(out_dir / f"fig_{ds_key}_val_vs_pearson.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

    # === paper-level summary.md ===
    with open(out_dir / "summary.md", "w", encoding="utf-8") as f:
        f.write(f"# 20-seeds 正式统计实验报告\n\n")
        f.write(f"- 方法：CVHI_Residual + MLP backbone + formula hints + L1 rollout\n")
        f.write(f"- seeds: {len(seeds)}\n")
        f.write(f"- epochs: {args.epochs}\n")
        f.write(f"- 红线: 训练中无 hidden 监督、无 anchor、无 pseudo-label、无外部协变量\n\n")

        f.write(f"## 一、paper-level 统计表\n\n")
        f.write(f"| 数据 | n | mean | std | 95% CI | median | min | max | "
                f"ρ(val,P) | ensemble (top-{args.top_k}) |\n")
        f.write(f"|---|---|---|---|---|---|---|---|---|---|\n")
        for ds_key in args.datasets:
            s = stats_table[ds_key]; p = s["pearson"]; e = s["ensemble_top_k"]
            f.write(f"| {ds_key} | {p['n']} | **{p['mean']:+.4f}** | {p['std']:.4f} | "
                    f"[{p['ci95_lo']:+.4f}, {p['ci95_hi']:+.4f}] | {p['median']:+.4f} | "
                    f"{p['min']:+.4f} | {p['max']:+.4f} | "
                    f"{s['rho_val_pearson']:+.3f} | {e['ensemble_pearson']:+.4f} |\n")

        f.write(f"\n## 二、对照基线\n\n")
        f.write(f"| 数据 | 本方法 | Linear Sparse+EM (有监督) | 相对比例 |\n")
        f.write(f"|---|---|---|---|\n")
        supervised_baselines = {"LV": 0.977, "Holling": 0.620, "Portal": 0.353}
        for ds_key in args.datasets:
            if ds_key not in supervised_baselines: continue
            m = stats_table[ds_key]["pearson"]["mean"]
            mx = stats_table[ds_key]["pearson"]["max"]
            b = supervised_baselines[ds_key]
            f.write(f"| {ds_key} | mean={m:+.3f}, max={mx:+.3f} | {b:.3f} | "
                    f"mean={m/b*100:.0f}%, max={mx/b*100:.0f}% |\n")

        f.write(f"\n## 三、d_ratio（动力学保真度）\n\n")
        f.write(f"将 hidden_true 直接塞进 learned dynamics，计算 recon(true)/recon(encoder)。"
                f"接近 1 表示 learned dynamics 结构上逼近真动力学。\n\n")
        f.write(f"| 数据 | mean d_ratio | median d_ratio |\n|---|---|---|\n")
        for ds_key in args.datasets:
            s = stats_table[ds_key]["d_ratio"]
            f.write(f"| {ds_key} | {s['mean']:.3f} | {s['median']:.3f} |\n")

        f.write(f"\n## 四、每个 seed 详细结果\n\n")
        for ds_key in args.datasets:
            ds = all_results[ds_key]
            f.write(f"### {ds['display_name']}\n\n")
            f.write(f"| seed | Pearson | val_recon | d_ratio | h_var | best_epoch |\n")
            f.write(f"|---|---|---|---|---|---|\n")
            for r in ds["results"]:
                f.write(f"| {r['seed']} | {r['pearson']:+.4f} | {r['val_recon']:.4f} | "
                        f"{r['d_ratio']:.3f} | {r['h_var']:.3f} | {r['best_epoch']} |\n")
            # top-K ensemble 信息
            e = stats_table[ds_key]["ensemble_top_k"]
            f.write(f"\n**Top-{args.top_k} (按 val_recon 选):** "
                    f"seeds={e['top_k_seeds']}, Pearsons={[f'{p:+.3f}' for p in e['top_k_pearsons']]}\n")
            f.write(f"**Ensemble Pearson (top-{args.top_k} 平均 h_mean):** "
                    f"{e['ensemble_pearson']:+.4f}\n\n")

    # Save JSON (excluding arrays)
    serializable = {}
    for ds_key in args.datasets:
        serializable[ds_key] = {
            "stats": stats_table[ds_key],
            "seeds_detail": [
                {k: v for k, v in r.items() if k not in ("h_mean", "h_scaled")}
                for r in all_results[ds_key]["results"]
            ],
        }
    with open(out_dir / "raw_stats.json", "w") as f:
        json.dump(serializable, f, indent=2, default=float)

    # Save npz with arrays
    save_dict = {}
    for ds_key in args.datasets:
        save_dict[f"{ds_key}_hidden_true"] = all_results[ds_key]["hidden"]
        for r in all_results[ds_key]["results"]:
            save_dict[f"{ds_key}_seed{r['seed']}_h_scaled"] = r["h_scaled"]
    np.savez(out_dir / "trajectories.npz", **save_dict)

    print(f"\n[OK] saved to: {out_dir}")


if __name__ == "__main__":
    main()
