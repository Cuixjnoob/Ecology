"""CVHI_Residual (无预设公式, 软形式发现) 在 LV 和 Holling 合成数据上对比.

L1only_K3 config (当前最佳 config), 纯无监督.
测试 "soft-preset, 自己发现"架构能否在不同动力学生成的数据上都 work.

同时诊断: 训练后模型在不同数据集上**实际选用了哪些 form gates**, 以判断:
  - LV 数据 → 模型是否把 LV bilinear gate 调高?
  - Holling 数据 → 模型是否把 Holling gate 调高?
(这是方法的"可解释性"验证)
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

from models.cvhi_residual import CVHI_Residual
from scripts.cvhi_residual_L1L3_diagnostics import (
    evaluate, pairwise_corr, make_model, _configure_matplotlib,
    hidden_true_substitution,
)


def load_lv():
    d = np.load("runs/analysis_5vs6_species/trajectories.npz")
    return (d["states_B_5species"].astype(np.float32) + 0.01,
            d["hidden_B"].astype(np.float32) + 0.01)


def load_holling():
    d = np.load("runs/20260413_100414_5vs6_holling/trajectories.npz")
    return (d["states_B_5species"].astype(np.float32) + 0.01,
            d["hidden_B"].astype(np.float32) + 0.01)


def train_one(visible, hidden_for_eval, device, seed, epochs=300,
               warmup_frac=0.2):
    torch.manual_seed(seed)
    T, N = visible.shape
    x = torch.tensor(visible, dtype=torch.float32, device=device).unsqueeze(0)
    model = make_model(N, is_portal=False).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=0.0008, weight_decay=1e-4)
    warmup_epochs = int(warmup_frac * epochs)
    ramp_epochs = max(1, int(0.2 * epochs))
    def lr_lambda(step):
        if step < 50: return step / 50
        p = (step - 50) / max(1, epochs - 50)
        return 0.5 * (1 + np.cos(np.pi * p))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    train_end = int(0.75 * T)
    best_val = float("inf"); best_state = None; best_epoch = -1
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
            h_weight=h_weight,
            lam_rollout=lam_rollout, rollout_weights=(1.0, 0.5, 0.25),
            lam_hf=0.0, lowpass_sigma=6.0,  # L3 off
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

    pear, h_scaled = evaluate(h_mean, hidden_for_eval)
    d = hidden_true_substitution(model, visible, hidden_for_eval, device)

    # Exact form gate analysis: per form type, mean gate across all edges
    gate_analysis = {"f_visible": {}, "G_field": {}}
    for layer_name, layers in [("f_visible", model.f_visible.layers),
                                ("G_field", model.G_field.layers)]:
        for l_idx, layer in enumerate(layers):
            gates = layer.get_gates().detach().cpu().numpy()  # (num_forms, N, N)
            coefs = layer.form_coefs.detach().cpu().numpy()
            for f_idx, form_name in enumerate(layer.FORM_NAMES):
                gm = float(gates[f_idx].mean())
                cm = float(np.abs(coefs[f_idx]).mean())
                eff = float((gates[f_idx] * np.abs(coefs[f_idx])).mean())
                gate_analysis[layer_name][f"layer{l_idx}_{form_name}"] = {
                    "gate_mean": gm, "coef_abs_mean": cm, "effective": eff,
                }

    return {
        "seed": seed, "pearson": pear, "h_mean": h_mean, "h_scaled": h_scaled,
        "val_recon": float(full["recon_full"]),
        "m_null": float(full["margin_null_obs"]),
        "h_var": float(full["h_var"]),
        "d_ratio": d["recon_true_scaled"] / d["recon_encoder"],
        "recon_true": d["recon_true_scaled"],
        "recon_encoder": d["recon_encoder"],
        "recon_null": d["recon_null"],
        "gate_analysis": gate_analysis,
        "best_epoch": best_epoch,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_seeds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=300)
    args = parser.parse_args()
    _configure_matplotlib()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"runs/{ts}_synthetic_comparison_LV_Holling")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_dir}\n")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    seeds = [42, 123, 456, 789, 2024][:args.n_seeds]

    datasets = {
        "LV": {"visible": None, "hidden": None, "load": load_lv,
                "desc": "Lotka-Volterra synthetic"},
        "Holling": {"visible": None, "hidden": None, "load": load_holling,
                     "desc": "Holling II + Allee synthetic"},
    }
    for key, d in datasets.items():
        d["visible"], d["hidden"] = d["load"]()

    all_results = {}
    for ds_key, ds in datasets.items():
        print(f"\n{'='*72}\n{ds['desc']}  (T={ds['visible'].shape[0]}, N={ds['visible'].shape[1]})")
        print(f"{'='*72}")
        results = []
        for seed in seeds:
            r = train_one(ds["visible"], ds["hidden"], device, seed, args.epochs)
            r["dataset"] = ds_key
            results.append(r)
            print(f"  seed {seed:5d}: P={r['pearson']:+.4f}  d_ratio={r['d_ratio']:.3f}  "
                  f"val={r['val_recon']:.4f}  h_var={r['h_var']:.3f}")
        all_results[ds_key] = results

    # Aggregated summary
    print(f"\n{'='*72}\nSUMMARY\n{'='*72}")
    print(f"{'Dataset':<15}{'mean P':<12}{'max P':<12}{'std P':<10}{'mean d_ratio':<15}")
    for ds_key, results in all_results.items():
        pears = np.array([r["pearson"] for r in results])
        ratios = np.array([r["d_ratio"] for r in results])
        print(f"{ds_key:<15}{pears.mean():<+12.4f}{pears.max():<+12.4f}{pears.std():<10.4f}"
              f"{ratios.mean():<15.3f}")

    # Per-form gate analysis (best seed per dataset)
    print(f"\n{'='*72}\nForm selection analysis (best seed by val_recon per dataset)")
    print(f"{'='*72}")
    for ds_key, results in all_results.items():
        best_r = min(results, key=lambda r: r["val_recon"])
        print(f"\n{ds_key} (seed {best_r['seed']}, P={best_r['pearson']:+.3f}):")
        for part in ["f_visible", "G_field"]:
            print(f"  {part}:")
            for form_key, stats in best_r["gate_analysis"][part].items():
                print(f"    {form_key:<35s}  gate={stats['gate_mean']:.3f}  "
                      f"|coef|={stats['coef_abs_mean']:.4f}  "
                      f"effective={stats['effective']:.4f}")

    # Average form usage across seeds
    print(f"\n{'='*72}\nMean form-effective magnitude across all seeds (discovers dynamics structure?)")
    print(f"{'='*72}")
    for ds_key, results in all_results.items():
        print(f"\n{ds_key}:")
        # Average effective across seeds
        avg_eff = {"f_visible": {}, "G_field": {}}
        for part in ["f_visible", "G_field"]:
            # get keys
            keys = list(results[0]["gate_analysis"][part].keys())
            for k in keys:
                vals = [r["gate_analysis"][part][k]["effective"] for r in results]
                avg_eff[part][k] = (np.mean(vals), np.std(vals))
            # Sort by effective
            sorted_keys = sorted(keys, key=lambda k: -avg_eff[part][k][0])
            print(f"  {part} (top effective forms):")
            for k in sorted_keys[:5]:
                m, s = avg_eff[part][k]
                print(f"    {k:<40s}  effective = {m:.4f} ± {s:.4f}")

    # Plots
    def save_single(title, plot_fn, path, figsize=(13, 5)):
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        plot_fn(ax)
        fig.suptitle(title, fontsize=13, fontweight="bold")
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)

    for ds_key, results in all_results.items():
        def plot_overlay(ax, ds=datasets[ds_key], results=results):
            ht = ds["hidden"]
            t_axis = np.arange(len(ht))
            ax.plot(t_axis, ht, color="black", linewidth=2.0, label="真实 hidden", zorder=10)
            colors = plt.cm.viridis(np.linspace(0, 1, len(results)))
            for i, r in enumerate(results):
                h = r["h_scaled"]
                L = min(len(h), len(t_axis))
                ax.plot(t_axis[:L], h[:L], color=colors[i], linewidth=1.0, alpha=0.85,
                        label=f"seed {r['seed']} (P={r['pearson']:.3f})")
            ax.set_xlabel("time"); ax.set_ylabel("hidden abundance")
            ax.legend(fontsize=10, ncol=2); ax.grid(alpha=0.25)
        pears = np.array([r["pearson"] for r in results])
        save_single(f"{datasets[ds_key]['desc']}: CVHI_Residual (mean P={pears.mean():.3f})",
                     plot_overlay, out_dir / f"fig_{ds_key}_overlay.png", figsize=(14, 5))

    # Form-selection bar plot (for f_visible layer 0)
    def plot_form_usage(ax, part="f_visible", layer=0):
        forms = ["Linear", "LV_bilin", "HollingII_lin", "HollingII_bilin", "FreeNN"]
        x = np.arange(len(forms))
        w = 0.35
        for i, (ds_key, results) in enumerate(all_results.items()):
            vals = [np.mean([r["gate_analysis"][part][f"layer{layer}_{f}"]["effective"]
                              for r in results]) for f in forms]
            color = "#1565c0" if ds_key == "LV" else "#c62828"
            ax.bar(x + (i - 0.5) * w, vals, w, color=color, label=ds_key)
            for j, v in enumerate(vals):
                ax.text(x[j] + (i - 0.5) * w, v + 0.002, f"{v:.3f}", ha="center", fontsize=9)
        ax.set_xticks(x); ax.set_xticklabels(forms, rotation=15)
        ax.set_ylabel("effective magnitude (gate × |coef|)")
        ax.legend(fontsize=11); ax.grid(alpha=0.25, axis="y")
    save_single("f_visible 层 0 形式使用（gate × |coef|）",
                 plot_form_usage, out_dir / "fig_form_usage_f_visible_layer0.png",
                 figsize=(12, 5))

    # Summary
    with open(out_dir / "summary.md", "w", encoding="utf-8") as f:
        f.write("# CVHI_Residual (无预设公式) 在 LV vs Holling 合成数据上对比\n\n")
        f.write(f"seeds: {seeds}, epochs={args.epochs}, config=L1only_K3\n\n")
        f.write("## 汇总\n\n")
        f.write("| Dataset | mean P | max P | std P | mean d_ratio |\n|---|---|---|---|---|\n")
        for ds_key, results in all_results.items():
            pears = np.array([r["pearson"] for r in results])
            ratios = np.array([r["d_ratio"] for r in results])
            f.write(f"| {ds_key} | {pears.mean():+.4f} | {pears.max():+.4f} | "
                    f"{pears.std():.4f} | {ratios.mean():.3f} |\n")
        f.write("\n## 各 seed 详细\n\n")
        for ds_key, results in all_results.items():
            f.write(f"### {ds_key}\n\n")
            f.write("| seed | Pearson | d_ratio | val_recon | h_var |\n|---|---|---|---|---|\n")
            for r in results:
                f.write(f"| {r['seed']} | {r['pearson']:+.4f} | {r['d_ratio']:.3f} | "
                        f"{r['val_recon']:.4f} | {r['h_var']:.3f} |\n")
            f.write("\n")
        f.write("## Form selection analysis (per layer, averaged across seeds)\n\n")
        for ds_key, results in all_results.items():
            f.write(f"### {ds_key}\n\n")
            for part in ["f_visible", "G_field"]:
                f.write(f"**{part}**:\n\n")
                f.write("| form | mean effective | std |\n|---|---|---|\n")
                keys = list(results[0]["gate_analysis"][part].keys())
                vals_s = [(k, np.mean([r["gate_analysis"][part][k]["effective"] for r in results]),
                             np.std([r["gate_analysis"][part][k]["effective"] for r in results]))
                            for k in keys]
                vals_s = sorted(vals_s, key=lambda x: -x[1])
                for k, m, s in vals_s:
                    f.write(f"| {k} | {m:.4f} | {s:.4f} |\n")
                f.write("\n")

    # Save numerical
    import json
    serializable = {}
    for ds_key, results in all_results.items():
        serializable[ds_key] = []
        for r in results:
            rs = {k: v for k, v in r.items() if k not in ("h_mean", "h_scaled")}
            serializable[ds_key].append(rs)
    with open(out_dir / "raw_summary.json", "w") as f:
        json.dump(serializable, f, indent=2, default=float)

    print(f"\n[OK] saved to: {out_dir}")


if __name__ == "__main__":
    main()
