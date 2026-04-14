"""CVHI_Residual + L1 (multi-step rollout) + L3 (low-freq hidden prior) 诊断.

对比前一轮 (无 L1/L3) 结果, 判断是否:
  - 减少 dynamics 多解 → Exp D 的 recon_true/recon_encoder ratio 降至 ~1
  - 减少 h 吸噪 → Portal 上 ρ(val_recon, Pearson) 变正常 (<= 0 更好)
  - 提升 Pearson 下限

仍然严格无 hidden supervision. hidden_true 仅用于最终诊断.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
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
    h_idx = TOP12.index(hidden_species)
    keep = [i for i in range(len(TOP12)) if i != h_idx]
    return matrix_s[:, keep] + 0.5, matrix_s[:, h_idx] + 0.5


def load_lv():
    d = np.load("runs/analysis_5vs6_species/trajectories.npz")
    return (d["states_B_5species"].astype(np.float32) + 0.01,
            d["hidden_B"].astype(np.float32) + 0.01)


def evaluate(h_pred, hidden_true):
    L = min(len(h_pred), len(hidden_true))
    h_pred = h_pred[:L]; hidden_true = hidden_true[:L]
    X = np.concatenate([h_pred.reshape(-1, 1), np.ones((L, 1))], axis=1)
    coef, _, _, _ = np.linalg.lstsq(X, hidden_true, rcond=None)
    h_scaled = X @ coef
    return float(np.corrcoef(h_scaled, hidden_true)[0, 1]), h_scaled


def pairwise_corr(h_list):
    K = len(h_list)
    M = np.ones((K, K))
    for i in range(K):
        for j in range(K):
            if i == j: continue
            L = min(len(h_list[i]), len(h_list[j]))
            M[i, j] = float(np.corrcoef(h_list[i][:L], h_list[j][:L])[0, 1])
    return M


def make_model(N, is_portal):
    if is_portal:
        return CVHI_Residual(
            num_visible=N,
            encoder_d=48, encoder_blocks=2, encoder_heads=4,
            takens_lags=(1, 2, 4, 8, 12), encoder_dropout=0.2,
            d_species_f=20, f_visible_layers=2, f_visible_top_k=4,
            d_species_G=12, G_field_layers=1, G_field_top_k=3,
            prior_std=1.0,
        )
    return CVHI_Residual(
        num_visible=N,
        encoder_d=64, encoder_blocks=2, encoder_heads=4,
        takens_lags=(1, 2, 4, 8), encoder_dropout=0.1,
        d_species_f=24, f_visible_layers=2, f_visible_top_k=4,
        d_species_G=16, G_field_layers=1, G_field_top_k=3,
        prior_std=1.0,
    )


def train_one(visible, hidden_for_eval, device, seed, epochs=300,
               warmup_frac=0.2, is_portal=False, lowpass_sigma=None):
    torch.manual_seed(seed)
    T, N = visible.shape
    x = torch.tensor(visible, dtype=torch.float32, device=device).unsqueeze(0)
    model = make_model(N, is_portal).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=0.0008, weight_decay=1e-4)
    warmup_epochs = int(warmup_frac * epochs)
    ramp_epochs = max(1, int(0.2 * epochs))  # ramp for h_weight

    if lowpass_sigma is None:
        lowpass_sigma = 6.0 if is_portal else 8.0

    def lr_lambda(step):
        if step < 50: return step / 50
        p = (step - 50) / max(1, epochs - 50)
        return 0.5 * (1 + np.cos(np.pi * p))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    train_end = int(0.75 * T)
    best_val = float("inf"); best_state = None; best_epoch = -1

    margin_null = 0.002 if is_portal else 0.003
    margin_shuf = 0.001 if is_portal else 0.002
    min_energy = 0.05 if is_portal else 0.02

    history = {"recon_full": [], "rollout": [], "hf_frac": [], "m_null": [],
                "h_var": [], "sigma": []}

    for epoch in range(epochs):
        # Schedule: warmup → h_weight ramp → full
        if epoch < warmup_epochs:
            h_weight = 0.0
            rollout_K = 0
            lam_rollout = 0.0
            lam_hf = 0.0
        else:
            post_warmup = epoch - warmup_epochs
            h_weight = min(1.0, post_warmup / ramp_epochs)
            # L1 schedule: K grows 0 → 3 over first 50% of post-warmup
            k_ramp = min(1.0, post_warmup / (epochs - warmup_epochs) * 2)
            rollout_K = int(round(k_ramp * 3))  # 0, 1, 2, 3
            rollout_K = max(1 if h_weight > 0 else 0, rollout_K)
            # L3 schedule: ramp after h_weight done
            lam_rollout = 0.5 * h_weight
            lam_hf = 0.5 * h_weight

        model.train(); opt.zero_grad()
        out = model(x, n_samples=2, rollout_K=rollout_K)
        train_out = model.slice_out(out, 0, train_end)
        losses = model.loss(
            train_out, beta_kl=0.03, free_bits=0.02,
            margin_null=margin_null, margin_shuf=margin_shuf,
            lam_necessary=5.0, lam_shuffle=3.0,
            lam_energy=2.0, min_energy=min_energy,
            lam_smooth=0.02,   # lower local smoothness (L3 subsumes it)
            lam_sparse=0.02, h_weight=h_weight,
            lam_rollout=lam_rollout,
            lam_hf=lam_hf, lowpass_sigma=lowpass_sigma,
        )
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        with torch.no_grad():
            val_out = model.slice_out(out, train_end, T)
            val_losses = model.loss(val_out, h_weight=1.0,
                                     margin_null=margin_null, margin_shuf=margin_shuf,
                                     lam_energy=2.0, min_energy=min_energy,
                                     lam_rollout=lam_rollout, lam_hf=lam_hf,
                                     lowpass_sigma=lowpass_sigma)
            val_recon = val_losses["recon_full"].item()

        history["recon_full"].append(losses["recon_full"].item())
        history["rollout"].append(losses["rollout"].item())
        history["hf_frac"].append(losses["hf_frac"].item())
        history["m_null"].append(losses["margin_null_obs"].item())
        history["h_var"].append(losses["h_var"].item())
        history["sigma"].append(losses["sigma_mean"].item())

        if epoch > warmup_epochs + 20 and val_recon < best_val:
            best_val = val_recon
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        out_eval = model(x, n_samples=30, rollout_K=3)
        h_mean = out_eval["h_samples"].mean(dim=0)[0].cpu().numpy()
        full_losses = model.loss(
            out_eval, h_weight=1.0,
            margin_null=margin_null, margin_shuf=margin_shuf,
            lam_energy=2.0, min_energy=min_energy,
            lam_rollout=0.5, lam_hf=0.5, lowpass_sigma=lowpass_sigma,
        )
        val_out = model.slice_out(out_eval, train_end, T)
        val_losses_final = model.loss(
            val_out, h_weight=1.0,
            margin_null=margin_null, margin_shuf=margin_shuf,
            lam_energy=2.0, min_energy=min_energy,
            lam_rollout=0.5, lam_hf=0.5, lowpass_sigma=lowpass_sigma,
        )

    pear, h_scaled = evaluate(h_mean, hidden_for_eval)
    return {
        "seed": seed, "model": model,
        "pearson": pear,
        "train_recon": float(full_losses["recon_full"]),
        "val_recon": float(val_losses_final["recon_full"]),
        "rollout_loss": float(full_losses["rollout"]),
        "rollout_per_step": full_losses.get("rollout_per_step", None),
        "hf_frac": float(full_losses["hf_frac"]),
        "m_null": float(full_losses["margin_null_obs"]),
        "m_shuf": float(full_losses["margin_shuf_obs"]),
        "h_var": float(full_losses["h_var"]),
        "kl": float(full_losses["kl"]),
        "sigma": float(full_losses["sigma_mean"]),
        "h_mean": h_mean,
        "h_scaled": h_scaled,
        "best_epoch": best_epoch,
        "history": history,
    }


def hidden_true_substitution(model, visible, hidden_true, device="cpu"):
    x = torch.tensor(visible, dtype=torch.float32, device=device).unsqueeze(0)
    h_true = torch.tensor(hidden_true, dtype=torch.float32, device=device).unsqueeze(0)
    h_true_centered = h_true - h_true.mean()
    with torch.no_grad():
        enc_out = model.encoder(x)
        if len(enc_out) == 2:
            mu_k, _ = enc_out
            encoder_h = mu_k[..., 0]                        # (B, T)
        else:
            # MoG: (mu, log_sigma, logits) each (B, T, K, 1)
            mu_K, _, logits_K = enc_out
            mu_K = mu_K[..., 0]                              # (B, T, K)
            logits_K = logits_K[..., 0]
            pi = F.softmax(logits_K, dim=-1)                  # (B, T, K)
            encoder_h = (pi * mu_K).sum(-1)                  # (B, T) π-weighted
        encoder_h_centered = encoder_h - encoder_h.mean()
        encoder_std = encoder_h_centered.std()
        h_true_scaled = h_true_centered * (encoder_std / (h_true_centered.std() + 1e-6))
    safe = torch.clamp(x, min=1e-6)
    actual = torch.log(safe[:, 1:] / safe[:, :-1])
    actual = torch.clamp(actual, -2.5, 2.5)
    with torch.no_grad():
        base = model.compute_f_visible(x)
        G = model.compute_G(x)
        pred_encoder = base + encoder_h.unsqueeze(-1) * G
        pred_true = base + h_true_scaled.unsqueeze(-1) * G
        pred_null = base
    recon_encoder = F.mse_loss(pred_encoder[:, :-1], actual).item()
    recon_true = F.mse_loss(pred_true[:, :-1], actual).item()
    recon_null = F.mse_loss(pred_null[:, :-1], actual).item()
    return {"recon_encoder": recon_encoder, "recon_true_scaled": recon_true,
            "recon_null": recon_null}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_seeds", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--datasets", nargs="+", default=["portal", "lv"])
    args = parser.parse_args()

    _configure_matplotlib()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"runs/{ts}_cvhi_residual_L1L3")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_dir}\n")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    seeds = [42, 123, 456, 789, 2024, 31415, 27182, 65537][:args.n_seeds]
    print(f"Seeds: {seeds}\n")

    datasets = {}
    if "portal" in args.datasets:
        v, h = load_portal("OT")
        datasets["portal"] = {"visible": v, "hidden": h, "is_portal": True,
                               "name": "Portal OT"}
    if "lv" in args.datasets:
        v, h = load_lv()
        datasets["lv"] = {"visible": v, "hidden": h, "is_portal": False,
                           "name": "Synthetic LV"}

    all_results = {}
    for ds_key, ds in datasets.items():
        print(f"\n{'='*72}\nExp A+: multi-seed on {ds['name']} (with L1+L3)\n{'='*72}")
        results = []
        for seed in seeds:
            r = train_one(ds["visible"], ds["hidden"], device, seed,
                           epochs=args.epochs, is_portal=ds["is_portal"])
            results.append(r)
            rps = r.get("rollout_per_step", None)
            if rps is not None:
                rps_str = " ".join([f"{v:.3f}" for v in rps.tolist()])
            else:
                rps_str = "N/A"
            print(f"  seed {seed:5d}: P={r['pearson']:+.4f}  val={r['val_recon']:.4f}  "
                  f"m_null={r['m_null']:+.4f}  hf_frac={r['hf_frac']:.3f}  "
                  f"h_var={r['h_var']:.3f}  roll_per_step=[{rps_str}]")
        all_results[ds_key] = {"data": ds, "results": results}

    # Analysis
    analysis = {}
    for ds_key, rr in all_results.items():
        results = rr["results"]
        ds = rr["data"]
        print(f"\n{'='*72}\nAnalysis: {ds['name']}\n{'='*72}")

        pearsons = np.array([r["pearson"] for r in results])
        val_recons = np.array([r["val_recon"] for r in results])
        m_nulls = np.array([r["m_null"] for r in results])
        h_vars = np.array([r["h_var"] for r in results])
        hf_fracs = np.array([r["hf_frac"] for r in results])

        rho_val, p_val = spearmanr(val_recons, pearsons)
        rho_mnull, _ = spearmanr(m_nulls, pearsons)
        rho_hvar, _ = spearmanr(h_vars, pearsons)
        rho_hf, _ = spearmanr(hf_fracs, pearsons)

        print(f"  Spearman ρ (vs Pearson):")
        print(f"    val_recon : {rho_val:+.3f} (p={p_val:.3f})")
        print(f"    m_null    : {rho_mnull:+.3f}")
        print(f"    h_var     : {rho_hvar:+.3f}")
        print(f"    hf_frac   : {rho_hf:+.3f}")

        # Ensemble
        sort_by_val = sorted(range(len(results)), key=lambda i: results[i]["val_recon"])
        top_k_idx = sort_by_val[:args.top_k]
        top_k_h = [results[i]["h_mean"] for i in top_k_idx]
        h_ens = np.mean(top_k_h, axis=0)
        ens_pear, _ = evaluate(h_ens, ds["hidden"])
        top_k_p = [results[i]["pearson"] for i in top_k_idx]

        print(f"\n  Top-{args.top_k} by val_recon:")
        print(f"    top-K seeds: {[results[i]['seed'] for i in top_k_idx]}")
        print(f"    top-K Pearsons: {[f'{p:+.3f}' for p in top_k_p]}")
        print(f"    Ensemble Pearson: {ens_pear:+.4f}")

        # Exp D: best-val model's hidden_true substitution
        best_model = results[top_k_idx[0]]["model"]
        d_result = hidden_true_substitution(best_model, ds["visible"], ds["hidden"], device)
        ratio = d_result["recon_true_scaled"] / d_result["recon_encoder"]
        print(f"\n  Exp D on best-val seed:")
        print(f"    recon_null   : {d_result['recon_null']:.4f}")
        print(f"    recon_encoder: {d_result['recon_encoder']:.4f}")
        print(f"    recon_true   : {d_result['recon_true_scaled']:.4f}")
        print(f"    ratio (true/encoder): {ratio:.3f}  "
              f"{'[OK: dynamics accepts true h]' if ratio < 1.02 else '[still pseudo]'}")

        # Mean Pearson, max, stability
        print(f"\n  Overall: mean={pearsons.mean():+.4f}  max={pearsons.max():+.4f}  "
              f"std={pearsons.std():.4f}")

        analysis[ds_key] = {
            "rho_val": rho_val, "rho_mnull": rho_mnull,
            "rho_hvar": rho_hvar, "rho_hf": rho_hf,
            "ens_pearson": ens_pear,
            "top_k_mean_p": float(np.mean(top_k_p)),
            "d_ratio": ratio,
            "recon_true": d_result['recon_true_scaled'],
            "recon_encoder": d_result['recon_encoder'],
            "recon_null": d_result['recon_null'],
            "pearsons": pearsons.tolist(),
            "val_recons": val_recons.tolist(),
            "hf_fracs": hf_fracs.tolist(),
            "mean_pearson": float(pearsons.mean()),
            "max_pearson": float(pearsons.max()),
            "std_pearson": float(pearsons.std()),
        }

    # Plots
    def save_single(title, plot_fn, path, figsize=(13, 5)):
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        plot_fn(ax)
        fig.suptitle(title, fontsize=13, fontweight="bold")
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)

    for ds_key, rr in all_results.items():
        ds = rr["data"]; results = rr["results"]

        # h mean vs truth overlay
        def plot_overlay(ax, ds=ds, results=results):
            ht = ds["hidden"]
            t_axis = np.arange(len(ht))
            ax.plot(t_axis, ht, color="black", linewidth=2.0, label="真实 hidden", zorder=10)
            colors = plt.cm.viridis(np.linspace(0, 1, len(results)))
            for i, r in enumerate(results):
                h = r["h_scaled"]
                L = min(len(h), len(t_axis))
                ax.plot(t_axis[:L], h[:L], color=colors[i], linewidth=1.0, alpha=0.85,
                        label=f"seed {r['seed']} (P={r['pearson']:.3f})")
            ax.set_xlabel("time"); ax.set_ylabel("hidden")
            ax.legend(fontsize=9, ncol=2); ax.grid(alpha=0.25)
        save_single(f"{ds['name']} L1+L3: 跨 seed hidden 恢复",
                     plot_overlay, out_dir / f"fig_{ds_key}_overlay.png")

        # hf_frac over training for each seed
        def plot_hf_frac(ax, results=results):
            colors = plt.cm.viridis(np.linspace(0, 1, len(results)))
            for i, r in enumerate(results):
                ax.plot(r["history"]["hf_frac"], color=colors[i], linewidth=1.0, alpha=0.85,
                        label=f"seed {r['seed']}")
            ax.set_xlabel("epoch"); ax.set_ylabel("hf_frac (low=low-freq hidden)")
            ax.axhline(0.3, color="green", linestyle="--", alpha=0.5, label="target <0.3")
            ax.legend(fontsize=9, ncol=2); ax.grid(alpha=0.25)
        save_single(f"{ds['name']}: h 的高频能量比例 over training",
                     plot_hf_frac, out_dir / f"fig_{ds_key}_hf_frac.png")

    # Comparison with pre-L1L3 results
    pre_l1l3_results = {
        "portal": {"mean": 0.146, "max": 0.277, "std": 0.081,
                    "rho_val": 0.738, "ens_pearson": 0.114, "d_ratio": 1.104},
        "lv": {"mean": 0.689, "max": 0.915, "std": 0.159,
                "rho_val": 0.405, "ens_pearson": 0.603, "d_ratio": 3.010},
    }

    with open(out_dir / "summary.md", "w", encoding="utf-8") as f:
        f.write("# CVHI_Residual + L1+L3 诊断\n\n")
        f.write("L1 = 3-step rollout (schedule 0→3 grow), L3 = Gaussian low-pass filter on h (σ=6/8)\n\n")
        f.write("## 对比 pre-L1L3 vs post-L1L3\n\n")
        f.write("| 数据 | 指标 | pre (L1L3=off) | post (L1L3=on) | 变化 |\n")
        f.write("|---|---|---|---|---|\n")
        for ds_key in all_results:
            pre = pre_l1l3_results[ds_key]
            post = analysis[ds_key]
            for metric, pre_key, post_key in [
                ("mean P", "mean", "mean_pearson"),
                ("max P", "max", "max_pearson"),
                ("std P", "std", "std_pearson"),
                ("ρ(val_recon, P)", "rho_val", "rho_val"),
                ("ens P (top-3)", "ens_pearson", "ens_pearson"),
                ("d_ratio", "d_ratio", "d_ratio"),
            ]:
                name = ds_key.upper()
                f.write(f"| {name} | {metric} | {pre[pre_key]:+.3f} | {post[post_key]:+.3f} | "
                        f"{post[post_key] - pre[pre_key]:+.3f} |\n")
        f.write("\n## 详细结果\n\n")
        for ds_key, rr in all_results.items():
            results = rr["results"]; ds = rr["data"]
            f.write(f"### {ds['name']}\n\n")
            f.write(f"| seed | Pearson | val_recon | m_null | hf_frac | h_var |\n")
            f.write(f"|---|---|---|---|---|---|\n")
            for r in results:
                f.write(f"| {r['seed']} | {r['pearson']:+.4f} | {r['val_recon']:.4f} | "
                        f"{r['m_null']:+.4f} | {r['hf_frac']:.3f} | {r['h_var']:.3f} |\n")

    print(f"\n[OK] saved to: {out_dir}")


if __name__ == "__main__":
    main()
