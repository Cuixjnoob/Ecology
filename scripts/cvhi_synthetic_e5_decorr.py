"""E5: HSIC-style decorrelation loss 堵 Layer 2.

在训练中加 L_decorr = λ · Σ_f corr(h, feature_f)²
其中 feature_f 遍历 {x_j, x_j², x_j/(1+x_j)} for all j

目的: 强迫 encoder 的 h 和任何可观测 visible 函数解相关,
      阻止"抄 x_2"的作弊行为.

扫 λ ∈ {0, 0.05, 0.1, 0.3, 1.0}, LV + Holling, 3 seeds each.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from models.cvhi_residual import CVHI_Residual
from scripts.cvhi_residual_synthetic_comparison import load_lv, load_holling
from scripts.cvhi_residual_L1L3_diagnostics import evaluate, hidden_true_substitution


SEEDS = [42, 123, 456]
LAMBDAS = [0.0, 0.05, 0.1, 0.3, 1.0]
EPOCHS = 300


def make_model(N, device):
    return CVHI_Residual(
        num_visible=N,
        encoder_d=64, encoder_blocks=2, encoder_heads=4,
        takens_lags=(1, 2, 4, 8), encoder_dropout=0.1,
        d_species_f=20, f_visible_layers=2, f_visible_top_k=4,
        d_species_G=12, G_field_layers=1, G_field_top_k=3,
        prior_std=1.0, gnn_backbone="mlp",
        use_formula_hints=True, use_G_field=True,
        num_mixture_components=1,
        G_anchor_first=True, G_anchor_sign=+1,
        hierarchical_h=False,
    ).to(device)


def build_feature_bank(x_full):
    """x_full: (1, T, N). Return feature bank (1, T, F) with all x_j, x_j², x_j/(1+x_j)."""
    bs, T, N = x_full.shape
    feats = []
    for j in range(N):
        xj = x_full[..., j]            # (1, T)
        feats.append(xj)                # linear
        feats.append(xj ** 2)           # squared
        feats.append(xj / (1 + xj))     # saturation
    return torch.stack(feats, dim=-1)  # (1, T, 3N)


def decorr_loss(h_samples, feature_bank):
    """h_samples: (S, B, T) — posterior samples of h
       feature_bank: (B, T, F)

       Return: mean over samples of sum_f corr(h, feature_f)²
    """
    S, B, T = h_samples.shape
    F_n = feature_bank.shape[-1]
    # Center
    h_c = h_samples - h_samples.mean(dim=-1, keepdim=True)     # (S, B, T)
    f_c = feature_bank - feature_bank.mean(dim=-2, keepdim=True)  # (B, T, F)

    h_std = h_c.std(dim=-1, keepdim=True) + 1e-6                # (S, B, 1)
    f_std = f_c.std(dim=-2, keepdim=True) + 1e-6                # (B, 1, F)

    # corr(h, f) = <h_c, f_c> / (T * h_std * f_std)
    # h_c: (S, B, T), f_c: (B, T, F) -> einsum
    inner = torch.einsum("sbt,btf->sbf", h_c, f_c) / T          # (S, B, F)
    corr = inner / (h_std * f_std)                              # (S, B, F)
    return (corr ** 2).sum(dim=-1).mean()                       # scalar


def train_with_decorr(visible, hidden, seed, device, lam_decorr, epochs=EPOCHS):
    torch.manual_seed(seed)
    T, N = visible.shape
    x = torch.tensor(visible, dtype=torch.float32, device=device).unsqueeze(0)

    model = make_model(N, device)
    opt = torch.optim.AdamW(model.parameters(), lr=0.0008, weight_decay=1e-4)
    warmup = int(0.2 * epochs)
    ramp = max(1, int(0.2 * epochs))
    def lr_lambda(step):
        if step < 50: return step / 50
        p = (step - 50) / max(1, epochs - 50)
        return 0.5 * (1 + np.cos(np.pi * p))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    train_end = int(0.75 * T)
    best_val = float("inf"); best_state = None

    # Precompute feature bank (x is constant, features don't change)
    feat_bank_full = build_feature_bank(x)  # (1, T, 3N)
    feat_bank_train = feat_bank_full[:, :train_end]

    for epoch in range(epochs):
        if epoch < warmup:
            h_w, K_r, lam_r = 0.0, 0, 0.0
        else:
            post = epoch - warmup
            h_w = min(1.0, post / ramp)
            k_ramp = min(1.0, post / max(1, epochs - warmup) * 2)
            K_r = max(1 if h_w > 0 else 0, int(round(k_ramp * 3)))
            lam_r = 0.5 * h_w

        # Input dropout aug
        if epoch > warmup:
            mask = (torch.rand(1, T, 1, device=device) > 0.05).float()
            x_train = x * mask + (1 - mask) * x.mean(dim=1, keepdim=True)
        else:
            x_train = x

        model.train(); opt.zero_grad()
        out = model(x_train, n_samples=2, rollout_K=K_r)
        tr_out = model.slice_out(out, 0, train_end)
        tr_out["visible"] = out["visible"][:, 0:train_end]
        tr_out["G"] = out["G"][:, 0:train_end]

        losses = model.loss(
            tr_out, beta_kl=0.03, free_bits=0.02,
            margin_null=0.003, margin_shuf=0.002,
            lam_necessary=5.0, lam_shuffle=3.0,
            lam_energy=2.0, min_energy=0.02,
            lam_smooth=0.02, lam_sparse=0.02,
            h_weight=h_w, lam_rollout=lam_r,
            rollout_weights=(1.0, 0.5, 0.25),
            lam_hf=0.0, lowpass_sigma=6.0,
            lam_rmse_log=0.1,
        )
        total = losses["total"]

        # === E5 decorrelation loss ===
        if lam_decorr > 0 and h_w > 0:
            # h_samples: (S=2, B=1, T_train) from the slice
            h_s = tr_out["h_samples"]  # (S, B, T_train)
            L_dec = decorr_loss(h_s, feat_bank_train)
            total = total + lam_decorr * h_w * L_dec

        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()

        with torch.no_grad():
            val_out = model.slice_out(out, train_end, T)
            val_out["visible"] = out["visible"][:, train_end:T]
            val_out["G"] = out["G"][:, train_end:T]
            val_losses = model.loss(
                val_out, h_weight=1.0,
                margin_null=0.003, margin_shuf=0.002,
                lam_energy=2.0, min_energy=0.02,
                lam_rollout=0.5, rollout_weights=(1.0, 0.5, 0.25),
                lam_hf=0.0, lowpass_sigma=6.0,
            )
            val_recon = val_losses["recon_full"].item()
        if epoch > warmup + 15 and val_recon < best_val:
            best_val = val_recon
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        out_eval = model(x, n_samples=30, rollout_K=3)
        h_mean = out_eval["h_samples"].mean(dim=0)[0].cpu().numpy()
    pear, _ = evaluate(h_mean, hidden)
    d = hidden_true_substitution(model, visible, hidden, device)

    # Also measure: current corr(h, x_j/(1+x_j)) for diagnosis
    corr_with_sats = {}
    for j in range(N):
        x_sat = visible[:, j] / (1 + visible[:, j])
        hh = h_mean - h_mean.mean()
        ss = x_sat - x_sat.mean()
        corr_with_sats[f"x_{j}_sat"] = float(
            (hh * ss).sum() / (np.sqrt((hh*hh).sum() * (ss*ss).sum()) + 1e-12))

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "pearson": pear,
        "d_ratio": d["recon_true_scaled"] / d["recon_encoder"],
        "val_recon": best_val,
        "corr_sats": corr_with_sats,
    }


def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"runs/{ts}_e5_decorr")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_dir}")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    lv_vis, lv_hid = load_lv()
    ho_vis, ho_hid = load_holling()

    datasets = {"LV": (lv_vis, lv_hid), "Holling": (ho_vis, ho_hid)}
    # results[ds][lam] = [list of per-seed dicts]
    results = {ds: {lam: [] for lam in LAMBDAS} for ds in datasets}

    total_runs = len(datasets) * len(LAMBDAS) * len(SEEDS)
    run_i = 0
    for ds_name, (vis, hid) in datasets.items():
        for lam in LAMBDAS:
            for seed in SEEDS:
                run_i += 1
                print(f"\n[{run_i}/{total_runs}] ds={ds_name}  lam={lam}  seed={seed}")
                t0 = datetime.now()
                r = train_with_decorr(vis, hid, seed, device, lam)
                dt = (datetime.now() - t0).total_seconds()
                print(f"  P={r['pearson']:+.4f}  d_ratio={r['d_ratio']:.2f}  "
                      f"val={r['val_recon']:.4f}  ({dt:.1f}s)")
                r["seed"] = seed; r["lam"] = lam; r["dataset"] = ds_name
                results[ds_name][lam].append(r)

    # Summary
    print(f"\n{'='*80}\nSUMMARY: E5 λ-scan Pearson\n{'='*80}")
    print(f"{'Dataset':<12}{'λ':<10}{'mean P':<12}{'max P':<12}{'std':<10}{'mean d_r':<12}")
    for ds_name in datasets:
        for lam in LAMBDAS:
            rs = results[ds_name][lam]
            P = np.array([r["pearson"] for r in rs])
            D = np.array([r["d_ratio"] for r in rs])
            print(f"{ds_name:<12}{lam:<10.3f}{P.mean():<+12.4f}{P.max():<+12.4f}"
                  f"{P.std():<10.4f}{D.mean():<12.2f}")

    # Plot: Pearson vs λ
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    for ax, ds_name in zip(axes, datasets):
        means = []; stds = []; maxes = []
        for lam in LAMBDAS:
            rs = results[ds_name][lam]
            P = np.array([r["pearson"] for r in rs])
            means.append(P.mean()); stds.append(P.std()); maxes.append(P.max())
        means = np.array(means); stds = np.array(stds); maxes = np.array(maxes)
        ax.errorbar(LAMBDAS, means, yerr=stds, marker="o", lw=2, capsize=5,
                    label="mean ± std", color="#1976d2")
        ax.plot(LAMBDAS, maxes, marker="^", linestyle="--", color="#c62828",
                label="max", alpha=0.8)
        ax.set_xscale("symlog", linthresh=0.01)
        ax.set_xlabel("λ (decorr strength)")
        ax.set_ylabel("Pearson")
        ax.set_title(f"{ds_name}: Pearson vs decorr λ")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle("E5: Decorrelation λ scan", fontweight="bold")
    fig.savefig(out_dir / "E5_lambda_scan.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # d_ratio plot
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for ds_name, color in zip(datasets, ["#1976d2", "#c62828"]):
        means = []
        for lam in LAMBDAS:
            rs = results[ds_name][lam]
            D = np.array([r["d_ratio"] for r in rs])
            means.append(D.mean())
        ax.plot(LAMBDAS, means, marker="o", lw=2, label=ds_name, color=color)
    ax.set_xscale("symlog", linthresh=0.01)
    ax.set_xlabel("λ"); ax.set_ylabel("mean d_ratio")
    ax.axhline(1.0, color="k", ls=":", alpha=0.5, label="ideal (=1)")
    ax.legend(); ax.grid(alpha=0.3)
    ax.set_title("d_ratio vs λ")
    fig.savefig(out_dir / "E5_dratio_scan.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # Write summary
    with open(out_dir / "summary.md", "w", encoding="utf-8") as f:
        f.write("# E5: Decorrelation λ-scan\n\n")
        f.write(f"Seeds: {SEEDS}, epochs={EPOCHS}\n\n")
        f.write("Feature bank: {x_j, x_j², x_j/(1+x_j)} for all j\n\n")
        f.write("L_decorr = λ · Σ_f corr(h, feature_f)² · h_weight(epoch)\n\n")
        f.write("## Results\n\n")
        f.write("| Dataset | λ | mean P | max P | std P | mean d_ratio |\n")
        f.write("|---|---|---|---|---|---|\n")
        for ds_name in datasets:
            for lam in LAMBDAS:
                rs = results[ds_name][lam]
                P = np.array([r["pearson"] for r in rs])
                D = np.array([r["d_ratio"] for r in rs])
                f.write(f"| {ds_name} | {lam:.3f} | {P.mean():+.4f} | {P.max():+.4f} | "
                        f"{P.std():.4f} | {D.mean():.2f} |\n")
        f.write("\n## Best λ per dataset\n\n")
        for ds_name in datasets:
            best_lam = max(LAMBDAS, key=lambda l: np.mean([r["pearson"] for r in results[ds_name][l]]))
            best_P = np.mean([r["pearson"] for r in results[ds_name][best_lam]])
            base_P = np.mean([r["pearson"] for r in results[ds_name][0.0]])
            f.write(f"- **{ds_name}**: best λ={best_lam:.3f}, mean P={best_P:+.4f} "
                    f"(vs λ=0 baseline {base_P:+.4f}, Δ={best_P - base_P:+.4f})\n")

    # JSON dump
    dump = {ds: {str(lam): results[ds][lam] for lam in LAMBDAS} for ds in datasets}
    with open(out_dir / "raw.json", "w") as f:
        json.dump(dump, f, indent=2, default=float)

    print(f"\n[OK] {out_dir}")


if __name__ == "__main__":
    main()
