"""E6: Causal species dropout — 堵 Layer 3.

机制: 每 epoch 随机把一个 visible 物种完全置零 (替换为该物种的时间均值),
      让 encoder 的 h 必须对任意单一物种的丢失鲁棒.
      如果 h 是"纯抄 x_2", 抄不到就崩.
      如果 h 是"从整体动力学推出", 单个丢失影响小.

加 consistency loss: λ · ||h_masked − h_full||²  (按 std 归一)

扫 λ ∈ {0, 0.1, 0.5, 2.0, 10.0}, LV + Holling, 3 seeds.
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
LAMBDAS = [0.0, 0.1, 0.5, 2.0, 10.0]
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


def encoder_h(model, x):
    """Extract encoder mean h (no sampling)."""
    enc_out = model.encoder(x)
    if len(enc_out) == 2:
        mu_k, _ = enc_out
        return mu_k[..., 0]      # (B, T)
    else:
        mu_K, _, logits_K = enc_out
        mu_K = mu_K[..., 0]; logits_K = logits_K[..., 0]
        pi = F.softmax(logits_K, dim=-1)
        return (pi * mu_K).sum(-1)


def train_with_causal_dropout(visible, hidden, seed, device, lam_cd, epochs=EPOCHS):
    torch.manual_seed(seed)
    T, N = visible.shape
    x_full = torch.tensor(visible, dtype=torch.float32, device=device).unsqueeze(0)
    x_mean_per_species = x_full.mean(dim=1, keepdim=True)  # (1, 1, N)

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

    for epoch in range(epochs):
        if epoch < warmup:
            h_w, K_r, lam_r = 0.0, 0, 0.0
        else:
            post = epoch - warmup
            h_w = min(1.0, post / ramp)
            k_ramp = min(1.0, post / max(1, epochs - warmup) * 2)
            K_r = max(1 if h_w > 0 else 0, int(round(k_ramp * 3)))
            lam_r = 0.5 * h_w

        # Input dropout aug (temporal, keep from baseline)
        if epoch > warmup:
            mask_t = (torch.rand(1, T, 1, device=device) > 0.05).float()
            x_train = x_full * mask_t + (1 - mask_t) * x_full.mean(dim=1, keepdim=True)
        else:
            x_train = x_full

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

        # === E6 causal dropout: mask one random species ===
        if lam_cd > 0 and h_w > 0:
            j_drop = int(torch.randint(0, N, (1,)).item())
            # Replace species j with its time-mean
            x_mask = x_train.clone()
            x_mask[..., j_drop] = x_mean_per_species[..., j_drop]
            # Get encoder h on masked input
            h_masked = encoder_h(model, x_mask)  # (1, T)
            # Get encoder h on full input (no sampling noise, deterministic mean)
            with torch.no_grad():
                h_full_frozen = encoder_h(model, x_train).detach()
            # Normalize both by std to be scale-invariant
            h_m_c = h_masked - h_masked.mean(dim=-1, keepdim=True)
            h_f_c = h_full_frozen - h_full_frozen.mean(dim=-1, keepdim=True)
            # Cosine-based consistency (scale-invariant)
            cos = (h_m_c * h_f_c).sum(-1) / (h_m_c.norm(dim=-1) * h_f_c.norm(dim=-1) + 1e-6)
            L_cd = (1 - cos).mean()
            total = total + lam_cd * h_w * L_cd

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
        out_eval = model(x_full, n_samples=30, rollout_K=3)
        h_mean = out_eval["h_samples"].mean(dim=0)[0].cpu().numpy()
    pear, _ = evaluate(h_mean, hidden)
    d = hidden_true_substitution(model, visible, hidden, device)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"pearson": pear,
            "d_ratio": d["recon_true_scaled"] / d["recon_encoder"],
            "val_recon": best_val}


def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"runs/{ts}_e6_causal_dropout")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_dir}")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    lv_vis, lv_hid = load_lv()
    ho_vis, ho_hid = load_holling()

    datasets = {"LV": (lv_vis, lv_hid), "Holling": (ho_vis, ho_hid)}
    results = {ds: {lam: [] for lam in LAMBDAS} for ds in datasets}

    total_runs = len(datasets) * len(LAMBDAS) * len(SEEDS)
    run_i = 0
    for ds_name, (vis, hid) in datasets.items():
        for lam in LAMBDAS:
            for seed in SEEDS:
                run_i += 1
                print(f"\n[{run_i}/{total_runs}] ds={ds_name}  lam={lam}  seed={seed}")
                t0 = datetime.now()
                r = train_with_causal_dropout(vis, hid, seed, device, lam)
                dt = (datetime.now() - t0).total_seconds()
                print(f"  P={r['pearson']:+.4f}  d_ratio={r['d_ratio']:.2f}  "
                      f"val={r['val_recon']:.4f}  ({dt:.1f}s)")
                r["seed"] = seed; r["lam"] = lam; r["dataset"] = ds_name
                results[ds_name][lam].append(r)

    # Summary
    print(f"\n{'='*80}\nSUMMARY: E6 causal-dropout λ-scan\n{'='*80}")
    print(f"{'Dataset':<12}{'λ':<10}{'mean P':<12}{'max P':<12}{'std':<10}{'mean d_r':<12}")
    for ds_name in datasets:
        for lam in LAMBDAS:
            rs = results[ds_name][lam]
            P = np.array([r["pearson"] for r in rs])
            D = np.array([r["d_ratio"] for r in rs])
            print(f"{ds_name:<12}{lam:<10.2f}{P.mean():<+12.4f}{P.max():<+12.4f}"
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
        ax.set_xscale("symlog", linthresh=0.1)
        ax.set_xlabel("λ (causal dropout consistency)")
        ax.set_ylabel("Pearson")
        ax.set_title(f"{ds_name}")
        ax.grid(alpha=0.3); ax.legend()
    fig.suptitle("E6: Causal-dropout λ scan", fontweight="bold")
    fig.savefig(out_dir / "E6_lambda_scan.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # d_ratio
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for ds_name, color in zip(datasets, ["#1976d2", "#c62828"]):
        means = []
        for lam in LAMBDAS:
            rs = results[ds_name][lam]
            D = np.array([r["d_ratio"] for r in rs])
            means.append(D.mean())
        ax.plot(LAMBDAS, means, marker="o", lw=2, label=ds_name, color=color)
    ax.set_xscale("symlog", linthresh=0.1)
    ax.set_xlabel("λ"); ax.set_ylabel("mean d_ratio")
    ax.axhline(1.0, color="k", ls=":", alpha=0.5, label="ideal (=1)")
    ax.legend(); ax.grid(alpha=0.3)
    ax.set_title("E6: d_ratio vs λ")
    fig.savefig(out_dir / "E6_dratio_scan.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    with open(out_dir / "summary.md", "w", encoding="utf-8") as f:
        f.write("# E6: Causal species dropout — λ scan\n\n")
        f.write(f"Seeds: {SEEDS}, epochs={EPOCHS}\n\n")
        f.write("Mechanism: each epoch mask one random species in encoder input, ")
        f.write("force cos_sim(h_masked, h_full) → 1.\n\n")
        f.write("## Results\n\n")
        f.write("| Dataset | λ | mean P | max P | std | mean d_ratio |\n")
        f.write("|---|---|---|---|---|---|\n")
        for ds_name in datasets:
            for lam in LAMBDAS:
                rs = results[ds_name][lam]
                P = np.array([r["pearson"] for r in rs])
                D = np.array([r["d_ratio"] for r in rs])
                f.write(f"| {ds_name} | {lam:.2f} | {P.mean():+.4f} | {P.max():+.4f} | "
                        f"{P.std():.4f} | {D.mean():.2f} |\n")
        f.write("\n## Best λ\n\n")
        for ds_name in datasets:
            best_lam = max(LAMBDAS, key=lambda l: np.mean([r["pearson"] for r in results[ds_name][l]]))
            best_P = np.mean([r["pearson"] for r in results[ds_name][best_lam]])
            base_P = np.mean([r["pearson"] for r in results[ds_name][0.0]])
            f.write(f"- **{ds_name}**: best λ={best_lam:.2f}, mean P={best_P:+.4f} "
                    f"(baseline {base_P:+.4f}, Δ={best_P - base_P:+.4f})\n")

    dump = {ds: {str(lam): results[ds][lam] for lam in LAMBDAS} for ds in datasets}
    with open(out_dir / "raw.json", "w") as f:
        json.dump(dump, f, indent=2, default=float)
    print(f"\n[OK] {out_dir}")


if __name__ == "__main__":
    main()
