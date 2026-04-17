"""h-dynamics self-consistency loss experiment on Huisman 1999.

Key idea: add constraint that h follows its own dynamics:
  h_{t+1} ≈ g(h_t, x_t)   where g is a small MLP

This compresses h's effective DoF from T (free scalars) down to
(initial cond + g params), making it much harder for encoder to
use h as a noise absorber.

Red line: g is learnable, trained on encoder's own h, no hidden_true.

Scan λ_h_dyn ∈ {0.0, 0.1, 0.3, 1.0} on Huisman chaos.
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
import torch.nn as nn
import torch.nn.functional as F

from models.cvhi_residual import CVHI_Residual
from scripts.cvhi_residual_L1L3_diagnostics import evaluate, hidden_true_substitution


SEEDS = [42, 123, 456]
LAMBDAS = [0.0, 0.1, 0.3, 1.0]
EPOCHS = 300
DETACH_UNTIL = 100   # first 100 epochs: g learns encoder's h w/o backprop to encoder
# Test species (representative mix: high/mid/low Pearson in baseline)
TEST_SPECIES_IDX = [0, 1, 3]   # sp1 (low), sp2 (high), sp4 (mid)
TEST_SPECIES_NAMES = ["sp1", "sp2", "sp4"]


class HiddenDynamicsNet(nn.Module):
    """Small MLP: (h_t, x_t) → ĥ_{t+1} prediction."""
    def __init__(self, n_visible, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1 + n_visible, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, h_t, x_t):
        """h_t: (B, T) or (B,), x_t: (B, T, N) or (B, N). Returns (B, T) or (B,)."""
        if h_t.dim() == 1:
            h_t = h_t.unsqueeze(-1)                  # (B, 1)
        elif h_t.dim() == 2:                          # (B, T)
            h_t = h_t.unsqueeze(-1)                   # (B, T, 1)
        inp = torch.cat([h_t, x_t], dim=-1)
        return self.net(inp).squeeze(-1)


def load_huisman():
    d = np.load("runs/huisman1999_chaos/trajectories.npz")
    full = np.concatenate([d["N_all"], d["resources"]], axis=1)
    full = full / (full.mean(axis=0, keepdims=True) + 1e-8)
    return full.astype(np.float32)


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


def train_with_hdyn(visible, hidden, seed, device, lam_hdyn, epochs=EPOCHS):
    torch.manual_seed(seed)
    T, N = visible.shape
    x_full = torch.tensor(visible, dtype=torch.float32, device=device).unsqueeze(0)

    model = make_model(N, device)
    h_dyn_net = HiddenDynamicsNet(n_visible=N, hidden_dim=32).to(device)

    params = list(model.parameters()) + list(h_dyn_net.parameters())
    opt = torch.optim.AdamW(params, lr=0.0008, weight_decay=1e-4)
    warmup = int(0.2 * epochs)
    ramp = max(1, int(0.2 * epochs))
    def lr_lambda(step):
        if step < 50: return step / 50
        p = (step - 50) / max(1, epochs - 50)
        return 0.5 * (1 + np.cos(np.pi * p))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    train_end = int(0.75 * T)
    best_val = float("inf"); best_state = None

    hdyn_loss_hist = []
    for epoch in range(epochs):
        if epoch < warmup:
            h_w, K_r, lam_r = 0.0, 0, 0.0
        else:
            post = epoch - warmup
            h_w = min(1.0, post / ramp)
            k_ramp = min(1.0, post / max(1, epochs - warmup) * 2)
            K_r = max(1 if h_w > 0 else 0, int(round(k_ramp * 3)))
            lam_r = 0.5 * h_w

        if epoch > warmup:
            mask = (torch.rand(1, T, 1, device=device) > 0.05).float()
            x_train = x_full * mask + (1 - mask) * x_full.mean(dim=1, keepdim=True)
        else:
            x_train = x_full

        model.train(); h_dyn_net.train(); opt.zero_grad()
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

        # === h-dynamics consistency loss ===
        if lam_hdyn > 0 and h_w > 0:
            # encoder's h mean, (S, B, T_train)
            h_samples = tr_out["h_samples"]
            h_mean_post = h_samples.mean(dim=0)        # (B, T_train)
            # x_train slice matching h
            T_h = h_mean_post.shape[-1]
            x_slice = x_train[:, :T_h]                 # (B, T_h, N)
            # Predict h_{t+1} from (h_t, x_t)
            h_prev = h_mean_post[:, :-1]               # (B, T_h-1)
            x_prev = x_slice[:, :-1]                   # (B, T_h-1, N)
            h_next_pred = h_dyn_net(h_prev, x_prev)    # (B, T_h-1)
            h_next_actual = h_mean_post[:, 1:]         # (B, T_h-1)

            if epoch < DETACH_UNTIL + warmup:
                # g learns encoder's h pattern (no gradient to encoder)
                h_next_target = h_next_actual.detach()
            else:
                # joint optimization: encoder pulled toward dynamics consistency
                h_next_target = h_next_actual

            L_hdyn = F.mse_loss(h_next_pred, h_next_target)
            total = total + lam_hdyn * h_w * L_hdyn
            hdyn_loss_hist.append(float(L_hdyn.item()))

        total.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
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

    # Diagnostic: h_dynamics consistency (how well g predicts encoder's h)
    with torch.no_grad():
        h_mean_t = torch.tensor(h_mean, dtype=torch.float32, device=device).unsqueeze(0)
        x_t = x_full[:, :len(h_mean)]
        h_pred_next = h_dyn_net(h_mean_t[:, :-1], x_t[:, :-1])
        hdyn_corr = F.cosine_similarity(h_pred_next.flatten(),
                                          h_mean_t[0, 1:].flatten(), dim=0).item()

    del model, h_dyn_net
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "pearson": pear,
        "d_ratio": d["recon_true_scaled"] / d["recon_encoder"],
        "val_recon": best_val,
        "h_mean": h_mean,
        "hdyn_final_loss": float(hdyn_loss_hist[-1]) if hdyn_loss_hist else float("nan"),
        "hdyn_consistency_corr": hdyn_corr,
    }


def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"runs/{ts}_huisman_hdyn")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_dir}")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    full = load_huisman()
    print(f"Huisman data: T={full.shape[0]}, channels={full.shape[1]}")
    print(f"Test species: {TEST_SPECIES_NAMES} (indices {TEST_SPECIES_IDX})")
    print(f"Lambdas: {LAMBDAS}, Seeds: {SEEDS}\n")

    results = {name: {lam: [] for lam in LAMBDAS} for name in TEST_SPECIES_NAMES}
    total_runs = len(TEST_SPECIES_NAMES) * len(LAMBDAS) * len(SEEDS)
    run_i = 0

    for sp_idx, sp_name in zip(TEST_SPECIES_IDX, TEST_SPECIES_NAMES):
        visible = np.delete(full, sp_idx, axis=1)
        hidden = full[:, sp_idx]
        for lam in LAMBDAS:
            for seed in SEEDS:
                run_i += 1
                print(f"[{run_i}/{total_runs}] {sp_name}  lam={lam}  seed={seed}")
                t0 = datetime.now()
                try:
                    r = train_with_hdyn(visible, hidden, seed, device, lam)
                    dt = (datetime.now() - t0).total_seconds()
                    print(f"  P={r['pearson']:+.4f}  d_r={r['d_ratio']:.2f}  "
                          f"hdyn_corr={r['hdyn_consistency_corr']:+.3f}  ({dt:.1f}s)")
                except Exception as e:
                    print(f"  FAILED: {e}")
                    import traceback; traceback.print_exc()
                    r = {"pearson": float("nan"), "d_ratio": float("nan"),
                         "val_recon": float("nan"), "h_mean": None,
                         "hdyn_final_loss": float("nan"),
                         "hdyn_consistency_corr": float("nan")}
                r["seed"] = seed; r["lam"] = lam; r["species"] = sp_name
                results[sp_name][lam].append(r)

    # Summary
    print(f"\n{'='*80}\nH-DYNAMICS λ SCAN SUMMARY\n{'='*80}")
    print(f"{'Species':<10}{'λ':<8}{'mean P':<12}{'max P':<12}{'std':<10}{'mean d_r':<10}{'hdyn_corr':<10}")
    for name in TEST_SPECIES_NAMES:
        for lam in LAMBDAS:
            rs = results[name][lam]
            P = np.array([r["pearson"] for r in rs if not np.isnan(r["pearson"])])
            D = np.array([r["d_ratio"] for r in rs if not np.isnan(r["d_ratio"])])
            C = np.array([r["hdyn_consistency_corr"] for r in rs
                           if not np.isnan(r.get("hdyn_consistency_corr", np.nan))])
            print(f"{name:<10}{lam:<8.2f}{P.mean():<+12.4f}{P.max():<+12.4f}"
                  f"{P.std():<10.4f}{D.mean():<10.2f}{C.mean():<+10.3f}")

    # Plot: Pearson vs λ
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    for ax, name in zip(axes, TEST_SPECIES_NAMES):
        means = [np.mean([r["pearson"] for r in results[name][lam]]) for lam in LAMBDAS]
        stds = [np.std([r["pearson"] for r in results[name][lam]]) for lam in LAMBDAS]
        maxes = [np.max([r["pearson"] for r in results[name][lam]]) for lam in LAMBDAS]
        ax.errorbar(LAMBDAS, means, yerr=stds, marker="o", lw=2, capsize=5,
                    color="#1976d2", label="mean ± std")
        ax.plot(LAMBDAS, maxes, marker="^", linestyle="--", color="#c62828",
                label="max", alpha=0.8)
        ax.set_xscale("symlog", linthresh=0.1)
        ax.set_xlabel("λ_h_dyn")
        ax.set_ylabel("Pearson")
        ax.set_title(f"{name}")
        ax.grid(alpha=0.3); ax.legend()
    fig.suptitle("h-dynamics consistency: Pearson vs λ (Huisman chaos)",
                 fontweight="bold")
    fig.savefig(out_dir / "fig_lambda_scan.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # Plot: overlay true vs recovered h for best lambda per species
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), constrained_layout=True)
    for ax, sp_idx, sp_name in zip(axes, TEST_SPECIES_IDX, TEST_SPECIES_NAMES):
        true_h = full[:, sp_idx]
        t = np.arange(len(true_h))
        ax.plot(t, true_h, color="black", lw=2, label="true", zorder=10)
        # find best lambda (highest mean Pearson)
        best_lam = max(LAMBDAS, key=lambda l: np.mean([r["pearson"]
                                                       for r in results[sp_name][l]]))
        best_rs = results[sp_name][best_lam]
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(best_rs)))
        for r, c in zip(best_rs, colors):
            h = r.get("h_mean")
            if h is None:
                continue
            L = min(len(h), len(true_h))
            a, b = np.polyfit(h[:L], true_h[:L], 1)
            ax.plot(t[:L], a * h[:L] + b, color=c, lw=1.0, alpha=0.8,
                    label=f"seed {r['seed']}  P={r['pearson']:+.3f}")
        mean_P_best = np.mean([r["pearson"] for r in best_rs])
        ax.set_title(f"{sp_name}  (λ*={best_lam}, mean P={mean_P_best:+.3f})")
        ax.set_ylabel(sp_name); ax.legend(fontsize=8, ncol=5); ax.grid(alpha=0.25)
    axes[-1].set_xlabel("time")
    fig.suptitle("Best λ_h_dyn: recovered h vs true", fontweight="bold")
    fig.savefig(out_dir / "fig_recovery.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # Save summary
    with open(out_dir / "summary.md", "w", encoding="utf-8") as f:
        f.write("# h-dynamics λ scan on Huisman 1999\n\n")
        f.write(f"Seeds: {SEEDS}, Epochs: {EPOCHS}, Test species: {TEST_SPECIES_NAMES}\n\n")
        f.write("New loss: L_h_dyn = MSE(encoder_h[1:], g(encoder_h[:-1], x[:-1]))\n")
        f.write("g: 2-layer MLP, hidden_dim=32. First 100 epochs: g learns w/o gradient "
                "to encoder; after: joint.\n\n")
        f.write("## Results\n\n")
        f.write("| Species | λ | mean P | max P | std | d_ratio | hdyn_corr |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for name in TEST_SPECIES_NAMES:
            for lam in LAMBDAS:
                rs = results[name][lam]
                P = np.array([r["pearson"] for r in rs if not np.isnan(r["pearson"])])
                D = np.array([r["d_ratio"] for r in rs if not np.isnan(r["d_ratio"])])
                C = np.array([r["hdyn_consistency_corr"] for r in rs
                               if not np.isnan(r.get("hdyn_consistency_corr", np.nan))])
                f.write(f"| {name} | {lam:.2f} | {P.mean():+.4f} | {P.max():+.4f} | "
                        f"{P.std():.4f} | {D.mean():.2f} | {C.mean():+.3f} |\n")

    # JSON (strip h_mean for size)
    dump = {name: {str(lam): [{k: (float(v) if isinstance(v, (int, float, np.floating))
                                     else v.tolist() if isinstance(v, np.ndarray)
                                     else v)
                                 for k, v in r.items() if k != "h_mean"}
                                for r in rs]
                   for lam, rs in d_lam.items()}
             for name, d_lam in results.items()}
    with open(out_dir / "raw.json", "w") as f:
        json.dump(dump, f, indent=2, default=float)
    print(f"\n[OK] {out_dir}")


if __name__ == "__main__":
    main()
