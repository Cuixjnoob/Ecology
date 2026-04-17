"""V2: h-dynamics with propagation + mixing (strong identifiability constraint).

V1 weakness: g just tracks encoder's h (MSE auxiliary), visible prediction
  still uses encoder's h directly → encoder can output arbitrary smooth h
  and g will fit it trivially. No real constraint on identifiability.

V2 fix: h_prop is built by propagating g from h_enc[0], then MIXED with
  h_enc in visible prediction:
    h_mix(α) = α·h_enc + (1-α)·h_prop
    visible_pred = f(x) + h_mix · G(x)

  α anneals from 1.0 (all encoder, warmup) → 0.3 (mostly propagated).

  If encoder cheats (outputs compensation), h_prop can't reproduce it
  → visible recon fails → encoder is pulled to match propagatable trajectories.

Design:
  - segment_len=20 (propagate in short chunks to avoid chaos explosion)
  - h_prop resets from h_enc at each segment start
  - Standard Huisman Lyapunov horizon ≈ 1/λ = 1/0.04 = 25 days = 12 steps
  - segment_len=15-20 sits just below horizon
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
LAMBDAS = [0.0, 0.3, 0.7, 1.0]   # weight on dynamics (when 0, h_mix = h_enc always)
EPOCHS = 400   # longer to give annealing room
SEGMENT_LEN = 15
TEST_SPECIES_IDX = [0, 1, 3]
TEST_SPECIES_NAMES = ["sp1", "sp2", "sp4"]


class HiddenDynamicsNet(nn.Module):
    """Small MLP: (h_t, x_t) → h_{t+1}."""
    def __init__(self, n_visible, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1 + n_visible, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def step(self, h_t, x_t):
        """h_t: (B,), x_t: (B, N). Returns (B,)."""
        inp = torch.cat([h_t.unsqueeze(-1), x_t], dim=-1)
        return self.net(inp).squeeze(-1)


def propagate_segmented(h_start, x, g, segment_len=15):
    """Propagate h through g in segments, resetting h at each segment start.

    h_start: (B, T) encoder's h
    x: (B, T, N)
    Returns h_prop: (B, T) with h_prop[t] resetting to h_start at segment boundaries.
    """
    B, T, N = x.shape
    h_prop = torch.zeros_like(h_start)

    for seg_start in range(0, T, segment_len):
        seg_end = min(seg_start + segment_len, T)
        # Reset h at segment start from encoder
        h_prop[:, seg_start] = h_start[:, seg_start]
        # Propagate within segment
        for t in range(seg_start + 1, seg_end):
            h_prop[:, t] = g.step(h_prop[:, t - 1], x[:, t - 1])

    return h_prop


def get_alpha(epoch, total_epochs, warmup_frac=0.25, anneal_end_frac=0.75,
              alpha_final=0.3):
    """α annealing: warmup → linear → constant.
    epoch 0 to warmup: α = 1.0
    warmup to anneal_end: α = 1.0 → alpha_final
    after anneal_end: α = alpha_final
    """
    w = int(warmup_frac * total_epochs)
    e = int(anneal_end_frac * total_epochs)
    if epoch < w:
        return 1.0
    elif epoch < e:
        return 1.0 - (1.0 - alpha_final) * (epoch - w) / (e - w)
    else:
        return alpha_final


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


def train_v2(visible, hidden, seed, device, lam_prop, epochs=EPOCHS):
    """lam_prop: weight for propagation loss (0 = disabled, baseline)."""
    torch.manual_seed(seed)
    T, N = visible.shape
    x_full = torch.tensor(visible, dtype=torch.float32, device=device).unsqueeze(0)

    model = make_model(N, device)
    g = HiddenDynamicsNet(n_visible=N, hidden_dim=32).to(device)

    params = list(model.parameters()) + list(g.parameters())
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

    alpha_hist = []
    prop_loss_hist = []

    for epoch in range(epochs):
        if epoch < warmup:
            h_w, K_r, lam_r = 0.0, 0, 0.0
        else:
            post = epoch - warmup
            h_w = min(1.0, post / ramp)
            k_ramp = min(1.0, post / max(1, epochs - warmup) * 2)
            K_r = max(1 if h_w > 0 else 0, int(round(k_ramp * 3)))
            lam_r = 0.5 * h_w

        # Alpha annealing (only engages when lam_prop > 0)
        if lam_prop > 0:
            alpha = get_alpha(epoch, epochs)
        else:
            alpha = 1.0   # pure baseline
        alpha_hist.append(alpha)

        if epoch > warmup:
            mask = (torch.rand(1, T, 1, device=device) > 0.05).float()
            x_train = x_full * mask + (1 - mask) * x_full.mean(dim=1, keepdim=True)
        else:
            x_train = x_full

        model.train(); g.train(); opt.zero_grad()
        out = model(x_train, n_samples=2, rollout_K=K_r)
        # h_samples: (S, B, T), use mean for propagation
        h_enc = out["h_samples"].mean(dim=0)    # (B, T)

        # Propagate and mix (only if lam_prop > 0 and h_weight active)
        if lam_prop > 0 and h_w > 0:
            h_prop = propagate_segmented(h_enc, x_train, g, segment_len=SEGMENT_LEN)
            h_mix = alpha * h_enc + (1.0 - alpha) * h_prop
            # Replace h_samples with h_mix (broadcast across sample dim)
            h_mix_samples = h_mix.unsqueeze(0).expand_as(out["h_samples"])
            out["h_samples"] = h_mix_samples
        else:
            h_prop = h_enc.detach().clone()   # placeholder for monitoring

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

        # Monitor prop loss (diagnostic only)
        with torch.no_grad():
            prop_loss = F.mse_loss(h_prop, h_enc).item()
            prop_loss_hist.append(prop_loss)

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
    model.eval(); g.eval()
    with torch.no_grad():
        out_eval = model(x_full, n_samples=30, rollout_K=3)
        h_mean = out_eval["h_samples"].mean(dim=0)[0].cpu().numpy()

        # Also compute final h_prop for diagnostic
        if lam_prop > 0:
            h_enc_eval = torch.tensor(h_mean, dtype=torch.float32, device=device).unsqueeze(0)
            h_prop_final = propagate_segmented(h_enc_eval, x_full, g,
                                                  segment_len=SEGMENT_LEN)
            h_prop_np = h_prop_final[0].cpu().numpy()
        else:
            h_prop_np = h_mean.copy()

    pear, _ = evaluate(h_mean, hidden)
    pear_prop, _ = evaluate(h_prop_np, hidden)
    d = hidden_true_substitution(model, visible, hidden, device)

    del model, g
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "pearson": pear,                           # Pearson of final encoder h
        "pearson_prop": pear_prop,                 # Pearson of propagated h (separate diagnostic)
        "d_ratio": d["recon_true_scaled"] / d["recon_encoder"],
        "val_recon": best_val,
        "h_mean": h_mean,
        "h_prop": h_prop_np,
        "final_alpha": alpha_hist[-1] if alpha_hist else 1.0,
        "final_prop_loss": prop_loss_hist[-1] if prop_loss_hist else float("nan"),
    }


def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"runs/{ts}_huisman_hdyn_v2")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_dir}")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    full = load_huisman()
    print(f"Huisman: T={full.shape[0]}, N={full.shape[1]}, segment_len={SEGMENT_LEN}")
    print(f"Test species: {TEST_SPECIES_NAMES}, Lambdas: {LAMBDAS}\n")

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
                    r = train_v2(visible, hidden, seed, device, lam)
                    dt = (datetime.now() - t0).total_seconds()
                    print(f"  P_enc={r['pearson']:+.4f}  P_prop={r['pearson_prop']:+.4f}  "
                          f"d_r={r['d_ratio']:.2f}  α_final={r['final_alpha']:.2f}  "
                          f"prop_loss={r['final_prop_loss']:.4f}  ({dt:.1f}s)")
                except Exception as e:
                    print(f"  FAILED: {e}")
                    import traceback; traceback.print_exc()
                    r = {"pearson": float("nan"), "pearson_prop": float("nan"),
                         "d_ratio": float("nan"), "val_recon": float("nan"),
                         "h_mean": None, "h_prop": None,
                         "final_alpha": float("nan"),
                         "final_prop_loss": float("nan")}
                r["seed"] = seed; r["lam"] = lam; r["species"] = sp_name
                results[sp_name][lam].append(r)

    print(f"\n{'='*90}\nV2 h-DYN+PROP+MIX SUMMARY\n{'='*90}")
    print(f"{'Species':<10}{'λ':<8}{'mean P_enc':<13}{'mean P_prop':<14}"
          f"{'max P_enc':<12}{'d_r':<8}{'α_f':<8}")
    for name in TEST_SPECIES_NAMES:
        for lam in LAMBDAS:
            rs = results[name][lam]
            P = np.array([r["pearson"] for r in rs if not np.isnan(r["pearson"])])
            Pp = np.array([r["pearson_prop"] for r in rs
                            if not np.isnan(r.get("pearson_prop", np.nan))])
            D = np.array([r["d_ratio"] for r in rs if not np.isnan(r["d_ratio"])])
            A = np.array([r["final_alpha"] for r in rs
                           if not np.isnan(r.get("final_alpha", np.nan))])
            print(f"{name:<10}{lam:<8.2f}{P.mean():<+13.4f}{Pp.mean():<+14.4f}"
                  f"{P.max():<+12.4f}{D.mean():<8.2f}{A.mean():<8.2f}")

    # Plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    for ax, name in zip(axes, TEST_SPECIES_NAMES):
        means = [np.mean([r["pearson"] for r in results[name][lam]]) for lam in LAMBDAS]
        means_prop = [np.mean([r["pearson_prop"] for r in results[name][lam]])
                      for lam in LAMBDAS]
        stds = [np.std([r["pearson"] for r in results[name][lam]]) for lam in LAMBDAS]
        maxes = [np.max([r["pearson"] for r in results[name][lam]]) for lam in LAMBDAS]
        ax.errorbar(LAMBDAS, means, yerr=stds, marker="o", lw=2, capsize=5,
                    color="#1976d2", label="mean P_enc ± std")
        ax.plot(LAMBDAS, means_prop, marker="s", linestyle=":", color="#2e7d32",
                label="mean P_prop", alpha=0.8)
        ax.plot(LAMBDAS, maxes, marker="^", linestyle="--", color="#c62828",
                label="max P_enc", alpha=0.8)
        ax.set_xlabel("λ_prop (0 = disabled baseline)")
        ax.set_ylabel("Pearson")
        ax.set_title(f"{name}"); ax.grid(alpha=0.3); ax.legend(fontsize=9)
    fig.suptitle("V2: h-dynamics + propagation + mixing (Huisman)",
                 fontweight="bold")
    fig.savefig(out_dir / "fig_v2_scan.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # Best config overlay
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), constrained_layout=True)
    for ax, sp_idx, sp_name in zip(axes, TEST_SPECIES_IDX, TEST_SPECIES_NAMES):
        true_h = full[:, sp_idx]
        t = np.arange(len(true_h))
        ax.plot(t, true_h, color="black", lw=2, label="true", zorder=10)
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
            ax.plot(t[:L], a * h[:L] + b, color=c, lw=1.0, alpha=0.85,
                    label=f"seed {r['seed']}  P={r['pearson']:+.3f}")
        mean_P = np.mean([r["pearson"] for r in best_rs])
        ax.set_title(f"{sp_name}  (best λ={best_lam}, mean P_enc={mean_P:+.3f})")
        ax.set_ylabel(sp_name); ax.legend(fontsize=8, ncol=5); ax.grid(alpha=0.25)
    axes[-1].set_xlabel("time")
    fig.suptitle("V2 best config: recovered h_enc vs true (scale-invariant aligned)",
                 fontweight="bold")
    fig.savefig(out_dir / "fig_v2_recovery.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # Save
    with open(out_dir / "summary.md", "w", encoding="utf-8") as f:
        f.write("# V2: h-dynamics + propagation + α-annealed mixing (Huisman)\n\n")
        f.write(f"Seeds: {SEEDS}, Epochs: {EPOCHS}, segment_len={SEGMENT_LEN}\n\n")
        f.write("Architecture: h_mix(α) = α·h_enc + (1-α)·propagate(g, h_enc)\n")
        f.write(f"α: 1.0 (warmup) → 0.3 (final); "
                f"warmup 25% epochs, anneal 25-75%, constant 75%+\n\n")
        f.write("## Results\n\n")
        f.write("| Species | λ_prop | mean P_enc | mean P_prop | max P_enc | d_r | α_final |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for name in TEST_SPECIES_NAMES:
            for lam in LAMBDAS:
                rs = results[name][lam]
                P = np.array([r["pearson"] for r in rs if not np.isnan(r["pearson"])])
                Pp = np.array([r["pearson_prop"] for r in rs
                                if not np.isnan(r.get("pearson_prop", np.nan))])
                D = np.array([r["d_ratio"] for r in rs
                               if not np.isnan(r["d_ratio"])])
                A = np.array([r["final_alpha"] for r in rs
                               if not np.isnan(r.get("final_alpha", np.nan))])
                f.write(f"| {name} | {lam:.2f} | {P.mean():+.4f} | {Pp.mean():+.4f} | "
                        f"{P.max():+.4f} | {D.mean():.2f} | {A.mean():.2f} |\n")

    dump = {name: {str(lam): [{k: (float(v) if isinstance(v, (int, float, np.floating))
                                     else v.tolist() if isinstance(v, np.ndarray)
                                     else v)
                                 for k, v in r.items() if k not in ("h_mean", "h_prop")}
                                for r in rs]
                   for lam, rs in d_lam.items()}
             for name, d_lam in results.items()}
    with open(out_dir / "raw.json", "w") as f:
        json.dump(dump, f, indent=2, default=float)
    print(f"\n[OK] {out_dir}")


if __name__ == "__main__":
    main()
