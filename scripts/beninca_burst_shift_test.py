"""Burst shift causal intervention test.

Question: does the model use burst timesteps or flat timesteps?

Procedure:
  1. Find burst timesteps: indices where |d log x| is in top 10%
  2. Build 3 versions of visible data:
      - original:         unchanged
      - burst_shifted:    cyclic-shift visible values AT burst timesteps by τ=100 steps
                          (burst values relocate to other timesteps; flat stays)
      - burst_ablated:    replace burst timesteps with local mean (smooth them)
  3. Train Stage 1b fresh on each version, evaluate hidden recovery.

Hypothesis (based on user's observation):
  - If model learns "mean" — ablating bursts shouldn't hurt much
  - If model uses bursts — shifting/ablating should destroy recovery
  - Diagnostic for whether model truly captures burst structure

Test species: 3 representative (Cyclopoids, Bacteria, Ostracods)
  to cover different flatness profiles.
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

from models.cvhi_residual import CVHI_Residual
from scripts.load_beninca import load_beninca
from scripts.cvhi_residual_L1L3_diagnostics import evaluate, hidden_true_substitution


SEEDS = [42, 123, 456]
EPOCHS = 500
BURST_PCT = 10   # top X% of |d log x| are "burst"
SHIFT_TAU = 100  # timesteps to shift bursts

BEST_HP = dict(
    encoder_d=96, encoder_blocks=3, encoder_dropout=0.1,
    takens_lags=(1, 2, 4, 8),
    lr=0.0006033475528697158,
    lam_smooth=0.02, lam_kl=0.017251789430967935,
    lam_hf=0.2, min_energy=0.14353013693386804,
    lam_cf=9.517725868477207,
)

TEST_SPECIES = ["Cyclopoids", "Bacteria", "Ostracods"]


def find_burst_timesteps(visible, pct=10, eps=1e-6):
    """Find timesteps where visible has large |d log x|.
    Returns boolean mask (T,) with True for burst timesteps.
    """
    safe = np.maximum(visible, eps)
    log_x = np.log(safe)
    dlog = np.abs(np.diff(log_x, axis=0))   # (T-1, N)
    mag = np.linalg.norm(dlog, axis=-1)      # (T-1,)
    threshold = np.percentile(mag, 100 - pct)
    burst = mag > threshold
    # Extend to length T (add one False at end)
    return np.concatenate([burst, [False]])


def create_perturbed_data(visible_orig, burst_mask, mode="shift", tau=100):
    """Create perturbed visible data.

    Modes:
      - 'shift':   cyclic-shift visible values at burst timesteps by tau
      - 'ablate':  replace burst timesteps with local mean (smooth)
    """
    T, N = visible_orig.shape
    visible = visible_orig.copy()

    if mode == "shift":
        # Take burst values and move them to new positions
        burst_idx = np.where(burst_mask)[0]
        new_idx = (burst_idx + tau) % T
        # Save original burst values and non-burst values at new_idx
        burst_values = visible_orig[burst_idx].copy()
        # Restore non-burst at burst positions (with neighbor interpolation)
        for b_i in burst_idx:
            # Find nearest non-burst neighbor
            left = b_i - 1
            while left >= 0 and burst_mask[left]:
                left -= 1
            right = b_i + 1
            while right < T and burst_mask[right]:
                right += 1
            # Average
            if left >= 0 and right < T:
                visible[b_i] = 0.5 * (visible_orig[left] + visible_orig[right])
            elif left >= 0:
                visible[b_i] = visible_orig[left]
            elif right < T:
                visible[b_i] = visible_orig[right]
        # Place burst values at shifted positions
        for i, (b_i, n_i) in enumerate(zip(burst_idx, new_idx)):
            visible[n_i] = burst_values[i]

    elif mode == "ablate":
        # Replace bursts with local smooth mean (window 21)
        from scipy.ndimage import uniform_filter1d
        smoothed = uniform_filter1d(visible_orig, size=21, axis=0, mode="nearest")
        visible[burst_mask] = smoothed[burst_mask]

    else:
        raise ValueError(f"Unknown mode: {mode}")

    return visible


def make_model(N, device):
    return CVHI_Residual(
        num_visible=N,
        encoder_d=BEST_HP["encoder_d"], encoder_blocks=BEST_HP["encoder_blocks"],
        encoder_heads=4,
        takens_lags=BEST_HP["takens_lags"], encoder_dropout=BEST_HP["encoder_dropout"],
        d_species_f=20, f_visible_layers=2, f_visible_top_k=4,
        d_species_G=12, G_field_layers=1, G_field_top_k=3,
        prior_std=1.0, gnn_backbone="mlp",
        use_formula_hints=True, use_G_field=True,
        num_mixture_components=1,
        G_anchor_first=True, G_anchor_sign=+1,
        hierarchical_h=False,
    ).to(device)


def alpha_schedule(epoch, epochs, start=0.5, end=0.95):
    f = epoch / max(1, epochs)
    if f <= start: return 1.0
    if f >= end: return 0.0
    return 1.0 - (f - start) / (end - start)


def train_one(visible, hidden, seed, device, epochs=EPOCHS):
    torch.manual_seed(seed)
    T, N = visible.shape
    x_full = torch.tensor(visible, dtype=torch.float32, device=device).unsqueeze(0)

    model = make_model(N, device)
    opt = torch.optim.AdamW(model.parameters(), lr=BEST_HP["lr"], weight_decay=1e-4)

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
        if hasattr(model, "G_anchor_alpha"):
            model.G_anchor_alpha = alpha_schedule(epoch, epochs)
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
        model.train(); opt.zero_grad()
        out = model(x_train, n_samples=2, rollout_K=K_r)
        tr_out = model.slice_out(out, 0, train_end)
        tr_out["visible"] = out["visible"][:, 0:train_end]
        tr_out["G"] = out["G"][:, 0:train_end]
        losses = model.loss(
            tr_out, beta_kl=BEST_HP["lam_kl"], free_bits=0.02,
            margin_null=0.002, margin_shuf=0.001,
            lam_necessary=BEST_HP["lam_cf"], lam_shuffle=BEST_HP["lam_cf"]*0.6,
            lam_energy=2.0, min_energy=BEST_HP["min_energy"],
            lam_smooth=BEST_HP["lam_smooth"], lam_sparse=0.02,
            h_weight=h_w, lam_rollout=lam_r,
            rollout_weights=(1.0, 0.5, 0.25),
            lam_hf=BEST_HP["lam_hf"], lowpass_sigma=6.0,
            lam_rmse_log=0.1,
        )
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        with torch.no_grad():
            val_out = model.slice_out(out, train_end, T)
            val_out["visible"] = out["visible"][:, train_end:T]
            val_out["G"] = out["G"][:, train_end:T]
            val_losses = model.loss(
                val_out, h_weight=1.0,
                margin_null=0.002, margin_shuf=0.001,
                lam_energy=2.0, min_energy=BEST_HP["min_energy"],
                lam_rollout=0.5, rollout_weights=(1.0, 0.5, 0.25),
                lam_hf=BEST_HP["lam_hf"], lowpass_sigma=6.0,
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
    return {"pearson": pear, "d_ratio": d["recon_true_scaled"] / d["recon_encoder"],
            "val_recon": best_val, "h_mean": h_mean}


def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"runs/{ts}_beninca_burst_shift")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_dir}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    full, species, _ = load_beninca()
    species = [str(s) for s in species]

    configs = ["original", "burst_shifted", "burst_ablated"]
    print(f"\n=== Burst shift causal test ===")
    print(f"Test species: {TEST_SPECIES}")
    print(f"Configs: {configs}")
    print(f"Total runs: {len(TEST_SPECIES)*len(configs)*len(SEEDS)}\n")

    # Build data for each config once per hidden species
    results = {sp: {cfg: [] for cfg in configs} for sp in TEST_SPECIES}

    for h_name in TEST_SPECIES:
        h_idx = species.index(h_name)
        visible_orig = np.delete(full, h_idx, axis=1).astype(np.float32)
        hidden_raw = full[:, h_idx].astype(np.float32)

        # Find bursts in ORIGINAL visible (before any manipulation)
        burst_mask = find_burst_timesteps(visible_orig, pct=BURST_PCT)
        print(f"\n--- hidden={h_name} ---")
        print(f"  Burst timesteps: {burst_mask.sum()}/{len(burst_mask)} "
              f"({burst_mask.mean():.1%})")

        # Save masked data snapshots for reproducibility
        visible_shifted = create_perturbed_data(visible_orig, burst_mask,
                                                  mode="shift", tau=SHIFT_TAU)
        visible_ablated = create_perturbed_data(visible_orig, burst_mask,
                                                  mode="ablate")

        data_by_cfg = {
            "original":      visible_orig,
            "burst_shifted": visible_shifted,
            "burst_ablated": visible_ablated,
        }

        for cfg in configs:
            vis = data_by_cfg[cfg]
            for seed in SEEDS:
                t0 = datetime.now()
                r = train_one(vis, hidden_raw, seed, device)
                dt = (datetime.now() - t0).total_seconds()
                print(f"  {cfg:<18}  seed={seed}  P={r['pearson']:+.3f}  "
                      f"d_r={r['d_ratio']:.2f}  ({dt:.1f}s)")
                r["seed"] = seed; r["config"] = cfg
                results[h_name][cfg].append(r)

    # Summary
    print(f"\n{'='*90}")
    print("BURST SHIFT CAUSAL TEST RESULTS")
    print('='*90)
    print(f"{'Species':<14}{'original':<14}{'burst_shifted':<16}{'burst_ablated':<16}"
          f"{'Δ(shift)':<12}{'Δ(ablate)':<12}")
    print('-'*90)
    for sp in TEST_SPECIES:
        orig_mean = np.mean([r["pearson"] for r in results[sp]["original"]])
        shift_mean = np.mean([r["pearson"] for r in results[sp]["burst_shifted"]])
        ablate_mean = np.mean([r["pearson"] for r in results[sp]["burst_ablated"]])
        d_shift = shift_mean - orig_mean
        d_ablate = ablate_mean - orig_mean
        print(f"{sp:<14}{orig_mean:<+14.3f}{shift_mean:<+16.3f}{ablate_mean:<+16.3f}"
              f"{d_shift:<+12.3f}{d_ablate:<+12.3f}")

    print('-'*90)
    print("\nInterpretation:")
    print("  If Δ(ablate) is very negative (bursts mattered):")
    print("    → model does use burst timesteps for hidden inference")
    print("  If Δ(ablate) ≈ 0 (bursts didn't matter):")
    print("    → model primarily learns from flat periods (mean-like)")
    print("  burst_shifted tests if model's h tracks burst *positions*.")

    # Visualization: compare recovered h across configs for one species
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), constrained_layout=True)
    for ax, sp in zip(axes, TEST_SPECIES):
        h_idx = species.index(sp)
        true_h = full[:, h_idx]
        t_axis = np.arange(len(true_h))
        ax.plot(t_axis, true_h, color="black", lw=1.5, label="true", zorder=20)

        colors = {"original": "#2e7d32", "burst_shifted": "#f57c00",
                   "burst_ablated": "#c62828"}
        for cfg in configs:
            # Use best seed by Pearson
            rs = results[sp][cfg]
            best = max(rs, key=lambda r: r["pearson"])
            hm = best["h_mean"]
            L = min(len(hm), len(true_h))
            a, b = np.polyfit(hm[:L], true_h[:L], 1)
            h_sc = a * hm[:L] + b
            ax.plot(t_axis[:L], h_sc, color=colors[cfg], lw=1.0, alpha=0.8,
                    label=f"{cfg}: P={best['pearson']:+.3f}")

        orig_mean = np.mean([r["pearson"] for r in results[sp]["original"]])
        shift_mean = np.mean([r["pearson"] for r in results[sp]["burst_shifted"]])
        ablate_mean = np.mean([r["pearson"] for r in results[sp]["burst_ablated"]])
        ax.set_title(f"{sp}:  orig={orig_mean:+.3f}  "
                     f"shifted={shift_mean:+.3f}  ablated={ablate_mean:+.3f}",
                     fontweight="bold", fontsize=11)
        ax.set_ylabel(sp)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("time step")
    fig.suptitle("Burst shift/ablate causal test: does model use burst timesteps?",
                 fontweight="bold", fontsize=13)
    fig.savefig(out_dir / "fig_burst_shift_recovery.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # Diagnostic: show burst mask + shifted data for one species
    h_idx = species.index("Bacteria")
    visible_orig = np.delete(full, h_idx, axis=1).astype(np.float32)
    burst_mask = find_burst_timesteps(visible_orig, pct=BURST_PCT)
    visible_shifted = create_perturbed_data(visible_orig, burst_mask,
                                              mode="shift", tau=SHIFT_TAU)
    visible_ablated = create_perturbed_data(visible_orig, burst_mask,
                                              mode="ablate")

    fig, axes = plt.subplots(3, 1, figsize=(14, 8), constrained_layout=True,
                               sharex=True)
    for ax, vis, name in zip(axes,
                              [visible_orig, visible_shifted, visible_ablated],
                              ["original", "burst_shifted", "burst_ablated"]):
        ax.plot(vis[:, 0], color="#1976d2", lw=0.9, label=f"channel 0")
        ax.plot(vis[:, 3], color="#c62828", lw=0.9, alpha=0.7, label=f"channel 3")
        # Overlay burst mask
        ax.fill_between(range(len(burst_mask)), 0,
                         (burst_mask.astype(float) *
                          max(vis.max(), 1) * 1.05),
                         color="gray", alpha=0.15, label="burst times (orig)")
        ax.set_ylabel(name); ax.legend(fontsize=8)
        ax.set_title(name, fontsize=10)
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("time step")
    fig.suptitle(f"Perturbation preview (Bacteria as hidden, 2 visible channels shown)",
                 fontweight="bold")
    fig.savefig(out_dir / "fig_data_preview.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # Save
    with open(out_dir / "summary.md", "w", encoding="utf-8") as f:
        f.write("# Burst shift causal intervention test\n\n")
        f.write(f"Burst definition: top {BURST_PCT}% of |d log x|\n")
        f.write(f"Shift tau: {SHIFT_TAU} timesteps\n\n")
        f.write("| Species | original | burst_shifted | burst_ablated | Δ(shift) | Δ(ablate) |\n")
        f.write("|---|---|---|---|---|---|\n")
        for sp in TEST_SPECIES:
            o = np.mean([r["pearson"] for r in results[sp]["original"]])
            s = np.mean([r["pearson"] for r in results[sp]["burst_shifted"]])
            a = np.mean([r["pearson"] for r in results[sp]["burst_ablated"]])
            f.write(f"| {sp} | {o:+.3f} | {s:+.3f} | {a:+.3f} | "
                    f"{s-o:+.3f} | {a-o:+.3f} |\n")

    def to_ser(v):
        if isinstance(v, (int, float, np.floating)):
            return float(v)
        if isinstance(v, np.ndarray):
            return v.tolist()
        return v
    dump = {sp: {cfg: [{k: to_ser(v) for k, v in r.items()}
                        for r in rs] for cfg, rs in d.items()}
            for sp, d in results.items()}
    with open(out_dir / "raw.json", "w") as f:
        json.dump(dump, f, default=float)

    print(f"\n[OK] {out_dir}")


if __name__ == "__main__":
    main()
