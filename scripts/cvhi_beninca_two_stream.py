"""Two-stream decomposition: h = h_slow + h_spike.

Based on burst_framework theory:
  - h_slow:  VAE encoder output, Gaussian prior (captures "skeleton" chaos)
  - h_spike: separate small encoder, Laplace prior via L1 (captures "resonance
              amplification events" per Benincà 2011)

Architecture:
  h_combined(t) = h_slow(t) + h_spike(t)
  dynamics: log(x_{t+1}/x_t) = f(x) + h_combined · G(x)
  loss:
    standard recon + CF + KL (using h_combined)
    + λ_spike · mean(|h_spike|)        ← sparsity
    (+ optional event weighting on recon)

Configs:
  - baseline (Stage 1b, no spike)
  - two_stream (λ_spike=0.1)
  - two_stream + event_α=1 (combined)
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
from scripts.load_beninca import load_beninca
from scripts.cvhi_residual_L1L3_diagnostics import evaluate, hidden_true_substitution


SEEDS = [42, 123, 456]
EPOCHS = 500

BEST_HP = dict(
    encoder_d=96, encoder_blocks=3, encoder_dropout=0.1,
    takens_lags=(1, 2, 4, 8),
    lr=0.0006033475528697158,
    lam_smooth=0.02, lam_kl=0.017251789430967935,
    lam_hf=0.2, min_energy=0.14353013693386804,
    lam_cf=9.517725868477207,
)

SPECIES_ORDER = ["Cyclopoids", "Calanoids", "Rotifers", "Nanophyto",
                 "Picophyto", "Filam_diatoms", "Ostracods",
                 "Harpacticoids", "Bacteria"]

CONFIGS = [
    # (name,              use_spike, lam_spike, event_alpha)
    ("baseline (S1b)",    False,     0.0,       0.0),
    ("two_stream",         True,      0.1,       0.0),
    ("two_stream+event1",  True,      0.1,       1.0),
]


class SpikeEncoder(nn.Module):
    """Burst detector: (x, log_ratio) → sparse h_spike signal.

    Small MLP. Input is current state + local log-ratio (which spikes during bursts).
    L1 regularization enforces sparsity.
    """
    def __init__(self, n_visible, hidden_dim=32):
        super().__init__()
        # Input: x_t (N) + log_ratio_t (N) = 2N dims per timestep
        self.net = nn.Sequential(
            nn.Linear(2 * n_visible, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x, log_ratio):
        """x: (B, T, N), log_ratio: (B, T, N). Returns h_spike: (B, T)."""
        feat = torch.cat([x, log_ratio], dim=-1)
        return self.net(feat).squeeze(-1)


def compute_log_ratio(x, eps=1e-6):
    """x: (B, T, N). Returns log_ratio: (B, T, N). Last timestep uses (T-1)."""
    safe = torch.clamp(x, min=eps)
    lr = torch.zeros_like(x)
    lr[:, :-1] = torch.log(safe[:, 1:] / safe[:, :-1])
    lr[:, -1] = lr[:, -2]   # replicate for last step
    return lr


def make_model(N, device, event_alpha=0.0):
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
        use_event_weighting=(event_alpha > 0),
        event_alpha=event_alpha if event_alpha > 0 else 1.0,
    ).to(device)


def alpha_schedule(epoch, epochs, start=0.5, end=0.95):
    f = epoch / max(1, epochs)
    if f <= start: return 1.0
    if f >= end: return 0.0
    return 1.0 - (f - start) / (end - start)


def train_one(visible_raw, hidden_raw, seed, device,
              use_spike, lam_spike, event_alpha, epochs=EPOCHS):
    torch.manual_seed(seed)
    T, N = visible_raw.shape
    x_full = torch.tensor(visible_raw, dtype=torch.float32, device=device).unsqueeze(0)

    model = make_model(N, device, event_alpha=event_alpha)
    spike_enc = SpikeEncoder(n_visible=N, hidden_dim=32).to(device) if use_spike else None

    params = list(model.parameters())
    if spike_enc is not None:
        params += list(spike_enc.parameters())
    opt = torch.optim.AdamW(params, lr=BEST_HP["lr"], weight_decay=1e-4)

    warmup = int(0.2 * epochs)
    ramp = max(1, int(0.2 * epochs))
    def lr_lambda(step):
        if step < 50: return step / 50
        p = (step - 50) / max(1, epochs - 50)
        return 0.5 * (1 + np.cos(np.pi * p))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    train_end = int(0.75 * T)
    best_val = float("inf"); best_state = None; best_spike = None

    spike_sparsity_hist = []

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

        model.train()
        if spike_enc is not None: spike_enc.train()
        opt.zero_grad()

        out = model(x_train, n_samples=2, rollout_K=K_r)

        # Compute spike component if enabled
        if use_spike and h_w > 0:
            log_ratio_t = compute_log_ratio(x_train)
            h_spike = spike_enc(x_train, log_ratio_t)   # (B, T)
            # Combine with main h_samples
            h_main = out["h_samples"]                     # (S, B, T)
            h_spike_expanded = h_spike.unsqueeze(0).expand_as(h_main)
            out["h_samples"] = h_main + h_spike_expanded
            # L1 sparsity
            L_spike_sparse = h_spike.abs().mean()
            spike_sparsity_hist.append(float(L_spike_sparse.item()))
        else:
            L_spike_sparse = torch.zeros((), device=device)

        tr_out = model.slice_out(out, 0, train_end)
        tr_out["visible"] = out["visible"][:, 0:train_end]
        tr_out["G"] = out["G"][:, 0:train_end]

        losses = model.loss(
            tr_out,
            beta_kl=BEST_HP["lam_kl"], free_bits=0.02,
            margin_null=0.002, margin_shuf=0.001,
            lam_necessary=BEST_HP["lam_cf"],
            lam_shuffle=BEST_HP["lam_cf"]*0.6,
            lam_energy=2.0, min_energy=BEST_HP["min_energy"],
            lam_smooth=BEST_HP["lam_smooth"], lam_sparse=0.02,
            h_weight=h_w, lam_rollout=lam_r,
            rollout_weights=(1.0, 0.5, 0.25),
            lam_hf=BEST_HP["lam_hf"], lowpass_sigma=6.0,
            lam_rmse_log=0.1,
        )
        total = losses["total"]

        # Add spike L1 penalty
        if use_spike and h_w > 0:
            total = total + lam_spike * h_w * L_spike_sparse

        total.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
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
            if spike_enc is not None:
                best_spike = {k: v.detach().cpu().clone()
                               for k, v in spike_enc.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    if spike_enc is not None and best_spike is not None:
        spike_enc.load_state_dict(best_spike)

    model.eval()
    if spike_enc is not None: spike_enc.eval()
    with torch.no_grad():
        out_eval = model(x_full, n_samples=30, rollout_K=3)
        h_main_eval = out_eval["h_samples"].mean(dim=0)[0].cpu().numpy()
        if use_spike:
            log_ratio_full = compute_log_ratio(x_full)
            h_spike_eval = spike_enc(x_full, log_ratio_full)[0].cpu().numpy()
            h_mean = h_main_eval + h_spike_eval
        else:
            h_spike_eval = np.zeros_like(h_main_eval)
            h_mean = h_main_eval

    pear, _ = evaluate(h_mean, hidden_raw)
    # Also compute pearson of each component alone for diagnostic
    pear_main, _ = evaluate(h_main_eval, hidden_raw)
    pear_spike = float("nan")
    if use_spike and h_spike_eval.std() > 1e-6:
        pear_spike, _ = evaluate(h_spike_eval, hidden_raw)

    d = hidden_true_substitution(model, visible_raw, hidden_raw, device)

    del model, spike_enc
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "pearson": pear,
        "pearson_main": pear_main,
        "pearson_spike": pear_spike,
        "d_ratio": d["recon_true_scaled"] / d["recon_encoder"],
        "val_recon": best_val,
        "h_mean": h_mean,
        "h_main": h_main_eval,
        "h_spike": h_spike_eval,
        "spike_final_mag": spike_sparsity_hist[-1] if spike_sparsity_hist else 0.0,
    }


def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"runs/{ts}_beninca_two_stream")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_dir}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    full, species, _ = load_beninca()
    species = [str(s) for s in species]

    print(f"\n=== Beninca two-stream h decomposition ===")
    print(f"h = h_main (VAE, Gaussian) + h_spike (small MLP, L1 sparse)")
    for cfg_name, us, ls, ev in CONFIGS:
        print(f"  {cfg_name}: spike={us}, λ_sparse={ls}, event_α={ev}")
    print(f"Total: {len(SPECIES_ORDER)*len(CONFIGS)*len(SEEDS)} runs\n")

    results = {cfg_name: {sp: [] for sp in SPECIES_ORDER}
               for cfg_name, _, _, _ in CONFIGS}

    total_runs = len(SPECIES_ORDER) * len(CONFIGS) * len(SEEDS)
    run_i = 0

    for h_name in SPECIES_ORDER:
        h_idx = species.index(h_name)
        visible_raw = np.delete(full, h_idx, axis=1).astype(np.float32)
        hidden_raw = full[:, h_idx].astype(np.float32)

        print(f"\n--- hidden={h_name} ---")
        for cfg_name, use_spike, lam_spike, event_alpha in CONFIGS:
            for seed in SEEDS:
                run_i += 1
                t0 = datetime.now()
                try:
                    r = train_one(visible_raw, hidden_raw, seed, device,
                                    use_spike, lam_spike, event_alpha)
                    dt = (datetime.now() - t0).total_seconds()
                    extra = ""
                    if use_spike:
                        extra = f" P_main={r['pearson_main']:+.3f} P_spike={r['pearson_spike']:+.3f}"
                    print(f"  [{run_i}/{total_runs}] {cfg_name:<22}  seed={seed}  "
                          f"P={r['pearson']:+.3f}{extra}  ({dt:.1f}s)")
                except Exception as e:
                    print(f"  [{run_i}/{total_runs}] FAILED: {e}")
                    import traceback; traceback.print_exc()
                    r = {"pearson": float("nan"), "pearson_main": float("nan"),
                         "pearson_spike": float("nan"),
                         "d_ratio": float("nan"), "val_recon": float("nan"),
                         "h_mean": None, "h_main": None, "h_spike": None,
                         "spike_final_mag": float("nan")}
                r["seed"] = seed; r["config"] = cfg_name
                results[cfg_name][h_name].append(r)

    stage1b_ref = {"Cyclopoids": 0.055, "Calanoids": 0.173, "Rotifers": 0.080,
                    "Nanophyto": 0.141, "Picophyto": 0.116, "Filam_diatoms": 0.031,
                    "Ostracods": 0.272, "Harpacticoids": 0.120, "Bacteria": 0.197}

    print(f"\n{'='*110}")
    print("TWO-STREAM RESULTS (ref: Stage 1b mean = 0.132)")
    print('='*110)
    header = f"{'Species':<18}{'S1b ref':<12}"
    for cfg_name, _, _, _ in CONFIGS:
        header += f"{cfg_name:<22}"
    print(header)
    print('-' * 110)

    config_means = {cfg_name: [] for cfg_name, _, _, _ in CONFIGS}
    for h in SPECIES_ORDER:
        line = f"{h:<18}{stage1b_ref[h]:<+12.3f}"
        for cfg_name, _, _, _ in CONFIGS:
            rs = results[cfg_name][h]
            P = np.array([r["pearson"] for r in rs if not np.isnan(r["pearson"])])
            m = float(P.mean()) if len(P) else float("nan")
            if not np.isnan(m):
                config_means[cfg_name].append(m)
            line += f"{m:<+22.3f}"
        print(line)

    print('-' * 110)
    avg_line = f"{'Overall':<18}{'0.132':<12}"
    for cfg_name, _, _, _ in CONFIGS:
        avg = np.mean(config_means[cfg_name])
        avg_line += f"{avg:<+22.4f}"
    print(avg_line)

    best_cfg = max(CONFIGS, key=lambda c: np.mean(config_means[c[0]]))
    best_avg = np.mean(config_means[best_cfg[0]])
    print(f"\nBest: {best_cfg[0]}  mean={best_avg:+.4f}  Δ vs S1b = {best_avg-0.132:+.4f}")

    # Save summary + raw
    with open(out_dir / "summary.md", "w", encoding="utf-8") as f:
        f.write("# Beninca two-stream (h_slow + h_spike) experiment\n\n")
        f.write(f"Seeds: {SEEDS}, Epochs: {EPOCHS}\n\n")
        f.write("Architecture: h_combined = h_main (VAE, Gaussian) + h_spike (MLP, L1-sparse)\n\n")
        f.write("| Species | S1b ref | " + " | ".join(c[0] for c in CONFIGS) + " |\n")
        f.write("|---|---|" + "---|" * len(CONFIGS) + "\n")
        for h in SPECIES_ORDER:
            row = f"| {h} | {stage1b_ref[h]:+.3f}"
            for cfg_name, _, _, _ in CONFIGS:
                rs = results[cfg_name][h]
                P = np.array([r["pearson"] for r in rs if not np.isnan(r["pearson"])])
                m = float(P.mean()) if len(P) else float("nan")
                row += f" | {m:+.3f}"
            f.write(row + " |\n")
        f.write(f"\n**Overall**:\n- Stage 1b ref: 0.132\n")
        for cfg_name, _, _, _ in CONFIGS:
            f.write(f"- {cfg_name}: {np.mean(config_means[cfg_name]):+.4f}\n")
        f.write(f"\nBest: **{best_cfg[0]}** = {best_avg:+.4f} "
                f"(Δ vs S1b = {best_avg-0.132:+.4f})\n")

    # Save raw with h_mean for recovery figures
    def to_ser(v):
        if isinstance(v, (int, float, np.floating)):
            return float(v)
        if isinstance(v, np.ndarray):
            return v.tolist()
        return v
    dump = {cfg: {h: [{k: to_ser(v) for k, v in r.items()}
                       for r in rs] for h, rs in d.items()}
            for cfg, d in results.items()}
    with open(out_dir / "raw.json", "w") as f:
        json.dump(dump, f, default=float)   # no indent for size

    print(f"\n[OK] {out_dir}")

    # Auto-generate figures
    print("\nGenerating figures...")
    import subprocess
    subprocess.run(["python", "-u", "-m", "scripts.post_experiment_figs",
                    "--exp-dir", str(out_dir)], check=False)


if __name__ == "__main__":
    main()
