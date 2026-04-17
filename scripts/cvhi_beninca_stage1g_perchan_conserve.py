"""Stage 1g - Per-channel conservation (standardized).

Fix Stage 1f's Filam-diatom domination (CV 7.52 >> others).
Instead of summing channels, constrain each channel independently:
    For each channel j:
        L_j = Var_t(x_std_j + c_j * h_std)
    L_conserve = mean_j L_j = N - mean_j corr(x_j, h)^2

Both x_j and h are **standardized per channel** (mean=0, std=1) so each
channel contributes equally, preventing any single high-CV channel
(e.g., Filam_diatoms) from dominating the loss.

Equivalently: pulls h toward the first PC of visible channel correlation
matrix (equal-weighted). This gives every species equal voice in what
h should represent.

Red line:
  - No hidden_true in loss
  - All signals from visible channels only
  - c_j per visible channel, learnable, no prior

Baseline: Stage 1b config (RMSE log + input dropout + G_anchor_first).
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.cvhi_residual import CVHI_Residual
from scripts.load_beninca import load_beninca
from scripts.cvhi_residual_L1L3_diagnostics import evaluate, hidden_true_substitution


SEEDS = [42, 123, 456, 789, 2024]
EPOCHS = 500
LAMBDA_CONSERVE = 0.1

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


def train_one(visible, hidden, seed, device, lam_conserve=LAMBDA_CONSERVE, epochs=EPOCHS):
    torch.manual_seed(seed)
    T, N = visible.shape
    x_full = torch.tensor(visible, dtype=torch.float32, device=device).unsqueeze(0)

    # Precompute per-channel mean and std for standardization
    x_mean = x_full.mean(dim=1, keepdim=True)   # (1, 1, N)
    x_std = x_full.std(dim=1, keepdim=True) + 1e-6

    model = make_model(N, device)
    # Per-channel learnable scale, unbounded (sign matters)
    c_per_chan = nn.Parameter(torch.zeros(N, device=device))
    params = list(model.parameters()) + [c_per_chan]
    opt = torch.optim.AdamW(params, lr=BEST_HP["lr"], weight_decay=1e-4)

    warmup = int(0.2 * epochs)
    ramp = max(1, int(0.2 * epochs))
    def lr_lambda(step):
        if step < 50: return step / 50
        p = (step - 50) / max(1, epochs - 50)
        return 0.5 * (1 + np.cos(np.pi * p))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    train_end = int(0.75 * T)
    best_val = float("inf"); best_state = None; best_c = None

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
            dp_mask = (torch.rand(1, T, 1, device=device) > 0.05).float()
            x_train = x_full * dp_mask + (1 - dp_mask) * x_full.mean(dim=1, keepdim=True)
        else:
            x_train = x_full

        model.train(); opt.zero_grad()
        out = model(x_train, n_samples=2, rollout_K=K_r)
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

        # === Per-channel conservation loss ===
        if lam_conserve > 0 and h_w > 0:
            # Standardize visible channels (across time, per channel)
            x_slice = x_train[:, :train_end]   # (1, T_train, N)
            mean_slice = x_slice.mean(dim=1, keepdim=True)
            std_slice = x_slice.std(dim=1, keepdim=True) + 1e-6
            x_std = (x_slice - mean_slice) / std_slice   # (1, T_train, N)

            # Standardize h
            h_mean_post = tr_out["h_samples"].mean(dim=0)   # (1, T_train)
            h_mu = h_mean_post.mean(dim=-1, keepdim=True)
            h_std_val = h_mean_post.std(dim=-1, keepdim=True) + 1e-6
            h_std = (h_mean_post - h_mu) / h_std_val         # (1, T_train)

            # Per-channel: x_std_j + c_j * h_std -> low variance
            # c_j unconstrained (signed). Expand h_std to (1, T_train, N)
            h_exp = h_std.unsqueeze(-1)                      # (1, T_train, 1)
            c_exp = c_per_chan.view(1, 1, N)                 # (1, 1, N)
            adjusted = x_std + c_exp * h_exp                 # (1, T_train, N)
            # Variance across time, then mean across channels
            L_conserve = adjusted.var(dim=1).mean()
            total = total + lam_conserve * h_w * L_conserve

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
            best_c = c_per_chan.detach().cpu().clone()

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        out_eval = model(x_full, n_samples=30, rollout_K=3)
        h_mean = out_eval["h_samples"].mean(dim=0)[0].cpu().numpy()
    pear, _ = evaluate(h_mean, hidden)
    d = hidden_true_substitution(model, visible, hidden, device)

    final_c = best_c.numpy().tolist() if best_c is not None else [float("nan")] * N
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "pearson": pear,
        "d_ratio": d["recon_true_scaled"] / d["recon_encoder"],
        "val_recon": best_val,
        "c_per_chan": final_c,
    }


def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"runs/{ts}_beninca_stage1g")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_dir}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    full, species, days = load_beninca()

    print(f"\n=== Stage 1g: Per-channel Conservation (standardized) ===")
    print(f"lambda_conserve = {LAMBDA_CONSERVE}")

    all_results = {}
    for h_name in SPECIES_ORDER:
        h_idx = species.index(h_name)
        visible_arr = np.delete(full, h_idx, axis=1)
        hidden_arr = full[:, h_idx]

        print(f"\n--- hidden={h_name} ---")
        rs = []
        for s in SEEDS:
            t0 = datetime.now()
            try:
                r = train_one(visible_arr, hidden_arr, s, device)
                dt = (datetime.now() - t0).total_seconds()
                c_summary = f"[{min(r['c_per_chan']):+.2f},{max(r['c_per_chan']):+.2f}]"
                print(f"  seed={s}  P={r['pearson']:+.3f}  d_r={r['d_ratio']:.2f}  "
                      f"val={r['val_recon']:.4f}  c={c_summary}  ({dt:.1f}s)")
            except Exception as e:
                print(f"  seed={s}  FAILED: {e}")
                import traceback; traceback.print_exc()
                r = {"pearson": float("nan"), "d_ratio": float("nan"),
                     "val_recon": float("nan"), "c_per_chan": []}
            r["seed"] = s
            rs.append(r)
        all_results[h_name] = rs

    # Baselines
    phase2 = {"Cyclopoids": 0.064, "Calanoids": 0.159, "Rotifers": 0.056,
              "Nanophyto": 0.204, "Picophyto": 0.062, "Filam_diatoms": 0.056,
              "Ostracods": 0.125, "Harpacticoids": 0.129, "Bacteria": 0.171}
    stage1b = {"Cyclopoids": 0.055, "Calanoids": 0.173, "Rotifers": 0.080,
               "Nanophyto": 0.141, "Picophyto": 0.116, "Filam_diatoms": 0.031,
               "Ostracods": 0.272, "Harpacticoids": 0.120, "Bacteria": 0.197}
    stage1f = {"Cyclopoids": 0.074, "Calanoids": 0.044, "Rotifers": 0.103,
               "Nanophyto": 0.069, "Picophyto": 0.037, "Filam_diatoms": 0.043,
               "Ostracods": 0.207, "Harpacticoids": 0.258, "Bacteria": 0.197}

    print(f"\n{'='*110}")
    print("STAGE 1g RESULTS - Per-channel Conservation")
    print('='*110)
    print(f"{'Species':<18}{'P2':<10}{'S1b':<10}{'S1f':<10}{'S1g':<12}{'D vs S1b':<12}{'D vs S1f':<12}")
    sum_p2=0; sum_s1b=0; sum_s1f=0; sum_s1g=0; cnt=0
    for h in SPECIES_ORDER:
        rs = all_results[h]
        P = np.array([r["pearson"] for r in rs if not np.isnan(r["pearson"])])
        s1g = float(P.mean()) if len(P) else float("nan")
        print(f"{h:<18}{phase2[h]:<+10.3f}{stage1b[h]:<+10.3f}{stage1f[h]:<+10.3f}"
              f"{s1g:<+12.3f}{s1g-stage1b[h]:<+12.3f}{s1g-stage1f[h]:<+12.3f}")
        if not np.isnan(s1g):
            sum_p2+=phase2[h]; sum_s1b+=stage1b[h]; sum_s1f+=stage1f[h]; sum_s1g+=s1g; cnt+=1

    avg_p2=sum_p2/cnt; avg_s1b=sum_s1b/cnt; avg_s1f=sum_s1f/cnt; avg_s1g=sum_s1g/cnt
    print(f"\nOverall:  P2={avg_p2:+.4f}  S1b={avg_s1b:+.4f}  S1f={avg_s1f:+.4f}  S1g={avg_s1g:+.4f}")
    print(f"D(S1g-S1b)={avg_s1g-avg_s1b:+.4f}  D(S1g-S1f)={avg_s1g-avg_s1f:+.4f}")

    if avg_s1g > avg_s1b + 0.005:
        verdict = "[OK] Stage 1g better than S1b - per-channel conservation works"
    elif avg_s1g > avg_s1b - 0.005:
        verdict = "[NEUTRAL] Stage 1g ~= S1b"
    else:
        verdict = "[FAIL] Stage 1g worse than S1b"
    print(f"\n{verdict}\n{'='*110}")

    with open(out_dir / "summary.md", "w", encoding="utf-8") as f:
        f.write("# Beninca Stage 1g - Per-channel Conservation (standardized)\n\n")
        f.write(f"Seeds: {len(SEEDS)}, Epochs: {EPOCHS}, lambda={LAMBDA_CONSERVE}\n\n")
        f.write("Loss: L = mean_j Var_t(x_std_j + c_j * h_std)\n")
        f.write("Each channel standardized (mean=0, std=1) for equal weighting.\n\n")
        f.write("## Results\n\n")
        f.write("| Species | P2 | S1b | S1f | **S1g** | D vs S1b | D vs S1f |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for h in SPECIES_ORDER:
            rs = all_results[h]
            P = np.array([r["pearson"] for r in rs if not np.isnan(r["pearson"])])
            s1g = float(P.mean()) if len(P) else float("nan")
            f.write(f"| {h} | {phase2[h]:+.3f} | {stage1b[h]:+.3f} | {stage1f[h]:+.3f} | "
                    f"**{s1g:+.3f}** | {s1g-stage1b[h]:+.3f} | {s1g-stage1f[h]:+.3f} |\n")
        f.write(f"\n**Overall**: P2={avg_p2:+.4f}, S1b={avg_s1b:+.4f}, "
                f"S1f={avg_s1f:+.4f}, **S1g={avg_s1g:+.4f}**\n\n")
        f.write(f"**{verdict}**\n")

    with open(out_dir / "raw.json", "w") as f:
        json.dump(all_results, f, indent=2, default=float)
    print(f"\n[OK] {out_dir}")


if __name__ == "__main__":
    main()
