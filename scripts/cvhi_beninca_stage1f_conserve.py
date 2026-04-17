"""Stage 1f - Mass conservation prior (no per-species rate).

Physics: Beninca closed mesocosm → visible total biomass + nutrients ≈ conserved.
If encoder_h correctly represents hidden biomass (up to a scale),
then adding h to visible total should REDUCE temporal variance.

Loss:
    adjusted_total(t) = sum_visible_channels(t) + softplus(c) * h(t)
    L_conserve = lambda * Var_t(adjusted_total)

Red line preserved:
  - hidden_true never used
  - encoder never sees any derived target
  - c is learnable, no prior
  - Only constraint: "if h is correct, total should be stable"

Baseline config: Stage 1b (RMSE log + input dropout + G_anchor_first, no MTE).
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
LAMBDA_CONSERVE = 0.1   # conservation loss weight

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


def train_one(visible, hidden, seed, device,
              lam_conserve=LAMBDA_CONSERVE, epochs=EPOCHS):
    torch.manual_seed(seed)
    T, N = visible.shape
    x_full = torch.tensor(visible, dtype=torch.float32, device=device).unsqueeze(0)

    model = make_model(N, device)
    # Learnable scale param: softplus ensures positive
    c_raw = nn.Parameter(torch.tensor(0.0, device=device))
    params = list(model.parameters()) + [c_raw]
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

        # Input dropout aug
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

        # === Conservation loss ===
        if lam_conserve > 0 and h_w > 0:
            # visible_total at each t: sum over all channels (normalized)
            visible_total = x_train.sum(dim=-1)       # (B, T)
            # h_samples: (S, B, T). Use mean over samples for stability.
            h_mean = tr_out["h_samples"].mean(dim=0)   # (B, T_train)
            T_h = h_mean.shape[-1]
            # Slice visible_total to match h's range
            visible_total_slice = visible_total[:, :T_h]
            c_pos = F.softplus(c_raw)
            adjusted = visible_total_slice + c_pos * h_mean   # (B, T_h)
            # Variance across time (target: low variance if conservation holds)
            L_conserve = adjusted.var(dim=-1).mean()
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
            best_c = c_raw.detach().cpu().clone()

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        out_eval = model(x_full, n_samples=30, rollout_K=3)
        h_mean = out_eval["h_samples"].mean(dim=0)[0].cpu().numpy()
    pear, _ = evaluate(h_mean, hidden)
    d = hidden_true_substitution(model, visible, hidden, device)

    final_c = float(F.softplus(best_c).item()) if best_c is not None else float("nan")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "pearson": pear,
        "d_ratio": d["recon_true_scaled"] / d["recon_encoder"],
        "val_recon": best_val,
        "learned_c": final_c,
    }


def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"runs/{ts}_beninca_stage1f")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_dir}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    full, species, days = load_beninca()

    print(f"\n=== Stage 1f: Mass Conservation (no per-species rate) ===")
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
                print(f"  seed={s}  P={r['pearson']:+.3f}  d_r={r['d_ratio']:.2f}  "
                      f"val={r['val_recon']:.4f}  c={r['learned_c']:.3f}  ({dt:.1f}s)")
            except Exception as e:
                print(f"  seed={s}  FAILED: {e}")
                import traceback; traceback.print_exc()
                r = {"pearson": float("nan"), "d_ratio": float("nan"),
                     "val_recon": float("nan"), "learned_c": float("nan")}
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

    print(f"\n{'='*100}")
    print("STAGE 1f RESULTS - Mass Conservation Prior")
    print('='*100)
    print(f"{'Species':<18}{'P2':<10}{'S1b':<10}{'S1f':<12}{'D vs P2':<12}{'D vs S1b':<12}")
    sum_p2=0; sum_s1b=0; sum_s1f=0; cnt=0
    for h in SPECIES_ORDER:
        rs = all_results[h]
        P = np.array([r["pearson"] for r in rs if not np.isnan(r["pearson"])])
        s1f = float(P.mean()) if len(P) else float("nan")
        print(f"{h:<18}{phase2[h]:<+10.3f}{stage1b[h]:<+10.3f}{s1f:<+12.3f}"
              f"{s1f-phase2[h]:<+12.3f}{s1f-stage1b[h]:<+12.3f}")
        if not np.isnan(s1f):
            sum_p2+=phase2[h]; sum_s1b+=stage1b[h]; sum_s1f+=s1f; cnt+=1

    avg_p2=sum_p2/cnt; avg_s1b=sum_s1b/cnt; avg_s1f=sum_s1f/cnt
    print(f"\nOverall:          P2={avg_p2:+.4f}  S1b={avg_s1b:+.4f}  S1f={avg_s1f:+.4f}")
    print(f"D(S1f-S1b)={avg_s1f-avg_s1b:+.4f}")

    if avg_s1f > avg_s1b + 0.005:
        verdict = "[OK] Stage 1f better than S1b - mass conservation works"
    elif avg_s1f > avg_s1b - 0.005:
        verdict = "[NEUTRAL] Stage 1f ~= S1b - no harm but no gain"
    else:
        verdict = "[FAIL] Stage 1f worse than S1b"
    print(f"\n{verdict}\n{'='*100}")

    with open(out_dir / "summary.md", "w", encoding="utf-8") as f:
        f.write("# Beninca Stage 1f - Mass Conservation Prior\n\n")
        f.write(f"Seeds: {len(SEEDS)}, Epochs: {EPOCHS}, lambda_conserve={LAMBDA_CONSERVE}\n\n")
        f.write("Loss: L_conserve = lambda * Var_t(visible_total + softplus(c)*h)\n")
        f.write("No per-species rate prior. Physics: closed mesocosm conservation.\n\n")
        f.write("| Species | P2 | S1b | **S1f** | D vs S1b |\n|---|---|---|---|---|\n")
        for h in SPECIES_ORDER:
            rs = all_results[h]
            P = np.array([r["pearson"] for r in rs if not np.isnan(r["pearson"])])
            s1f = float(P.mean()) if len(P) else float("nan")
            f.write(f"| {h} | {phase2[h]:+.3f} | {stage1b[h]:+.3f} | "
                    f"**{s1f:+.3f}** | {s1f-stage1b[h]:+.3f} |\n")
        f.write(f"\n**Overall**: P2={avg_p2:+.4f}, S1b={avg_s1b:+.4f}, "
                f"**S1f={avg_s1f:+.4f}**, D(S1f-S1b)={avg_s1f-avg_s1b:+.4f}\n\n")
        f.write(f"**{verdict}**\n")

    with open(out_dir / "raw.json", "w") as f:
        json.dump(all_results, f, indent=2, default=float)
    print(f"\n[OK] {out_dir}")


if __name__ == "__main__":
    main()
