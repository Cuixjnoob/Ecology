"""Phase 2: 用 Optuna 最佳超参 (Trial 13) 跑全 9 hidden × 5 seeds."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import torch

from models.cvhi_residual import CVHI_Residual
from scripts.cvhi_residual_L1L3_diagnostics import (
    evaluate, _configure_matplotlib, hidden_true_substitution,
)
from scripts.load_beninca import load_beninca


SEEDS_5 = [42, 123, 456, 789, 2024]

# Phase 1 Optuna Trial 13 best params
BEST_HP = dict(
    encoder_d=96,
    encoder_blocks=3,
    encoder_dropout=0.1,
    takens_lags=(1, 2, 4, 8),
    lr=0.0006033475528697158,
    epochs=500,
    lam_smooth=0.02,
    lam_kl=0.017251789430967935,
    lam_hf=0.2,
    min_energy=0.14353013693386804,
    lam_cf=9.517725868477207,
)


def alpha_schedule(epoch, epochs, start=0.5, end=0.95):
    f = epoch / max(1, epochs)
    if f <= start: return 1.0
    if f >= end: return 0.0
    return 1.0 - (f - start) / (end - start)


def train_one(visible, hidden_eval, device, seed, hp=BEST_HP):
    torch.manual_seed(seed)
    T, N = visible.shape
    x = torch.tensor(visible, dtype=torch.float32, device=device).unsqueeze(0)

    model = CVHI_Residual(
        num_visible=N,
        encoder_d=hp["encoder_d"], encoder_blocks=hp["encoder_blocks"], encoder_heads=4,
        takens_lags=hp["takens_lags"], encoder_dropout=hp["encoder_dropout"],
        d_species_f=20, f_visible_layers=2, f_visible_top_k=4,
        d_species_G=12, G_field_layers=1, G_field_top_k=3,
        prior_std=1.0, gnn_backbone="mlp",
        use_formula_hints=True, use_G_field=True,
        num_mixture_components=1,
        G_anchor_first=True, G_anchor_sign=+1,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=1e-4)
    epochs = hp["epochs"]
    warmup = int(0.2 * epochs); ramp = max(1, int(0.2 * epochs))

    def lr_lambda(step):
        if step < 50: return step / 50
        p = (step - 50) / max(1, epochs - 50)
        return 0.5 * (1 + np.cos(np.pi * p))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    train_end = int(0.75 * T)
    best_val = float("inf"); best_state = None

    m_null, m_shuf = 0.002, 0.001
    for epoch in range(epochs):
        model.G_anchor_alpha = alpha_schedule(epoch, epochs)
        if epoch < warmup:
            h_w, K_r, lam_r = 0.0, 0, 0.0
        else:
            post = epoch - warmup
            h_w = min(1.0, post / ramp)
            k_ramp = min(1.0, post / (epochs - warmup) * 2)
            K_r = max(1 if h_w > 0 else 0, int(round(k_ramp * 3)))
            lam_r = 0.5 * h_w
        model.train(); opt.zero_grad()
        out = model(x, n_samples=2, rollout_K=K_r)
        tr_out = model.slice_out(out, 0, train_end)
        losses = model.loss(
            tr_out, beta_kl=hp["lam_kl"], free_bits=0.02,
            margin_null=m_null, margin_shuf=m_shuf,
            lam_necessary=hp["lam_cf"], lam_shuffle=hp["lam_cf"]*0.6,
            lam_energy=2.0, min_energy=hp["min_energy"],
            lam_smooth=hp["lam_smooth"], lam_sparse=0.02,
            h_weight=h_w, lam_rollout=lam_r,
            rollout_weights=(1.0, 0.5, 0.25),
            lam_hf=hp["lam_hf"], lowpass_sigma=6.0,
        )
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        with torch.no_grad():
            val_out = model.slice_out(out, train_end, T)
            val_losses = model.loss(val_out, h_weight=1.0,
                margin_null=m_null, margin_shuf=m_shuf,
                lam_necessary=hp["lam_cf"], lam_shuffle=hp["lam_cf"]*0.6,
                lam_energy=2.0, min_energy=hp["min_energy"],
                lam_rollout=lam_r, rollout_weights=(1.0, 0.5, 0.25),
                lam_hf=hp["lam_hf"], lowpass_sigma=6.0,
            )
            val_recon = val_losses["recon_full"].item()
        if epoch > warmup + 15 and val_recon < best_val:
            best_val = val_recon
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        out_eval = model(x, n_samples=30, rollout_K=3)
        h_mean = out_eval["h_samples"].mean(dim=0)[0].cpu().numpy()
    pear, _ = evaluate(h_mean, hidden_eval)
    diag = hidden_true_substitution(model, visible, hidden_eval, device)
    d_ratio = diag["recon_true_scaled"] / diag["recon_encoder"]
    return {"seed": seed, "pearson": pear, "val_recon": best_val, "d_ratio": d_ratio}


def main():
    _configure_matplotlib()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"runs/{ts}_beninca_phase2")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_dir}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    full, species, days = load_beninca()
    # 只跑 9 species (不跑 nutrients)
    SPECIES_NAMES = ["Cyclopoids", "Calanoids", "Rotifers", "Nanophyto",
                      "Picophyto", "Filam_diatoms", "Ostracods",
                      "Harpacticoids", "Bacteria"]

    all_results = {}
    for h_name in SPECIES_NAMES:
        if h_name not in species: continue
        h_idx = species.index(h_name)
        visible = np.delete(full, h_idx, axis=1)
        hidden = full[:, h_idx]
        print(f"\n{'='*70}\n{h_name}  visible={visible.shape}\n{'='*70}")
        rs = []
        for s in SEEDS_5:
            t0 = datetime.now()
            r = train_one(visible, hidden, device, s)
            dt = (datetime.now() - t0).total_seconds()
            rs.append(r)
            print(f"  seed={s}  P={r['pearson']:+.3f}  d_r={r['d_ratio']:.2f}  ({dt:.1f}s)")
        all_results[h_name] = rs

    # Summary
    summary = {}
    for h in all_results:
        P = np.array([r["pearson"] for r in all_results[h]])
        D = np.array([r["d_ratio"] for r in all_results[h]])
        summary[h] = {
            "mean_P": float(P.mean()), "std_P": float(P.std(ddof=1)),
            "max_P": float(P.max()), "min_P": float(P.min()),
            "mean_d_r": float(D.mean()),
        }

    print(f"\n{'='*90}\nBENINCA PHASE 2 RESULTS (best HP from Optuna Trial 13, 5 seeds)\n{'='*90}")
    print(f"{'Hidden':<20s}{'mean P':<12s}{'std':<10s}{'max':<8s}{'min':<8s}{'d_r':<8s}")
    # Sort by mean_P descending
    sorted_hiddens = sorted(summary.keys(), key=lambda h: -summary[h]["mean_P"])
    for h in sorted_hiddens:
        s = summary[h]
        print(f"{h:<20s}{s['mean_P']:+.4f}      {s['std_P']:.4f}    "
               f"{s['max_P']:+.3f}   {s['min_P']:+.3f}   {s['mean_d_r']:.3f}")

    # Overall
    all_P = [p for h in summary for p in [summary[h]["mean_P"]]]
    print(f"\nOverall mean across 9 hidden: {np.mean(all_P):+.4f}")
    print(f"Overall max across 9 hidden:  {max(all_P):+.4f}")

    # Save
    with open(out_dir / "summary.md", "w", encoding="utf-8") as f:
        f.write("# Beninca Phase 2 — Best HP Applied to All 9 Hidden\n\n")
        f.write(f"HP (Optuna Trial 13): {BEST_HP}\n\n")
        f.write("| Hidden | mean P | std | max | min | d_ratio |\n|---|---|---|---|---|---|\n")
        for h in sorted_hiddens:
            s = summary[h]
            f.write(f"| {h} | {s['mean_P']:+.4f} | {s['std_P']:.4f} | "
                     f"{s['max_P']:+.3f} | {s['min_P']:+.3f} | {s['mean_d_r']:.3f} |\n")
        f.write(f"\n**Overall mean**: {np.mean(all_P):+.4f}  \n")
        f.write(f"**Overall max**: {max(all_P):+.4f}\n")

    with open(out_dir / "raw.json", "w") as f:
        json.dump({"best_hp": {k: v if not isinstance(v, tuple) else list(v)
                                 for k, v in BEST_HP.items()},
                   "results": {h: [{k: float(v) if isinstance(v,(int,float,np.floating)) else v
                                    for k,v in r.items()} for r in rs]
                               for h, rs in all_results.items()}},
                   f, indent=2, default=float)
    print(f"\n[OK] {out_dir}")


if __name__ == "__main__":
    main()
