"""Beninca Stage 1: Ecology-informed unsupervised improvements.

严格冻结 HP (from Optuna Trial 13), 严禁训练中用 hidden_true.
加入:
  1. MTE-based G magnitude prior  (body mass 来源文献, 不是监督)
  2. log(x) amplitude-aware reconstruction
  3. Input dropout augmentation (data-level chaos robustness)

跑 9 species × 10 seeds.
"""
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


SEEDS_10 = [42, 123, 456, 789, 2024, 31415, 27182, 65537, 7, 11]

# Phase 1 Optuna Trial 13 best params (固定)
BEST_HP = dict(
    encoder_d=96, encoder_blocks=3, encoder_dropout=0.1,
    takens_lags=(1, 2, 4, 8),
    lr=0.0006033475528697158, epochs=500,
    lam_smooth=0.02, lam_kl=0.017251789430967935,
    lam_hf=0.2, min_energy=0.14353013693386804,
    lam_cf=9.517725868477207,
)

# Body mass (grams, 文献值 for Beninca mesocosm species)
# refs: Peters 1983 "Ecological Implications of Body Size", Cermeno & Figueiras 2008
BODY_MASS = {
    "Bacteria":      1e-15,   # ~1 fg per cell
    "Picophyto":     1e-13,   # ~0.1 pg, ~2 μm
    "Nanophyto":     1e-11,   # ~10 pg, ~10 μm
    "Filam_diatoms": 5e-9,    # filament cluster ~50 μm
    "Rotifers":      5e-7,    # ~0.1-0.3 mm
    "Harpacticoids": 5e-6,    # ~0.5 mm
    "Ostracods":     1e-5,    # ~1 mm
    "Cyclopoids":    2e-5,    # ~1-2 mm
    "Calanoids":     5e-5,    # ~2 mm
}

# Nutrients: no body mass (abiotic)
# For nutrients as visible, use "reference mass" = 0 (no MTE effect)


def compute_mte_prior(species_list, target_idx):
    """MTE: G 对 target species i 的 magnitude ∝ M_i^(-0.25).
    小物种 mass-specific response 大.
    """
    masses = []
    for s in species_list:
        if s in BODY_MASS:
            masses.append(BODY_MASS[s])
        else:
            # nutrient or unknown: use average
            masses.append(1e-10)
    masses = np.array(masses, dtype=np.float32)
    # target = M^(-0.25)
    mte = masses ** (-0.25)
    # normalize
    mte = mte / mte.sum()
    return torch.tensor(mte, dtype=torch.float32)


def alpha_schedule(epoch, epochs, start=0.5, end=0.95):
    f = epoch / max(1, epochs)
    if f <= start: return 1.0
    if f >= end: return 0.0
    return 1.0 - (f - start) / (end - start)


def train_one(visible, hidden_eval, device, seed, hp=BEST_HP,
              mte_target=None, lam_mte=0.0, lam_rmse_log=0.0,
              input_dropout_prob=0.05):
    torch.manual_seed(seed)
    T, N = visible.shape
    x_full = torch.tensor(visible, dtype=torch.float32, device=device).unsqueeze(0)

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

    if mte_target is not None:
        mte_target_dev = mte_target.to(device)
    else:
        mte_target_dev = None

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

        # Stage 1: Input dropout augmentation (training only)
        if input_dropout_prob > 0 and epoch > warmup:
            # randomly mask a few timesteps
            mask = (torch.rand(1, T, 1, device=device) > input_dropout_prob).float()
            x = x_full * mask + (1 - mask) * x_full.mean(dim=1, keepdim=True)
        else:
            x = x_full

        model.train(); opt.zero_grad()
        out = model(x, n_samples=2, rollout_K=K_r)
        tr_out = model.slice_out(out, 0, train_end)
        # propagate 'visible' and 'G' into sliced dict for loss to use
        tr_out["visible"] = out["visible"][:, 0:train_end]
        tr_out["G"] = out["G"][:, 0:train_end]
        losses = model.loss(
            tr_out, beta_kl=hp["lam_kl"], free_bits=0.02,
            margin_null=m_null, margin_shuf=m_shuf,
            lam_necessary=hp["lam_cf"], lam_shuffle=hp["lam_cf"]*0.6,
            lam_energy=2.0, min_energy=hp["min_energy"],
            lam_smooth=hp["lam_smooth"], lam_sparse=0.02,
            h_weight=h_w, lam_rollout=lam_r,
            rollout_weights=(1.0, 0.5, 0.25),
            lam_hf=hp["lam_hf"], lowpass_sigma=6.0,
            lam_rmse_log=lam_rmse_log,
            lam_mte_prior=lam_mte, mte_prior_target=mte_target_dev,
        )
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()

        # Validation (no dropout/augmentation)
        with torch.no_grad():
            out_val = model(x_full, n_samples=2, rollout_K=K_r)
            val_out = model.slice_out(out_val, train_end, T)
            val_out["visible"] = out_val["visible"][:, train_end:T]
            val_out["G"] = out_val["G"][:, train_end:T]
            val_losses = model.loss(val_out, h_weight=1.0,
                margin_null=m_null, margin_shuf=m_shuf,
                lam_necessary=hp["lam_cf"], lam_shuffle=hp["lam_cf"]*0.6,
                lam_energy=2.0, min_energy=hp["min_energy"],
                lam_rollout=lam_r, rollout_weights=(1.0, 0.5, 0.25),
                lam_hf=hp["lam_hf"], lowpass_sigma=6.0,
                lam_rmse_log=0.0, lam_mte_prior=0.0,  # val 只看 recon
            )
            val_recon = val_losses["recon_full"].item()
        if epoch > warmup + 15 and val_recon < best_val:
            best_val = val_recon
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        out_eval = model(x_full, n_samples=30, rollout_K=3)
        h_mean = out_eval["h_samples"].mean(dim=0)[0].cpu().numpy()
    pear, _ = evaluate(h_mean, hidden_eval)
    diag = hidden_true_substitution(model, visible, hidden_eval, device)
    d_ratio = diag["recon_true_scaled"] / diag["recon_encoder"]
    return {"seed": seed, "pearson": pear, "val_recon": best_val, "d_ratio": d_ratio}


def main():
    _configure_matplotlib()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"runs/{ts}_beninca_stage1")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_dir}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    full, species, days = load_beninca()
    print(f"Species order: {species}")

    SPECIES_NAMES = ["Cyclopoids", "Calanoids", "Rotifers", "Nanophyto",
                      "Picophyto", "Filam_diatoms", "Ostracods",
                      "Harpacticoids", "Bacteria"]

    # Stage 1 config
    CONFIG = dict(
        lam_mte=0.5,           # MTE prior strength
        lam_rmse_log=0.1,      # log(x) 重构
        input_dropout_prob=0.05,  # input 增强
    )
    print(f"Stage 1 config: {CONFIG}")

    all_results = {}
    for h_name in SPECIES_NAMES:
        if h_name not in species: continue
        h_idx = species.index(h_name)
        # 构造 visible list without hidden
        visible_species = [s for i, s in enumerate(species) if i != h_idx]
        visible = np.delete(full, h_idx, axis=1)
        hidden = full[:, h_idx]

        # MTE prior for visible species
        mte_target = compute_mte_prior(visible_species, None)

        print(f"\n{'='*70}\n{h_name}  visible={visible.shape}  MTE target sum={mte_target.sum():.3f}\n{'='*70}")
        rs = []
        for s in SEEDS_10:
            t0 = datetime.now()
            r = train_one(visible, hidden, device, s, hp=BEST_HP,
                           mte_target=mte_target, **CONFIG)
            dt = (datetime.now() - t0).total_seconds()
            rs.append(r)
            print(f"  seed={s}  P={r['pearson']:+.3f}  d_r={r['d_ratio']:.2f}  ({dt:.1f}s)")
        all_results[h_name] = rs

    # Summary
    summary = {}
    for h in all_results:
        P = np.array([r["pearson"] for r in all_results[h]])
        D = np.array([r["d_ratio"] for r in all_results[h]])
        V = np.array([r["val_recon"] for r in all_results[h]])
        # val-weighted ensemble
        w = 1.0 / (V + 1e-6); w = w / w.sum()
        p_vw = float((P * w).sum())
        # top-2 by val
        idx_sort = np.argsort(V)
        p_top2 = float(P[idx_sort[:2]].mean())
        summary[h] = {
            "mean_P": float(P.mean()), "std_P": float(P.std(ddof=1)),
            "max_P": float(P.max()), "min_P": float(P.min()),
            "mean_d_r": float(D.mean()),
            "val_weighted_P": p_vw,
            "val_top2_P": p_top2,
        }

    print(f"\n{'='*100}\nBENINCA STAGE 1 RESULTS (MTE + RMSE log + augmentation, 10 seeds)\n{'='*100}")
    print(f"{'Hidden':<18s}{'mean P':<12s}{'std':<8s}{'val_top2':<10s}{'max':<8s}{'min':<8s}{'d_r':<8s}")
    sorted_hiddens = sorted(summary.keys(), key=lambda h: -summary[h]["mean_P"])
    for h in sorted_hiddens:
        s = summary[h]
        print(f"{h:<18s}{s['mean_P']:+.4f}      {s['std_P']:.4f}    "
               f"{s['val_top2_P']:+.4f}    {s['max_P']:+.3f}   {s['min_P']:+.3f}   {s['mean_d_r']:.3f}")

    all_P = [summary[h]["mean_P"] for h in summary]
    all_top2 = [summary[h]["val_top2_P"] for h in summary]
    print(f"\nOverall mean across 9: {np.mean(all_P):+.4f}  (Phase 2 was +0.1141)")
    print(f"Overall val_top2 mean: {np.mean(all_top2):+.4f}  (ensemble bonus)")
    print(f"Overall max across 9:  {max(all_P):+.4f}")

    with open(out_dir / "summary.md", "w", encoding="utf-8") as f:
        f.write("# Beninca Stage 1 — Ecology-informed Unsupervised\n\n")
        f.write(f"Config: {CONFIG}  HP: frozen from Optuna Trial 13\n\n")
        f.write("| Hidden | mean P | std | val_top2 | max | min | d_ratio |\n|---|---|---|---|---|---|---|\n")
        for h in sorted_hiddens:
            s = summary[h]
            f.write(f"| {h} | {s['mean_P']:+.4f} | {s['std_P']:.4f} | "
                     f"{s['val_top2_P']:+.4f} | {s['max_P']:+.3f} | "
                     f"{s['min_P']:+.3f} | {s['mean_d_r']:.3f} |\n")
        f.write(f"\n- Overall mean (unif): {np.mean(all_P):+.4f}\n")
        f.write(f"- Overall val_top2 mean: {np.mean(all_top2):+.4f}\n")

    with open(out_dir / "raw.json", "w") as f:
        json.dump({"config": CONFIG, "best_hp": {k: v if not isinstance(v, tuple) else list(v)
                                                   for k, v in BEST_HP.items()},
                   "body_mass": BODY_MASS,
                   "results": {h: [{k: float(v) if isinstance(v,(int,float,np.floating)) else v
                                    for k,v in r.items()} for r in rs]
                               for h, rs in all_results.items()}},
                   f, indent=2, default=float)
    print(f"\n[OK] {out_dir}")


if __name__ == "__main__":
    main()
