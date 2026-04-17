"""Stage 1b 快测: 隔离 MTE 影响.

配置: Phase 2 HP + RMSE log + input augmentation, **NO MTE prior**
用 fast_utils 加速 (torch.compile + val reuse 优化).

对比:
  Phase 2 baseline:        0.114 overall mean
  Stage 1 (全生态 prior):    0.080 (失败)
  Stage 1b (无 MTE):       ???  ← 此次要测的
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
from scripts.load_beninca import load_beninca
from scripts.train_utils_fast import train_one_fast


SEEDS_3 = [42, 123, 456]

BEST_HP = dict(
    encoder_d=96, encoder_blocks=3, encoder_dropout=0.1,
    takens_lags=(1, 2, 4, 8),
    lr=0.0006033475528697158, epochs=500,
    lam_smooth=0.02, lam_kl=0.017251789430967935,
    lam_hf=0.2, min_energy=0.14353013693386804,
    lam_cf=9.517725868477207,
)


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
        hierarchical_h=False,     # no hierarchical
    ).to(device)


def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"runs/{ts}_beninca_stage1b")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_dir}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    full, species, days = load_beninca()
    SPECIES = ["Cyclopoids", "Calanoids", "Rotifers", "Nanophyto",
                "Picophyto", "Filam_diatoms", "Ostracods",
                "Harpacticoids", "Bacteria"]

    # Stage 1b config: RMSE + aug, NO MTE, NO EMA/snapshot (pure isolation test)
    config_label = "RMSE+aug only (NO MTE)"
    print(f"Config: {config_label}")
    loss_kwargs = dict(
        beta_kl=BEST_HP["lam_kl"], lam_smooth=BEST_HP["lam_smooth"],
        lam_hf=BEST_HP["lam_hf"], min_energy=BEST_HP["min_energy"],
        lam_necessary=BEST_HP["lam_cf"], lam_shuffle=BEST_HP["lam_cf"]*0.6,
        margin_null=0.002, margin_shuf=0.001,
        lam_rmse_log=0.1,       # ✓ RMSE log
        lam_mte_prior=0.0,       # ✗ NO MTE
    )

    all_results = {}
    for h_name in SPECIES:
        h_idx = species.index(h_name)
        visible = np.delete(full, h_idx, axis=1)
        hidden = full[:, h_idx]
        print(f"\n--- {h_name} ---")
        rs = []
        for s in SEEDS_3:
            torch.manual_seed(s)
            model = make_model(visible.shape[1], device)
            t0 = datetime.now()
            try:
                r = train_one_fast(
                    model, visible, hidden, device=device,
                    epochs=BEST_HP["epochs"], lr=BEST_HP["lr"],
                    input_dropout_prob=0.05,   # ✓ augmentation
                    mte_prior_target=None,
                    use_compile=True,
                    use_ema=False,              # isolation test
                    use_snapshot_ensemble=False,
                    **loss_kwargs,
                )
                dt = (datetime.now() - t0).total_seconds()
                print(f"  seed={s}  P={r['pearson']:+.3f}  d_r={r['d_ratio']:.2f}  ({dt:.1f}s)")
            except Exception as e:
                print(f"  seed={s}  FAILED: {e}")
                r = {"pearson": float("nan"), "val_recon": float("nan"), "d_ratio": float("nan"),
                     "per_method": {}}
            rs.append({k: float(v) if isinstance(v,(int,float,np.floating)) else v
                        for k, v in r.items() if k != "h_mean"})
            del model
            torch.cuda.empty_cache()
        all_results[h_name] = rs

    # Summary vs Phase 2 baseline
    phase2 = {"Cyclopoids": 0.064, "Calanoids": 0.159, "Rotifers": 0.056,
              "Nanophyto": 0.204, "Picophyto": 0.062, "Filam_diatoms": 0.056,
              "Ostracods": 0.125, "Harpacticoids": 0.129, "Bacteria": 0.171}
    stage1_orig = {"Cyclopoids": 0.062, "Calanoids": 0.102, "Rotifers": 0.046,
                    "Nanophyto": 0.072, "Picophyto": 0.105, "Filam_diatoms": 0.044,
                    "Ostracods": 0.146, "Harpacticoids": 0.063, "Bacteria": 0.082}

    print(f"\n{'='*90}")
    print(f"STAGE 1b RESULTS — RMSE + aug only (NO MTE)")
    print(f"{'='*90}")
    print(f"{'Species':<18s}{'Phase2':<10s}{'Stage1':<10s}{'Stage1b':<12s}{'Δ vs P2':<10s}{'Δ vs S1':<10s}")
    sum_p2 = 0; sum_s1 = 0; sum_s1b = 0; count = 0
    for h in SPECIES:
        rs = all_results[h]
        P = np.array([r["pearson"] for r in rs if not np.isnan(r["pearson"])])
        s1b_mean = P.mean() if len(P) else float("nan")
        d_p2 = s1b_mean - phase2[h]
        d_s1 = s1b_mean - stage1_orig[h]
        print(f"{h:<18s}{phase2[h]:<+10.3f}{stage1_orig[h]:<+10.3f}{s1b_mean:<+12.3f}"
               f"{d_p2:<+10.3f}{d_s1:<+10.3f}")
        if not np.isnan(s1b_mean):
            sum_p2 += phase2[h]; sum_s1 += stage1_orig[h]; sum_s1b += s1b_mean; count += 1

    print(f"\nOverall:          P2={sum_p2/count:+.4f}  S1={sum_s1/count:+.4f}  "
           f"S1b={sum_s1b/count:+.4f}")
    diff_p2 = sum_s1b/count - sum_p2/count
    diff_s1 = sum_s1b/count - sum_s1/count
    print(f"Δ(S1b-P2)={diff_p2:+.4f}  Δ(S1b-S1)={diff_s1:+.4f}")

    # Conclusion
    print(f"\n{'='*90}")
    if sum_s1b/count > sum_p2/count + 0.01:
        print("✓ Stage 1b 超过 Phase 2 — RMSE+aug 有益, MTE 是元凶")
    elif abs(sum_s1b/count - sum_p2/count) < 0.01:
        print("≈ Stage 1b 接近 Phase 2 — RMSE+aug 中性, MTE 是元凶")
    else:
        print("✗ Stage 1b < Phase 2 — RMSE 或 aug 也有问题")
    print(f"{'='*90}")

    with open(out_dir / "summary.md", "w", encoding="utf-8") as f:
        f.write("# Beninca Stage 1b — RMSE + aug, NO MTE\n\n")
        f.write(f"Seeds: 3, Epochs: 500, fast_utils enabled\n\n")
        f.write("| Species | Phase 2 | Stage 1 | Stage 1b | Δ vs P2 | Δ vs S1 |\n|---|---|---|---|---|---|\n")
        for h in SPECIES:
            rs = all_results[h]
            P = np.array([r["pearson"] for r in rs if not np.isnan(r["pearson"])])
            s1b_mean = float(P.mean()) if len(P) else float("nan")
            f.write(f"| {h} | {phase2[h]:+.3f} | {stage1_orig[h]:+.3f} | {s1b_mean:+.3f} | "
                     f"{s1b_mean - phase2[h]:+.3f} | {s1b_mean - stage1_orig[h]:+.3f} |\n")
        f.write(f"\n**Overall**: P2={sum_p2/count:+.4f}, S1={sum_s1/count:+.4f}, S1b={sum_s1b/count:+.4f}\n")

    with open(out_dir / "raw.json", "w") as f:
        json.dump(all_results, f, indent=2, default=float)
    print(f"\n[OK] {out_dir}")


if __name__ == "__main__":
    main()
