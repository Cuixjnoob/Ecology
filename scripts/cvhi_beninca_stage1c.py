"""Stage 1c — Corrected MTE shape prior on f_visible (NOT G).

Stage 1 failed (mean 0.080) 原因 (Agent 深读 Glazier/Kremer/Clarke/Brown 后定位):
  1. 应用在 G 上 (G 是 hidden-coupling, MTE 不管)
  2. 用 universal b=0.75 → 对 pelagic 物种 5 倍过强
  3. 强制绝对 magnitude (Clarke 2025: B_0 不可识别)

Stage 1c 修正:
  ✓ 在 f_visible 上约束 intrinsic rate shape
  ✓ Taxon-specific b (Glazier 2005 + Kremer 2017):
      Bacteria   b=0.60  →  b-1 = -0.40
      Picophyto  b=0.95  →  b-1 = -0.05  (Kremer 实测 -0.054)
      Nanophyto  b=0.95  →  b-1 = -0.05
      Filam_diat b=0.95  →  b-1 = -0.05
      Rotifers   b=0.88  →  b-1 = -0.12  (pelagic boost, Glazier)
      Cyclopoids b=0.88  →  b-1 = -0.12
      Calanoids  b=0.88  →  b-1 = -0.12
      Harpacti.  b=0.75  →  b-1 = -0.25  (benthic copepod)
      Ostracods  b=0.75  →  b-1 = -0.25
  ✓ Shape-only (Pearson corr distance), 不约束 absolute
  ✓ λ 小 (0.02)
  ✓ 保留 RMSE + aug (Stage 1b 验证有效)
  ✗ 不动 G (MTE 不管)
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


SEEDS_5 = [42, 123, 456, 789, 2024]

BEST_HP = dict(
    encoder_d=96, encoder_blocks=3, encoder_dropout=0.1,
    takens_lags=(1, 2, 4, 8),
    lr=0.0006033475528697158, epochs=500,
    lam_smooth=0.02, lam_kl=0.017251789430967935,
    lam_hf=0.2, min_energy=0.14353013693386804,
    lam_cf=9.517725868477207,
)


SPECIES_ORDER = ["Cyclopoids", "Calanoids", "Rotifers", "Nanophyto",
                 "Picophyto", "Filam_diatoms", "Ostracods",
                 "Harpacticoids", "Bacteria"]

# Approx body mass (μg dry weight per individual) — order of magnitude from literature
BODY_MASS_UG = {
    "Bacteria":      1e-6,
    "Picophyto":     1e-5,
    "Nanophyto":     1e-3,
    "Filam_diatoms": 1e-2,
    "Rotifers":      0.5,
    "Harpacticoids": 5.0,
    "Cyclopoids":    20.0,
    "Calanoids":     50.0,
    "Ostracods":     50.0,
}

# Glazier 2005 / Kremer 2017 corrected b
TAXON_B = {
    "Bacteria":      0.60,
    "Picophyto":     0.95,
    "Nanophyto":     0.95,
    "Filam_diatoms": 0.95,
    "Rotifers":      0.88,
    "Cyclopoids":    0.88,
    "Calanoids":     0.88,
    "Harpacticoids": 0.75,
    "Ostracods":     0.75,
}


def compute_mte_target(visible_species):
    """Given ordered list of visible species/channel names, return (N,) target log_r tensor.

    log r_i = (b_i - 1) * log10(M_i)   (constant offset absorbed by correlation).
    Non-species channels (nutrients) → NaN (skipped in correlation).
    Bacteria: clip |log_r| to 0.6 to avoid outlier-driven correlation leverage
              (Clarke 2025 notes bacterial scaling is highly uncertain).
    """
    vals = []
    for s in visible_species:
        if s not in TAXON_B:
            vals.append(float("nan"))
            continue
        b = TAXON_B[s]
        M = BODY_MASS_UG[s]
        log_r = (b - 1.0) * np.log10(M)
        # clip outliers so no single point dominates correlation
        log_r = float(np.clip(log_r, -0.6, 0.6))
        vals.append(log_r)
    return torch.tensor(vals, dtype=torch.float32)


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


def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"runs/{ts}_beninca_stage1c")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_dir}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    full, species, days = load_beninca()

    print("\n=== Stage 1c: Corrected MTE (shape-only, f_visible, taxon-specific b) ===")
    print("Taxon-specific scaling (b-1):")
    for s in SPECIES_ORDER:
        log_r = (TAXON_B[s] - 1.0) * np.log10(BODY_MASS_UG[s])
        print(f"  {s:<16s}  b={TAXON_B[s]:.2f}  M={BODY_MASS_UG[s]:.2e}μg  log_r={log_r:+.3f}")

    loss_kwargs = dict(
        beta_kl=BEST_HP["lam_kl"], lam_smooth=BEST_HP["lam_smooth"],
        lam_hf=BEST_HP["lam_hf"], min_energy=BEST_HP["min_energy"],
        lam_necessary=BEST_HP["lam_cf"], lam_shuffle=BEST_HP["lam_cf"]*0.6,
        margin_null=0.002, margin_shuf=0.001,
        lam_rmse_log=0.1,        # ✓ RMSE (Stage 1b 证实)
        lam_mte_prior=0.0,       # ✗ disable deprecated MTE on G
        lam_mte_shape=0.02,      # ✓ corrected shape prior
    )

    all_results = {}
    for h_name in SPECIES_ORDER:
        h_idx = species.index(h_name)
        visible = np.delete(full, h_idx, axis=1)
        hidden = full[:, h_idx]
        # visible species names in order (全部除去 h_name)
        visible_species = [s for s in species if s != h_name]
        mte_target = compute_mte_target(visible_species)
        print(f"\n--- {h_name} (hidden) ---  visible target shape: {list(mte_target.numpy().round(3))}")
        rs = []
        for s in SEEDS_5:
            torch.manual_seed(s)
            model = make_model(visible.shape[1], device)
            t0 = datetime.now()
            try:
                r = train_one_fast(
                    model, visible, hidden, device=device,
                    epochs=BEST_HP["epochs"], lr=BEST_HP["lr"],
                    input_dropout_prob=0.05,
                    mte_target_log_r=mte_target,
                    use_compile=True,
                    use_ema=False,
                    use_snapshot_ensemble=False,
                    **loss_kwargs,
                )
                dt = (datetime.now() - t0).total_seconds()
                print(f"  seed={s}  P={r['pearson']:+.3f}  d_r={r['d_ratio']:.2f}  ({dt:.1f}s)")
            except Exception as e:
                print(f"  seed={s}  FAILED: {e}")
                import traceback; traceback.print_exc()
                r = {"pearson": float("nan"), "val_recon": float("nan"),
                     "d_ratio": float("nan"), "per_method": {}}
            rs.append({k: float(v) if isinstance(v,(int,float,np.floating)) else v
                       for k, v in r.items() if k != "h_mean"})
            del model
            torch.cuda.empty_cache()
        all_results[h_name] = rs

    # Baselines
    phase2 = {"Cyclopoids": 0.064, "Calanoids": 0.159, "Rotifers": 0.056,
              "Nanophyto": 0.204, "Picophyto": 0.062, "Filam_diatoms": 0.056,
              "Ostracods": 0.125, "Harpacticoids": 0.129, "Bacteria": 0.171}
    stage1b = {"Cyclopoids": 0.055, "Calanoids": 0.173, "Rotifers": 0.080,
               "Nanophyto": 0.141, "Picophyto": 0.116, "Filam_diatoms": 0.031,
               "Ostracods": 0.272, "Harpacticoids": 0.120, "Bacteria": 0.197}

    print(f"\n{'='*95}")
    print("STAGE 1c RESULTS — Corrected MTE (shape-only, f_visible)")
    print('='*95)
    print(f"{'Species':<18s}{'Phase2':<10s}{'Stage1b':<12s}{'Stage1c':<12s}{'Δ vs P2':<10s}{'Δ vs S1b':<10s}")
    sum_p2 = 0; sum_s1b = 0; sum_s1c = 0; count = 0
    for h in SPECIES_ORDER:
        rs = all_results[h]
        P = np.array([r["pearson"] for r in rs if not np.isnan(r["pearson"])])
        s1c_mean = float(P.mean()) if len(P) else float("nan")
        d_p2 = s1c_mean - phase2[h]
        d_s1b = s1c_mean - stage1b[h]
        print(f"{h:<18s}{phase2[h]:<+10.3f}{stage1b[h]:<+12.3f}{s1c_mean:<+12.3f}"
              f"{d_p2:<+10.3f}{d_s1b:<+10.3f}")
        if not np.isnan(s1c_mean):
            sum_p2 += phase2[h]; sum_s1b += stage1b[h]; sum_s1c += s1c_mean; count += 1

    avg_p2 = sum_p2/count; avg_s1b = sum_s1b/count; avg_s1c = sum_s1c/count
    print(f"\nOverall:          P2={avg_p2:+.4f}  S1b={avg_s1b:+.4f}  S1c={avg_s1c:+.4f}")
    print(f"Δ(S1c-P2) = {avg_s1c-avg_p2:+.4f}   Δ(S1c-S1b) = {avg_s1c-avg_s1b:+.4f}")

    print(f"\n{'='*95}")
    if avg_s1c > avg_s1b + 0.005:
        print("✓ Stage 1c 超过 Stage 1b — 修正后的 MTE shape prior 有益")
    elif avg_s1c > avg_s1b - 0.005:
        print("≈ Stage 1c ~= Stage 1b — MTE shape prior 中性 (无副作用, 可作学术 prior)")
    else:
        print("✗ Stage 1c < Stage 1b — shape prior 也有副作用, 需重新设计")
    print('='*95)

    with open(out_dir / "summary.md", "w", encoding="utf-8") as f:
        f.write("# Beninca Stage 1c — Corrected MTE (shape on f_visible)\n\n")
        f.write(f"Seeds: {len(SEEDS_5)}, Epochs: {BEST_HP['epochs']}, λ_mte_shape=0.02\n\n")
        f.write("## Design\n\n")
        f.write("- MTE applied to f_visible intrinsic rate (NOT G), via Pearson-correlation distance\n")
        f.write("- Shape only (no absolute magnitude constraint; Clarke 2025)\n")
        f.write("- Taxon-specific b (Glazier 2005 / Kremer 2017)\n\n")
        f.write("| Species | b | M (μg) | log r target |\n|---|---|---|---|\n")
        for s in SPECIES_ORDER:
            log_r = (TAXON_B[s]-1.0) * np.log10(BODY_MASS_UG[s])
            f.write(f"| {s} | {TAXON_B[s]:.2f} | {BODY_MASS_UG[s]:.1e} | {log_r:+.3f} |\n")
        f.write("\n## Results\n\n")
        f.write("| Species | Phase 2 | Stage 1b | Stage 1c | Δ vs P2 | Δ vs S1b |\n")
        f.write("|---|---|---|---|---|---|\n")
        for h in SPECIES_ORDER:
            rs = all_results[h]
            P = np.array([r["pearson"] for r in rs if not np.isnan(r["pearson"])])
            s1c_mean = float(P.mean()) if len(P) else float("nan")
            f.write(f"| {h} | {phase2[h]:+.3f} | {stage1b[h]:+.3f} | {s1c_mean:+.3f} | "
                    f"{s1c_mean-phase2[h]:+.3f} | {s1c_mean-stage1b[h]:+.3f} |\n")
        f.write(f"\n**Overall**: P2={avg_p2:+.4f}, S1b={avg_s1b:+.4f}, S1c={avg_s1c:+.4f}\n")

    with open(out_dir / "raw.json", "w") as f:
        json.dump(all_results, f, indent=2, default=float)
    print(f"\n[OK] {out_dir}")


if __name__ == "__main__":
    main()
