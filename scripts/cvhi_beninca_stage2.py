"""Stage 2 — Klausmeier-Droop sign priors on f_visible.

Per Klausmeier, Litchman & Levin 2004 L&O (Droop-Liebig):
  - Nutrients (N, P) positively drive phytoplankton growth:
      d μ_phyto / d [nutrient] > 0
  - Phytoplankton consumes nutrients:
      d [nutrient]/dt has negative contribution from phyto biomass

Implementation: soft sign prior via finite-difference on f_visible.
  - For each (phyto_i, nutrient_j) pair: ReLU(-∂base_i/∂x_j)  penalty
  - For each (nutrient_j, phyto_i) pair: ReLU(+∂base_j/∂x_i)  penalty
λ small (0.02).

Baseline config: Stage 1b (RMSE + aug, no MTE).  Purpose: isolate Klausmeier sign prior.
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

PHYTO = {"Nanophyto", "Picophyto", "Filam_diatoms"}
NUTRIENTS = {"NO2", "NO3", "NH4", "SRP"}


def build_klausmeier_pairs(visible_species):
    """Return (pos_pairs, neg_pairs) of channel indices for soft sign prior.

    pos: (phyto_i, nutrient_j) — d base_phyto / d x_nutrient > 0
    neg: (nutrient_j, phyto_i) — d base_nutrient / d x_phyto < 0
    """
    pos_pairs = []
    neg_pairs = []
    for i, s_i in enumerate(visible_species):
        if s_i not in PHYTO:
            continue
        for j, s_j in enumerate(visible_species):
            if s_j not in NUTRIENTS:
                continue
            pos_pairs.append((i, j))   # nutrient→phyto positive
            neg_pairs.append((j, i))   # phyto→nutrient negative
    return tuple(pos_pairs), tuple(neg_pairs)


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
    out_dir = Path(f"runs/{ts}_beninca_stage2")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_dir}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    full, species, days = load_beninca()
    print(f"Full channels: {species}")

    loss_kwargs = dict(
        beta_kl=BEST_HP["lam_kl"], lam_smooth=BEST_HP["lam_smooth"],
        lam_hf=BEST_HP["lam_hf"], min_energy=BEST_HP["min_energy"],
        lam_necessary=BEST_HP["lam_cf"], lam_shuffle=BEST_HP["lam_cf"]*0.6,
        margin_null=0.002, margin_shuf=0.001,
        lam_rmse_log=0.1,
        lam_mte_prior=0.0,
        lam_mte_shape=0.0,      # Stage 2 only isolates Klausmeier
        lam_stoich_sign=0.02,
    )

    all_results = {}
    for h_name in SPECIES_ORDER:
        h_idx = species.index(h_name)
        visible = np.delete(full, h_idx, axis=1)
        visible_species = [s for s in species if s != h_name]
        pos_pairs, neg_pairs = build_klausmeier_pairs(visible_species)
        hidden = full[:, h_idx]
        print(f"\n--- {h_name} (hidden) ---  {len(pos_pairs)} pos pairs, {len(neg_pairs)} neg pairs")
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
                    use_compile=True,
                    use_ema=False, use_snapshot_ensemble=False,
                    stoich_pos_pairs=pos_pairs,
                    stoich_neg_pairs=neg_pairs,
                    **loss_kwargs,
                )
                dt = (datetime.now() - t0).total_seconds()
                print(f"  seed={s}  P={r['pearson']:+.3f}  d_r={r['d_ratio']:.2f}  ({dt:.1f}s)")
            except Exception as e:
                import traceback; traceback.print_exc()
                print(f"  seed={s}  FAILED: {e}")
                r = {"pearson": float("nan"), "val_recon": float("nan"),
                     "d_ratio": float("nan"), "per_method": {}}
            rs.append({k: float(v) if isinstance(v,(int,float,np.floating)) else v
                       for k, v in r.items() if k != "h_mean"})
            del model
            torch.cuda.empty_cache()
        all_results[h_name] = rs

    phase2 = {"Cyclopoids": 0.064, "Calanoids": 0.159, "Rotifers": 0.056,
              "Nanophyto": 0.204, "Picophyto": 0.062, "Filam_diatoms": 0.056,
              "Ostracods": 0.125, "Harpacticoids": 0.129, "Bacteria": 0.171}
    stage1b = {"Cyclopoids": 0.055, "Calanoids": 0.173, "Rotifers": 0.080,
               "Nanophyto": 0.141, "Picophyto": 0.116, "Filam_diatoms": 0.031,
               "Ostracods": 0.272, "Harpacticoids": 0.120, "Bacteria": 0.197}

    print(f"\n{'='*95}")
    print("STAGE 2 — Klausmeier sign priors (nutrient↔phyto)")
    print('='*95)
    sum_p2 = 0; sum_s1b = 0; sum_s2 = 0; count = 0
    print(f"{'Species':<18s}{'Phase2':<10s}{'S1b':<10s}{'S2':<10s}{'Δ S1b':<10s}")
    for h in SPECIES_ORDER:
        rs = all_results[h]
        P = np.array([r["pearson"] for r in rs if not np.isnan(r["pearson"])])
        s2_m = float(P.mean()) if len(P) else float("nan")
        print(f"{h:<18s}{phase2[h]:<+10.3f}{stage1b[h]:<+10.3f}{s2_m:<+10.3f}"
              f"{s2_m-stage1b[h]:<+10.3f}")
        if not np.isnan(s2_m):
            sum_p2 += phase2[h]; sum_s1b += stage1b[h]; sum_s2 += s2_m; count += 1
    print(f"\nOverall:  P2={sum_p2/count:+.4f}  S1b={sum_s1b/count:+.4f}  S2={sum_s2/count:+.4f}")

    with open(out_dir / "summary.md", "w", encoding="utf-8") as f:
        f.write("# Beninca Stage 2 — Klausmeier sign priors\n\n")
        f.write(f"Seeds: {len(SEEDS_5)}, λ_stoich=0.02\n\n")
        f.write("| Species | Phase 2 | Stage 1b | Stage 2 | Δ vs S1b |\n|---|---|---|---|---|\n")
        for h in SPECIES_ORDER:
            rs = all_results[h]
            P = np.array([r["pearson"] for r in rs if not np.isnan(r["pearson"])])
            s2_m = float(P.mean()) if len(P) else float("nan")
            f.write(f"| {h} | {phase2[h]:+.3f} | {stage1b[h]:+.3f} | {s2_m:+.3f} | "
                    f"{s2_m-stage1b[h]:+.3f} |\n")
        f.write(f"\n**Overall**: P2={sum_p2/count:+.4f}, S1b={sum_s1b/count:+.4f}, S2={sum_s2/count:+.4f}\n")
    with open(out_dir / "raw.json", "w") as f:
        json.dump(all_results, f, indent=2, default=float)
    print(f"\n[OK] {out_dir}")


if __name__ == "__main__":
    main()
