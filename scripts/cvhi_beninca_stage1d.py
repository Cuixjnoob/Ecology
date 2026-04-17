"""Stage 1d — Generic food-web sign prior (NO dataset-derived correlations).

Motivation (after Stage 1c 失败):
  - Beninca 2008 Fig 3: 9 物种 Lyapunov 几乎相等 → 不存在 per-species rate ordering
  - Clarke 2025 Table 3: Bacteria b=1.28, 我们给 0.60 (方向反了)
  - Kremer 2017: phyto group-intercept 差异 >> slope
  → MTE 作 per-species quantitative prior 从根本上 ill-posed
  → 转向 sign-level prior (食物网方向), 同 Stage 2 Klausmeier 同源

Design — only use universal biology signs, NO dataset measurements:

  Generic Baltic plankton food web (not from this dataset):

  1. Nutrients (NO2/NO3/NH4/SRP) ↔ Phyto (Pico/Nano/Filam)
       d base_phyto / d x_nutrient > 0   (nutrients 驱动 phyto)
       d base_nutrient / d x_phyto < 0   (phyto 消耗 nutrient)

  2. Bacteria ↔ microbial loop filter-feeders (Rotifers, Ostracods, Harpacticoids)
       d base_filter / d x_bacteria > 0  (Bacteria 作为食物)
       d base_bacteria / d x_filter < 0  (被捕食消耗)

  3. Phyto ↔ Herbivorous zoo (Rotifers, Cyclo, Cala, Harpact, Ostra)
       d base_zoo / d x_phyto > 0        (phyto 作为食物)
       d base_phyto / d x_zoo < 0        (被捕食)

  4. Large copepods ↔ Rotifers (intraguild, 大吃小 generic biology)
       d base_rotifer / d x_copepod < 0
       d base_copepod / d x_rotifer > 0

All pairs are GENERIC biology sign (predator-prey direction), NOT from Beninca
Table 1 correlations. Unsupervised red line preserved.
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

# Functional groups (generic biology, not dataset-specific)
PHYTO       = {"Nanophyto", "Picophyto", "Filam_diatoms"}
NUTRIENTS   = {"NO2", "NO3", "NH4", "SRP"}
HERBIVORE_ZOO = {"Rotifers", "Cyclopoids", "Calanoids", "Harpacticoids", "Ostracods"}
FILTER_FEEDER = {"Rotifers", "Ostracods", "Harpacticoids"}   # eat bacteria/detritus
LARGE_COPEPOD = {"Calanoids", "Cyclopoids"}                    # intraguild predation on small zoo


def build_foodweb_pairs(visible_species):
    """Return (pos_pairs, neg_pairs) enforcing generic food-web sign priors.

    pos_pairs: [(i_target, j_source)] where d base_i / d x_j should be > 0
    neg_pairs: [(i_target, j_source)] where d base_i / d x_j should be < 0

    Only generic biology (predator-prey direction), no dataset correlations.
    """
    pos, neg = [], []
    idx = {s: i for i, s in enumerate(visible_species)}

    # 1. Nutrients ↔ Phyto
    for ph in PHYTO:
        if ph not in idx: continue
        for nut in NUTRIENTS:
            if nut not in idx: continue
            pos.append((idx[ph], idx[nut]))   # nutrient→phyto  +
            neg.append((idx[nut], idx[ph]))   # phyto→nutrient  −

    # 2. Bacteria ↔ filter-feeders (microbial loop)
    if "Bacteria" in idx:
        for ff in FILTER_FEEDER:
            if ff not in idx: continue
            pos.append((idx[ff], idx["Bacteria"]))   # bacteria→filter  +  (食物)
            neg.append((idx["Bacteria"], idx[ff]))   # filter→bacteria  −  (捕食)

    # 3. Phyto ↔ Herbivorous zoo
    for ph in PHYTO:
        if ph not in idx: continue
        for zoo in HERBIVORE_ZOO:
            if zoo not in idx: continue
            pos.append((idx[zoo], idx[ph]))    # phyto→zoo  +
            neg.append((idx[ph], idx[zoo]))    # zoo→phyto  −

    # 4. Large copepod ↔ Rotifers (intraguild, big eat small)
    if "Rotifers" in idx:
        for lc in LARGE_COPEPOD:
            if lc not in idx: continue
            pos.append((idx[lc], idx["Rotifers"]))    # rotifers→largeCop  +
            neg.append((idx["Rotifers"], idx[lc]))    # largeCop→rotifers −

    return tuple(pos), tuple(neg)


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
    out_dir = Path(f"runs/{ts}_beninca_stage1d")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_dir}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    full, species, days = load_beninca()
    print(f"Channels: {species}")

    loss_kwargs = dict(
        beta_kl=BEST_HP["lam_kl"], lam_smooth=BEST_HP["lam_smooth"],
        lam_hf=BEST_HP["lam_hf"], min_energy=BEST_HP["min_energy"],
        lam_necessary=BEST_HP["lam_cf"], lam_shuffle=BEST_HP["lam_cf"]*0.6,
        margin_null=0.002, margin_shuf=0.001,
        lam_rmse_log=0.1,         # keep Stage 1b proven
        lam_mte_prior=0.0,
        lam_mte_shape=0.0,        # drop MTE (Stage 1c failed)
        lam_stoich_sign=0.02,     # food-web sign (Stage 1d)
    )

    all_results = {}
    for h_name in SPECIES_ORDER:
        h_idx = species.index(h_name)
        visible = np.delete(full, h_idx, axis=1)
        visible_species = [s for s in species if s != h_name]
        pos_pairs, neg_pairs = build_foodweb_pairs(visible_species)
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
        # flush incrementally
        with open(out_dir / "raw.json", "w") as f:
            json.dump(all_results, f, indent=2, default=float)

    phase2 = {"Cyclopoids": 0.064, "Calanoids": 0.159, "Rotifers": 0.056,
              "Nanophyto": 0.204, "Picophyto": 0.062, "Filam_diatoms": 0.056,
              "Ostracods": 0.125, "Harpacticoids": 0.129, "Bacteria": 0.171}
    s1b = {"Cyclopoids": 0.055, "Calanoids": 0.173, "Rotifers": 0.080,
           "Nanophyto": 0.141, "Picophyto": 0.116, "Filam_diatoms": 0.031,
           "Ostracods": 0.272, "Harpacticoids": 0.120, "Bacteria": 0.197}
    s1c = {"Cyclopoids": 0.053, "Calanoids": 0.079, "Rotifers": 0.087,
           "Nanophyto": 0.094, "Picophyto": 0.061, "Filam_diatoms": 0.021,
           "Ostracods": 0.169, "Harpacticoids": 0.156, "Bacteria": 0.052}

    print("\n" + "="*95)
    print("STAGE 1d RESULTS - generic food-web sign prior")
    print("="*95)
    print(f"{'Species':<18s}{'P2':<10s}{'S1b':<10s}{'S1c':<10s}{'S1d':<10s}{'dS1b':<10s}")
    sum_p2 = sum_s1b = sum_s1c = sum_s1d = 0.0; count = 0
    for h in SPECIES_ORDER:
        rs = all_results[h]
        P = np.array([r["pearson"] for r in rs if not np.isnan(r["pearson"])])
        s1d_m = float(P.mean()) if len(P) else float("nan")
        print(f"{h:<18s}{phase2[h]:<+10.3f}{s1b[h]:<+10.3f}{s1c[h]:<+10.3f}{s1d_m:<+10.3f}"
              f"{s1d_m-s1b[h]:<+10.3f}")
        if not np.isnan(s1d_m):
            sum_p2 += phase2[h]; sum_s1b += s1b[h]; sum_s1c += s1c[h]; sum_s1d += s1d_m; count += 1
    print(f"\nOverall:          P2={sum_p2/count:+.4f}  S1b={sum_s1b/count:+.4f}  "
          f"S1c={sum_s1c/count:+.4f}  S1d={sum_s1d/count:+.4f}")
    ds1b = sum_s1d/count - sum_s1b/count
    print(f"dS1b = {ds1b:+.4f}")

    with open(out_dir / "summary.md", "w", encoding="utf-8") as f:
        f.write("# Beninca Stage 1d - Generic food-web sign prior\n\n")
        f.write(f"Seeds: {len(SEEDS_5)}, Epochs: {BEST_HP['epochs']}, lam_stoich=0.02\n\n")
        f.write("Unsupervised red line: preserved (only generic biology signs, no dataset correlations)\n\n")
        f.write("| Species | Phase 2 | S1b | S1c | **S1d** | vs S1b |\n")
        f.write("|---|---|---|---|---|---|\n")
        for h in SPECIES_ORDER:
            rs = all_results[h]
            P = np.array([r["pearson"] for r in rs if not np.isnan(r["pearson"])])
            s1d_m = float(P.mean()) if len(P) else float("nan")
            f.write(f"| {h} | {phase2[h]:+.3f} | {s1b[h]:+.3f} | {s1c[h]:+.3f} | "
                    f"**{s1d_m:+.3f}** | {s1d_m-s1b[h]:+.3f} |\n")
        f.write(f"\n**Overall**: P2={sum_p2/count:+.4f}, S1b={sum_s1b/count:+.4f}, "
                f"S1c={sum_s1c/count:+.4f}, **S1d={sum_s1d/count:+.4f}**  "
                f"(Delta S1b = {ds1b:+.4f})\n")
    with open(out_dir / "raw.json", "w") as f:
        json.dump(all_results, f, indent=2, default=float)
    print(f"\n[OK] {out_dir}")


if __name__ == "__main__":
    main()
