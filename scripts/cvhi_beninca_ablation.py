"""Beninca 大消融实验 — 2026-04-15 夜跑.

在 Stage 1b (mean 0.132) 基础上, 测试生态 prior + 经典 NN 改进 + ML pure-win 堆叠.

Configs:
  A0  baseline          : Phase 2 原始 (RMSE+aug off, no MTE, no klaus)
  A1  +RMSE+aug         : Stage 1b 复现
  A2  +MTE shape        : Stage 1c 复现 (预期 neg)
  A3  +Klausmeier       : Stage 2 复现
  A4  +EMA+Snapshot     : 经典 NN (low-cost)
  A5  +Hier h           : 架构 slow/fast
  A6  Eco combo         : A1+A3 (跳过 A2 若证实有害)
  A7  Classical NN      : A1+A4+A5
  A8  All combined      : A6 + A7

9 species × 5 seeds × 9 configs = 405 runs.
~30s/run = ~3h compute.
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


# 3 seeds for time efficiency (9 configs × 9 species × 3 seeds = 243 runs × ~46s = ~3h).
# After ablation finishes, top-3 configs can be re-run at 5 seeds for tighter CIs.
SEEDS_5 = [42, 123, 456]

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

BODY_MASS_UG = {"Bacteria": 1e-6, "Picophyto": 1e-5, "Nanophyto": 1e-3,
                "Filam_diatoms": 1e-2, "Rotifers": 0.5, "Harpacticoids": 5.0,
                "Cyclopoids": 20.0, "Calanoids": 50.0, "Ostracods": 50.0}
TAXON_B = {"Bacteria": 0.60, "Picophyto": 0.95, "Nanophyto": 0.95,
           "Filam_diatoms": 0.95, "Rotifers": 0.88, "Cyclopoids": 0.88,
           "Calanoids": 0.88, "Harpacticoids": 0.75, "Ostracods": 0.75}
PHYTO = {"Nanophyto", "Picophyto", "Filam_diatoms"}
NUTRIENTS = {"NO2", "NO3", "NH4", "SRP"}


def compute_mte_target(visible_species):
    vals = []
    for s in visible_species:
        if s not in TAXON_B:
            vals.append(float("nan")); continue
        b = TAXON_B[s]; M = BODY_MASS_UG[s]
        log_r = (b - 1.0) * np.log10(M)
        vals.append(float(np.clip(log_r, -0.6, 0.6)))
    return torch.tensor(vals, dtype=torch.float32)


def build_klausmeier_pairs(visible_species):
    pos, neg = [], []
    for i, si in enumerate(visible_species):
        if si not in PHYTO: continue
        for j, sj in enumerate(visible_species):
            if sj not in NUTRIENTS: continue
            pos.append((i, j)); neg.append((j, i))
    return tuple(pos), tuple(neg)


CONFIGS = {
    "A0_baseline":      dict(rmse_aug=False, mte_shape=False, klaus=False, ema=False, hier=False),
    "A1_rmse_aug":      dict(rmse_aug=True,  mte_shape=False, klaus=False, ema=False, hier=False),
    "A2_mte_shape":     dict(rmse_aug=True,  mte_shape=True,  klaus=False, ema=False, hier=False),
    "A3_klausmeier":    dict(rmse_aug=True,  mte_shape=False, klaus=True,  ema=False, hier=False),
    "A4_ema_snap":      dict(rmse_aug=True,  mte_shape=False, klaus=False, ema=True,  hier=False),
    "A5_hier_h":        dict(rmse_aug=True,  mte_shape=False, klaus=False, ema=False, hier=True),
    "A6_eco_combo":     dict(rmse_aug=True,  mte_shape=False, klaus=True,  ema=False, hier=False),
    "A7_classic_nn":    dict(rmse_aug=True,  mte_shape=False, klaus=False, ema=True,  hier=True),
    "A8_all":           dict(rmse_aug=True,  mte_shape=False, klaus=True,  ema=True,  hier=True),
}


def make_model(N, device, hier=False):
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
        hierarchical_h=hier,
    ).to(device)


def run_one(config_name, cfg, h_name, seed, full, species, device):
    h_idx = species.index(h_name)
    visible = np.delete(full, h_idx, axis=1)
    hidden = full[:, h_idx]
    visible_species = [s for s in species if s != h_name]

    input_dropout = 0.05 if cfg["rmse_aug"] else 0.0
    lam_rmse_log = 0.1 if cfg["rmse_aug"] else 0.0
    lam_mte_shape = 0.02 if cfg["mte_shape"] else 0.0
    lam_stoich = 0.02 if cfg["klaus"] else 0.0

    mte_target = compute_mte_target(visible_species) if cfg["mte_shape"] else None
    pos_pairs, neg_pairs = (build_klausmeier_pairs(visible_species) if cfg["klaus"]
                             else ((), ()))

    loss_kwargs = dict(
        beta_kl=BEST_HP["lam_kl"], lam_smooth=BEST_HP["lam_smooth"],
        lam_hf=BEST_HP["lam_hf"], min_energy=BEST_HP["min_energy"],
        lam_necessary=BEST_HP["lam_cf"], lam_shuffle=BEST_HP["lam_cf"]*0.6,
        margin_null=0.002, margin_shuf=0.001,
        lam_rmse_log=lam_rmse_log,
        lam_mte_prior=0.0,
        lam_mte_shape=lam_mte_shape,
        lam_stoich_sign=lam_stoich,
    )

    torch.manual_seed(seed)
    model = make_model(visible.shape[1], device, hier=cfg["hier"])
    t0 = datetime.now()
    try:
        r = train_one_fast(
            model, visible, hidden, device=device,
            epochs=BEST_HP["epochs"], lr=BEST_HP["lr"],
            input_dropout_prob=input_dropout,
            mte_target_log_r=mte_target,
            stoich_pos_pairs=pos_pairs, stoich_neg_pairs=neg_pairs,
            use_compile=True,
            use_ema=cfg["ema"], use_snapshot_ensemble=cfg["ema"],
            **loss_kwargs,
        )
        dt = (datetime.now() - t0).total_seconds()
        out = dict(pearson=float(r["pearson"]), d_ratio=float(r["d_ratio"]),
                   val_recon=float(r["val_recon"]), time=dt,
                   per_method=r.get("per_method", {}))
    except Exception as e:
        import traceback; traceback.print_exc()
        out = dict(pearson=float("nan"), d_ratio=float("nan"),
                   val_recon=float("nan"), time=-1, error=str(e))
    del model
    torch.cuda.empty_cache()
    return out


def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"runs/{ts}_beninca_ablation")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_dir}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    full, species, days = load_beninca()

    results = {c: {sp: [] for sp in SPECIES_ORDER} for c in CONFIGS}

    total = len(CONFIGS) * len(SPECIES_ORDER) * len(SEEDS_5)
    n_done = 0
    t_global = datetime.now()
    for cname, cfg in CONFIGS.items():
        print(f"\n{'='*80}\nCONFIG: {cname}  {cfg}\n{'='*80}")
        for sp in SPECIES_ORDER:
            ps = []
            for s in SEEDS_5:
                r = run_one(cname, cfg, sp, s, full, species, device)
                results[cname][sp].append(r)
                ps.append(r["pearson"])
                n_done += 1
                elapsed = (datetime.now() - t_global).total_seconds()
                eta = elapsed / max(1, n_done) * (total - n_done)
                print(f"  [{n_done}/{total}] {sp:<16s} seed={s} P={r['pearson']:+.3f} "
                      f"({r['time']:.1f}s) ETA {eta/60:.1f}min")
            ps_valid = [p for p in ps if not np.isnan(p)]
            if ps_valid:
                print(f"  → {sp}: mean={np.mean(ps_valid):+.3f} ({len(ps_valid)}/5 seeds)")
            with open(out_dir / "raw.json", "w") as f:
                json.dump(results, f, indent=2, default=float)

    # Summary
    print(f"\n{'='*95}")
    print("ABLATION SUMMARY (mean Pearson per species)")
    print('='*95)
    config_means = {c: [] for c in CONFIGS}
    print(f"{'Species':<18s}" + "".join(f"{c[:13]:<15s}" for c in CONFIGS))
    for sp in SPECIES_ORDER:
        row = f"{sp:<18s}"
        for c in CONFIGS:
            ps = [r["pearson"] for r in results[c][sp] if not np.isnan(r["pearson"])]
            m = float(np.mean(ps)) if ps else float("nan")
            row += f"{m:<+15.3f}"
            if not np.isnan(m):
                config_means[c].append(m)
        print(row)
    print(f"\n{'Overall':<18s}" + "".join(
        f"{(np.mean(config_means[c]) if config_means[c] else float('nan')):<+15.4f}" for c in CONFIGS))

    with open(out_dir / "summary.md", "w", encoding="utf-8") as f:
        f.write("# Beninca Big Ablation — 2026-04-15\n\n")
        f.write(f"Configs: {len(CONFIGS)}, Species: {len(SPECIES_ORDER)}, Seeds: {len(SEEDS_5)}\n\n")
        f.write("## Config spec\n\n")
        f.write("| ID | rmse_aug | mte_shape | klaus | ema | hier |\n|---|---|---|---|---|---|\n")
        for c, cfg in CONFIGS.items():
            f.write(f"| {c} | {cfg['rmse_aug']} | {cfg['mte_shape']} | {cfg['klaus']} | "
                    f"{cfg['ema']} | {cfg['hier']} |\n")
        f.write("\n## Per-species mean Pearson\n\n")
        f.write("| Species | " + " | ".join(CONFIGS.keys()) + " |\n")
        f.write("|" + "---|" * (len(CONFIGS)+1) + "\n")
        for sp in SPECIES_ORDER:
            vals = []
            for c in CONFIGS:
                ps = [r["pearson"] for r in results[c][sp] if not np.isnan(r["pearson"])]
                m = float(np.mean(ps)) if ps else float("nan")
                vals.append(f"{m:+.3f}")
            f.write(f"| {sp} | " + " | ".join(vals) + " |\n")
        f.write(f"\n## Overall (species-averaged)\n\n")
        f.write("| Config | Mean Pearson | vs A0 |\n|---|---|---|\n")
        a0_mean = float(np.mean(config_means["A0_baseline"])) if config_means["A0_baseline"] else 0
        for c in CONFIGS:
            m = float(np.mean(config_means[c])) if config_means[c] else float("nan")
            f.write(f"| {c} | {m:+.4f} | {m-a0_mean:+.4f} |\n")
    print(f"\n[OK] {out_dir}")


if __name__ == "__main__":
    main()
