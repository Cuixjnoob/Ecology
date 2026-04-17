"""Fill Huisman h_dyn on sp3/sp5/sp6 (we only tested sp1/sp2/sp4 before)."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from scripts.cvhi_huisman_hdyn import train_with_hdyn, load_huisman, EPOCHS


SEEDS = [42, 123, 456]
LAMBDAS = [0.0, 0.3]
MISSING_SPECIES_IDX = [2, 4, 5]   # sp3, sp5, sp6
MISSING_NAMES = ["sp3", "sp5", "sp6"]


def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"runs/{ts}_huisman_hdyn_complete")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_dir}")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    full = load_huisman()

    results = {name: {lam: [] for lam in LAMBDAS} for name in MISSING_NAMES}
    total = len(MISSING_NAMES) * len(LAMBDAS) * len(SEEDS)
    run_i = 0

    for sp_idx, sp_name in zip(MISSING_SPECIES_IDX, MISSING_NAMES):
        visible = np.delete(full, sp_idx, axis=1)
        hidden = full[:, sp_idx]
        for lam in LAMBDAS:
            for seed in SEEDS:
                run_i += 1
                print(f"[{run_i}/{total}] {sp_name}  lam={lam}  seed={seed}")
                t0 = datetime.now()
                try:
                    r = train_with_hdyn(visible, hidden, seed, device, lam)
                    dt = (datetime.now() - t0).total_seconds()
                    print(f"  P={r['pearson']:+.4f}  d_r={r['d_ratio']:.2f}  "
                          f"hdyn_corr={r['hdyn_consistency_corr']:+.3f}  ({dt:.1f}s)")
                except Exception as e:
                    print(f"  FAILED: {e}")
                    r = {"pearson": float("nan"), "d_ratio": float("nan"),
                         "val_recon": float("nan"), "h_mean": None,
                         "hdyn_final_loss": float("nan"),
                         "hdyn_consistency_corr": float("nan")}
                r["seed"] = seed; r["lam"] = lam
                results[sp_name][lam].append(r)

    # Print summary combined with earlier results
    # V1 earlier results (at λ=0.3):
    prev_hdyn = {"sp1": 0.303, "sp2": 0.809, "sp4": 0.607}
    prev_base = {"sp1": 0.179, "sp2": 0.677, "sp4": 0.356}

    print(f"\n{'='*90}\nFULL HUISMAN 6-SPECIES h_dyn SUMMARY\n{'='*90}")
    print(f"{'Species':<10}{'base (λ=0)':<14}{'hdyn (λ=0.3)':<14}{'Δ':<10}{'max hdyn':<12}{'hdyn_corr':<10}")

    full_base = {}
    full_hdyn = {}
    for sp in ["sp1", "sp2", "sp3", "sp4", "sp5", "sp6"]:
        if sp in MISSING_NAMES:
            rs_base = results[sp][0.0]
            rs_hdyn = results[sp][0.3]
            b = np.mean([r["pearson"] for r in rs_base])
            h_mean = np.mean([r["pearson"] for r in rs_hdyn])
            h_max = np.max([r["pearson"] for r in rs_hdyn])
            c = np.mean([r["hdyn_consistency_corr"] for r in rs_hdyn])
        else:
            b = prev_base[sp]
            h_mean = prev_hdyn[sp]
            h_max = h_mean   # approximation for earlier data
            c = float("nan")
        full_base[sp] = b
        full_hdyn[sp] = h_mean
        delta = h_mean - b
        print(f"{sp:<10}{b:<+14.3f}{h_mean:<+14.3f}{delta:<+10.3f}{h_max:<+12.3f}{c:<+10.3f}")

    base_mean = np.mean(list(full_base.values()))
    hdyn_mean = np.mean(list(full_hdyn.values()))
    print(f"\nMean across 6 species:  base={base_mean:+.4f}  hdyn={hdyn_mean:+.4f}  "
          f"Δ={hdyn_mean-base_mean:+.4f}")

    # Plot bar chart
    species_list = ["sp1", "sp2", "sp3", "sp4", "sp5", "sp6"]
    base_vals = [full_base[s] for s in species_list]
    hdyn_vals = [full_hdyn[s] for s in species_list]

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    x = np.arange(len(species_list))
    w = 0.35
    ax.bar(x - w/2, base_vals, w, color="#90a4ae", label="baseline (no h_dyn)")
    ax.bar(x + w/2, hdyn_vals, w, color="#c62828", label="+ h_dyn (λ=0.3)")
    for i, (b, h) in enumerate(zip(base_vals, hdyn_vals)):
        ax.text(i - w/2, b + 0.02, f"{b:.2f}", ha="center", fontsize=9)
        ax.text(i + w/2, h + 0.02, f"{h:.2f}", ha="center", fontsize=9)
    ax.axhline(0.6, color="green", ls="--", alpha=0.5, label="target 0.6")
    ax.set_xticks(x); ax.set_xticklabels(species_list)
    ax.set_ylabel("Pearson")
    ax.set_title(f"Huisman 1999 (λ=0.043): h_dyn all 6 species\n"
                 f"baseline mean={base_mean:+.3f} → h_dyn mean={hdyn_mean:+.3f}")
    ax.legend(); ax.grid(alpha=0.25, axis="y")
    fig.savefig(out_dir / "fig_full_species.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    with open(out_dir / "summary.md", "w", encoding="utf-8") as f:
        f.write("# Huisman 6-species h_dyn complete results\n\n")
        f.write("| Species | baseline | + h_dyn (λ=0.3) | Δ |\n|---|---|---|---|\n")
        for sp in species_list:
            f.write(f"| {sp} | {full_base[sp]:+.3f} | {full_hdyn[sp]:+.3f} | "
                    f"{full_hdyn[sp]-full_base[sp]:+.3f} |\n")
        f.write(f"\n**Mean**: baseline={base_mean:+.4f}, hdyn={hdyn_mean:+.4f}, "
                f"Δ={hdyn_mean-base_mean:+.4f}\n")

    def to_ser(v):
        if isinstance(v, (int, float, np.floating)):
            return float(v)
        if isinstance(v, np.ndarray):
            return v.tolist()
        return v
    dump = {sp: {str(lam): [{k: to_ser(v) for k, v in r.items() if k != "h_mean"}
                              for r in rs] for lam, rs in d.items()}
            for sp, d in results.items()}
    with open(out_dir / "raw.json", "w") as f:
        json.dump(dump, f, indent=2, default=float)
    print(f"\n[OK] {out_dir}")


if __name__ == "__main__":
    main()
