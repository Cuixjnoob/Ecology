"""Quantify 'flatness': fraction of time series below X% of peak."""
from __future__ import annotations
import numpy as np
from scripts.load_beninca import load_beninca

SPECIES_ORDER = ["Cyclopoids", "Calanoids", "Rotifers", "Nanophyto",
                 "Picophyto", "Filam_diatoms", "Ostracods",
                 "Harpacticoids", "Bacteria"]


def flatness_stats(x):
    """Return metrics about how 'flat/sparse' the signal is."""
    T = len(x)
    peak = x.max()
    med = np.median(x)
    # How much time below various thresholds of peak
    pct_below_5 = (x < 0.05 * peak).mean()
    pct_below_10 = (x < 0.10 * peak).mean()
    pct_below_25 = (x < 0.25 * peak).mean()
    # Log-ratio magnitude (burst indicator)
    lr = np.log((x[1:] + 1e-6) / (x[:-1] + 1e-6))
    lr_abs = np.abs(lr)
    # Fraction of timesteps carrying most of log-ratio action
    lr_sorted = np.sort(lr_abs)[::-1]
    cumsum = np.cumsum(lr_sorted)
    total = cumsum[-1]
    # How many timesteps contain 80% of the total |log_ratio| magnitude
    frac_for_80pct = (np.argmax(cumsum >= 0.8 * total) + 1) / len(lr_sorted)
    # Effective sample size (ESS) of log-ratio (how much signal concentrates)
    w = lr_abs / (lr_abs.sum() + 1e-8)
    ess = 1.0 / (w ** 2).sum() / len(w)   # 0=concentrated, 1=uniform

    return dict(
        peak=float(peak),
        median=float(med),
        pct_below_5_pct_peak=float(pct_below_5),
        pct_below_10_pct_peak=float(pct_below_10),
        pct_below_25_pct_peak=float(pct_below_25),
        lr_mean=float(lr_abs.mean()),
        lr_max=float(lr_abs.max()),
        frac_for_80pct_burstiness=float(frac_for_80pct),
        ess_normalized=float(ess),
    )


def main():
    full, species, _ = load_beninca()
    species = [str(s) for s in species]

    print("="*100)
    print("BENINCA FLATNESS ANALYSIS")
    print("="*100)
    print("[Interpretation: signal mostly flat when pct_below_10 > 50% or ESS < 0.3]")
    print(f"{'Species':<18}{'peak/med':<12}{'<5% peak':<12}{'<10% peak':<12}"
          f"{'<25% peak':<12}{'frac>80%':<12}{'ESS_lr':<10}")
    for sp in SPECIES_ORDER:
        idx = species.index(sp)
        s = flatness_stats(full[:, idx])
        # peak/median ratio - how extreme are peaks
        pm = s["peak"] / (s["median"] + 1e-8)
        print(f"{sp:<18}{pm:<12.1f}{s['pct_below_5_pct_peak']:<12.1%}"
              f"{s['pct_below_10_pct_peak']:<12.1%}"
              f"{s['pct_below_25_pct_peak']:<12.1%}"
              f"{s['frac_for_80pct_burstiness']:<12.1%}"
              f"{s['ess_normalized']:<10.3f}")

    # Compare with Huisman
    print("\n" + "="*100)
    print("HUISMAN FLATNESS (for contrast)")
    print("="*100)
    h_data = np.load("runs/huisman1999_chaos/trajectories.npz")
    N_all = h_data["N_all"]
    print(f"{'Species':<10}{'peak/med':<12}{'<5% peak':<12}{'<10% peak':<12}"
          f"{'<25% peak':<12}{'frac>80%':<12}{'ESS_lr':<10}")
    for i in range(6):
        s = flatness_stats(N_all[:, i])
        pm = s["peak"] / (s["median"] + 1e-8)
        print(f"sp{i+1:<7}{pm:<12.1f}{s['pct_below_5_pct_peak']:<12.1%}"
              f"{s['pct_below_10_pct_peak']:<12.1%}"
              f"{s['pct_below_25_pct_peak']:<12.1%}"
              f"{s['frac_for_80pct_burstiness']:<12.1%}"
              f"{s['ess_normalized']:<10.3f}")


if __name__ == "__main__":
    main()
