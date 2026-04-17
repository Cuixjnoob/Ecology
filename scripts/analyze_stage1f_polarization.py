"""Analyze why Stage 1f polarized species into winners/losers.

Hypothesis: L_conserve = Var_t(visible_total + c*h) forces encoder_h to
ANTI-correlate with visible_total (so their variance cancels).
If hidden_true naturally anti-correlates with visible_total → Pearson up
If hidden_true naturally positive-correlates with visible_total → Pearson down

Also decompose: which channels dominate visible_total variance?
"""
from __future__ import annotations

import numpy as np
from scripts.load_beninca import load_beninca


SPECIES_ORDER = ["Cyclopoids", "Calanoids", "Rotifers", "Nanophyto",
                 "Picophyto", "Filam_diatoms", "Ostracods",
                 "Harpacticoids", "Bacteria"]

STAGE1F_DELTA = {
    "Cyclopoids":    +0.019,
    "Calanoids":     -0.129,
    "Rotifers":      +0.023,
    "Nanophyto":     -0.072,
    "Picophyto":     -0.079,
    "Filam_diatoms": +0.012,
    "Ostracods":     -0.065,
    "Harpacticoids": +0.138,
    "Bacteria":      +0.000,
}


def pearson(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / (np.sqrt((a*a).sum() * (b*b).sum()) + 1e-12))


def main():
    full, species, days = load_beninca()
    species = [str(s) for s in species]
    T, N = full.shape
    print(f"Data: T={T}, N={N}, species+nutrients: {species}\n")

    # Compute visible_total for each hidden rotation
    print(f"{'Hidden':<18}{'Δ vs S1b':<12}{'corr(h,vis_tot)':<20}{'Var(h)':<10}{'Var(vis_tot)':<15}{'dominant contributor'}")
    print("="*110)

    # Also compute per-channel CV to see which dominate
    cv_per_channel = np.std(full, axis=0) / (np.mean(full, axis=0) + 1e-8)
    print(f"\nPer-channel CV (higher = more variance in normalized data):")
    for i, s in enumerate(species):
        print(f"  {s}: {cv_per_channel[i]:.3f}")

    print(f"\n{'='*110}")
    print("Analysis per hidden rotation:")
    print('='*110)

    results = []
    for h_name in SPECIES_ORDER:
        h_idx = species.index(h_name)
        hidden = full[:, h_idx]
        visible = np.delete(full, h_idx, axis=1)

        # Option 1: simple sum
        visible_total = visible.sum(axis=1)
        # Option 2: sum weighted by inverse CV (dominant channels get less weight)
        visible_species_only = visible[:, :8]  # first 8 are species (since 1 hidden removed)

        # Correlation with hidden
        corr_sum = pearson(hidden, visible_total)

        # Which channel is most correlated with visible_total (dominant contributor)?
        remaining_species = [s for s in species if s != h_name]
        corrs_with_total = [pearson(visible[:, i], visible_total) for i in range(visible.shape[1])]
        dom_idx = np.argmax(np.abs(corrs_with_total))
        dom_name = remaining_species[dom_idx]
        dom_corr = corrs_with_total[dom_idx]

        # Correlation of hidden with the dominant-contributor
        corr_h_dom = pearson(hidden, visible[:, dom_idx])

        results.append({
            "hidden": h_name,
            "delta": STAGE1F_DELTA[h_name],
            "corr_h_vis_total": corr_sum,
            "var_h": float(hidden.var()),
            "var_vis_total": float(visible_total.var()),
            "dominant": dom_name,
            "dom_corr_with_total": dom_corr,
            "corr_h_dom": corr_h_dom,
        })

        print(f"{h_name:<18}{STAGE1F_DELTA[h_name]:<+12.3f}"
              f"{corr_sum:<+20.4f}{hidden.var():<10.3f}{visible_total.var():<15.3f}"
              f"{dom_name} (r={dom_corr:+.3f} w/total, r={corr_h_dom:+.3f} w/hidden)")

    print(f"\n{'='*110}")
    print("KEY INSIGHT CHECK:")
    print(f"If L_conserve forces h ~ -visible_total, winners should have corr(h_true, vis_tot) < 0")
    print('='*110)
    print(f"{'Hidden':<18}{'Δ S1f':<10}{'sign(corr)':<12}{'Expected':<12}{'Match?':<10}")
    for r in results:
        sign_corr = "+" if r["corr_h_vis_total"] > 0 else "-"
        # Hypothesis: Δ > 0 when corr < 0
        expected_sign = "+" if r["corr_h_vis_total"] < 0 else "-"
        actual_sign = "+" if r["delta"] > 0.01 else "-" if r["delta"] < -0.01 else "≈"
        match = "✓" if (r["delta"] > 0 and r["corr_h_vis_total"] < 0) or \
                       (r["delta"] < -0.01 and r["corr_h_vis_total"] > 0) else "✗"
        print(f"{r['hidden']:<18}{r['delta']:<+10.3f}{sign_corr:<12}"
              f"{'winner' if expected_sign == '+' else 'loser':<12}{match:<10}")


if __name__ == "__main__":
    main()
