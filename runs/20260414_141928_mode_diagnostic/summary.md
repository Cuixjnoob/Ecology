# Mode Diagnostic (Step 0)

- 20 seeds × Portal K=1 full config × 300 epochs

## Per-seed

Pearson vs hidden_true: mean=+0.113 std=0.108 min=+0.004 max=+0.305

## Pairwise Pearson across 190 seed pairs

mean=-0.001 std=0.205 median=+0.000

- pairs < 0.2: 173/190 (91%)
- pairs > 0.5: 7/190  (4%)

## Clustering (hierarchical on 1−|corr|)

- 2-cluster sizes=[3, 17] cluster mean Pearson=['+0.019', '+0.130']
- 3-cluster sizes=[3, 16, 1] cluster mean Pearson=['+0.019', '+0.134', '+0.053']
