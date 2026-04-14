# Path A: G ≥ 0 约束实验

- seeds: 5  epochs: 300

## Per-config Pearson + pairwise corr stats

| cfg | ds | mean P | std | max | pair μ | pair σ | %>0.5 | %<-0.3 |
|---|---|---|---|---|---|---|---|---|
| baseline | LV | +0.8053 | 0.0747 | +0.873 | -0.187 | 0.761 | 40% | 60% |
| baseline | Holling | +0.6585 | 0.2314 | +0.849 | -0.101 | 0.666 | 30% | 50% |
| baseline | Mendota | +0.1355 | 0.0429 | +0.201 | -0.180 | 0.859 | 40% | 60% |
| G_anchor_first | LV | +0.7849 | 0.0764 | +0.879 | +0.807 | 0.107 | 100% | 0% |
| G_anchor_first | Holling | +0.6646 | 0.1067 | +0.755 | +0.965 | 0.022 | 100% | 0% |
| G_anchor_first | Mendota | +0.1412 | 0.0336 | +0.175 | +0.916 | 0.024 | 100% | 0% |

## Δ vs baseline

| cfg | ds | ΔP | Δpair_mean_corr |
|---|---|---|---|
| G_anchor_first | LV | -0.0204 | +0.993 |
| G_anchor_first | Holling | +0.0061 | +1.065 |
| G_anchor_first | Mendota | +0.0057 | +1.096 |
