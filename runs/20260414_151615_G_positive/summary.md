# Path A: G ≥ 0 约束实验

- seeds: 20  epochs: 300

## Per-config Pearson + pairwise corr stats

| cfg | ds | mean P | std | max | pair μ | pair σ | %>0.5 | %<-0.3 |
|---|---|---|---|---|---|---|---|---|
| baseline | Portal | +0.1130 | 0.1108 | +0.305 | -0.001 | 0.205 | 4% | 7% |
| G_positive | Portal | +0.1170 | 0.0745 | +0.310 | +0.363 | 0.210 | 28% | 0% |
| G_anchor_first | Portal | +0.1968 | 0.0615 | +0.269 | +0.743 | 0.350 | 81% | 0% |

## Δ vs baseline

| cfg | ds | ΔP | Δpair_mean_corr |
|---|---|---|---|
| G_positive | Portal | +0.0041 | +0.363 |
| G_anchor_first | Portal | +0.0839 | +0.744 |
