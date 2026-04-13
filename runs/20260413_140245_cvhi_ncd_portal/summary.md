# CVHI-NCD on Portal OT

架构: PosteriorEncoder + PerSpeciesTemporalAttn + SpeciesGNN_SoftForms

## 结果

| Seed | Pearson | RMSE | best_ep |
|---|---|---|---|
| 42 | +0.1482 | 8.115 | 93 |
| 123 | +0.1032 | 8.162 | 136 |
| 456 | +0.1847 | 8.064 | 144 |

Mean = +0.1453 ± 0.0333

## 对比

- Linear Sparse+EM: 0.353
- CVHI original: 0.33 ± 0.21
- **CVHI-NCD**: +0.1453 ± 0.0333
