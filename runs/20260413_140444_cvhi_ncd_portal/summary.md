# CVHI-NCD on Portal OT

架构: PosteriorEncoder + PerSpeciesTemporalAttn + SpeciesGNN_SoftForms

## 结果

| Seed | Pearson | RMSE | best_ep |
|---|---|---|---|
| 42 | +0.1087 | 8.157 | 138 |
| 123 | +0.0423 | 8.198 | 214 |
| 456 | +0.1695 | 8.087 | 144 |

Mean = +0.1068 ± 0.0520

## 对比

- Linear Sparse+EM: 0.353
- CVHI original: 0.33 ± 0.21
- **CVHI-NCD**: +0.1068 ± 0.0520
