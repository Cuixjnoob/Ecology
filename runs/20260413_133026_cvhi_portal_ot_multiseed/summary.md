# Portal OT — CVHI 5-seed 验证

数据: Portal Project, T=520 months, hidden = OT (Onychomys torridus)

Visible: 11 物种 (top-12 minus OT)

## 结果（5 seeds）

| Seed | Pearson | RMSE |
|---|---|---|
| 42 | +0.4713 | 7.237 |
| 123 | +0.0136 | 8.205 |
| 456 | +0.1739 | 8.080 |
| 789 | +0.5933 | 6.605 |
| 2024 | +0.3902 | 7.555 |

- **Mean Pearson = +0.3285 ± 0.2085**
- Median Pearson = +0.3902
- Cross-seed stability (pairwise Pearson) = +0.6356 ± 0.2064

## 与 Linear Baseline 对比

- Linear Sparse+EM (OT as hidden): Pearson = +0.3534
- CVHI mean: Pearson = +0.3285
- 提升 Δ = -0.0249
