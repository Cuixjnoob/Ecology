# CVHI_Residual — 纯无监督 hidden recovery

架构: f_visible(x) + h·G(x) (残差分解) + 反事实必要性

训练: 无 anchor, 无 hidden 监督, 反事实 null/shuffle margins 强制 h 必要

## Portal top-12 hidden=OT

T = 520, N = 11

| Seed | Pearson | RMSE | m_null_final | h_var | best_ep |
|---|---|---|---|---|---|
| 42 | +0.1461 | 8.117 | +0.0232 | 0.182 | 421 |
| 123 | +0.1612 | 8.098 | +0.0149 | 0.191 | 206 |
| 456 | +0.2865 | 7.861 | -0.0000 | 0.047 | 123 |

**Mean = +0.1979 ± 0.0629**
Max  = 0.2865

## Synthetic LV (5+1)

T = 820, N = 5

| Seed | Pearson | RMSE | m_null_final | h_var | best_ep |
|---|---|---|---|---|---|
| 42 | +0.3040 | 0.278 | +0.0106 | 0.159 | 481 |
| 123 | +0.8190 | 0.168 | +0.0216 | 0.311 | 452 |
| 456 | +0.5981 | 0.234 | +0.0202 | 0.305 | 496 |

**Mean = +0.5737 ± 0.2109**
Max  = 0.8190

## 对照 (其他方法)

| 方法 | Portal OT | 合成 LV | 是否用 hidden 监督 |
|---|---|---|---|
| Linear Sparse + EM | 0.35 | 0.98 | 是 (投影步骤) |
| CVHI 原版 + anchor | 0.33 ± 0.21 | 0.88 | 间接 (anchor 来自 Linear) |
| CVHI-NCD + anchor (v4) | 0.23 ± 0.002 | 0.84 ± 0.0002 | 间接 |
| **CVHI_Residual (本次)** | ? | ? | **无 (纯无监督)** |
