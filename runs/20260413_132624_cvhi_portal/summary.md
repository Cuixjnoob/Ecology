# CVHI (GNN) 真实数据 Portal top-12

数据: Portal Project rodents, T=520 months

## 对比: CVHI vs Linear Sparse+EM

| Hidden | Linear baseline | CVHI coarse (EM) | CVHI posterior | Δ |
|---|---|---|---|---|
| DO | +0.4289 | +0.1767 | +0.4282 | -0.0007 |
| OT | +0.3534 | +0.3120 | +0.4713 | +0.1179 |
| PP | +0.3416 | +0.3328 | +0.3204 | -0.0212 |
| PF | +0.3072 | +0.2697 | +0.1343 | -0.1729 |

## 参数

- Epochs: 500
- Encoder d=64, Dynamics d=32
- β_max=0.02
