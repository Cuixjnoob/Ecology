# Dynamics Operator + Forward Simulation 结果

## 核心思想

GNN 学 dynamics operator f: (x_t, h_t) → (x_{t+1}, h_{t+1})，
不是 static hidden decoder。结构化 linear sparse + GNN 残差 correction。

## LV

- Stage 1 h_coarse Pearson: 0.9773
- Dynamics Operator params: 5,430
- Best epoch: 1290

### Forward Simulation Visible RMSE (越低越好)

| Horizon | Mean RMSE | Median RMSE |
|---|---|---|
| 5 steps | 0.1011 | 0.0960 |
| 10 steps | 0.1540 | 0.0991 |
| 20 steps | 0.2463 | 0.1743 |
| 50 steps | 0.4706 | 0.4306 |
| 100 steps | 0.5454 | 0.4870 |
| 200 steps | 0.6641 | 0.7464 |

## Holling

- Stage 1 h_coarse Pearson: 0.8762
- Dynamics Operator params: 5,430
- Best epoch: 1816

### Forward Simulation Visible RMSE (越低越好)

| Horizon | Mean RMSE | Median RMSE |
|---|---|---|
| 5 steps | 0.2241 | 0.2104 |
| 10 steps | 0.3283 | 0.3092 |
| 20 steps | 0.4250 | 0.3928 |
| 50 steps | 0.6230 | 0.5943 |
| 100 steps | 0.8714 | 0.8076 |
| 200 steps | 1.1687 | 1.2435 |

