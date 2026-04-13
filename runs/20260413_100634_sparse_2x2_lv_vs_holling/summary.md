# 2x2 对比: LV vs Holling × sparse LV 软约束

## LV 数据

- Best sweep: λ=0.5, Pearson=0.9703, RMSE=0.0706
- Best EM: iter 1, Pearson=0.9773, RMSE=0.0620
- A vs true 5x5: sign_acc=0.846, Pearson=+0.658
- A vs renormalized (包含 hidden 间接): sign_acc=0.846, Pearson=+0.573

## Holling 数据

- Best sweep: λ=2.0, Pearson=0.8602, RMSE=0.3017
- Best EM: iter 3, Pearson=0.8969, RMSE=0.2616
- A vs true 5x5: sign_acc=1.000, Pearson=+0.718
- A vs renormalized (包含 hidden 间接): sign_acc=1.000, Pearson=+0.711

## 关键对比

| 数据 | 最佳 Pearson | 最佳 RMSE | A sign_acc |
|---|---|---|---|
| LV (匹配) | 0.9773 | 0.0620 | 0.846 |
| Holling (非 LV) | 0.8969 | 0.2616 | 1.000 |
