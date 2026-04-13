# 方法 3 (两阶段) 参数恢复分析报告

## 方法

- **Stage 1**: visible → encoder → hidden 恢复（复用已训模型）
- **Stage 2**: (visible + hidden) → Ricker 对数线性回归 → 参数

Ricker 关系: `log(x_{t+1}/x_t) = r + A·x_t + noise`，对每个物种独立做最小二乘。

## 三种方法

- **Oracle**: 用真实 hidden 做 Stage 2（参数恢复的上界）
- **Pipeline**: 用 Stage 1 恢复的 hidden 做 Stage 2（实际方法 3 流水线）
- **Baseline**: 直接用端到端模型的参数（之前的做法）

## LV数据+LV先验+tanh

Run: `runs/20260412_165241_partial_lv_lv_guided_stochastic_refined`

Stage 2 拟合质量 (Oracle): R^2 = 0.963, residual std = 0.0506
Stage 2 拟合质量 (Pipeline): R^2 = 0.957, residual std = 0.0571

| 指标 | Oracle | Pipeline | Baseline |
|------|--------|----------|----------|
| Growth 符号准确率 | 1.000 | 0.833 | 0.667 |
| Growth Spearman | 0.200 | 0.486 | 0.314 |
| Growth Pearson | 0.135 | 0.516 | 0.306 |
| Growth 相对 L2 | 0.549 | 0.607 | 0.761 |
| Growth Scale 比 | 0.924 | 0.909 | 0.445 |
| Diagonal 符号 | 1.000 | 1.000 | 1.000 |
| Diagonal Spearman | 0.086 | 0.086 | -0.086 |
| Diagonal Scale 比 | 0.828 | 0.764 | 4.535 |
| Interaction 有意义边符号 | 0.947 | 0.947 | 0.895 |
| Interaction Pearson | 0.943 | 0.957 | 0.765 |
| Interaction Spearman | 0.919 | 0.941 | 0.709 |
| Interaction Scale 比 | 1.124 | 1.011 | 1.005 |

## LV数据+无LV先验

Run: `runs/20260412_165541_exp_lv_data_no_lv_prior`

Stage 2 拟合质量 (Oracle): R^2 = 0.963, residual std = 0.0506
Stage 2 拟合质量 (Pipeline): R^2 = 0.946, residual std = 0.0589

| 指标 | Oracle | Pipeline | Baseline |
|------|--------|----------|----------|
| Growth 符号准确率 | 1.000 | 1.000 | 1.000 |
| Growth Spearman | 0.200 | 0.257 | nan |
| Growth Pearson | 0.135 | 0.315 | nan |
| Growth 相对 L2 | 0.549 | 0.419 | 0.666 |
| Growth Scale 比 | 0.924 | 0.976 | 0.343 |
| Diagonal 符号 | 1.000 | 1.000 | 1.000 |
| Diagonal Spearman | 0.086 | 0.143 | 0.029 |
| Diagonal Scale 比 | 0.828 | 0.700 | 4.098 |
| Interaction 有意义边符号 | 0.947 | 1.000 | 0.684 |
| Interaction Pearson | 0.943 | 0.925 | 0.754 |
| Interaction Spearman | 0.919 | 0.924 | 0.588 |
| Interaction Scale 比 | 1.124 | 0.979 | 0.605 |

## 非线性+LV先验+tanh

Run: `runs/20260412_165809_exp_nonlinear_data_with_lv_prior`

Stage 2 拟合质量 (Oracle): R^2 = 0.973, residual std = 0.1225
Stage 2 拟合质量 (Pipeline): R^2 = 0.973, residual std = 0.1236

| 指标 | Oracle | Pipeline | Baseline |
|------|--------|----------|----------|
| Growth 符号准确率 | 0.667 | 0.333 | 1.000 |
| Growth Spearman | -0.371 | -0.371 | 0.314 |
| Growth Pearson | -0.238 | -0.196 | 0.334 |
| Growth 相对 L2 | 3.236 | 4.786 | 0.501 |
| Growth Scale 比 | 3.313 | 4.728 | 0.567 |
| Diagonal 符号 | 0.333 | 0.333 | 0.333 |
| Diagonal Spearman | -0.886 | -0.543 | -0.543 |
| Diagonal Scale 比 | 23.870 | 24.911 | 91.112 |
| Interaction 有意义边符号 | 0.875 | 0.875 | 1.000 |
| Interaction Pearson | 0.770 | 0.680 | 0.745 |
| Interaction Spearman | 0.636 | 0.575 | 0.697 |
| Interaction Scale 比 | 0.524 | 0.622 | 0.288 |

## LV数据+LV先验+Ricker

Run: `runs\20260412_173105_exp_lv_data_ricker_form`

Stage 2 拟合质量 (Oracle): R^2 = 0.963, residual std = 0.0506
Stage 2 拟合质量 (Pipeline): R^2 = 0.952, residual std = 0.0612

| 指标 | Oracle | Pipeline | Baseline |
|------|--------|----------|----------|
| Growth 符号准确率 | 1.000 | 0.833 | 1.000 |
| Growth Spearman | 0.200 | 0.486 | 0.771 |
| Growth Pearson | 0.135 | 0.575 | 0.791 |
| Growth 相对 L2 | 0.549 | 0.559 | 0.347 |
| Growth Scale 比 | 0.924 | 0.900 | 0.694 |
| Diagonal 符号 | 1.000 | 1.000 | 1.000 |
| Diagonal Spearman | 0.086 | 0.200 | 0.429 |
| Diagonal Scale 比 | 0.828 | 1.035 | 3.626 |
| Interaction 有意义边符号 | 0.947 | 1.000 | 0.579 |
| Interaction Pearson | 0.943 | 0.793 | 0.577 |
| Interaction Spearman | 0.919 | 0.834 | 0.550 |
| Interaction Scale 比 | 1.124 | 1.131 | 0.923 |
