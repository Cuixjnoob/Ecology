# 实验 README

- 时间戳：`20260412_161938`
- 实验名：`exp_nonlinear_data_no_lv_prior`
- 模式：**hidden recovery-centric**（不做未来预测，专注隐藏物种恢复）

## 核心方法
- 目标：从可见物种的已知时间序列中恢复未观测的隐藏物种
- 训练：在已知数据上做滑动窗口重构，visible rollout 作为训练信号（非预测目标）
- 评估：以 hidden recovery quality (RMSE/Pearson) 为核心指标
- 架构：LV soft guidance + stochastic residual + hidden fast innovation

## 本次结果
- Hidden Test RMSE: `0.1555`
- Hidden Test Pearson: `0.9770`
- Hidden Val Pearson: `0.9862`
- 数据 regime: `moderate_complexity`
- 诊断: `good hidden recovery`