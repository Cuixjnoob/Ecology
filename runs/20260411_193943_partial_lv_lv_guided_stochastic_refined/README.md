# 实验 README

- 时间戳：`20260411_193943`
- 实验名：`partial_lv_lv_guided_stochastic_refined`

## 本次实验修改了什么
- 保留上一版的 LV soft guidance + stochastic residual 主框架，但不再做模型对照，只修正 rollout 训练噪声和 hidden/environment 解耦。
- rollout training noise 被压到一个很小的组合扫描范围内，并固定 particle rollout 为 `K=4`、`mean aggregation` 的辅助项。
- hidden/environment 解耦约束增强为：更强相关性惩罚、正交惩罚、方差下界、以及 environment 更慢 / hidden 更快的时间尺度先验。

## 当前总算法是什么样子的
1. 数据层：仍使用 moderate_complexity 的 5 个 visible + 1 个 hidden + 1 个真实 environment 的合成生态系统。
2. 动力学层：状态更新保持 `state_t + LV_guided_drift + neural_residual + stochastic_noise`，LV 仍是 soft backbone。
3. 训练层：主要目标是降低过强噪声带来的平线化预测，并压低 hidden/environment 纠缠。
4. 评估层：继续报告 sliding-window rollout、full-context forecast、hidden recovery、LV/residual 比例和 latent disentanglement 统计。

## 本次结果一句话
- 数据 regime: `moderate_complexity`
- 选中的噪声配置: `{'training_input_noise': 0.01, 'training_rollout_noise': 0.05, 'training_latent_perturb': 0.01}`
- 当前主诊断: `rollout noise too strong`