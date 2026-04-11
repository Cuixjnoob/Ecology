# 实验 README

- 时间戳：`20260411_004141`
- 实验名：`partial_lv_hidden_environment_compare`

## 本次实验修改了什么
- 不再做 shallow/deep 对照，而是改成 Model A（hidden-only）和 Model B（hidden + environment latent）对照。
- 真实数据生成器把 hidden species、environment driver、pulse forcing 显式拆开，避免单一 hidden 节点同时吸收多类驱动。
- visible loss 从单纯 RMSE 升级成 base + peak-aware + turning-point/slope + amplitude-preservation 的联合损失，专门缓解预测塌成均值平线的问题。
- 保留 sliding-window rollout 和 full-context long-horizon 两种评估方式，分别衡量局部递推能力和整段未来预测能力。

## 当前总算法是什么样子的
1. 数据层：生成 5 个 visible + 1 个真实 hidden 的离散广义 LV / Ricker 风格系统，同时加入单独的 environment driver 和适度 pulse forcing，并通过自动筛选保证数据属于 moderate complexity。
2. 模型层：Model A 使用 visible 编码 + delay embedding + GRU memory + 1 个 hidden latent；Model B 在此基础上新增 1 个独立的 environment latent，并通过单独递推和 decorrelation regularization 与 hidden latent 分开建模。
3. 损失层：联合优化 visible one-step、visible rollout、peak-aware、slope、amplitude-preservation、hidden recovery、interaction regularization，以及 Model B 的 environment smoothness / stability / disentanglement。
4. 评估层：同时报告 sliding-window visible rollout、full-context visible forecast、hidden recovery，并保存少量最关键图表用于人工诊断。

## 本次结果一句话
- 数据 regime: `moderate_complexity`
- hidden 更优模型: `A`
- sliding-window visible 更优模型: `B`
- full-context visible 更优模型: `similar`
- 主诊断: `hidden/environment entanglement`