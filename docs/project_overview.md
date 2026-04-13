# 项目深度总览与当前状态判断

> 最后更新：2026-04-12
> 目的：为接手者或未来的自己提供一份"不打开代码也能理解项目到底在哪里"的状态描述

---

## 一、核心目标

本项目研究的核心问题是：**在只能观测到部分物种的条件下，能否从可见物种的时间序列中恢复出未观测的隐藏物种和环境驱动？**

这不是一个普通的时间序列预测问题。核心评判标准不是"预测未来有多准"，而是：
1. 恢复出的 hidden species 是否与真实值一致（recovery quality）
2. 恢复出的 hidden 和 environment 是否功能分离（disentanglement）
3. 加入 hidden/env 后是否对 visible 动力学有额外解释力（extra explanatory power）

## 二、当前代码实现进度

### 已完成的部分
- 合成数据生成器（5 visible + 1 hidden + 1 environment，Ricker-style LV，820 步）
- 完整的 4-way rollout 模型（LV drift + residual + hidden_fast + noise + OU env）
- 17 项损失函数的训练器，含 curriculum learning 和噪声退火
- 6 种评估指标 + 6 张自动图表
- 噪声配置网格搜索
- 4 轮 Codex 迭代实验（有完整日志和定量指标）

### 尚未完成的关键部分
- **visible-only baseline 对比**：没有无 hidden/env 的基线模型，无法系统化验证"额外解释力"
- **消融实验**：OU env / hidden_fast / curriculum 各自独立贡献未量化
- **v2 配置完整实验**：multiscale + local_variance + residual_energy 三个新损失已实现但未跑完整对比
- **多种子实验**：当前只用 seed=42，结果的稳定性未验证
- **真实数据接入**：完全未开始

## 三、当前项目状态判断（2026-04-12）

### 3.1 最可信的结论

基于 best run (`20260411_115901`) 和 4 轮迭代实验，以下结论有较充分的证据支持：

1. **Hidden recovery 是可行的**：Pearson 0.90 说明从 visible dynamics 恢复 hidden 在合成数据上是可以做到的。这是项目最重要的正面结果。

2. **Hidden/env disentanglement 约束有效**：correlation 从无约束的 0.91 降至 0.10，多重约束策略（correlation + orthogonality + variance floor + timescale prior + env smooth）起了作用。

3. **Full-context visible prediction 是当前主瓶颈**：RMSE 0.81，Pearson 仅 0.24。长段预测质量远低于短窗口预测（sliding RMSE 0.79 但 Pearson 0.66）。

4. **Residual 分支仍然过强**：dominates fraction 0.87 意味着 87% 的时间步中 residual 贡献大于 LV。LV 结构化先验未充分发挥作用。

### 3.2 三个最可能存在的问题/不确定点

**问题 1：Residual 与 LV 的分工失衡可能是架构层面的问题，不仅是训练策略问题**

观察：4 轮迭代中，硬限 residual (Iter 1) 导致 hidden 崩溃，软约束 (residual_energy/residual_magnitude) 效果有限。residual dominates 0.87 在多次实验中稳定出现。

可能原因：residual network 的输入包含 `log_state + interaction_feature + env + memory + context`，信息量远大于 LV drift（只有 growth_rates + interaction + env coupling）。在反向传播中，residual 分支更容易被优化。

不确定性：不清楚这是因为 LV 先验本身对合成数据不够好，还是 residual 优化路径太容易。需要消融实验来区分。

**问题 2：v2 配置（multiscale + local_variance + residual_energy）的效果完全未验证**

观察：v2 配置已实现在代码中并写好了 YAML，但从未跑过完整实验。这三个损失项是为解决 amplitude collapse 和 residual 过强而设计的。

不确定性：不知道这些损失项是否会引入新的训练不稳定性，也不知道它们与现有 17 项损失的交互效果。

**问题 3：Iteration 3 的 structured visible driver 是唯一显著改善 full-context 的方向，但两次尝试均以 hidden 崩溃告终**

观察：Iteration 3 将 sliding visible RMSE 从 0.79 降到 0.35（大幅改善），但 hidden Pearson 崩溃到 -0.09。Iteration 4 试图约束 hidden 一致性但又导致 amplitude 崩溃。

可能原因：hidden→visible 的结构化路径使模型可以通过"扭曲 hidden 表示来拟合 visible"来走捷径，绕过了 hidden 的真实身份约束。

不确定性：不清楚是否存在一种约束方式能同时保持 visible 改善和 hidden 一致性，还是这两个目标在当前架构下本质上冲突。

### 3.3 下一步最值得做的 1 个小实验

**推荐实验：用 v2 配置跑完整实验，与原版 v1 结果对比**

理由：
1. **风险最低**：不需要改代码，只需换配置文件运行
2. **信息量最大**：一次实验同时验证三个新损失项（multiscale, local_variance, residual_energy）
3. **直接对准瓶颈**：multiscale 和 local_variance 针对 visible 预测质量，residual_energy 针对 LV/residual 分工
4. **已有明确基线**：v1 best run 提供了完整的对比指标
5. **符合单变量原则**：虽然 v2 同时启用了 3 个新损失，但它们是作为一个配置方案整体设计的，且权重已经过初步调整

预期观察：
- 如果 residual_energy 有效 → residual dominates fraction 应从 0.87 下降
- 如果 multiscale + local_variance 有效 → amplitude collapse 应保持低位且 visible RMSE 可能改善
- 需要警惕：hidden recovery 是否被影响（Pearson 不应低于 0.85）

---

## 四、项目时间线概述

| 时间 | 里程碑 |
|------|--------|
| 2026-04-11 凌晨 | 首批 MVP 实验（context 编码验证） |
| 2026-04-11 上午 | hidden/environment 对比实验 → 确认 env 必要性 |
| 2026-04-11 上午 | LV-guided stochastic 架构确立 |
| 2026-04-11 白天 | refined 主线迭代 → best run 115901 |
| 2026-04-11 下午 | Codex 4 轮迭代（均失败/部分保留） |
| 2026-04-12 | 仓库清理 + 文档体系建设 |
| 2026-04-12 | **方法论转向**：从预测到重构，hidden recovery-centric |
