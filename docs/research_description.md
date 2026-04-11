# 研究描述文书

## 题目

**部分观测生态动力学系统中的隐藏物种推断与生态一致性建模**

Inference of Hidden Species and Ecologically Consistent Modeling in Partially Observed Ecological Dynamical Systems

---

## 研究背景

生态系统的观测通常是不完整的：在一个含有 $N$ 个物种的多物种系统中，实际被直接监测的往往只有其中一部分。未被观测的物种（hidden species）可能通过竞争、捕食、资源限制、代谢约束或环境驱动等机制显著影响可观测物种的动态演化。忽略这些隐藏成分，仅用 visible-only 模型拟合数据，会导致：

1. 隐藏生态作用被错误地压缩为残差噪声或参数漂移；
2. 模型的长期预测能力和泛化能力下降；
3. 无法发现关键的跨物种交互关系。

传统生态学方法（如 Lotka-Volterra 参数估计）要求完整观测，而纯数据驱动的神经网络方法虽然灵活，却缺乏生态可解释性，且容易学出冗余表示。本研究旨在在这两个极端之间找到平衡。

---

## 研究问题

在部分观测条件下，如何联合推断：

1. **visible dynamics** — 可见物种的未来演化；
2. **hidden latent states** — 未观测的隐藏物种状态与环境驱动；
3. **额外解释力** — 恢复出的隐藏成分是否真的提升了对系统动力学的解释？

---

## 核心研究主张

1. **hidden detection / recovery 比 forecast 更重要。** 研究目的不是未来值本身，而是恢复未观测的生态成分。
2. **ecological residual 不能只被视为噪声。** 它可能是 hidden species、hidden environment、未建模交互或时滞机制的投影。
3. **评估必须通过"接回正向模型后的额外解释力"来完成。** 否则 hidden recovery 容易沦为不可验证的自由表示。

---

## 方法

### 问题形式化

设系统有 $n_v$ 个可见物种和 $n_h$ 个隐藏物种。观测数据为可见物种的离散时间序列：

$$\mathbf{x}_t = [x_{t,1}, x_{t,2}, \ldots, x_{t,n_v}]^\top, \quad t = 1, 2, \ldots, T$$

目标是从 $\{\mathbf{x}_t\}_{t=1}^T$ 中恢复隐藏状态序列 $\{\mathbf{h}_t\}$ 和环境驱动 $\{e_t\}$。

### 模型架构

采用"结构化先验 + 神经残差 + 随机噪声"的混合架构。状态更新公式：

$$\mathbf{s}_{t+1} = \mathbf{s}_t + \underbrace{\alpha_{\text{lv}} \cdot f_{\text{LV}}(\mathbf{s}_t, e_t)}_{\text{LV drift}} + \underbrace{\alpha_{\text{res}} \cdot c(t) \cdot g_\theta(\mathbf{s}_t, e_t, \mathbf{m}_t)}_{\text{neural residual}} + \underbrace{\beta_h \cdot q_\phi(\mathbf{x}_t, e_t, \mathbf{m}_t)}_{\text{hidden fast}} + \underbrace{\boldsymbol{\epsilon}_t}_{\text{noise}}$$

其中：
- $\mathbf{s}_t = [\mathbf{x}_t, \mathbf{h}_t]$ 为完整状态
- $f_{\text{LV}}$ 为可学习的 Lotka-Volterra drift（含交互矩阵 $\mathbf{A}$ 和环境耦合）
- $g_\theta$ 为神经残差网络
- $q_\phi$ 为隐藏快创新网络（仅作用于 hidden species）
- $c(t) \in [0.3, 1.0]$ 为 curriculum 系数
- $\boldsymbol{\epsilon}_t$ 为可学习方差的过程噪声

环境状态采用 Ornstein-Uhlenbeck 过程更新：

$$e_{t+1} = e_t + \tau_{\text{env}} \cdot (\hat{e}_{t}^{\text{target}} - e_t) + \eta_t$$

其中 $\tau_{\text{env}} \in [0.03, 0.12]$ 为可学习的时间常数，确保环境作为慢变量。

### 编码器

从 visible 历史序列提取上下文：
1. **Takens delay embedding** → MLP 编码
2. **GRU 序列编码** → 全局 context
3. **趋势斜率** → 局部动态信号
4. 三者融合 → hidden / environment 初始化

### 损失函数

共 17 项损失的加权组合，涵盖：
- **拟合准确性**：one-step MSE、rollout MSE、peak-weighted MSE、multiscale L1、local variance
- **生态约束**：LV guidance、interaction sparsity、amplitude preservation
- **解耦约束**：hidden-env correlation、timescale prior、variance floor、orthogonality
- **平衡约束**：LV/residual 能量比、residual magnitude

### 训练策略

- 噪声网格搜索选择最优噪声配置
- Sliding-window + full-context 双轨训练
- Curriculum learning：residual 强度渐进增长
- 噪声退火：训练过程中噪声逐步降低

---

## 评估体系

### 主评估：hidden recovery
- **逐步滑窗 hidden recovery**：从 visible 历史恢复 hidden 初始值，计算 RMSE / Pearson
- **hidden/env disentanglement**：correlation、roughness ratio、autocorrelation 对比
- **interaction matrix recovery**：hidden 边的符号准确率

### 辅助评估：visible dynamics
- Sliding-window rollout RMSE / Pearson
- Full-context forecast
- Amplitude collapse score
- LV/residual 能量分析

### 额外解释力检验（待完成）
- visible-only baseline vs visible + hidden model 的系统对比
- 交叉验证与多种子实验

---

## 当前进展

### 实验数据
合成系统：5 visible + 1 hidden + 1 environment，离散 LV (Ricker-style)，820 时间步。

### 当前最佳结果
| 指标 | 值 | 说明 |
|------|-----|------|
| hidden RMSE | 0.166 | 隐藏物种恢复误差 |
| hidden Pearson | 0.902 | 恢复与真值的相关性 |
| hidden/env correlation | 0.099 | 解耦质量（越低越好）|
| sliding visible RMSE | 0.794 | 短窗口可见预测 |
| full-context visible RMSE | 0.807 | 长段可见预测 |
| amplitude collapse | 0.058 | 振幅保持（越低越好）|

### 已验证的关键发现
1. **Hidden recovery 不是瓶颈**：hidden Pearson 达 0.90，说明从 visible dynamics 恢复 hidden 是可行的。
2. **Full-context visible prediction 是主瓶颈**：长期 visible 预测质量仍不理想。
3. **Residual 仍然过强**：LV 结构化分支贡献不足（residual dominates 87%），需要更好的平衡策略。
4. **Environment disentanglement 有效**：hidden/env correlation 从无约束的 0.91 降至 0.10。

### 4 轮迭代实验总结
| 迭代 | 假设 | 结果 | 保留 |
|------|------|------|------|
| 1 | 硬限 residual budget | hidden 崩溃 | 否 |
| 2 | 多 cut-point full-context | 全面恶化 | 否 |
| 3 | structured hidden/env → visible driver | visible 大幅改善，hidden 崩溃 | 部分 |
| 4 | 约束 hidden-visible path | amplitude 崩溃 | 否 |

---

## 下一步方向

1. **v2 机制分离实验**：启用 multiscale loss、local variance loss、residual energy 约束
2. **消融实验**：验证 OU env / hidden_fast / curriculum 的独立贡献
3. **visible-only baseline 对比**：建立"额外解释力"的系统化检验
4. **structured visible driver 改进**：在约束 hidden 一致性的前提下重试 Iteration 3 方向
5. **真实数据探索**：长远目标

---

## 技术栈

- **框架**：PyTorch（CPU 训练）
- **数据**：合成 LV 动力学
- **Python 环境**：`.venv/` + `requirements.txt`

---

## 仓库结构

```
models/partial_lv_recovery_model.py   ← 核心模型（四路分工 + OU 环境）
train/partial_lv_mvp_trainer.py       ← 训练器（17 项损失 + curriculum）
scripts/run_partial_lv_mvp.py         ← 主线入口（数据→训练→评估→绘图）
data/partial_lv_mvp.py                ← 合成数据生成器
configs/partial_lv_mvp*.yaml          ← 配置文件
CLAUDE.md                             ← 项目入口记忆文件
codex_iteration_log.md                ← 实验迭代日志
notes/                                ← 项目文档集
```
