# 研究笔记

> 记录较长期的思考、模型设计理解、潜在问题、失败原因分析、备选方向。  
> 按主题组织，非按时间排列。每条标注日期和置信度。

---

## 0.5. LV 先验的真实价值（2026-04-12 2×2 实验后，中等置信度）

### 实验设置
2×2 对比：{LV 数据, 非线性数据} × {带 LV 先验, 无 LV 先验}。非线性数据使用 Holling type II + Allee effect + 时滞。

### 结果（Hidden Test Pearson / RMSE）

|          | LV 先验 | 无 LV 先验 |
|----------|--------|----------|
| LV 数据 | 0.989 / 0.063 | 0.989 / 0.107 |
| 非线性数据 | 0.979 / 0.124 | 0.977 / 0.156 |

### 关键解读

**LV 先验的实际作用比理论预期要小得多。**

原本的假设：LV 先验在 LV 数据上显著帮助，在非 LV 数据上拖累。
实际观察：LV 先验在两种数据上都**只带来轻微改善**，没有出现 misspecification 陷阱。

### 三个可能的解释

1. **Residual 网络吸收了所有"数据 - LV"的差异。** LV 解释不了的，residual 补上。无论 LV 先验对不对，最终拟合质量由 residual 决定。证据：residual_dominates_fraction 在所有实验中都 > 0.8。

2. **Encoder 映射路径主导了 hidden recovery。** Hidden 恢复主要通过 encoder(visible → hidden)，不经过 rollout 动力学。所以 rollout 里的 LV/residual 平衡对 hidden Pearson 影响小。

3. **当前评估方式 (encoder-based recovery) 对动力学错误不敏感。** 如果改用"动力学恢复"评估（用初始 hidden 做 rollout 对比），可能会看到 LV vs 非 LV 的更大差异。

### 研究含义

- LV 先验不是 hidden recovery 的必要条件
- LV 先验的价值主要在**可解释性**（给出 interaction matrix）
- 去掉 LV 先验，hidden recovery 几乎不变，但失去生态学解释锚点
- 这是研究叙事的关键结论：**"为什么还要保留 LV？"的答案不是 "因为它让预测更准"，而是"因为它让结果可解释"**

### 更新：interaction matrix 恢复分析（2026-04-12 下午，高置信度）

分析 4 个 run 的 npz 后，对上述结论有**量化支持**：

| | LV 数据 | 非线性数据 |
|---|---|---|
| 带 LV 先验 meaningful sign acc | 0.895 | **1.000** |
| 无 LV 先验 meaningful sign acc | 0.684 | 0.750 |
| 带 LV 先验 matrix correlation | 0.765 | 0.745 |
| 无 LV 先验 matrix correlation | 0.754 | **0.293** |

**最强的证据**：
- 非线性数据 + 无 LV 先验时，matrix correlation 跌到 0.293，几乎是随机
- 其他三种配置（ABC）都维持在 0.74-0.77
- 这说明 LV 先验 + residual 的组合在各种动力学上都能稳定恢复交互方向
- 纯 residual 模型（无 LV）在数据动力学远离 LV 时会失去 matrix 的生态含义

**新的假设（待验证）**：LV 先验的主要机制是为 interaction matrix 参数提供**稀疏性约束的搜索空间**。没有 LV 先验时，`off_diagonal` 参数矩阵可以任意取值（因为它不直接影响 visible 预测，只通过 residual 间接影响），所以优化压力不足以把它推向真实值。

---

## 0. 方法论转向：从预测到重构（2026-04-12，高置信度）

### 决策
彻底放弃 future prediction 作为项目目标和评估标准。项目核心重新聚焦为 **hidden species recovery**。

### 推理过程

**起点**：项目目标是"从 visible dynamics 中恢复 hidden ecological structure"，这是一个关于已发生的事情的推断问题，不是预测问题。

**关键观察**：
1. 之前 val_score 中 visible 预测占 0.79 权重，hidden 仅占 0.16 — 完全偏离研究目标
2. 4 轮 Codex 迭代全部在追求 full-context visible RMSE，方向错误
3. Hidden Pearson 已达 0.90，说明 hidden recovery 本身并非瓶颈
4. 已知的 visible 数据可以分成"输入段"和"目标段"来做重构，不需要预测任何真正的未来

**具体改动**：
- `val_score` 重新分配：hidden recovery 0.40 + disentanglement 0.20 + visible 重构 0.20 + amplitude 0.10 + residual balance 0.10
- `_noise_scan` 评分同步调整
- 去掉 `evaluate_full_context("val")` 在 fit 循环中的调用
- 去掉 full-context visible prediction 的核心地位
- run 脚本输出以 hidden 为中心重组
- 图表更新：新增全序列 hidden recovery 图，去掉 full-context visible 对比图

**保留的部分**：
- Sliding-window rollout 训练机制保留（它本质是在已知数据内做重构，是 hidden 的间接训练信号）
- `_full_context_train_step()` 保留（在 train 段内做长 rollout 重构）
- Visible 重构误差作为辅助指标保留（非核心目标）

### 对后续实验的影响
- 不再做任何以"改善 visible RMSE"为目标的实验
- 消融实验和 baseline 对比以 hidden recovery 为核心评判标准
- "额外解释力"的检验方式变为：with-hidden vs without-hidden 模型在已知数据重构上的对比

---

## 1. LV/Residual 分工失衡的深层原因（2026-04-12，中等置信度）

### 现象
在 best run 及所有后续迭代中，residual dominates fraction 持续在 0.85-0.87。即使加入 `residual_magnitude` 和 `lv_guidance` 损失，LV 分支始终未能成为主力。

### 可能的解释

**解释 A：信息不对称。** residual network 的输入是 `log_state + interaction_feature + env + memory + context`（维度约为 2*6+1+2*40=93），而 LV drift 只使用 `growth_rates + interaction_matrix + env_coupling`（参数化的简单公式）。在反向传播中，高维 residual 分支的梯度信号更丰富，优化更容易。

**解释 B：LV 先验的表达能力限制。** 合成数据虽然是 LV 生成的，但 Ricker-style 离散动力学 + 环境驱动 + 脉冲扰动使得真实动态比简单连续 LV 更复杂。模型中的 LV drift 可能无法精确匹配数据生成过程。

**解释 C：curriculum 设计的问题。** curriculum 从 0.3 开始就已经允许 residual 有 30% 的强度。如果 LV 分支在早期训练中来不及建立有效的结构化表示，residual 就会抢先占据优化空间。

### 对后续实验的启示
- 消融实验（去掉 residual，只保留 LV）可以揭示 LV 先验的上限
- 如果 LV-only 的 visible RMSE 远差于当前，说明 residual 确实承担了 LV 无法覆盖的部分
- 如果 LV-only 的 visible RMSE 接近当前 → 说明 LV 有潜力但被 residual 抢占了

---

## 2. Structured visible driver 的"捷径"问题（2026-04-12，高置信度）

### 现象
Codex Iteration 3 加入 hidden→visible 和 env→visible 的结构化路径后，visible RMSE 大幅改善（0.79→0.35），但 hidden Pearson 崩溃到 -0.09。

### 分析
这是一个典型的"信息泄漏"或"捷径学习"问题。hidden→visible driver 的输入不仅仅是 hidden state，还包含 context（来自 visible 历史编码）。模型可以通过 context→hidden→visible 的路径绕过 hidden 的真实身份约束，把 hidden latent 当作纯中间表示来用。

### 为什么 Iteration 4 的修复也失败了
Iteration 4 试图限制 hidden→visible path 只能使用 hidden state 作为输入。但这使得 hidden→visible driver 的表达能力不足（只有 1 维 hidden scalar 作为输入），导致 amplitude 崩溃。

### 潜在的更好方案（待验证）
1. **梯度阻断 + 辅助损失**：对 hidden→visible path 中的 hidden state 做 partial stop_gradient（如 0.5 的比例），同时增强 hidden MSE 损失权重。这样 hidden→visible path 可以使用 hidden 信息，但训练信号不会过度通过此路径回传到 hidden head。
2. **hidden identity 正则化**：不直接限制输入，而是要求 hidden→visible driver 输出与 hidden state 的某种一致性（如 hidden state 的变化方向应与 hidden→visible driver 的变化方向一致）。
3. **两阶段训练**：先训练 hidden recovery 到收敛，冻结 hidden 相关参数，再训练 structured visible driver。

### 置信度说明
"捷径学习"的诊断置信度较高（有 Iteration 3/4 的定量证据）。但上述三个修复方案都未经验证，属于推测。

---

## 3. Full-context visible prediction 瓶颈的性质（2026-04-12，中等置信度）

### 现象
sliding-window visible RMSE (0.79) 和 full-context visible RMSE (0.81) 数值接近，但 Pearson 差异巨大：sliding 0.66 vs full-context 0.24。这说明 full-context 预测的整体误差并不比 sliding 大很多，但**动态结构（相关性）大幅退化**。

### 可能的解释
- Full-context 预测从训练段末尾开始做长距离 rollout（约 164 步 val 或 test）。误差累积导致预测序列逐步偏离真实轨迹的相位和幅度。
- 但 amplitude collapse 只有 0.058，说明振幅本身保持得不错。问题更可能在于**相位漂移**而非振幅坍缩。
- Full-context Pearson 低意味着预测的峰谷时机与真实值不匹配，即使峰谷幅度差不多。

### 这意味着什么
- 改善 full-context 需要更好的长期相位跟踪能力，而不是更好的幅度保持
- 这可能需要更强的 memory 机制（当前 rollout_memory 是一个 GRUCell，可能记忆容量不足）
- 也可能需要更频繁的 "re-anchor" 机制（在 rollout 中间重新对齐）

---

## 4. Particle rollout 为什么没用（2026-04-12，低置信度）

### 现象
`particle_rollout_helpful: false`。单粒子 RMSE 0.7936 vs 多粒子 0.7937，几乎无差异。

### 猜测
- K=4 太少，不足以覆盖有意义的动态多样性
- mean 聚合抹平了粒子间差异
- 噪声退火后，粒子间差异本身就很小

### 优先级
低。这不是当前瓶颈。等更重要的问题解决后再考虑。

---

## 5. 合成数据的局限性（2026-04-12，高置信度）

### 已知局限
- 只有 1 个 hidden species，真实系统可能有多个
- 固定 seed=42 的单次生成，数据特性可能不具代表性
- Ricker-style 离散 LV 可能与模型中的连续 LV drift 有系统性偏差
- 环境驱动是简单的多频正弦 + AR(1)，真实环境更复杂

### 对当前结果的影响
- hidden Pearson 0.90 的结论只在"这一组合成数据"上成立
- 多种子实验（至少 5 个不同 seed）是发表前必须做的
- 但在方法开发阶段，单 seed 可以接受（节省时间）
