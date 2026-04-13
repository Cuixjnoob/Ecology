# 实验日志

> 格式：每次实验按时间倒序记录。  
> 早期实验（2026-04-11 Codex 迭代）的完整记录在 `codex_iteration_log.md`。  
> 本文件从 2026-04-12 文档体系建立后开始记录新实验。

---

## 实验编号规则

`EXP-YYYYMMDD-NN`，其中 NN 为当日序号。

---

## 历史实验摘要（详见 codex_iteration_log.md）

| 编号 | 日期 | 假设 | 结果 | 决策 |
|------|------|------|------|------|
| Codex-Iter-1 | 2026-04-11 | 硬限 visible residual budget | hidden 崩溃 (Pearson -0.57) | revert |
| Codex-Iter-2 | 2026-04-11 | 多 cut-point full-context 训练 | visible/hidden 全面恶化 | revert |
| Codex-Iter-3 | 2026-04-11 | structured hidden→vis / env→vis driver | visible 大幅改善 (0.79→0.35) 但 hidden 崩溃 | partial keep |
| Codex-Iter-4 | 2026-04-11 | 约束 hidden-vis path 仅用 hidden 输入 | amplitude 崩溃 (0.71) | revert |

**旧 best run (visible-centric val_score)**: `runs/20260411_115901_partial_lv_lv_guided_stochastic_refined`
**新 best run (hidden-centric val_score)**: `runs/20260412_063300_partial_lv_lv_guided_stochastic_refined`

---

## EXP-20260412-00：方法论转向 — 从预测到重构

### 状态：已完成

### 改动目的
将项目从"visible future prediction"导向转为"hidden recovery-centric"导向。这不是实验，是基础设施层面的方法论调整。

### 改了什么
1. **trainer val_score**：hidden 从 0.16 提升到 0.60（0.40 hidden_rmse + 0.20 h/e_corr），visible 从 0.79 降到 0.20
2. **trainer fit()**：去掉 `evaluate_full_context("val")`，不再在每个 epoch 做 future prediction 评估
3. **noise_scan 评分**：hidden 从 0.14 提升到 0.35，去掉 full_context_visible
4. **run 脚本评估**：去掉 `full_context_test`，新增 `hidden_val`，summary 以 hidden 为主
5. **图表**：fig4 从 full-context visible 改为全序列 hidden recovery 图
6. **诊断函数**：以 hidden Pearson 为核心诊断标准
7. **输出格式**：打印和 README 全部以 hidden recovery 为中心

### 验证
- py_compile 全部通过
- smoke test（2 epochs, 1 noise config）成功运行

### 未改动的部分
- 模型架构（PartialLVRecoveryModel）完全不变
- 训练损失函数的计算逻辑不变（17 项损失都保留）
- 滑动窗口训练机制不变
- `_full_context_train_step()` 保留（train 段内长 rollout 重构）
- 配置文件不变

---

## EXP-20260412-01：hidden-centric 基线实验

### 状态：已完成

### 实验目的
在方法论转向后，用新 val_score（hidden-centric）跑完整实验，建立新基线。

### 关键假设
新 val_score 会选出对 hidden recovery 更优的 best epoch（而非对 visible prediction 最优的 epoch）。

### 运行命令
```bash
python -m scripts.run_partial_lv_mvp --config configs/partial_lv_mvp.yaml
```

### 结果对比

| 指标 | 旧 best (115901, vis-centric) | 新 best (063300, hid-centric) | 变化 |
|------|------|------|------|
| **hidden RMSE** | 0.166 | **0.125** | **改善 24%** |
| **hidden Pearson** | 0.902 | **0.947** | **改善 5%** |
| **h/e correlation** | 0.099 | **0.040** | **改善 60%** |
| visible RMSE | 0.794 | 0.811 | 略微退步 2% |
| amplitude collapse | 0.058 | 0.094 | 略微退步 |
| residual dominates | 0.871 | 0.838 | 略有改善 |
| best epoch | 39 | 32 | 更早收敛 |

### 分析

**核心发现：仅改变 val_score 权重，不改任何代码/损失/架构，hidden recovery 就显著改善。**

- Hidden RMSE 从 0.166 降至 0.125（改善 24%），Pearson 从 0.902 提升至 0.947
- Hidden/env correlation 从 0.099 降至 0.040（改善 60%），disentanglement 大幅提升
- Visible RMSE 从 0.794 略升至 0.811（退步 2%），amplitude 从 0.058 升至 0.094
- Best epoch 从 39 降至 32：旧 val_score 选的是 visible 最优的 epoch，新 val_score 选的是 hidden 最优的 epoch — 说明 hidden 和 visible 的最优 epoch 不同

**解读**：之前的模型并非"hidden 学不会"，而是 val_score 在选 epoch 时偏向了 visible，错过了 hidden 更好�� epoch。这验证了方法论转向的正确性。

### 下一步建议
已被 EXP-20260412-02 取代（彻底去掉未来预测）。

---

## EXP-20260412-02：彻底去掉未来预测

### 状态：已完成

### 实验目的
第一次尝试（纯 hidden val_score，无任何 visible 信号）导致 hidden Pearson 从 0.947 崩溃到 0.296。
原因：visible rollout 虽然不是目标，但它是 hidden 动力学一致性的间接验证。完全去掉导致模型选择失灵。

第二次尝试：用 **train 段内的 visible 重构质量**（已知数据，非预测）作为间接信号，加入 val_score。

### 改了什么
- `fit()` 中：去掉 `_validation_metrics(val_loader)` 和 `_iterate_loader(val_loader)`
- 新增 `_iterate_loader(train_loader, training=False)` 作为 train 段内重构评估
- val_score = 0.40*hidden_rmse + 0.20*|h/e_corr| + 0.15*(1-pearson) + 0.15*train_vis_recon + 0.10*train_amplitude
- `run_experiment`：去掉 `evaluate_loader(test)`、`forecast_case()`、所有 sliding_eval/lv_guidance/particle 指标
- 图表精简为 5 张（fig1 真实轨迹 / fig2 test hidden / fig3 全序列 hidden / fig4 训练曲线 / fig5 诊断）

### 结果

| 指标 | 旧 best (vis-centric) | EXP-01 (hid+vis val) | EXP-02 (纯重构) |
|------|---|---|---|
| Hidden Test Pearson | 0.902 | 0.947 | 0.911 |
| Hidden Test RMSE | 0.166 | 0.125 | 0.187 |
| H/E Correlation | 0.099 | 0.040 | **0.010** |
| Best epoch | 39 | 32 | 38 |

### 分析

- Hidden Pearson 0.911：合理，比旧 best 略好，比 EXP-01 差（因为 EXP-01 仍用了 val 段预测信号）
- H/E correlation 0.010：三次实验中最好，解耦非常干净
- 代价：失去了 val 段 visible 预测带来的间接 hidden 信号，Pearson 从 0.947 降到 0.911
- **关键教训**：visible 重构不是目标，但它提供的动力学一致性信号有价值。用 train 段内重构（已知数据）替代 val 段预测是合理的折中

### 当前接受为新基线
Run: `runs/20260412_065155_partial_lv_lv_guided_stochastic_refined`

---

## EXP-20260412-03：砍掉 environment latent

### 状态：已完成

### 动机
发现 environment 恢复质量极差（与真实 environment 几乎无关），但 hidden 恢复却很好。分析后认为 environment 是"有害的干扰源"：它进入 residual/LV 输入传递错误信号，且消耗了 6 项 disentanglement 损失的训练容量。

### 改了什么
- 模型：完全移除 environment_head、environment_target_network、tau_env、environment_to_species、environment 噪声、OU 过程
- 训练器：移除 _environment_terms()、_sequence_diagnostics()、6 项 env 相关损失
- run 脚本：移除 disentanglement_metrics、环境图表

### 结果

| 指标 | 有 environment | 无 environment | 变化 |
|------|---|---|---|
| Hidden Test Pearson | 0.911 | **0.989** | **+8.5%** |
| Hidden Test RMSE | 0.187 | **0.063** | **-66%** |

### 结论
砍掉 environment 后 hidden recovery 大幅提升。Environment 在当前架构下**弊大于利**。

Run: `runs/20260412_072614_partial_lv_lv_guided_stochastic_refined`

---

## EXP-20260412-04：2×2 对比（LV 先验 × 数据类型）

### 状态：已完成

### 实验目的
回答核心方法论问题：**LV 先验的价值依赖数据是否真的是 LV 吗？**

合成数据存在循环论证问题：用 LV 造数据，用 LV 先验建模，当然有效。需要在非 LV 数据上重测。

### 设计

**新数据生成器** `data/partial_nonlinear_mvp.py`：
- **Holling type II** 捕食函数（饱和响应）：`a*x/(1+h*x)`，LV 无法表达
- **Allee effect**：部分物种低密度时增长率为负
- **时滞反馈**：`delay_coef * (x_t - x_{t-4})`
- 保持 5 visible + 1 hidden 结构，便于对比

**四个实验配置**：

| 配置 | 数据 | LV 先验 | Run |
|------|------|---------|-----|
| A | LV | ✅ | 20260412_072614 |
| B | LV | ❌ | 20260412_161153 |
| C | 非线性 | ✅ | 20260412_161625 |
| D | 非线性 | ❌ | 20260412_161938 |

### 结果

#### Hidden Test Pearson (越高越好)

|          | LV 先验 (use_lv_guidance=true) | 无 LV 先验 (use_lv_guidance=false) |
|----------|-----|-----|
| **LV 数据** | 0.9887 (A) | 0.9894 (B) |
| **非线性数据** | 0.9789 (C) | 0.9770 (D) |

#### Hidden Test RMSE (越低越好)

|          | LV 先验 | 无 LV 先验 |
|----------|-----|-----|
| **LV 数据** | 0.063 (A) | 0.107 (B) |
| **非线性数据** | 0.124 (C) | 0.156 (D) |

#### Val Hidden Pearson (验证集)

|          | LV 先验 | 无 LV 先验 |
|----------|-----|-----|
| **LV 数据** | 0.987 (A) | 0.982 (B) |
| **非线性数据** | 0.988 (C) | 0.986 (D) |

### 关键发现

**发现 1：LV 先验的价值有限，但在 LV 数据上仍有小幅优势**
- LV 数据：LV 先验 RMSE 0.063 vs 无 LV 0.107（RMSE 提升 41%，但 Pearson 几乎持平）
- 这说明 LV 先验主要改善**幅度拟合**（RMSE），对相位跟踪（Pearson）影响小

**发现 2：非线性数据上 LV 先验仍然有用，没有明显拖累**
- 非线性数据 + LV 先验: Pearson 0.979, RMSE 0.124
- 非线性数据无 LV 先验: Pearson 0.977, RMSE 0.156
- LV 先验在 misspecified 数据上没有变成拖累，反而略有帮助

**发现 3：数据类型对 hidden recovery 质量有影响**
- LV 数据的 hidden 恢复比非线性数据更好（RMSE 0.06-0.11 vs 0.12-0.16）
- 但两者 Pearson 都在 0.97 以上，整体质量都很高
- 说明**编码器映射路径**对两种数据都有效，不特别依赖动力学形式

**发现 4：LV 先验 + residual 架构有鲁棒性**
- 即使数据不是 LV 生成的，LV 先验也不会显著拖累性能
- residual 网络有能力"吸收"LV 无法表达的部分
- 这验证了"LV as soft prior"的设计是合理的

### 最重要的研究意义

**当前实验证据支持以下结论（置信度：中等）**：

> LV 先验在本项目中的实际价值不是"让模型学得更好"，而是"让 interaction matrix 可解释"。去掉 LV 先验只损失轻微的 RMSE，但失去 interaction matrix 的生态学含义。

这意味着：
- 如果研究目标是 **hidden recovery** → LV 先验不是必需的
- 如果研究目标是 **interaction matrix recovery** → LV 先验是必需的
- 真实数据场景下，LV 先验预计不会造成灾难性 misspecification

### 后续分析：interaction matrix 恢复（2026-04-12 下午）

分析 4 个 run 的 `data_snapshot.npz` 中 `interaction_true` 和 `interaction_pred`：

| 指标 | A (LV+LV先验) | B (LV+无先验) | C (非线性+LV先验) | D (非线性+无先验) |
|------|---|---|---|---|
| 有意义边符号准确率 (\|true\|>0.05) | 0.895 | 0.684 | **1.000** | 0.750 |
| Hidden 边符号准确率 | 1.000 | 1.000 | 1.000 | 1.000 |
| 可见-可见边符号准确率 | 0.846 | 0.538 | **1.000** | 0.600 |
| 矩阵相关性 | 0.765 | 0.754 | 0.745 | **0.293** |
| 相对 L2 误差 | 0.702 | 0.675 | 0.831 | 0.959 |

### 反转的结论

**LV 先验的真正价值不是 hidden recovery，是 interaction matrix recovery：**

1. LV 数据上: 带 LV 先验的符号准确率 0.895 vs 无 LV 先验 0.684 (下降 24%)
2. 非线性数据上: 带 LV 先验 1.000 vs 无 LV 先验 0.750
3. 最戏剧性: 非线性数据 + 无 LV 先验时，matrix correlation **崩盘到 0.293**（A/B/C 都在 0.74-0.77）
4. Hidden 边在所有配置下都 100% 正确 — 说明 hidden-visible 的主要交互方向不依赖 LV

### 研究叙事（修正版）

> **LV 先验对 hidden recovery 的影响小（Pearson 差异 < 0.01），但对 interaction matrix recovery 至关重要。**
>
> 即使数据不是 LV 生成的，LV 先验仍能帮助恢复生态交互的符号结构（direction）。这意味着 LV 先验的价值不限于"数据是 LV 时才有用"，而是作为**稀疏可解释的方向约束**，在多种动力学形式下都提供价值。
>
> 反之，去掉 LV 先验时，模型仍能做 hidden recovery（靠 encoder 映射），但 interaction matrix 退化为无意义的残差参数，失去生态学解释。

---

## EXP-20260412-05：Ricker 形式参数恢复

### 状态：已完成

### 动机
参数恢复分析（EXP-04 补充）揭示 tanh 形式下模型参数和数据参数不可对比（参数化差异）：
- 数据用 `x_{t+1} = x * exp(r + Ax)`（Ricker）
- 模型用 `x_{t+1} = x + lv_scale * x * tanh(r + Ax) + residual`

tanh 饱和使 `r + Ax` 的绝对值不敏感，growth_rates 恢复失败（spearman 0.31）。

### 改动
- 模型新增 `lv_form: ricker` 选项
- Ricker drift: `x * (exp(clamp(r+Ax, -1.12, 0.92)) - 1)`
- 加 clamp 防数值爆炸（范围和数据生成器一致）

### 结果（LV 数据 + LV 先验下对比 tanh vs Ricker）

| 参数 | 指标 | tanh | Ricker | 变化 |
|------|------|------|--------|------|
| Growth rates | Sign | 0.67 | **1.00** | +50% |
| | Spearman | 0.31 | **0.77** | +146% |
| | L2 误差 | 0.76 | **0.35** | -54% |
| Diagonal | Spearman | -0.09 | 0.43 | +520% |
| | Scale ratio | 4.54 | 3.63 | 略好 |
| Off-diagonal | Sign | **0.90** | 0.58 | **-35%** |
| | Spearman | **0.71** | 0.55 | -22% |
| Hidden Pearson | - | 0.989 | 0.986 | 持平 |

### 关键发现

**Ricker 形式是 trade-off 不是全面改善**：

- **Growth rates 大幅改善**：从几乎无恢复到 Spearman 0.77
- **Diagonal 改善有限**：scale 仍错 3.6 倍
- **Off-diagonal 变差**：`exp` 对弱交互敏感，符号准确率从 0.90 跌到 0.58
- **Hidden recovery 不变**：0.989 vs 0.986，说明 encoder 路径主导

### 研究解读

这揭示了**不同参数化适合不同目的**：
- tanh 饱和→结构恢复稳定（交互方向）
- Ricker 匹配→参数数值可恢复（增长率）

没有一种形式能同时做好两件事。**这正好支持两网络分离的设计（用户提出的方法 3）**：
- Stage 1 用 tanh encoder 做 hidden recovery
- Stage 2 用 Ricker fitter 在完整数据上恢复参数

### 训练稳定性
Ricker 形式下 3 epochs 内 Pearson 到 0.97，40 epochs 最终 0.986。clamp 有效防止爆炸。

Run: `runs/20260412_172547_exp_lv_data_ricker_form`（实际时间戳以目录为准）

---

## EXP-20260412-06（原 01）：v2 机制分离配置完整实验

### 状态：降级为低优先级（方法论转向后，visible 改善不再是核心目标）

### 实验目的
验证 v2 配置中新增的三个损失项（multiscale, local_variance, residual_energy）是否能改善 visible 预测质量和 LV/residual 分工，同时不破坏 hidden recovery。

### 关键假设
1. `residual_energy` 损失（lambda=0.15，LV 能量占比 >= 0.55 目标）会降低 residual dominates fraction
2. `multiscale` 损失（lambda=0.40，scale=2,4 的差分 L1）会捕捉中期动态趋势
3. `local_variance` 损失（lambda=0.35）会惩罚预测振幅坍缩

### v2 vs v1 配置差异

| 配置键 | v1 (partial_lv_mvp.yaml) | v2 (partial_lv_mvp_v2_mechanism.yaml) |
|--------|--------------------------|---------------------------------------|
| epochs | 40 | 50 |
| early_stopping_patience | 10 | 12 |
| lambda_multiscale | 0 (未设置) | 0.40 |
| lambda_local_variance | 0 (未设置) | 0.35 |
| lambda_residual_energy | 0 (未设置) | 0.15 |
| lambda_env_smooth | 0.090 | 0.12 |
| lambda_disentangle | 0.45 | 0.55 |
| lambda_orthogonality | 0.30 | 0.35 |
| lambda_variance_floor | 0.12 | 0.15 |
| lambda_timescale_prior | 0.25 | 0.30 |
| lambda_lv_guidance | 0.08 | 0.10 |
| lambda_residual_magnitude | 0.12 | 0.18 |
| hidden_variance_floor | 0.09 | 0.10 |
| environment_smoother_ratio | 0.45 | 0.35 |
| environment_autocorr_margin | 0.10 | 0.12 |

**注意**：v2 不仅新增了 3 个损失项，还微调了多个已有损失权重和训练参数。严格来说不是纯单变量实验，但作为一个整体配置方案的首次验证，这是合理的起点。

### 预计改动
无代码修改。仅使用不同配置文件。

### 运行命令
```bash
source .venv/bin/activate
python -m scripts.run_partial_lv_mvp --config configs/partial_lv_mvp_v2_mechanism.yaml
```

### 评价指标（对比基线 = best run 115901）

| 指标 | v1 基线 | 预期方向 | 警戒线 |
|------|---------|----------|--------|
| sliding visible RMSE | 0.794 | 持平或改善 | > 1.0 则失败 |
| full-context visible RMSE | 0.807 | 改善 | > 1.0 则失败 |
| hidden RMSE | 0.166 | 持平 | > 0.25 则警告 |
| hidden Pearson | 0.902 | 持平 | < 0.85 则警告 |
| amplitude collapse | 0.058 | 持平或改善 | > 0.20 则警告 |
| residual dominates fraction | 0.871 | 下降 | 无硬限 |
| hidden/env correlation | 0.099 | 持平或改善 | > 0.20 则警告 |

### 预期观察现象
- residual dominates fraction 应从 0.87 有所下降（residual_energy 约束的直接效果）
- amplitude collapse 应保持低位（local_variance 损失的效果）
- 如果 visible RMSE 改善而 hidden 不受影响 → v2 配置可作为新基线
- 如果 hidden 恶化 → 需要检查 disentangle/variance_floor 增强是否干扰了 hidden 训练

### 实际结果
（待实验后填写）

### 分析
（待实验后填写）

### 下一步建议
（待实验后填写）
