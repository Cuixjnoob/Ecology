# 项目总览

> 最后更新：2026-04-11

## 1. 项目背景

真实生态系统通常是不完全观测的。即使部分物种未被直接测量，它们仍可能通过竞争、捕食、资源限制等机制影响可见物种的演化。仅用 visible-only 模型拟合数据，往往会把隐藏生态作用错误地压缩为噪声或参数漂移。

**本项目不是普通时间序列 forecast 项目。** 核心是从 visible dynamics 中恢复 hidden ecological structure（隐藏物种、环境驱动），并用生态一致性与额外解释力来检验其有效性。

## 2. 问题定义

在部分观测条件下，联合推断：
1. visible dynamics — 可见物种的未来演化
2. hidden latent states — 未观测的隐藏物种/环境驱动
3. 额外解释力 — recovered hidden 是否真的提升了系统动力学的解释

### 评估立场
- hidden recovery > visible forecast（重要性排序）
- ecological residual 不是噪声，可能是 hidden/environment/未建模交互的投影
- 成功标准：接回正向模型后的额外解释力，而非拟合分数

## 3. 方法总览

### 3.1 数据
合成生态系统（`data/partial_lv_mvp.py`）：
- 5 visible + 1 hidden + 1 environment，离散 LV (Ricker) 动力学
- 820 步，train/val/test = 60/20/20
- 数据自动筛选，拒绝 too_flat / too_periodic

### 3.2 模型架构
`PartialLVRecoveryModel`（`models/partial_lv_recovery_model.py`）：

**编码阶段**：
- Takens delay embedding → `delay_encoder`（MLP）
- 全序列 GRU 编码 → `history_encoder`
- 趋势斜率 → slope summary
- 三者融合 → `context_refiner` → refined context

**初始化**：
- `hidden_head(context)` → hidden 初始值（softplus 确保正）
- `environment_head(context)` → env 初始值（tanh）

**Rollout**（每步四路分工）：
1. **LV-guided drift**：可学习的 growth rates + interaction matrix + environment coupling
2. **Neural residual**：残差网络 + curriculum 渐进
3. **Hidden fast innovation**：仅作用于 hidden species 的快时间尺度网络
4. **Stochastic noise**：可学习的 species-specific 噪声强度

环境状态独立更新为 OU 过程：`env + τ_env × (target - env) + noise`

### 3.3 训练
`PartialLVMVPTrainer`（`train/partial_lv_mvp_trainer.py`）：
- 17 项损失函数的加权组合（见 `CLAUDE.md` 第四节）
- 噪声配置网格搜索 → 选最优噪声组合
- 每 epoch 一次 sliding-window 训练 + 一次 full-context 训练
- Early stopping 基于复合 val_score
- Curriculum learning：residual 强度从 0.3 线性增长到 1.0（前 60% epochs）

### 3.4 评估
- hidden recovery（逐步滑窗 hidden_initial 恢复 → RMSE / Pearson）
- visible rollout / full-context（sliding-window + 全段预测）
- disentanglement（hidden-env correlation / roughness / autocorrelation）
- LV/residual 能量分析
- interaction matrix recovery（hidden 边符号准确率）

## 4. 代码做到了哪里

### ✅ 已完成
- 合成数据生成与筛选
- 完整的 partial-observation 模型（四路分工 + OU 环境）
- 17 项损失 + 噪声网格搜索 + curriculum 训练
- 6 种评估指标
- 6 张标准图表自动生成
- 4 轮 Codex 迭代实验（有完整日志）
- 26 个实验运行存档

### 🔄 进行中
- full-context visible prediction 仍是主瓶颈
- residual/LV 分工比例仍不理想
- v2 机制分离配置已就绪但未跑完整实验

### ❌ 未开始
- 真实数据接入
- visible-only baseline 正式对比（旧 pipeline 有，但与当前主线不一致）
- 消融实验（对 hidden_fast / curriculum / OU env 的独立贡献评估）
- 额外解释力的系统化检验

## 5. 项目书与代码现状的一致性说明

项目书（用户在本次对话开头给出）描述了研究意图和方法框架。以下是与代码现状的对应关系：

| 项目书要求 | 代码现状 | 一致性 |
|------------|----------|--------|
| hidden recovery 为中心 | 评估体系中 hidden recovery 权重 0.08+0.08=0.16，visible 相关权重 0.67 | ⚠️ 评估权重仍以 visible 为主，但 hidden 有完整评估管线 |
| LV 结构先验 | 实现为 LV-guided drift + 可学习交互矩阵 | ✅ |
| 正向检验（接回模型后的额外解释力） | 尚无 visible-only vs visible+hidden 的系统对照 | ❌ 缺少 |
| 不做黑箱拟合器 | 四路分工中 residual 仍是主力（dominates 87%） | ⚠️ 方向正确但尚未达成 |
| GNN / message passing | 存在于旧主线（`models/gnn.py`, `full_model.py`），当前主线未使用 | ⚠️ 旧代码保留但不在主线 |

## 6. 三条实验线简述

| 线路 | 脚本 | 状态 |
|------|------|------|
| A. 旧图模型 forecast | `run_train.py` / `run_pipeline.py` | 历史线，已弃用 |
| B. hidden-only recovery | `run_hidden_inference_experiment.py` | 阶段性验证，不是当前主线 |
| C. partial-observation 联合 recovery | `run_partial_lv_mvp.py` | **当前主线** |
