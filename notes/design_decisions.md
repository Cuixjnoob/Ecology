# 设计决策

> 最后更新：2026-04-11  
> 每条决策标注：理论驱动 / 工程折中 / 经验调优  
> 否决方向标注：❌ 不建议继续

---

## 1. 合成数据而非真实数据

**类型**：工程折中  
**决策**：使用合成 LV 系统（5 visible + 1 hidden + 1 environment），而非真实生态观测数据。  
**理由**：合成数据提供 ground truth hidden/environment/interaction matrix，使评估有明确标准。这是"先证明方法可行"的 MVP 策略。  
**代价**：结论的生态迁移性未经验证。  
**代码位置**：`data/partial_lv_mvp.py` → `generate_partial_lv_mvp_system()`

---

## 2. LV-guided drift 作为 soft backbone

**类型**：理论驱动  
**决策**：状态更新以 Lotka-Volterra 作为可学习结构化 drift，而非纯神经 delta。  
**理由**：
- LV 是生态动力学的最基础先验
- 使 interaction matrix 可解释
- 防止模型退化为纯残差拟合  
**Evidence**：`codex_iteration_log.md` 显示 residual dominates 87%，说明 LV 占比仍不够。但去掉 LV 后模型完全不可解释。  
**代码位置**：`models/partial_lv_recovery_model.py` forward() 中的 `lv_raw` / `lv_drift` 计算

---

## 3. 四路分工架构

**类型**：理论驱动 + 经验调优  
**决策**：rollout 分为 LV drift + neural residual + hidden fast innovation + stochastic noise 四路。  
**演化历程**：
- v0：纯神经 delta
- v1：LV drift + residual + noise 三路
- v2（当前）：加入 hidden_fast_network，仅作用于 hidden species  
**理由**：hidden species 需要一个独立的快时间尺度通道，因为 LV drift 对 hidden 的建模不如 visible 充分（hidden 不直接参与 LV 训练信号）。  
**代码位置**：`models/partial_lv_recovery_model.py` → `hidden_fast_network`, `hidden_fast_scale_unconstrained`

---

## 4. 环境 OU 过程（慢变量设计）

**类型**：理论驱动  
**决策**：环境状态更新为 OU 过程 `env + τ_env × (target - env)`，而非直接神经网络预测。  
**理由**：
- 生态环境通常是慢变量（温度、降水、营养盐循环）
- OU 过程天然产生均值回复、时间尺度可控的动态
- `τ_env ∈ [0.03, 0.12]` 确保环境变化慢于 species dynamics  
**Evidence**：`timescale_prior` 损失验证了 env roughness < hidden roughness  
**代码位置**：`models/partial_lv_recovery_model.py` → `tau_env_unconstrained`, `environment_target_network`

---

## 5. Residual curriculum learning

**类型**：经验调优  
**决策**：residual 贡献乘以 curriculum factor = 0.3 + 0.7 × min(1.0, epoch / (total_epochs × 0.6))  
**理由**：
- 训练初期让 LV 先建立结构化理解
- 后期再放开 residual 拟合残余部分
- 防止 residual 一开始就主导  
**代码位置**：`train/partial_lv_mvp_trainer.py` fit() 中设置 `model.residual_curriculum_progress`

---

## 6. Hidden/environment disentanglement 多重约束

**类型**：理论驱动 + 经验调优  
**决策**：使用 5 个互补约束：correlation 惩罚 + diff orthogonality + variance floor + timescale prior + env smoothness  
**理由**：单一约束（如只用 correlation）不足以实现功能分离。需要从统计相关性、变化率、方差下界等多维度施压。  
**Evidence**：
- `20260411_004141` 对比实验显示无约束时 hidden/env correlation 达 0.912
- 加入约束后 `115901` 降至 0.099  
**代码位置**：`train/partial_lv_mvp_trainer.py` → `_environment_terms()`

---

## 7. 噪声网格搜索 + 退火

**类型**：工程折中  
**决策**：先用短训练（6 epochs × 3 patience）在噪声网格上搜索，选最优组合后正式训练。训练过程中噪声退火（从 1.0 线性降到 0.25）。  
**理由**：噪声配置对结果影响极大（过强→平线化、过弱→过拟合），但最优值数据依赖。  
**代码位置**：`scripts/run_partial_lv_mvp.py` → `_noise_scan()`

---

## 8. Full-context 训练作为补充

**类型**：经验调优  
**决策**：每 epoch 额外做一次 full-context train step（用 train 段前 72% 为历史，后 28% 为 target）。  
**理由**：sliding-window 训练让模型看到的 rollout horizon 较短，不足以学习长期动态。  
**注意**：`codex_iteration_log.md` Iteration 2 尝试多 cut-point 增强此策略但失败，说明**更多 full-context 不等于更好**。  
**代码位置**：`train/partial_lv_mvp_trainer.py` → `_full_context_train_step()`

---

## 9. Multiscale loss + local variance preservation

**类型**：经验调优  
**决策**：v2 新增多尺度差分 L1（scale=2,4）和局部窗口方差保持损失。  
**理由**：标准 MSE 不惩罚"平滑化"预测。multiscale 捕捉中期趋势，local variance 惩罚振幅坍缩。  
**状态**：已实现于代码中。在 `partial_lv_mvp.yaml` 中权重为 0（未启用），在 `partial_lv_mvp_v2_mechanism.yaml` 中启用（0.40 / 0.35）。  
**代码位置**：`train/partial_lv_mvp_trainer.py` → `_visible_loss_terms()` 中的 `multiscale_loss` / `local_var_loss`

---

## 10. LV 能量占比约束

**类型**：经验调优  
**决策**：约束 LV 能量占比 ≥ 0.55（目标值），通过 `residual_energy` 惩罚实现。  
**理由**：residual dominates fraction 一直偏高（0.87），需要能量层面的约束来提升 LV 分支的实际作用。  
**代码位置**：`train/partial_lv_mvp_trainer.py` → `_lv_terms()` 中的 `lv_energy_fraction`

---

## 11. 数据质量筛选

**类型**：工程折中  
**决策**：生成合成数据后检查 too_flat / too_periodic，不合格则重新生成（最多 640 次尝试）。  
**理由**：某些随机种子产生的数据缺乏足够复杂性，在这样的数据上训练/评估没有意义。  
**代码位置**：`data/partial_lv_mvp.py` → `generate_partial_lv_mvp_system()`

---

## 12. Takens delay embedding 作为输入编码

**类型**：理论驱动  
**决策**：使用 delay embedding（length=6, stride=2）编码历史序列，再用 MLP 映射到 delay embedding 空间。  
**理由**：Takens 嵌入定理保证从单变量延迟坐标可重建系统吸引子。对多变量 visible species 的延迟编码可保留局部动力学结构。  
**代码位置**：`models/partial_lv_recovery_model.py` → `_build_delay_features()`, `delay_encoder`

---

## ❌ 已否决方向

### ❌ 硬限 visible residual budget
**迭代**：Codex Iteration 1  
**结果**：hidden 崩溃（RMSE 0.17→0.40, Pearson 0.90→-0.57）、amplitude 恶化至 0.67  
**教训**：方向正确（减弱 residual），但硬限太粗暴。应通过软约束（能量占比、curriculum）而非硬 clamp。

### ❌ 多 cut-point full-context 训练
**迭代**：Codex Iteration 2  
**结果**：visible 和 hidden 全面恶化  
**教训**：更多长 rollout 训练不等于更好。瓶颈在 visible generation 的结构分工，不在训练数据覆盖。

### ⚠️ structured hidden→visible / env→visible driver（有条件继续）
**迭代**：Codex Iteration 3  
**结果**：sliding visible 大幅改善（0.79→0.35），但 hidden 崩溃（Pearson→-0.09）  
**教训**：方向极有前途，但需防止 hidden→visible 路径绕过 hidden identity。值得在约束 hidden 一致性的前提下重试。

### ❌ 约束 hidden-visible 路径仅用 hidden 输入
**迭代**：Codex Iteration 4  
**结果**：amplitude 崩溃（0.71），整体不如 best run  
**教训**：hidden→visible path 不能只依赖 hidden state，需要一定的 context 输入。下次应尝试更温和的约束方式。
