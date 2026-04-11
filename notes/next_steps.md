# 下一步任务

> 最后更新：2026-04-11  
> 状态标记：⬜ open question / ✅ 已实现 / 🔄 已尝试但效果不佳

---

## 立即可做（1-2 天）

### 1. ✅ 多尺度 loss + 局部方差损失
**已实现**。代码位于 `train/partial_lv_mvp_trainer.py` → `_visible_loss_terms()` 中的 `multiscale_loss` / `local_var_loss`。  
在 `configs/partial_lv_mvp.yaml` 中权重为 0（未启用），在 `configs/partial_lv_mvp_v2_mechanism.yaml` 中启用（0.40 / 0.35）。  
**待做**：用 v2 配置跑完整实验，对比原版。

### 2. ✅ OU 环境慢变量
**已实现**。环境更新为 `env + τ_env × (target - env)`，`τ_env ∈ [0.03, 0.12]`。  
**待做**：消融实验——对比 OU env vs 直接神经网络预测 env。

### 3. ✅ Hidden fast innovation network
**已实现**。`hidden_fast_network` 仅作用于 hidden species，`hidden_fast_scale ∈ [0.03, 0.15]`。  
**待做**：消融实验——去掉 hidden_fast 看 hidden recovery 变化。

### 4. ⬜ v2 配置完整实验
**描述**：用 `partial_lv_mvp_v2_mechanism.yaml` 跑完整实验，与 `partial_lv_mvp.yaml` 结果对比。  
**依赖**：无，直接可跑 `python scripts/run_partial_lv_mvp.py --config configs/partial_lv_mvp_v2_mechanism.yaml`  
**风险**：低  
**预期产出**：判断 multiscale + local_variance + residual_energy 三个 v2 损失是否有效

### 5. ⬜ 消融实验套件
**描述**：对以下组件做独立消融：
- hidden_fast_network（去掉 → hidden recovery 是否下降）
- OU 环境（改回直接预测 → timescale separation 是否消失）
- residual curriculum（去掉 → residual dominates 是否更严重）
- multiscale + local_variance loss（v2 vs v1）  
**依赖**：`models/partial_lv_recovery_model.py` 支持通过 flag 关闭各组件  
**风险**：低，但需要约 4 次完整训练（每次 ~20 分钟 on CPU）

### 6. ⬜ visible-only baseline 正式对比
**描述**：训练一个无 hidden / 无 environment 的 visible-only 模型，用同样的评估管线对比。这是项目书要求的"额外解释力"检验的前提。  
**依赖**：需要修改 `PartialLVRecoveryModel` 或创建简化版模型  
**风险**：中等，需确保评估管线兼容

---

## 中期重构（1-2 周）

### 7. 🔄 structured hidden→visible / env→visible driver（需改进后重试）
**描述**：Codex Iteration 3 证明此方向能显著改善 full-context visible（RMSE 0.81→0.75, sliding 0.79→0.35），但 hidden 崩溃。需要在 hidden 一致性约束下重试。  
**改进思路**：
- 对 hidden→visible driver 增加 hidden MSE 梯度传递（stop_gradient 控制比例）
- 或使用 contrastive loss 保证 hidden state 的 identity
- 或限制 hidden→visible driver 只能使用 hidden state 的低维投影  
**依赖**：`models/partial_lv_recovery_model.py` 需要新增 sub-network  
**风险**：高——Iteration 3/4 已经两次失败于此方向。但收益也高（唯一显著改善 full-context 的方向）

### 8. ⬜ 训练策略优化
**描述**：
- 当前 noise 退火是线性的，可能需要 cosine 或 step schedule
- 验证集综合分数的权重可能需要调整（hidden 权重偏低）
- Full-context 训练的比例（72% history / 28% rollout）可能不是最优  
**依赖**：`train/partial_lv_mvp_trainer.py`  
**风险**：低

### 9. ⬜ 评估管线增强
**描述**：
- 系统化的 visible-only vs visible+hidden 额外解释力对比
- 交叉验证或多种子实验（当前只用 seed=42）
- interaction matrix recovery 的更细致评估（不只是符号准确率）  
**依赖**：需要 baseline 模型（见任务 6）  
**风险**：低

---

## 高风险研究问题（探索性）

### 10. ⬜ 真实数据接入
**描述**：从合成数据迁移到真实生态时间序列。  
**挑战**：
- 没有 ground truth hidden species
- 需要设计"可能存在 hidden 影响"的间接证据
- 数据格式 / 缺失值 / 不规则采样  
**依赖**：先在合成数据上完成方法验证  
**风险**：极高

### 11. ⬜ hidden 的可解释性
**描述**：当前 recovered hidden 是一个 scalar latent，不能直接说它"是某个真实物种"。需要发展从 recovered hidden 到生态解释的桥梁。  
**可能方向**：
- hidden 与真实 hidden species 的对应关系分析
- hidden 对 visible dynamics 的贡献分解
- interaction matrix 的生态解释  
**风险**：高（涉及因果推断）

### 12. ⬜ 多 hidden species
**描述**：当前只支持 1 个 hidden species。真实系统可能有多个未观测物种。  
**挑战**：
- 多 hidden 之间的 identifiability
- hidden 数量的自动确定
- 计算复杂度增长  
**风险**：极高

### 13. ⬜ particle rollout 的有效利用
**描述**：当前 particle rollout (K=4) 对结果几乎没有帮助。`codex_iteration_log.md` 的 pre-iteration baseline 也确认了这一点。  
**可能原因**：
- particle 数量太少
- aggregation 方式（mean）过于简单
- 噪声分布不正确  
**风险**：中等
