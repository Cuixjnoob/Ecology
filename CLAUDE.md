# CLAUDE.md — 项目入口记忆文件

> 最后更新：2026-04-12（方法论转向：从预测到重构，hidden recovery-centric）  
> 维护方式：每次重大代码/实验修改后必须同步更新本文件

---

## 一、项目一句话

**部分观测生态动力学中的隐藏物种推断与生态一致性建模。**  
核心是从 visible dynamics 中恢复 hidden ecological structure，并用生态一致性与额外解释力来检验其有效性。
**不做未来预测。** Visible rollout 仅作为训练信号（在已知数据内重构），评估以 hidden recovery quality 为核心。

---

## 二、当前主线

### 主线脚本
- **入口**：`scripts/run_partial_lv_mvp.py`
- **模型**：`models/partial_lv_recovery_model.py` → `PartialLVRecoveryModel`
- **训练**：`train/partial_lv_mvp_trainer.py` → `PartialLVMVPTrainer`
- **配置**：
  - `configs/partial_lv_mvp.yaml`（原版，生产用）
  - `configs/partial_lv_mvp_v2_mechanism.yaml`（v2 机制分离版，实验用）

### 当前架构：四路分工
模型状态更新公式（每个 rollout step）：

```
state_{t+1} = state_t
    + α_lv  × LV_guided_drift           ← 结构化生态先验（Lotka-Volterra）
    + α_res × curriculum × residual      ← 神经残差网络（curriculum 控制强度渐进）
    + hidden_fast_scale × hidden_fast    ← 仅作用于 hidden species 的快创新网络
    + noise                              ← 过程噪声（可关闭）
```

环境状态更新（OU 过程）：
```
env_{t+1} = env_t + τ_env × (target - env_t) + noise
```
其中 `target` 由 `environment_target_network` 输出，`τ_env ∈ [0.03, 0.12]` 可学习。这一设计使环境成为慢变量（relaxation toward target），而非直接预测。

### 关键可学习参数
| 参数 | 含义 | 约束范围 |
|------|------|----------|
| `alpha_lv` | LV 贡献权重 | [0.10, 0.95] via sigmoid |
| `alpha_res` | residual 贡献权重 | [0.08, 0.90] via sigmoid |
| `tau_env` | 环境 OU 时间常数（越小环境越慢） | [0.03, 0.12] via sigmoid |
| `hidden_fast_scale` | hidden 快创新强度 | [0.03, 0.15] via sigmoid |
| `residual_curriculum_progress` | 外部设定的课程学习进度 | [0, 1]，由 trainer 在训练循环中设置 |
| `growth_rates` | LV 内禀增长率 | 可学习 |
| `off_diagonal` / `diagonal_unconstrained` | LV 交互矩阵 | 对角线强制为负 |
| `environment_to_species` | 环境对各物种的影响系数 | 可学习 |

### 子网络清单
| 网络 | 输入 | 输出 | 用途 |
|------|------|------|------|
| `delay_encoder` | delay 窗口特征 | delay embedding | Takens 风格时序编码 |
| `history_encoder` | visible 标准化序列 (GRU) | hidden state | 历史上下文压缩 |
| `rollout_memory` | 每步 visible → GRUCell | 更新后 memory | rollout 中递进记忆 |
| `context_refiner` | delay emb + GRU state + slope | refined context | 上下文融合 |
| `hidden_head` | context | hidden 初始值 | softplus 确保正 |
| `environment_head` | context | env 初始值 | tanh |
| `residual_network` | log_state + interaction + env + memory + context | 全物种 delta | 神经残差 |
| `environment_target_network` | log_state + env + memory + context | env target | OU 过程的目标值 |
| `hidden_fast_network` | visible + env + memory | hidden-only delta | 快时间尺度隐藏创新 |

---

## 三、数据生成

**合成数据**（`data/partial_lv_mvp.py`）：
- 5 个 visible species + 1 个 hidden species + 1 个 environment driver
- 离散 Lotka-Volterra (Ricker-style) 动力学，820 步（含 160 步 warmup）
- environment：准周期驱动（多频正弦叠加 + AR(1) 平滑，AR 系数 0.88）
- pulse：随机稀疏脉冲（概率 0.018，衰减 0.82）
- 交互矩阵含捕食链 (0→1→2→3→4→0) + 稀疏竞争 + hidden↔visible 耦合
- 数据筛选：拒绝 too_flat / too_periodic，保留 moderate_complexity

**数据划分**：train 60% / val 20% / test 20%（默认 492/164/164 步）

---

## 四、损失函数体系（共 17 项）

### Visible 损失
| 损失 | 配置键 | 说明 |
|------|--------|------|
| `visible_one_step` | `lambda_visible_one_step` | 首步 MSE |
| `visible_rollout` | `lambda_visible_rollout` | 全窗口 rollout MSE |
| `peak_visible` | `lambda_peak_visible` | 高分位加权 MSE |
| `slope` | `lambda_slope` | 差分趋势 L1 + 方向一致性 |
| `amplitude` | `lambda_amplitude` | 振幅/标准差坍缩惩罚 |
| `multiscale` | `lambda_multiscale` | 多尺度差分 L1（scale=2,4）|
| `local_variance` | `lambda_local_variance` | 局部窗口方差保持 |

### Hidden 损失
| 损失 | 配置键 | 说明 |
|------|--------|------|
| `hidden_initial` | `lambda_hidden_initial` | 初始 hidden 与真值 MSE |
| `hidden_rollout` | `lambda_hidden_rollout` | rollout hidden 与真值 MSE |
| `hidden_smooth` | `lambda_hidden_smooth` | 二阶差分平滑 |
| `interaction_hidden` | `lambda_interaction_hidden` | hidden 边交互矩阵 MSE |
| `interaction_sparse` | `lambda_interaction_sparse` | 非对角稀疏 |

### 环境/解耦损失
| 损失 | 配置键 | 说明 |
|------|--------|------|
| `env_smooth` | `lambda_env_smooth` | 环境一阶差分平滑 |
| `env_stability` | `lambda_env_stability` | 环境值域稳定 |
| `disentangle` | `lambda_disentangle` | hidden-env 相关性惩罚 |
| `orthogonality` | `lambda_orthogonality` | diff 序列正交 |
| `variance_floor` | `lambda_variance_floor` | hidden/env 方差下界 |
| `timescale_prior` | `lambda_timescale_prior` | env 应比 hidden 更慢的先验 |

### LV/Residual 平衡损失
| 损失 | 配置键 | 说明 |
|------|--------|------|
| `lv_guidance` | `lambda_lv_guidance` | deterministic vs lv-only 接近度 |
| `residual_magnitude` | `lambda_residual_magnitude` | residual/LV 比值 + 能量惩罚 |
| `residual_energy` | `lambda_residual_energy` | LV 能量占比 ≥ 0.55 目标 |

### Full-context 训练
每个 epoch 额外做一次 full-context train step，使用 train 段前 72% 作为历史、后 28% 作为 rollout 目标。

---

## 五、评估体系

### 以 hidden recovery 为中心的评估
1. **hidden recovery**：`recover_hidden_on_split()` — 逐步滑窗用 `hidden_initial` 恢复 hidden，计算 RMSE / Pearson
2. **hidden/env disentanglement**：correlation / roughness / autocorrelation
3. **interaction recovery**：hidden 边的符号准确率

### 辅助评估
4. **sliding-window visible rollout**：标准窗口 rollout RMSE / Pearson
5. **full-context visible forecast**：从训练段末尾预测验证/测试段
6. **amplitude collapse score**：预测振幅 vs 真值振幅比
7. **LV/residual 能量分析**：能量占比、visible-specific 比值

### 验证集综合分数（early stopping 用，hidden recovery-centric）
```python
val_score = 0.40 * hidden_recovery_rmse
           + 0.20 * |hidden_env_correlation|
           + 0.15 * sliding_visible_rmse
           + 0.05 * peak_visible_error
           + 0.10 * amplitude_collapse_score
           + 0.10 * residual_dominates_fraction
```

---

## 六、当前最佳 run 与瓶颈

### 当前接受的 best run
`runs/20260411_115901_partial_lv_lv_guided_stochastic_refined`

来源：`codex_iteration_log.md` Iteration 4 结束后的决策：回退至此 run。

| 指标 | 值 |
|------|-----|
| sliding visible RMSE | 0.794 |
| full-context visible RMSE | 0.807 |
| hidden RMSE | 0.166 |
| hidden Pearson | 0.902 |
| amplitude collapse | 0.058 |
| hidden/env correlation | 0.099 |
| LV/residual ratio mean | 1.645 |
| residual dominates fraction | 0.871 |

### 当前状态（2026-04-12 方法论转向后）
1. **Hidden recovery 质量良好**（Pearson 0.90），但需在新 val_score 下重新验证
2. **residual 分支仍偏强**（dominates fraction 0.87），结构化 LV 分支未充分发挥
3. **hidden/env 解耦有效**（corr 0.099），多重约束策略起作用
4. **不再追求 full-context visible prediction**，该方向已放弃

---

## 七、已尝试并否决的方向

以下方向均在 `codex_iteration_log.md` 中有详细记录（包含定量指标）：

| 方向 | 迭代 | 结果 | 原因 |
|------|------|------|------|
| 硬限 visible residual budget | Iteration 1 | **revert** | hidden 崩溃（RMSE 0.17→0.40, Pearson 0.90→-0.57）、amplitude 恶化 |
| 多 cut-point full-context 训练 | Iteration 2 | **revert** | visible/hidden 均恶化 |
| structured hidden→visible / env→visible driver | Iteration 3 | **partial keep** | sliding visible 大幅改善（0.79→0.35）但 hidden 崩溃（Pearson→-0.09） |
| 约束 hidden-visible 路径仅用 hidden 输入 | Iteration 4 | **revert** | amplitude 崩溃 (0.71) |

**关键教训**：
- "减弱 residual" 方向正确，但硬限太粗暴
- structured visible driver 能改善 full-context，但需防止 hidden bypass
- 多 cut-point 训练没有帮助
- Iteration 3 的 structured channel 是唯一显著改善 full-context visible 的尝试，值得沿此方向继续但需同时约束 hidden 一致性

---

## 八、常用命令

```bash
# 激活环境
source .venv/bin/activate

# 用原版配置运行主线实验
python -m scripts.run_partial_lv_mvp --config configs/partial_lv_mvp.yaml

# 用 v2 机制分离配置运行
python -m scripts.run_partial_lv_mvp --config configs/partial_lv_mvp_v2_mechanism.yaml

# 编译检查（不运行）
python -m py_compile models/partial_lv_recovery_model.py
python -m py_compile train/partial_lv_mvp_trainer.py
python -m py_compile scripts/run_partial_lv_mvp.py

# 快速 smoke test（建议修改 epochs=2, noise_scan_epochs=1）
```

---

## 九、修改原则

1. **先理解再改**：读完本文件 + `notes/` + `codex_iteration_log.md` 后再动手
2. **不要重复失败实验**：查看第七节已否决方向及 `codex_iteration_log.md` 全文
3. **小步验证**：每次只改一个机制，立即 smoke test
4. **以 hidden recovery 为中心**：不要为了 visible forecast 而牺牲 hidden 质量
5. **保持向后兼容**：新配置键用 `.get(key, default)` 读取，旧 YAML 不需要更新
6. **记录一切**：每次实验需在 `codex_iteration_log.md` 追加完整条目（假设→变更→指标→keep/revert）

---

## 十、项目文件索引

| 文件/目录 | 用途 |
|-----------|------|
| `CLAUDE.md` | 本文件，项目入口 |
| `configs/partial_lv_mvp.yaml` | 原版配置（生产用） |
| `configs/partial_lv_mvp_v2_mechanism.yaml` | v2 机制分离配置（实验用） |
| `models/partial_lv_recovery_model.py` | 核心模型（4-way rollout + OU 环境） |
| `models/encoders.py` | MLP 等基础网络构件 |
| `train/partial_lv_mvp_trainer.py` | 全上下文训练器（17 项损失） |
| `train/utils.py` | 工具函数（seed / data loader / save_json） |
| `scripts/run_partial_lv_mvp.py` | 主实验入口脚本 |
| `data/partial_lv_mvp.py` | 合成生态系统数据生成器 |
| `data/dataset.py` | TimeSeriesBundle + 窗口采样 |
| `data/transforms.py` | Log-ZScore 标准化 |
| `notes/` | 项目文档（overview / codebase_map / experiment / design / next_steps） |
| `codex_iteration_log.md` | Codex 4 轮迭代的完整实验日志（**最重要的决策文档**） |
| `docs/research_description.md` | 研究描述文书 |
| `runs/` | 保留的关键实验结果（3 个里程碑 run） |
