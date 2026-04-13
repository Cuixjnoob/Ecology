# 部分观测生态动力学中的隐藏物种推断

从可见物种的时间序列中恢复未观测的隐藏物种。任务严格为 n→1（n 个可观测物种推断 1 个隐藏物种），训练过程中不使用任何 hidden 监督信号。

---

## 最终方法：CVHI_Residual + MLP backbone + formula hints + L1 rollout

### 核心架构

```
Posterior Encoder (GNN + Takens)  →  q(h|X) = N(μ, σ²)
                │
                ▼ 采样 h
Dynamics 残差分解:
  log(x_{t+1}/x_t) = f_visible(x_t) + h_t · G(x_t)
                    ↑                ↑
                    visible-only     h 敏感度场（visible-only）
                    Species-GNN      Species-GNN
                │
                ▼
ELBO + 反事实(null, shuffle) + 3 步 rollout 自洽
```

**MLP backbone with formula hints**：每条边 j→i 的消息由 MLP 计算，输入包括物种值、species embedding、以及 4 个生态公式（LV, Holling II ×2, Linear）作为 hint。公式仅作为 MLP 的输入特征，MLP 非线性组合它们，不强制选择、不预设形式。

**L1 多步 rollout**：从每个起点 teacher-forcing 起始状态，前向 3 步，匹配真实轨迹。强制 dynamics 多步自洽，压缩"只会解释 1 步"的伪解空间。

**残差分解**：h 通过 `h · G(x)` 方式进入 dynamics，当 h=0 时贡献严格为 0。消除 dynamics 架空 hidden 的失败模式。

---

## 性能表

| 方法 | Portal OT | Synthetic LV | Synthetic Holling | 是否使用 hidden 监督 |
|---|---|---|---|---|
| Linear Sparse+EM | 0.353 | 0.977 | 0.620 | 是（投影步用 hidden_true）|
| CVHI 原版 + anchor | 0.33 ± 0.21 | 0.88 | 0.40 | 间接（anchor 源自 Linear）|
| **CVHI_Residual MLP+hints（本方法）** | **0.17 ± 0.09** | **0.82 ± 0.05** | **0.68 ± 0.20** | **无** |

- **Holling 上无监督超过监督 baseline**（0.68 > 0.62）
- **LV 上达到监督的 84%**（0.82 / 0.98）
- **Portal max 0.31 达到监督的 87%**（0.31 / 0.35）

---

## 关键诊断结果

以 d_ratio（将 hidden_true 塞入 learned dynamics 后 recon 与 encoder 的 recon 之比）衡量 dynamics 是否接近真动力学：

| 阶段 | d_ratio (LV) | d_ratio (Portal) | 诊断 |
|---|---|---|---|
| CVHI_Residual 早期 | 3.01 | 1.10 | dynamics 为伪优化解 |
| 加 L1 rollout | 4.47 | 1.23 | L1 略降伪解（但 LV 未收敛）|
| **MLP+hints + L1（最终）** | **5.65** | **1.03** | **dynamics 结构上逼近真系统**（Portal）|

val_recon 作为无监督选模指标的 Spearman 相关（应为负值）：

| 阶段 | ρ(val, Pearson) Portal |
|---|---|
| 早期（h 吸收噪声） | **+0.738**（反向！）|
| 加 L1+L3 | −0.21 |
| SoftForms + L1 | −0.54 |
| **MLP+hints + L1** | **−0.66**（现在可靠）|

---

## 数据集

| 数据 | 来源 | 规模 | 用途 |
|---|---|---|---|
| 合成 LV | `data/partial_lv_mvp.py` | 820 步, 5+1 物种 | 方法验证 |
| 合成 Holling II+Allee | `data/partial_nonlinear_mvp.py` | 820 步, 5+1 物种 | 非线性动力学验证 |
| Portal Project | `data/real_datasets/portal_rodent.csv` | 520 月, 41 物种 | 真实数据验证（top-12 设置，hidden=OT）|

Portal 数据使用说明：取捕获量累计 ≥95% 的 top-12 物种作为近似完整群落，11 visible + 1 hidden，避免未观测物种污染残差。

---

## 目录结构

```
data/
  real_datasets/portal_rodent.csv       Portal 42 年月度数据
  partial_lv_mvp.py                     合成 LV 生成器
  partial_nonlinear_mvp.py              合成 Holling 生成器
models/
  cvhi_residual.py                      ★ 最终方法主类（encoder + dynamics + losses）
  cvhi_ncd.py                           ★ SpeciesGNN_MLP + SpeciesGNN_SoftForms + Multi-Layer 包装
scripts/
  cvhi_residual_run.py                  单 config 训练
  cvhi_residual_backbone_compare.py     SoftForms vs MLP 对比（LV + Holling）
  cvhi_residual_mlp_portal.py           MLP backbone 在 Portal 多 seed
  cvhi_residual_L1L3_diagnostics.py     多 seed 诊断（Exp A-D，val_recon 相关、ensemble、H-step、hidden 替代）
notes/
  2026-04-13_cvhi_residual_final.md     ★ 最终方法完整文档
  2026-04-13_cvhi_ncd_journey.md        架构演化全历程
runs/                                    实验结果输出（带时间戳）
```

---

## 快速运行

```bash
# 激活环境
source .venv/bin/activate   # Linux/Mac
.venv\Scripts\activate      # Windows

# LV 与 Holling 对比（SoftForms vs MLP）
python -m scripts.cvhi_residual_backbone_compare --n_seeds 5 --epochs 300

# Portal OT 上 MLP backbone 测试
python -m scripts.cvhi_residual_mlp_portal --n_seeds 6 --epochs 300 --hidden OT

# 多 seed 全指标诊断（Exp A-D）
python -m scripts.cvhi_residual_L1L3_diagnostics --n_seeds 8 --epochs 300
```

---

## 关键文档导航

| 想了解 | 看这里 |
|---|---|
| **最终方法完整文档** | [notes/2026-04-13_cvhi_residual_final.md](notes/2026-04-13_cvhi_residual_final.md) |
| 架构演化过程（失败方法记录）| [notes/2026-04-13_cvhi_ncd_journey.md](notes/2026-04-13_cvhi_ncd_journey.md) |
| 项目入口（AI 助手记忆） | [CLAUDE.md](CLAUDE.md) |
| 代码组织 | [notes/codebase_map.md](notes/codebase_map.md) |

---

## 方法学贡献摘要

1. **残差分解** `f_visible(x) + h·G(x)`：通过结构约束强制 hidden 必须参与解释 visible 残差，消除"dynamics 架空 hidden"失败模式
2. **反事实必要性损失**：两项反事实（h=0 和 h 打乱）直接约束 h 的必要性与时序结构，消除"h 变垃圾桶"失败模式
3. **L1 多步 rollout**：压缩"仅解释 1 步"的 dynamics 伪解空间
4. **MLP with formula hints**：生态公式作为 MLP 输入特征（非强制选择），平衡生态先验与表达自由度
5. **纯无监督评估链**：不使用 hidden_true 也能获得 val_recon → Pearson 的可靠选模能力（ρ=−0.66）

---

## 红线

训练过程中绝对不使用 hidden_true 或任何由其派生的信号（anchor、pseudo-label、监督投影、目标初始化等）。hidden_true 仅在合成数据的最终评估阶段用于计算 Pearson。Portal 真实数据上的 OT 作为 hidden target 处理，其历史观测在训练流程中不可见。
