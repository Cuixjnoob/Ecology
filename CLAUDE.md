# CLAUDE.md — 项目入口记忆文件

> 最后更新：2026-04-13（CVHI_Residual + MLP backbone + formula hints + L1 rollout 定型）

---

## 一、项目一句话

**部分观测生态动力学中的隐藏物种推断**：从 n 个 visible 物种时间序列恢复 1 个 hidden 物种，不预设动力学公式，严格无 hidden 监督。

---

## 二、当前方法（已定型）

### CVHI_Residual + MLP backbone + formula hints + L1 rollout

**核心架构**：
```
Posterior Encoder (GNN + Takens)
    ↓ q(h|X) = N(μ, σ²)
Dynamics:  log(x_{t+1}/x_t) = f_visible(x_t) + h_t · G(x_t)
    │        (两个 Species-GNN, MLP backbone)
    ▼
Loss: recon + rollout(3-step) + 反事实(null, shuffle) + KL + sparsity
```

**MLP backbone with formula hints**：每条边 j→i 的消息由 MLP 从 `[x_i, x_j, s_i, s_j, LV_hint, Holling_lin_hint, Holling_bi_hint, linear_hint]` 计算。公式仅作为输入特征，不强制选择。

**L1 rollout**：3 步 teacher-forced rollout 强制 dynamics 多步自洽。

**残差分解**：`h · G(x)` 结构确保 h=0 时贡献为 0，硬约束消除架空。

---

## 三、主要文件

| 文件 | 角色 |
|---|---|
| `models/cvhi_residual.py` | CVHI_Residual 主类，包含 forward/loss/rollout/lowpass 等 |
| `models/cvhi_ncd.py` | `SpeciesGNN_MLP`（MLP backbone） + `SpeciesGNN_SoftForms`（soft-preset 对照版） + `MultiLayerSpeciesGNN`（包装） + `PerSpeciesTemporalAttn` + `MultiChannelPosteriorEncoder` |
| `models/cvhi.py` | 原版 CVHI（已弃用，保留作为 anchor-based 对照） |
| `scripts/cvhi_residual_backbone_compare.py` | 最终对比脚本：SoftForms vs MLP（LV + Holling） |
| `scripts/cvhi_residual_mlp_portal.py` | MLP backbone 在 Portal OT 上的多 seed |
| `scripts/cvhi_residual_L1L3_diagnostics.py` | 多 seed 诊断（Exp A-D） |

---

## 四、当前结果

| 数据 | Pearson（mean ± std） | max | 对照 Linear 监督 baseline |
|---|---|---|---|
| LV (5+1) | 0.82 ± 0.05 | 0.87 | 0.98 |
| Holling (5+1) | 0.68 ± 0.20 | 0.86 | **0.62**（无监督超越监督）|
| Portal OT (11+1) | 0.17 ± 0.09 | 0.31 | 0.35 |

**d_ratio**（hidden_true 代入 learned dynamics 的 recon 比）：LV 5.65, Portal **1.03**（接近 1 说明 dynamics 结构上正确）

**ρ(val_recon, Pearson) on Portal**：−0.66（方向正确，可做无监督选模）

---

## 五、关键约束（红线）

1. **训练中绝对不用 hidden_true**：不作监督目标、不作 anchor、不作 pseudo-label、不作初始化
2. **不引入外部协变量**（降雨/NDVI 等）
3. **任务严格 n→1**，单 hidden
4. **节点必须是物种**（GNN 语义保留）

---

## 六、方法演化路径（可查阅的详细记录）

- [notes/2026-04-13_cvhi_ncd_journey.md](notes/2026-04-13_cvhi_ncd_journey.md) — 完整 10 次迭代记录
- [notes/2026-04-13_cvhi_residual_final.md](notes/2026-04-13_cvhi_residual_final.md) — 最终方法完整文档

演化摘要：
1. CVHI 原版 + anchor（违反无监督红线）
2. CVHI-NCD + soft-preset forms（gates 无分化）
3. CVHI_Residual 去 anchor（Portal 上 val_recon 反向）
4. 加 L1 rollout（LV 提升，Portal 改善 val 相关）
5. 加 L3 低频先验（证伪：hidden 不是慢变量）
6. MLP backbone + formula hints（**最终版，各项指标全面改善**）

---

## 七、诊断实验套件

`scripts/cvhi_residual_L1L3_diagnostics.py` 包含四组关键诊断：

- **Exp A**：多 seed 扫描 val_recon、m_null、m_shuf、h_var 与 Pearson 的相关性
- **Exp B**：top-K（按 val_recon）ensemble + 跨 seed 相似度
- **Exp C**：H-step 原型（固定 dynamics，从 4 种 h_init 内循环优化 h_free）
- **Exp D**：hidden_true 替代诊断（测试 learned dynamics 能否接受真 h）

这些诊断在定位"dynamics 伪解"作为根本问题上起了决定性作用。

---

## 八、已否决的方向（不要重复）

| 方向 | 为什么否决 |
|---|---|
| Anchor from Linear Sparse+EM | anchor 来自监督投影，违反红线 |
| CVHI-NCD 软混合 5 种预设 | gates 在 LV 和 Holling 上几乎相同，未发生形式选择 |
| L3 低频先验 | LV 真实 hidden 本身含高频成分，低频约束会压掉真信号 |
| Val-based top-K ensemble（SoftForms 阶段）| Portal 上 val_recon 反向选解，ensemble 更差 |
| D-H-Q 交替优化 | Exp C 证明 h 空间已凸，问题在 dynamics 层 |
| 外部协变量 | 用户明确禁止 |

---

## 九、未完成工作（paper-ready 清单）

### 必做
1. 多 seed 正式统计（20-30 seeds, 95% CI）
2. 消融实验独立跑：MLP vs SoftForms, with/without hints, L1 vs no L1
3. 跨 hidden 物种（Portal 上 DO, PP, PF, PE）

### 可选
4. Ensemble by val-selection（ρ 已翻正，top-3 应能提 Portal 到 0.25+）
5. 公式 hint 库扩充（Holling III, Ivlev, Beverton-Holt）
6. 可解释性分析（MLP 对各 hint 的扰动敏感度）

---

## 十、常用命令

```bash
# 环境
source .venv/bin/activate   # Linux/Mac
.venv\Scripts\activate       # Windows

# 最终方法 LV+Holling 对比（SoftForms vs MLP）
python -m scripts.cvhi_residual_backbone_compare --n_seeds 5 --epochs 300

# Portal 上 MLP backbone
python -m scripts.cvhi_residual_mlp_portal --n_seeds 6 --epochs 300 --hidden OT

# 多 seed 诊断
python -m scripts.cvhi_residual_L1L3_diagnostics --n_seeds 8 --epochs 300

# 代码编译检查
python -m py_compile models/cvhi_residual.py
python -m py_compile models/cvhi_ncd.py
```

---

## 十一、修改原则

1. 先读 [notes/2026-04-13_cvhi_residual_final.md](notes/2026-04-13_cvhi_residual_final.md) 理解现有架构
2. 结构性改动前先用诊断实验（Exp A-D）定位瓶颈，避免盲目调参
3. 每次改动单一组件，立即多 seed 验证
4. 红线绝不触碰：训练流程中不出现 hidden_true
5. d_ratio 和 ρ(val_recon, Pearson) 是两个核心无监督指标，改动后都要检查
