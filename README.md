# 部分观测生态动力学中的隐藏物种推断

从可见物种的时间序列中恢复未观测的隐藏物种和环境驱动，并用生态一致性与额外解释力来检验恢复结果。

目标投稿 SCI Q2 期刊，方法论核心是**数据驱动、无预设公式**的 hidden recovery。

---

## 当前最佳方法（2026-04-13 更新）

### 🥇 最佳方法：Linear Sparse + EM（闭式 baseline）

```
Step 1: 拟合 log(x_{t+1}/x_t) = r + A·x, L1 稀疏 A
Step 2: EM 迭代 — 带 hidden 项再拟合
Step 3: residual 投到 hidden_true 做 direction fit
```

**性能**：

| 数据 | Pearson |
|---|---|
| 合成 LV (5+1) | **0.977** |
| 合成 Holling (5+1) | 0.62（系统偏差）|
| **真实 Portal OT (11+1)** | **0.35** |

**特点**：
- 闭式优化，10 秒内跑完
- 最稳定、最可靠
- 所有其他方法的基线对照

**代码**：`scripts/real_data_portal_topk.py`（top-12 λ sweep 版）

---

### 🥈 次选方法：CVHI-NCD（GNN + 软预设 + 变分）

**架构**：
```
Posterior Encoder (GNN + Takens + 双轴 attention)
       ↓ q(h|X) = N(μ, σ²)
采样 h ~ posterior
       ↓
Multi-Layer Species-GNN Dynamics Operator:
  - 节点 = 物种 (N_visible + k_hidden)
  - 每边 5 种软预设形式 {Linear, LV, Holling II×2, Free NN}
  - Per-edge coef + gate (L1 稀疏) → 形式自动选择
  - Top-k attention 稀疏邻居
       ↓
预测 log(x_{t+1}/x_t) 对 visible MSE
       ↓
ELBO + L1 + anti-bypass + smoothness
```

**性能**：

| 数据 | Pearson |
|---|---|
| 合成 LV (5+1) | **0.84 ± 0.0002** |
| 真实 Portal OT (11+1) | 0.23 ± 0.002 |
| 真实 Portal OT (最佳 seed) | 0.59（CVHI 原版）|

**特点**：
- GNN 核心，物种作节点，边是物种间作用
- **软预设生态形式**（LV/Holling）+ **Free NN**（自由发现）
- 多层堆叠，深度表达
- 可 decode 发现的方程（可解释性）
- Pearson 稳定性极高（std < 0.002）
- **Pearson 暂未超越 Linear baseline**（在真实数据上）

**代码**：
- 模型：`models/cvhi_ncd.py`
- 训练（合成 LV）：`scripts/cvhi_ncd_synthetic_lv.py`
- 训练（真实 Portal）：`scripts/cvhi_ncd_simplified_portal.py`

---

### 🥉 备选方法：CVHI 原版（简单 GNN + anchor）

**架构**：Posterior Encoder (GNN) + Factorized DynamicsOperator（linear A + 小 GAT + hidden 线性耦合）

**性能**：

| 数据 | Pearson |
|---|---|
| 合成 LV | 0.88 |
| 合成 Holling | 0.40 |
| 真实 Portal OT | 0.33 ± 0.21（5 seeds）|
| 真实 Portal OT 最佳 seed | **0.59** |

**特点**：
- 比 CVHI-NCD 简单，训练快
- 合成数据上 Pearson 更高（0.88 vs 0.84）
- 真实数据上 seed 之间**极不稳定**（0.01 ~ 0.59）
- Val recon 可作无监督 seed 选择器

**代码**：`models/cvhi.py` + `scripts/train_cvhi.py` + `scripts/cvhi_portal.py`

---

## 项目目标

在部分观测条件下，联合推断：
1. **visible dynamics** — 可见物种动力学形式
2. **hidden latent states** — 未观测的隐藏物种 / 环境驱动
3. **生态结构** — 物种交互网络、作用形式

**不做未来预测**。核心是从 visible 的时间演化**重构** hidden ecological structure。

---

## 关键数据集

| 数据 | 来源 | 规模 | 用途 |
|---|---|---|---|
| 合成 LV (5+1) | `data/partial_lv_mvp.py` | 820 步 | 方法验证 |
| 合成 Holling (5+1) | `data/partial_lv_mvp_holling.py` | 820 步 | 非线性拓展 |
| **Portal Project** | `data/real_datasets/portal_rodent.csv` | 520 月, 41 物种 | **真实数据**主验证 |
| Lynx-Hare | `data/real_datasets/lynx_hare_long.csv` | 57 年, 2 物种 | 太小, 仅备注 |

**Portal 使用说明**：必须选 top-K 物种覆盖 ≥95% 总捕获量（= **top-12**），否则剩余未观测物种会污染 residual（详见 `notes/2026-04-13_cvhi_ncd_journey.md`）。

---

## 目录结构

```
data/
  real_datasets/                   真实数据集
    portal_rodent.csv              Portal 42 年月度啮齿动物 ★
    lynx_hare_long.csv             Hudson Bay 57 年 2 物种
  partial_lv_mvp.py                合成 LV 数据生成
  partial_lv_mvp_holling.py        合成 Holling 数据生成

models/
  cvhi_ncd.py                      ★ CVHI-NCD（最新次选方法）
  cvhi.py                          CVHI 原版（备选方法）
  partial_lv_recovery_model.py     早期 4-way rollout（已弃用主线）

scripts/
  real_data_portal_topk.py         ★ Linear Sparse+EM 在 top-12 (最佳方法)
  cvhi_ncd_synthetic_lv.py         ★ CVHI-NCD 在合成 LV
  cvhi_ncd_simplified_portal.py    ★ CVHI-NCD 在 Portal OT
  train_cvhi.py                    CVHI 原版训练（合成）
  cvhi_portal.py                   CVHI 原版（真实 Portal）
  cvhi_portal_ot_multiseed.py      CVHI 5-seed 验证
  portal_dynamics_form_comparison.py  6 种动力学形式对比
  identifiability_analysis.py      多 seed 辨识性分析

notes/
  2026-04-13_cvhi_ncd_journey.md   ★ 本日完整实验日志（10 次迭代）
  project_overview.md              项目总览
  codebase_map.md                  代码说明
  experiment_status.md             实验运行状态
  design_decisions.md              设计决策
  next_steps.md                    待办

runs/                              实验结果输出（带时间戳）

CLAUDE.md                          AI 助手入口
README.md                          本文件
```

---

## 快速运行

```bash
# 激活环境
source .venv/bin/activate    # Linux/Mac
# or
.venv\Scripts\activate       # Windows

# 最佳方法：Linear Sparse + EM 在 Portal top-12 (λ sweep)
python -m scripts.real_data_portal_topk

# 次选方法：CVHI-NCD 在合成 LV (信号干净, 验证架构)
python -m scripts.cvhi_ncd_synthetic_lv

# 次选方法：CVHI-NCD 在 Portal OT (真实数据)
python -m scripts.cvhi_ncd_simplified_portal

# 备选方法：CVHI 原版 5-seed 稳定性验证
python -m scripts.cvhi_portal_ot_multiseed

# 动力学形式对比（6 forms × 3 species × 3 seeds）
python -m scripts.portal_dynamics_form_comparison
```

---

## 技术栈

- **框架**：PyTorch 2.x + CUDA (RTX 4060)
- **数据**：合成 LV/Holling 动力学 + Portal Project 真实月度数据
- **Python 环境**：`.venv/` + `requirements.txt`

---

## 关键文档导航

| 想了解... | 看这里 |
|---|---|
| **本日实验流水账（CVHI-NCD 设计过程）** | [notes/2026-04-13_cvhi_ncd_journey.md](notes/2026-04-13_cvhi_ncd_journey.md) ★ |
| 项目目标与方法框架 | [CLAUDE.md](CLAUDE.md) |
| 代码文件功能说明 | [notes/codebase_map.md](notes/codebase_map.md) |
| 设计决策 | [notes/design_decisions.md](notes/design_decisions.md) |
| 历史迭代日志（早期 4 轮） | [codex_iteration_log.md](codex_iteration_log.md) |

---

## 方法学结论摘要

1. **真实数据 setup 至关重要**：必须近似完整群落（top-K 覆盖 ≥95%），否则未观测物种污染 residual
2. **log-ratio (Ricker) 预设不够**：真实数据有密度饱和，需要 Holling II 风形式
3. **Species-as-Nodes + Soft-Preset Forms 是正确的 GNN 语义**：节点=物种，边=作用，形式软选择
4. **CVHI-NCD Pearson 天花板**：合成 LV 可逼近 anchor (0.84 vs 0.97)，真实 Portal **暂未超越 Linear baseline** (0.23 vs 0.35)
5. **论文建议**：不单追 Pearson，强调 CVHI-NCD 的**结构化输出**（可解释方程 / 交互图 / 多通道分解）

详见 [notes/2026-04-13_cvhi_ncd_journey.md](notes/2026-04-13_cvhi_ncd_journey.md)。
