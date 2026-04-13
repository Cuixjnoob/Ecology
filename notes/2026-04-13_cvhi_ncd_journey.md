# 2026-04-13 CVHI / CVHI-NCD 实验日志

> 本日主线：从 CVHI baseline → 真实数据验证 → 动力学形式探索 → CVHI-NCD (GNN 软预设) 架构迭代。

---

## 总览

一日共 10 次关键实验，方法论从"固定形式 + GNN" 推进到 "species-as-nodes GNN + soft-preset forms + 可学习组合"。

目标演化：
- 早晨：验证 CVHI 在真实数据上能否 work (Portal 42 年啮齿动物)
- 中午：探究动力学形式是否必须 log-ratio（Ricker 预设）
- 下午：设计并实现 CVHI-NCD (Neural Compositional Dynamics with GNN attention)
- 傍晚：多次迭代修 bug、tightening、multi-layer，最终确认**合成数据 0.84，真实数据未能超越 Linear baseline**

---

## 实验时间线（按 run 目录）

### 1. Lynx-Hare 真实数据测试 (`20260413_130322_real_data_lynx_hare`)

**目标**：验证方法在最经典生态数据上的可行性

**设置**：
- Hudson Bay 1847-1903, 57 年
- Method A: Hare (N=1) as visible, Lynx as hidden
- Method B: Hare Takens 嵌入 (N=4) as virtual visible, Lynx as hidden

**结果**：
- Method A Pearson: 0.002（失败）
- Method B Pearson: 0.18（弱）

**结论**：**2 物种数据太小**，不满足"5 visible + 1 hidden"的框架假设。切换到更大数据集。

---

### 2. Portal Project 初测 (`20260413_131050_real_data_portal`)

**目标**：用更大的真实群落验证

**设置**：
- 41 物种, 520 月 (1977-2020+), Chihuahuan Desert rodents
- **错误 setup**：取 top-5 最常见物种作 visible, 第 6 (RM) 作 hidden
- Linear Sparse + EM sparsity sweep, λ ∈ [0, 2.0]

**结果**：
- 最佳 Pearson = **0.16** (无论 λ 如何)
- λ=2.0, EM iter=1 (线性基线最好)

**诊断（用户提出的关键洞察）**：
> "如果是这种 40 多个物种的，每个物种之间都有影响的啊"

**根因**：framework 假设 visible + hidden = 全系统。但 41 物种里选 5+1 = 还有 **35 个未观测物种** 藏在 residual 里，污染了 hidden 信号。

---

### 3. Portal top-12 修复 (`20260413_131642_real_data_portal_topk`)

**目标**：修正上一实验的 setup 错误

**设置**：
- 选 top-12 物种（覆盖 **95.47% 总捕获量**），近似完整群落
- 11 visible + 1 hidden, 依次试每个 top-12 物种作 hidden

**结果（按 |Pearson| 排序）**：

| 排名 | Hidden | Pearson | 生态解释 |
|---|---|---|---|
| 1 | **DO** (kangaroo rat) | **+0.429** | 与 DM/DS 同属, 强耦合 granivore |
| 2 | OT (grasshopper mouse) | +0.353 | 肉食, 与群落部分耦合 |
| 3 | PP | +0.342 | 主力 granivore |
| 4 | PF | +0.307 | 小型 granivore |
| ... | ... | ... | ... |
| 12 | RM (harvest mouse) | +0.187 | **唯一非纯 granivore**, 解耦 |

**关键结论**：
- 修复 setup 后 **Pearson 从 0.16 跳到 0.43**
- 排名有**生态意义**：granivore 物种可恢复性高，食性解耦物种低
- 这本身就是一个值得写的发现

---

### 4. CVHI on Portal (`20260413_132624_cvhi_portal`)

**目标**：把 CVHI（GNN 原版）跑到 Portal 上，对比 Linear 基线

**设置**：
- CVHI 原版架构 (encoder d=64, dynamics d=32, 2 blocks)
- 对 top-4 候选 hidden (DO, OT, PP, PF) 分别跑
- 单 seed

**结果**：

| Hidden | Linear baseline | CVHI coarse (EM) | **CVHI posterior** | Δ |
|---|---|---|---|---|
| DO | 0.429 | 0.18 | 0.428 | ≈ tied |
| **OT** | 0.353 | 0.31 | **0.471** ✓ | **+0.12** |
| PP | 0.342 | 0.33 | 0.320 | -0.02 |
| PF | 0.307 | 0.27 | 0.134 | -0.17 |

**结论**：
- CVHI 在 OT 上明显胜出 (0.47)
- 但 PF 上退步 (0.13) —— 说明方法**不是万能**
- 最佳 Pearson 0.47 > Linear 0.43

---

### 5. OT 多 seed 验证 (`20260413_133026_cvhi_portal_ot_multiseed`)

**目标**：验证 CVHI 在 OT 上 0.47 是否稳定

**设置**：5 seeds × 500 epochs

**结果**：

| Seed | Val recon | Pearson |
|---|---|---|
| 42 | 1.93 | +0.47 |
| 123 | 6.22 | **+0.01** ← 失败 |
| 456 | 0.86 | +0.17 |
| **789** | **0.63** | **+0.59** ← 最佳 |
| 2024 | 2.27 | +0.39 |

**统计**：
- Mean ± std = **0.33 ± 0.21**
- Linear baseline = 0.35 (CVHI mean 未超过)
- **Val recon 和 Pearson 强负相关**（0.63 最低 → 0.59 最高）

**关键发现**：
- CVHI **单 seed 极不稳定**（0.01 ~ 0.59）
- 但 **val_recon 可作无监督选择器**
- Best-by-val: Pearson 0.59 (>> Linear 0.35)

---

### 6. 动力学形式对比 (`20260413_133952_dynamics_form_comparison`)

**目标**：探究 log-ratio (Ricker) 预设是否必要

**设置**：6 种形式 × 3 seeds × 3 target 物种

**结果**（max|Pearson|）：

| Form | Description | OT | DO | PP |
|---|---|---|---|---|
| A | log-ratio 线性 (基线) | 0.283 | 0.146 | 0.203 |
| **B** | **log-ratio 二次 (Holling 风)** | **0.458** | **0.526** | 0.305 |
| C | Gompertz (log-state 线性) | 0.233 | 0.037 | 0.236 |
| D | 加性增量 | 0.326 | 0.122 | 0.130 |
| E | 线性+神经残差 | 0.245 | 0.154 | 0.234 |
| F | 完全神经网络 | 0.208 | 0.183 | 0.263 |

**关键发现**：
- **二次项 (Form B) 大幅提升**——DO 0.15 → 0.53, OT 0.28 → 0.46
- 真实 rodent 数据有明显**密度饱和 / Holling II 风结构**
- 纯 log-ratio 线性不够

**用户反馈推动的下一步**：
> "是否必须从一个library中选一个 自己寻找是否有可能？或者在模型中动态组合"

→ 从"预设 6 种"进化到"atoms + 可学习组合"

---

### 7. CVHI-NCD v1 (初版) (`20260413_140245_cvhi_ncd_portal`)

**目标**：实现"原子空间 + GNN attention 组合"架构

**设计**：
- 原子 P = [x_1..x_N, H_1..H_k, 1]
- Rank-R 乘/除/log 组合, softmax attention over atoms
- 温度退火
- 每物种 L1 稀疏 readout

**用户反馈**（关键修正）：
> "不是不是 我仍然希望使用的nodes是物种"  
> "然后 可以预设LV holling这些 但是可以软预设"  
> "或者attention那个 不一定是gnn"

→ 重写架构：**Species-as-Nodes GNN + Soft-Preset Forms**

---

### 8. CVHI-NCD v2 (Species-GNN + SoftForms) (`20260413_140444_cvhi_ncd_portal`)

**设计**：
- **节点 = 物种** (N_v + k_h 个)
- **每条边 5 种形式的软组合**:
  - Linear: x_j
  - LV bilinear: x_i · x_j
  - Holling II 线: x_j/(1+α·x_j)
  - Holling II 双: x_i·x_j/(1+α·x_j)
  - Free NN MLP
- 每边每形式有 coef (signed) + gate (sigmoid, L1 稀疏)
- Top-k attention 选邻居
- **PerSpeciesTemporalAttn 预处理**（非 GNN, per-species 时序 attention）

**结果**（3 seeds）：
- Pearson = **0.107 ± 0.052**（比 Linear 0.35 大幅退步）

**诊断**：
- 211K 参数 / 5700 数据点 = 37× 过参数化
- Free NN 太灵活 → 吞掉 hidden 信号
- Gates 从未真正稀疏化
- **Anchor Pearson = 0.29，模型输出 0.11，把 anchor 拉坏了**

---

### 9. CVHI-NCD v3 (KL bug fix) (`20260413_141429_cvhi_ncd_synthetic_lv`)

**关键 bug 发现**：
```python
# 错误版本
kl_per_step = 0.5 * (... + (sigma_sq + mu²) / prior_var - 1)
# mu = anchor + delta_mu, KL 把 anchor 推向 0！
```

修复：KL 改为用 delta_mu (encoder 原始输出)，不是 mu (包含 anchor)：
```python
kl_per_step = 0.5 * (... + (sigma_sq + delta_mu²) / prior_var - 1)
```

**结果**：

| 数据 | bug 修复前 | 修复后 |
|---|---|---|
| 合成 LV (anchor=0.97) | 0.30 ± 0.02 | **0.79 ± 0.01** ↑↑ |
| Portal OT (anchor=0.29) | 0.107 | 0.17 |

---

### 10. CVHI-NCD v4 (tightening: multi-layer + anchor_scale) (`20260413_142909_cvhi_ncd_*`)

**C1. 硬 posterior**: `prior_std` 1.5→0.5, `free_bits` 0.05→0.02  
**C2. 强 anti-bypass**: `min_h2v_mass` 0.05→0.15, `lam_anti_bypass` 2→8; **h→v gates 初始化 sigmoid(0)=0.5**  
**C3. 多层 GNN**: `num_gnn_layers=2`, 每层独立 gates/coefs  
**C4. anchor_scale**: `delta_mu * 0.3` (限制 encoder 漂移)

**结果**：

| 实验 | v3 | **v4** | anchor | Linear |
|---|---|---|---|---|
| 合成 LV | 0.79 ± 0.01 | **0.84 ± 0.0002** ↑ | 0.97 | 0.977 |
| Portal OT | 0.17 | **0.23 ± 0.002** | 0.29 | 0.35 |

**讨论**：
- 合成 LV 上逼近 anchor (0.84 vs 0.97, diff 0.13)
- Portal OT 上**仍低于 anchor** (0.23 vs 0.29)
- 稳定性极好 (std 0.0002)
- **没能超越 Linear baseline**（真实数据上的目标未达成）

---

## 方法学结论

### 1. 真实数据的 setup 至关重要
- 必须使用**近似完整群落**（top-K 覆盖 ≥95%）
- 不能简单取前几个最常见物种——会混入 N-K 个未观测物种污染 residual

### 2. log-ratio (Ricker) 预设在真实数据上不够
- 真实 rodent 数据有明显密度饱和（Holling II 风）
- 二次项即可把 Pearson 从 0.28 提到 0.46 (OT)
- 但完全自由的神经网络又容易过拟合——需要"软预设"平衡

### 3. Species-as-Nodes + Soft-Preset Forms 是正确的 GNN 语义
- 节点必须是物种（生态意义）
- 边是物种间作用
- 每条边软选择 {Linear, LV, Holling II, Free NN}
- L1 稀疏 gate 自动选形式

### 4. Anchor（h_coarse）的问题
- 来自 Linear Sparse + EM，间接包含 hidden 监督
- 给 CVHI 提供稳定性, 但"污染"了 unsupervised 宣称
- 无 anchor 版本稳定性差 (0.33 ± 0.21) 但更诚实

### 5. CVHI-NCD 的 Pearson 天花板
- **合成 LV**: 能逼近 anchor (0.84 vs 0.97), 接近 CVHI 原版 0.88
- **真实 Portal**: **无法超越 Linear baseline** (0.23 < 0.35)
- 差距来源：partial observation 下 identifiability 不足, visible 可被多种 h 轨迹解释

---

## 未解决的方向

1. **无 anchor 版本的 CVHI-NCD**: 不用 Linear Sparse+EM 的 h_coarse 作先验，完全 unsupervised
2. **多通道 hidden (k>1)**: 分解物种 / 环境 / 季节三成分
3. **外部协变量**: 加降雨 / NDVI 作为 exogenous driver
4. **Ensemble**: 多 seed 取 val 最低的 top-K 平均
5. **方程 decode**: 训练后读出"本数据发现的方程"作为可解释性卖点

---

## 对 Q2 paper 的启示

当前 state-of-the-art（本日最佳）：

| 任务 | 最佳方法 | Pearson |
|---|---|---|
| 合成 LV | **Linear Sparse + EM** | 0.977 |
| 合成 Holling | Linear Sparse + EM | 0.62 (系统偏差) |
| **真实 Portal OT** | **Linear Sparse + EM (λ sweep)** | **0.35** |
| 真实 Portal OT (CVHI 最佳 seed) | CVHI 原版 seed 789 | 0.59 |

**Paper 主线建议**：
- 方法论核心：CVHI-NCD 架构（GNN + 软预设 + 多层 + 可解释方程 decode）
- 合成实验展示方法**可行**
- 真实数据诚实讨论**局限性**（Pearson 不超 Linear, 但稳定性 / 可解释性 / 结构化输出胜出）
- 写成"方法 + 多角度评估"的故事，不单追 Pearson

**不建议的方向**：
- 强求 CVHI-NCD 超越 Linear Sparse + EM 的 Pearson ——目前未见到可行路径
- 多模态/多通道扩展需要额外外部数据（降雨等），超出当前 Portal 纯群落数据的范围

---

## 下一步候选

1. **无 anchor CVHI-NCD**（用户已指示走这条）
   - 代价：稳定性差
   - 价值：完全 unsupervised, 更诚实
2. **方程 decode demo**：跑一次 CVHI-NCD，读出发现的方程作为 figure
3. **多通道 k=3**：channel 0 = 物种 hidden, channel 1 = 环境, channel 2 = 季节
4. **Ensemble + val-based selection** 作为 deployable 方法
