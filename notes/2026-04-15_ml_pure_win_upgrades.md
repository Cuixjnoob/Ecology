# ML / GNN pure-win upgrades — 2026-04-15

> 目标：列出"加上去基本只有好处没有害处"的 ML 侧改进, 每项都要实验验证.
> 原则：不脱离生态语义, AI 与生态**融合** — 很多 ML 技巧可以拿生态先验反向强化 (mass ratio → edge weight, trophic layer → depth, etc).

## 一、已落地 (Stage 1b 验证)
- [x] **torch.compile** (Triton 可用时) — JIT, 0 accuracy 影响. Windows 无 triton 已静默跳过
- [x] **Input dropout augmentation** (p=0.05) — 混沌序列抗过拟合. Stage 1b 实测有效
- [x] **RMSE log reconstruction** — log-scale reconstruction as auxiliary loss

## 二、已实现未测 (在 train_utils_fast.py)
- [x] **EMA of weights** (decay=0.999, warmup-aware) — 权重 EMA, 泛化更好
- [x] **Snapshot ensemble** (last 15%, N=5) — 末期 checkpoint 平均
- [x] **Hierarchical h** (slow+fast channels) — 已接入 encoder

## 三、拟新增 (ML pure-win 候选, ablation 验证)

### A. 训练器升级 (几乎无副作用)

| 组件 | 预期效果 | 代价 | 风险 |
|---|---|---|---|
| **SWA** (Stochastic Weight Averaging) | +0.5-1% Pearson, 参数平均泛化 | 末期保 SWA buffer | 0 |
| **CosineAnnealingWarmRestarts** | 逃离 local min, +0-2% | 0 | 0 |
| **Lookahead 封装 AdamW** | 更稳定 update, +0-1% | k 步额外存储 | 0 |
| **Gradient noise 注入** (annealed → 0) | 逃离 saddle, +0-1% | 0 | 极低 |
| **SAM** (Sharpness-Aware Min) | +1-2% robustness | 2× compute | 若陡峭 manifold 可能漂移 |
| **DropPath / 随机深度** 在 transformer block | +0-2%, 弱化依赖 | 0 | 0 |
| **ReZero** 初始化 (残差 scale=0 出发) | 训练更稳 | 微小 | 0 |

### B. GNN 专门升级

| 组件 | 预期效果 | 生态 hook |
|---|---|---|
| **Edge-drop augmentation** (物种-物种图) | +0.5-1.5% 泛化 | 用 1 − 食物网重叠概率 加权 drop (保护已知相关边) |
| **GraphNorm** 替代 LayerNorm | +0-1% (GNN 特化) | 0 |
| **Laplacian PE** / 随机游走 PE | +0.5% 位置感 | 可结合 functional-trait 相似度图 |
| **Virtual node** (全局均值聚合) | +0-1% | 作为"群落总体状态" 解读 |

### C. Posterior 改进

| 组件 | 预期效果 | 生态 hook |
|---|---|---|
| **IWAE bound** (K=5 importance) | tighter ELBO, 更好 h 后验 | 直接替换现 MoG |
| **DReG estimator** (Tucker 2019) | 更低方差 IWAE 梯度 | 0 |
| **VRNN 时间耦合** posterior | h_t 间显式耦合 | Beninca Lyapunov 预测 horizon 自然匹配 |
| **Latent ODE h** (Rubanova 2019) | h_t 由 NeuralODE 演化, 内生平滑 | **Portal 不规则采样天然** |

### D. 自蒸馏 / 一致性正则

| 组件 | 方法 | 生态 hook |
|---|---|---|
| **EMA-teacher self-distillation** | EMA 模型出 h_teacher, 学生 MSE(h_student, h_teacher.detach()) | 替代已证伪 L3 smooth |
| **Consistency regularization** | 两种弱 augment (dropout + 时间 shift) 下 h 相近 | 对混沌系统鲁棒 |

### E. 数据层

| 组件 | 方法 |
|---|---|
| **TimeWarp 增广** | 局部非线性时间拉伸 (生态系统 rate 变化常见) |
| **RandomShift** | 起始时间相位随机 |
| **Cutout** 时间段屏蔽 | 模拟采样缺失 |

### F. 优化器细节

| 组件 | 影响 |
|---|---|
| **Weight decay 分层 tuning** (bias/norm 不 decay) | +0-0.5% |
| **Gradient clipping 1.0** | 已加 |
| **LR warm-up 50 step** | 已加 |

---

## 四、AI × 生态融合 (非 pure-win, 但创新点高)

### F1. MTE-informed edge attention bias (新)
边 j→i 的 attention logit 加 bias `log(M_j / M_i) · α_mte`.
— 物理意义: 质量比 predict 消费关系方向, 大→小概率高
— 仅作为 soft bias, α_mte 可学习 (init small)
— **实现位置**: `MultiChannelPosteriorEncoder` 的 coupling attention

### F2. Trophic-layer-informed depth (新)
GNN 第 k 层 aggregation 只允许 trophic level < k 的物种 → k 的物种.
— 违反时 soft penalty
— **落地**: Beninca 可用 Bacteria(0) → Phyto(1) → Zoo(2) → top(3) 粗分层

### F3. Phylogeny-informed species embedding 初始化
species_emb[i, :] 初始化为 taxon distance encoding (one-hot taxon + mass + trait).
— Cyclopoida/Calanoida 共享 sub-space 而非完全正交.

### F4. EMA-teacher + h sign-invariance 自蒸馏
h 有 ±1 对称. EMA 教师的 h 与学生 h 的 |corr| 作为监督 (不约束方向).
— 我们已有 G_anchor_first 破对称, 自蒸馏进一步稳定.

### F5. Lyapunov-informed prediction horizon
Beninca Lyapunov ≈ 0.05/day, 1/λ ≈ 20 days ≈ 5 step (dt=4day).
— rollout_K = 3 目前是经验选择. 5 步更接近 Lyapunov horizon.
— Sanderse 2024 closure-model 一致性理论给支撑.

### F6. 生态 prior as PE (创新 framing)
把 (log M, trophic level, phylogeny) 作为 node PE, 替代/补充 Laplacian PE.
— 这是"**生物-physics informed positional encoding**", 可 claim 为新 contribution.

---

## 五、优先级 (在大 ablation 里测)

**Tier 1 (先测, 0 风险, 实现 < 30 min)**:
- SWA
- CosineAnnealingWarmRestarts
- DropPath in transformer
- GraphNorm
- Edge-drop augmentation

**Tier 2 (中实现, 较确定正收益)**:
- EMA-teacher self-distillation
- IWAE + DReG
- Laplacian PE
- Lookahead optimizer

**Tier 3 (高创新, 高风险)**:
- Latent-ODE h (Stage 3 候选)
- MTE-informed edge attention bias (F1)
- SAM

**Tier 4 (论文卖点, 不一定 +Pearson)**:
- Trophic-layer-informed depth (F2)
- σ(t) regime-shift detector
- HSIC CF margin (Anakok 2025)

---

## 六、ablation 扩充版 (从 A0-A7 扩到 A0-A15)

| ID | 配置 | 测什么 |
|---|---|---|
| A0 | Phase 2 baseline | 参考 |
| A1 | +RMSE+aug (S1b) | 已测 |
| A2 | +MTE shape (S1c) | 生态 prior |
| A3 | +Klausmeier hints (S2) | 生态公式 |
| A4 | +EMA+Snapshot | 经典 NN |
| A5 | +Hier h | 架构 |
| A6 | Ecology combo (A1+A2+A3) | 生态堆叠 |
| A7 | Classical NN (A4+A5 +SWA) | NN 堆叠 |
| **A8** | **+DropPath+GraphNorm** | GNN 正则 |
| **A9** | **+EMA-teacher 自蒸馏** | 一致性 |
| **A10** | **+CosineRestart+Lookahead** | 优化器 |
| **A11** | **+Edge-drop aug (生态加权)** | F1 融合 |
| **A12** | **+Laplacian PE** | GNN PE |
| **A13** | **+IWAE K=5 bound** | Posterior |
| **A14** | **All Tier1 (A8+A9+A10+A11+A12)** | ML pure-win 堆叠 |
| **A15** | **All (A6 + A14)** | 完整最终 |

5 seeds × 9 species × 16 configs = 720 runs, ~5h. 可减到 3 seeds × 5 重点 species 作快测, 确定 top-5 configs 再 5 seeds × 9 species.

---

## 七、期望最终 Pearson

保守叠加 (仅取显著有效):
```
Phase 2        +0.114
+RMSE+aug      +0.018 → 0.132 (已验)
+MTE shape     +0.005~0.015 (若 Stage 1c 成)
+Klausmeier    +0.010~0.020
+EMA+Snapshot  +0.005~0.015
+Hier h        +0.005~0.015
+ML pure-win   +0.010~0.025 (累计 Tier 1)
─────────────
预期 ceiling:  0.17 ~ 0.22
```
Portal: 若 Latent-ODE h 有效, 类似 +0.02 ~ 0.05.

Q2 门槛: **Beninca 稳定 > 0.20, 至少 2 个 species > 0.30**.
