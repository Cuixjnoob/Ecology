# Brainstorm Round 2 — 2026-04-15 夜

> 在 Stage 1c 失败 + Beninca 深读之后的重新 brainstorm. 更激进, 更生态-ML 融合.
> 原则: 1) 无监督红线; 2) 生态-AI 融合 (不偏一头); 3) 每个想法标注 "只好不坏" vs "可能坏".

---

## 一、重新审视 "我们到底在解什么问题"

之前的 framing: "n→1 hidden species recovery". 太窄了. 换个角度:

### 新 framing 候选
1. **"Residual hidden-state closure" (Sanderse 2024)** — 当前 frame, 强调封闭性
2. **"Counterfactual identifiability of latent dynamics"** — 强调 CF loss 作 identifiability 约束
3. **"Graph-structured system identification with missing nodes"** — 工程侧 framing
4. **"Information-bottleneck recovery of ecological attractor dimensions"** — 混沌理论视角 (Beninca Lyapunov 信息)

**推荐**: 主 frame 用 1, sub-claim 用 2. 这样 methodology 和 identifiability 都覆盖.

---

## 二、突破 0.13 平台的 10 个创新方向

### Tier 0 — 极高收益, 低风险 (可立即试)

#### 【1】 Rolling-window 局部训练 + test-time ensemble
**想法**: Beninca 混沌 Lyapunov ≈ 0.05/day. 未来 20 天后动力学漂移. 目前用 T=493 一口气训练, 后期段动力学可能被前期主导.  
**做法**: 用 rolling window (W=150, stride=50) 训练多个 dynamics head, 测试时用最近 window 的 h. Temporal ensemble 替代全局单模型.  
**生态-AI 融合点**: Beninca 混沌 horizon 决定 window 大小 (W ≈ 3×1/λ = 60 steps, 取 150 留 buffer).  
**好坏**: 只好, 代价 train 时间 ×3. 预期 +0.02~0.05 (后期段改善).

#### 【2】 Residual-coupling coupling matrix sparsity 正则
**想法**: GNN top_k=4/3 已经强制 sparse. 但我们没直接正则化 learned coupling matrix 本身的 spectral 性质.  
**做法**: 在 GNN messages 汇总成 effective coupling W_eff(t) 后, 加 λ·||W_eff||_* (nuclear norm 或 Frobenius).  
**生态-AI 融合点**: 生态系统 food-web 天然低秩 (species roles 有限). 对应 May 1972 "food web complexity" 理论.  
**好坏**: 只好 (假设 λ 足够小). 预期 +0.01~0.03.

#### 【3】 Multi-task hidden: 同时训练 K hidden 轮换
**想法**: 目前单 hidden 训练. 但 9 次训练的 dynamics 参数共享会不会更好?  
**做法**: 每个 batch 随机 mask 一个 species (像 MAE). 共享的 f_visible + 物种 embedding. Eval 时在每个具体 hidden 上评 Pearson.  
**生态-AI 融合点**: masked autoencoder 思想 + 物种对等性 (每个 species 都可能缺).  
**好坏**: 可能更好 (更多数据, 更鲁棒) 或可能更差 (averaging 丢失物种特异). 预期 ±0.02, 需实验.  
**风险**: 若 hidden 身份被编码到 species embedding 中就有 leakage 风险 — 需特别设计.

#### 【4】 σ(t) 驱动的 adaptive loss weighting
**想法**: encoder 的 posterior σ(t) 本质是 model 自评的不确定性. 高 σ(t) 的时刻可能是 regime shift (Trifonova / McClintock), recon loss 应该降权.  
**做法**: `loss_recon(t) *= 1/(1 + α·σ(t)²)`, α 可学习.  
**生态-AI 融合点**: uncertainty-aware ecology (与 McClintock 2020 HMM state 检测对齐).  
**好坏**: 只好 (自 adjust, 无额外 prior). 预期 +0.01~0.02, 可能稳定训练.

### Tier 1 — 高收益, 中风险 (值得试)

#### 【5】 Rank-consistency loss between visible channels
**想法**: 生态系统有 "quasi-conservation laws" (总生物量 / 总 N). 虽然 Beninca 是开放系统 (有外加 nutrient), 但有近似守恒.  
**做法**: 软约束 `sum(base_visible)` 变化小. 或 rank-preserving constraint on biomass.  
**生态-AI 融合点**: 物质守恒 + Redfield stoichiometry 混合.  
**好坏**: 可能坏 (若破坏 data-driven 信号). 需扫 λ.

#### 【6】 食物网 sign prior 的学习版: 不固定 sign, 学 sign confidence
**想法**: Stage 1d 硬性给 sign (通用 biology). 但不同 pair 的 confidence 可能不同 (phyto↔zoo 很强, large↔small zoo 较弱).  
**做法**: 每个 pair 学一个 sign_confidence 标量, 乘到 ReLU loss 上. 初始化全 1, 训练中自调整.  
**生态-AI 融合点**: "soft trust" 生态先验 + meta-learning.  
**好坏**: 可能更稳健 (避免硬约束拉偏). 预期 +0.01.

#### 【7】 Latent-ODE h 替换 VAE-style mu/sigma
**想法**: Rubanova 2019. h_t 由 neural ODE 演化而非独立高斯.  
**做法**: 保留 encoder, 但输出 h_0; 然后 h_t = ODESolve(f_ode, h_0, t=[0..T]).  
**生态-AI 融合点**: Lyapunov exponent 直接约束 ODE Jacobian 的 spectral radius (生态 prior).  
**好坏**: 可能 +0.02~0.05 (Portal 更大). 代码改动中等. 需警惕混沌系统 stiff ODE.

#### 【8】 HSIC-based counterfactual (Anakok 2025)
**想法**: 替代 margin-based null/shuffle.  
**做法**: `loss = -HSIC(h_samples, residual)` 最大化 h 和 residual 的统计依赖性.  
**好坏**: 只好 (理论更规范). 预期 +0.005~0.015, 主要是 paper story 升级.

### Tier 2 — 高风险, 高创新

#### 【9】 Attractor-level CCM validation loss
**想法**: Sugihara 2012 CCM 能检测 2 个时序是否在同一 attractor. 用它作 **无监督 self-supervision**: reconstructed hidden h(t) 应该和 visible 在同一 attractor.  
**做法**: 每 epoch 随机选 1 个 visible, 用 CCM 算 predictive skill(h, visible). 作为额外 loss 最大化.  
**生态-AI 融合点**: **这是真正的生态-AI 融合** — attractor-theoretic + neural.  
**好坏**: 可能更好 (attractor consistency 是本质约束). 实现复杂 (CCM 不可微, 需 differentiable proxy).

#### 【10】 Mutual information bottleneck on h
**想法**: 经典 IB: h 要尽可能包含 visible → target 的信息, 但信息量要小 (avoid memorization).  
**做法**: loss += β·I(h, x_future) - α·I(h, x_past). 用 MINE 或 InfoNCE 估计.  
**好坏**: 理论强, 实现有争议 (MINE 估计不稳). 预期 +0.01~0.03, 主要方法论价值.

---

## 三、生态-AI 融合的 5 个最高创新点 (for paper)

### Innovation 1: "Lyapunov-bounded GNN message passing"
每层 GNN 的 update step 显式约束 || Δx_i || ≤ λ_lyap · dt · ||x||, λ_lyap 从 Beninca 2008 取 0.05/day.  
**生态 hook**: 混沌系统 Lyapunov 作 architectural prior.  
**ML hook**: 类似 Lipschitz-constrained network (GAN 稳定性).  
**Paper hook**: 新 architectural design principle.

### Innovation 2: "Unsupervised HSIC-margin counterfactual"
替换 MSE margin 为 HSIC-based independence test. 严格证明 h 和 residual dependence.  
**生态 hook**: 因果推断 ("h 确实驱动 visible 变化").  
**ML hook**: Anakok 2025 HSIC in GNN.  
**Paper hook**: methodology 重大升级, 引用 Gretton 2005 + Anakok 2025.

### Innovation 3: "Taxon-stratified prior with uncertainty quantification"
不是 per-species MTE, 而是 per-functional-group (phyto/zoo/bacteria/nutrient) 的 intercept prior, 带 Kremer 2017 的 95% CI 作 loss 权重.  
**生态 hook**: 层次生态学 (function groups > individual species).  
**ML hook**: Hierarchical Bayesian prior.  
**Paper hook**: 修正 Stage 1c 失败的正解, 可写 "Lessons from negative MTE result".

### Innovation 4: "Frequency-domain predator-prey resonance prior"
Benincà 2011 发现 plankton 与 temperature 共振, τ=T/2π. 在 h_t 的频谱上加 soft prior: 应在生态共振频率有 peak.  
**生态 hook**: Frequency-domain ecology.  
**ML hook**: Fourier-domain regularization.  
**Paper hook**: 独特, 此前无人做过.

### Innovation 5: "Dynamic identifiability via counterfactual necessity"
证明 counterfactual null/shuffle loss 是 **h→x 动力学的 identifiability 条件**. 给 formal statement.  
**生态 hook**: identifiability 是生态模型核心问题 (Cobelli 2001).  
**ML hook**: VAE identifiability (Khemakhem 2020).  
**Paper hook**: 升级 "empirical loss" 为 "identifiability theorem".

---

## 四、对当前组件的去留判断

| 组件 | 判定 | 理由 |
|---|---|---|
| f_visible GNN | **必留** | 核心 |
| G field | **必留** | 核心 |
| G_anchor_first | **必留** | 破对称, 已验证 |
| L1 rollout | **必留** | 多步自洽, Sanderse 理论支撑 |
| RMSE log loss | **必留** | 已验证 +0.018 |
| Input dropout aug | **必留** | 已验证 |
| MoG K>1 | **可弃** | Beninca K=1 够 |
| L3 low-freq | **已弃** | 证伪 |
| MTE on G (Stage 1) | **已弃** | 位置错 |
| MTE shape on f_visible (Stage 1c) | **已弃** | 数值 prior 不适用 |
| Food-web sign (Stage 1d) | **待定** | 运行中 |
| Klausmeier N↔phyto (Stage 2) | **合并到 Stage 1d** | 子集 |
| EMA weights | **待测** | 可能只好 |
| Snapshot ensemble | **待测** | 可能只好 |
| Hierarchical h | **待测** | 可能对 Filam 有用 |
| Formula hints (LV/Holling) | **必留** | Phase 2 验证 |

---

## 五、我们的"不可替代创新点" (被抄袭的风险最低)

1. **h·G 残差分解 + counterfactual necessity** — 和 DSEM / EDM / JSDM 都不同
2. **Unsupervised hidden species recovery with zero hidden supervision** — 红线明确
3. **Ecology-informed soft priors on GNN dynamics** — Klausmeier-style sign, Lyapunov-bounded updates (若加)
4. **Negative finding on quantitative MTE prior** — 少有人明确 publish 这种

---

## 六、建议下一步 (待用户决策)

### Plan A: 先验证 Stage 1d, 若成功再扩
- 等 Stage 1d 完成 (~30min)
- 若 Overall > S1b 0.132 → 保留, 测大 ablation
- 若 ≈ S1b → 可作为"合规但无收益"的 negative finding
- 若 < S1b → 撤

### Plan B: 同时跑 Innovation 1 (Lyapunov-bounded)
- 简单实现: 在 f_visible 输出 clamp 到 [-0.05*4, 0.05*4]/day
- 可能瞬间稳定训练

### Plan C: 实现 Innovation 4 (σ(t) adaptive weighting)
- 只改 3 行 loss 代码
- 预期 +0.01~0.02

### Plan D: 高风险高回报 - 实现 Latent ODE h
- 代码改动较大 (~100 行)
- 需要 torchdiffeq 依赖
- 预期 +0.02~0.05

**我的建议**: 等 S1d 结果 → 根据结果选 Plan A/C (低风险快速迭代) vs Plan B/D (更大创新).
