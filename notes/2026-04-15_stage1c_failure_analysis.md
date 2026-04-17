# Stage 1c 失败深度诊断 — MTE Shape Prior 为什么崩？

日期: 2026-04-15
状态: Stage 1c (MTE shape prior) overall Pearson 从 Stage 1b 的 0.132 跌到 0.086

---

## 1. Beninca 数据集深读 — 我们之前不知道的关键信息

### 1.1 Beninca et al. 2008 Nature "Chaos in a long-term experiment with a plankton community"

**身份确认**: `Chaos_in_a_long_term_experiment_with_a_p.pdf` 即 Beninca 2008 Nature 原文
(Vol 451, 14 Feb 2008, pp. 822-825, doi:10.1038/nature06512)

**关键数值** (Fig. 3, p. 824):
直接法 (time-delay embedding, dim=6) 计算的 per-species Lyapunov exponents:
- Picophytoplankton: λ = 0.057 /day
- Nanophytoplankton: λ = 0.059 /day
- Calanoids: λ = 0.054 /day
- Phosphorus: λ = 0.063 /day
- Nitrogen: λ = 0.059 /day
- Rotifers: λ = 0.066 /day
- Ostracods: λ = 0.053 /day
- Harpacticoids: λ = 0.051 /day
- Bacteria: λ = 0.056 /day

全部均值: **λ ≈ 0.057 /day, s.d. 0.005, n=9**
间接法 (Jacobian from neural net deterministic skeleton): λ ≈ 0.04 /day, 95% CI 下界 0.03 /day

**对我们的重要启示** (这是我们原来错估的):
1. **所有 9 个物种的 Lyapunov 几乎一致** (s.d. 仅 0.005), 说明它们共享同一个 chaotic attractor。这意味着 per-species intrinsic rate 的 ordering **不由内禀代谢决定**, 而由 **attractor 上的位置** 决定。MTE prior 尝试给每物种定一个 ordering (Bacteria 最快, Ostracods 最慢), 这与 "same attractor, similar divergence" 的经验事实 **直接冲突**。
2. 时间序列原始采样: 2 次/周 × 2319 天 = 690 点/物种 (我们用的 dt=4day, T=658 步, 基本一致)。
3. Filamentous diatoms, protozoa, cyclopoid copepods 在 Beninca 原论文的 Lyapunov 和 correlation 分析中 **都被排除**, 原因是 "time series contained too many zeros" (p. 824 Fig 3 caption, p. 823 Table 1 caption)。**这是 Filam_diatoms 在我们实验中极难预测 (Phase 2 仅 0.056, Stage 1c 0.021) 的直接原因** — 它本身就是稀疏事件型动力学, 不是连续代谢驱动的。
4. 预处理细节 (p. 825): 原始数据经过 **fourth-root power transformation** + 300-day Gaussian-kernel detrending + normalization。我们如果直接用原始 log 做 target log_r, **和 Beninca 的 stationarity 假设 不一致**。

**per-species 生态学解释** (p. 822-823, Fig 1a 食物网, Table 1 相关性):

- **Bacteria**: 微生物环 (microbial loop) 的一员, 分解 detritus, 被 ostracods & harpacticoids 取食。Table 1: bacteria 与 ostracods 显著负相关 (-0.24***), 与 phosphorus 正相关 (0.19***), 与 rotifers 正相关 (0.30***)。Bacteria 动力学 **由 detritus 供给 + 捕食压力** 驱动, 不是温度-size 驱动的生长。MTE 给它 +0.6 的 target log_r **完全忽略了这一点** — Bacteria 在 closed mesocosm 里不是"快速生长者", 而是"被捕食者限制的周转者"。这解释了为什么 Stage 1c 把 Bacteria 从 0.197 打到 0.052 (-0.145 大崩)。

- **Calanoids (Eurytemora affinis)**: Table 1 显示 calanoids 与 picophytoplankton 正相关 (0.26***), 与 nanophytoplankton 负相关 (-0.14*) — Beninca 解释为 "indirect mutualism: the enemy of my enemy is my friend" (p. 822)。**Calanoids 动力学强烈被 food-web indirect effects 驱动**, 不是内禀代谢。MTE 给 calanoids -0.204 的 target (慢), 但它在 mesocosm 里其实被 picophyto 通过 cascade 推动, 表现出与 "MTE慢" 对立的快速耦合。这正是 Stage 1c 把 calanoids 从 0.173 崩到 0.079 (-0.094) 的原因。

- **Filam_diatoms (Melosira moniliformis)**: 大型丝状硅藻, 极端间歇性出现 (Fig 1d, 偶发爆发)。Beninca 说 "peaks of pico/nano/filam alternated with little or no overlap" (p. 822) — 这是 **竞争排斥性的时间分离**, 不是 MTE 代谢 ordering。

- **Rotifers (Brachionus plicatilis)**: 30-day 周期为主, 与 bacteria 正相关 (0.30***)。Rotifers 在 Stage 1c 反而 **没崩** (+0.007) — 可能因为 target log_r = +0.036 几乎为 0, prior 约束弱, 不干扰。

- **Harpacticoids (Halectinosoma curticorne)**: benthic 的 copepod, 居然在 Stage 1c 变好 (+0.036)! 原因: target = -0.175 (慢), 而它本身就是 benthic detritivore, 缓慢周转, prior 和 data 方向一致。这是**唯一**受益的物种, 说明 shape prior 本身不是完全错, 只是对大多数物种错。

### 1.2 Beninca et al. 2011 American Naturalist "Resonance of Plankton Communities with Temperature Fluctuations"

**身份确认**: `661902.pdf` = Beninca, Dakos, Van Nes, Huisman, Scheffer (2011) Am Nat 178(4): E85-E95, doi:10.1086/661902

**核心发现**: Plankton 对温度红噪声 (red noise) 的 resonance 效应。

**关键公式** (p. E90, eq 10):
```
τ_max = T / (2π)
```
其中 T 是 predator-prey 内禀周期, τ_max 是最共振的红噪声特征时标。

**对 Bacteria / Calanoids 的直接影响** (Fig 6, Table 1):
- Beninca 2011 Fig 6 明确给出: 不同 zooplankton 的 τ_max 与 max growth rate 的关系。
- 浮游生物 mesocosm 的"特征振荡"周期不是 MTE 代谢给定的, 而是 **predator-prey 共振** 给定的。
- 按 Eq 10: Beninca 食物链 30-day 周期对应 τ ≈ 30/(2π) ≈ 4.8 day (接近我们的 dt=4day!)。**这意味着 f_visible 学到的 "base rate" 本质上是 resonance-driven frequency, 不是 metabolic rate**。强推 MTE shape 于事无补。

**对我们 Stage 1c 诊断的价值**:
Beninca 2011 p. E93 明确警告: "In natural systems, temperature fluctuations will have different effects on different components of the population dynamics ... it is clearly a simplification of the complex ways in which organisms may respond." — 把内禀 rate 约束到 MTE ordering 是 oversimplification。

### 1.3 Rogers, Munch, Matsuzaki, Symons 2023 Ecology Letters "Intermittent instability is widespread in plankton communities"

**身份确认**: `noaa_49245_DS1.pdf` = Rogers et al. 2023 Ecol Lett 26: 470-481, doi:10.1111/ele.14168

**核心发现** (这是我们之前不知道的重点!):

1. **Local instability is seasonal and intermittent**, 不是稳态。Species level 52% of time series are chaotic; 58% of non-chaotic series show intermittent local instability (Fig 1, p. 474)。

2. **Aggregation stabilizes** (p. 475, Fig 2, Fig 5):
   - Species level: 50% of local eigenvalues positive
   - Functional group: 39%
   - Trophic level: 21%
   Variability (CV) 从 species 到 trophic 下降 3x, R² 从 species 到 trophic 上升 ~2x (Fig 5)。

3. **Local instability peaks in spring, coincides with max growth rate** (p. 476, Fig S3): "month with greatest local instability tended to coincide with, or just follow, the month with maximum growth rate" — 说明 **内禀 rate 不是常数, 是 time-varying**。

4. **VER (variance expansion ratio) predicts step-ahead forecast error** (Fig 4): 高 VER → 高 residual。且 **species 层比 functional group 层关系更强** — 单物种动力学受 local eigenvalue 支配更强。

**对 CVHI-Residual 的关键启示**:
- 我们把 target log_r_i 做成 **常数 vector**, 但论文明确告诉我们 "instability varies seasonally and is associated with periods of high growth" — **rate 是 time-varying 的, 常数 prior 从根上错了**。
- 论文建议 "aggregated functional group is more predictable" — 如果我们要 MTE prior, **应该在 functional group level 做, 不在 species level 做**。

---

## 2. MTE 再审视 — 小样本 / 多物种 / ML prior 场景的 uncertainty

### 2.1 MTE scaling exponent 在 N=9 下能否作为定量约束?  驳回。

**Clarke 2025 Ecological Monographs** (这篇综述是 2025 年最权威的 MTE 现状评估) 的明确数据:

- Table 3 (p. 6): 不同门类的 scaling exponent 实测范围:
  - Prokaryotes: b = **1.28** (isometric 以上!)
  - Protists: b = 1.00
  - Diploblastic (Porifera, Cnidaria): b ≈ 0.92-0.97
  - Polychaetes: b = **0.61**
  - Echinoderms: b = 0.74
  - Decapod crustaceans: b = **0.69**
  - Insects: b = 0.82
  - Cephalopods: b = 0.81
  - Planktonic crustacea (copepods, euphausiids, decapods): b ≈ 0.70-0.75
  - Bivalves: b = 0.76

  **实测范围 [0.61, 1.28] 远宽于我们 Stage 1c 假定的 [0.60, 0.95]**。Bacteria 的 b=0.60 在 Clarke 2025 是**不支持的** — prokaryotes 实际是 **isometric 以上 (b=1.28)** (Table 3, 源自 DeLong et al. 2010)。我们给 Bacteria b=0.60 是完全错误的 — 原始文献是说 "unicells are isometric to super-isometric"。**这是 Bacteria Pearson 大崩 -0.145 的直接 prior misspecification 原因**。

- Clarke 2025 p. 17 明确警告: "most studies using the MTE central equation have been concerned solely with the consequences of the scaling central tendency, almost always assuming the canonical values of 0.75 for the scaling exponent ... This approach bypasses the nuances and details."

**结论 (驳回猜测 #3)**: taxon-specific b 对 Baltic mesocosm 不只是"可能不准确", 而是 **Bacteria 的 b 方向都反了**。小样本 N=9 下, MTE 作为 quantitative rate ordering prior 是 **不可行的** — 实测 b 区间过宽, per-species uncertainty 巨大。

### 2.2 Kremer 2017 phytoplankton -0.054 scaling 的 uncertainty

**直接从 Kremer et al. 2017 Limnol Oceanogr 62: 1658-1670 读取**:

- p. 1663, Table 2 + Supporting Table S.4: phytoplankton mass scaling α = **-0.054**, p<0.005, **95% CI = [-0.089, -0.018]**
- Width of 95% CI = 0.071, 相对于中心值 0.054 是 **±130% 的不确定性**
- Kremer 明确指出 (p. 1666): "the strength of this effect was **much weaker than predicted** (α = -0.25)" — MTE 理论值 -0.25 不在 95% CI 内!
- 温度 activation energy E = **0.30 eV**, 95% CI [0.233, 0.368] (vs MTE 光合预测 0.32 eV, vs 先前 Eppley 估值 0.41 eV, Fig 4)
- **四大功能类群 intercept 差异显著** (Table 2): Cyanobacteria -1.52, Diatoms 正偏 +0.70, Greens +0.82, Dinoflagellates 0.05. 即使 size + temperature 校正后, functional-group intercept difference 仍达 2 个数量级。

**对我们的直接警告**:
- 我们 Stage 1c 给 Nanophyto b=0.95, Picophyto b=0.95, Filam b=0.95 — 但 Kremer 说 phytoplankton 整体 **α = -0.054** (mass scaling), 换算回 **b ≈ 1 - 0.054 ≈ 0.95 这个数值恰好巧合但意义不同**。我们把它用到 volume ordering 上, 假定 Bacteria (1e-6 μg) → Picophyto → Nanophyto → Filam 线性 +0.6, +0.25, +0.15, +0.10, 但 Kremer 明确说 **"functional group intercepts differ far more than size effect"** — 也就是说 **intercept > slope**。我们只给了 slope prior, 忽略了更大的 intercept 差, 所以 prior 必然和数据冲突。

### 2.3 Clarke 2025 对 B₀ 不可识别的 precise statement

**Clarke 2025 p. 3**: 
> "The pre-exponential factor B₀ sets the level of metabolism; it varies, for example, between endotherms and ectotherms, between different animal groups, and with ecology. Unlike the scaling exponent, **the value of B₀ cannot be predicted from the WBE model; it is a free parameter that has to be determined empirically by fitting the model to data**. In other words, **we cannot predict the metabolic rate of any organism from first principles**; biological systems are just too complex."

**这确认了我们的猜测 #5**: B₀ 确实不可从理论识别。更重要的是, Clarke 2025 p. 18 还补充 (在讨论 deep-sea metabolism 时):
> "The key problem here is that **the MTE cannot predict the metabolic rate of an organism from its body mass and temperature alone**; the pre-exponential constant that defines the intensity of metabolic rate has to be determined by fitting the WBE model to empirical data."

**但** Clarke 也说了可接受的 prior 形式 (p. 3 + p. 12):
> "MTE central equation ... captures the consequences of the scaling central tendency"
> "Size and functional group membership are ... critical" (这是 Kremer 2017 的观点)

**可接受的 prior 形式**:
1. **Group-difference / intercept difference** (functional group level): 在 Kremer 2017 实测下, intercept 差异比 slope 更大, 所以 group indicator 是强先验。
2. **Sign / bound** on scaling exponent in a group: 如 "phytoplankton within-group scaling should be negative" — 这是 bounded qualitative prior, 不是 quantitative pointwise。
3. **Rank / ordering** within closely-related taxa (e.g., pico < nano < filam within phytoplankton): 但 **across major functional groups 不可排序** (bacteria 和 phyto 本身 scaling mechanism 不同)。

**不可接受的 prior 形式** (就是我们 Stage 1c 做的):
- **Point-mass quantitative shape across 9 taxonomically disparate species**: 把 prokaryote + eukaryote autotroph + metazoan heterotroph 混在一起, 强制线性 log_r ordering — 这在 Clarke 2025 Table 3 明确显示的门类 scaling 不一致面前站不住。

### 2.4 是否有论文讨论 MTE 作 ML prior?

直接检索的 4 篇关键 MTE 文献 (Brown JH MA, glazier2005a, Kremer 2017, Clarke 2025) 中, **没有一篇** 把 MTE 用作机器学习模型的 prior (physics-informed or soft constraint)。最接近的是:

- **Clarke 2025 p. 20** 讨论 ecosystem models "sophisticated ecosystem models, but **key to meaningful predictions will be getting the parameterization correct, and also allowing for the variability evident in nature**". 
- **Kremer 2017 p. 1667**: "it is important to know which relationships apply broadly ... and which differ by group. ... rigorous ecosystem models ... must also employ functional groups that capture the most important differences among species."

**这两句话合起来就是我们 Stage 1c 违反的原则**: "allow for variability" + "parameterize at functional group level, not species level"。

---

## 3. Stage 1c 失败综合诊断 — 验证/驳回 5 个猜测

| 猜测 | 结论 | 论据 |
|---|---|---|
| (1) Bacteria log_r=+0.6 outlier dominates 9-point Pearson | **部分成立** | Bacteria 是 outlier (target +0.6 vs 其他都在 [-0.43, +0.25]), 但真正的问题不只是"数值大", 而是**方向错** — Clarke 2025 Table 3 显示 prokaryotes b≈1.28 (isometric+), MTE WBE 模型 **根本不适用于 prokaryotes** (p. 7: "Unicellular organisms ... fall outside the scope of the WBE model"). 我们给 Bacteria +0.6 是把 body mass 外推的 artefact 当 prior. |
| (2) 9-sample global Pearson 易被单点影响 | **成立但非主因** | 的确 n=9 的 Pearson 统计极不稳定 (任一点残差可主导), 但即使用 Spearman rank 也一样错 (因为 ordering 本身错). 真正主因是 target ordering 与 data ordering 冲突. |
| (3) Taxon-specific b 对 Baltic mesocosm 不准确 | **强烈成立** | Clarke 2025 Table 3: prokaryote b=1.28 (我们给 0.60), 跨 phyla b 从 0.61 到 1.28. Kremer 2017 95% CI 宽度 ±130% 中心值. n=9 下 MTE 作定量约束**不可行**. |
| (4) Target log_r 是常数, 与 data-driven rate 冲突 | **最强烈成立, 这是核心机制** | Rogers 2023 Ecol Lett 明证 "local instability is seasonal, peaks in spring, coincides with max growth rate" — **rate 是 state-dependent + time-varying 的**. Beninca 2008 Lyapunov 对 9 个物种都近乎相等 (0.051-0.066) — **没有稳定的 per-species ordering**. 我们的 target 向量本质上是 ill-posed. |
| (5) B₀ 不可识别 (Clarke 2025 警告) | **完全成立, 且比我们想的更严重** | Clarke 2025 p. 3 + p. 18 两次明示. 并且 Kremer 2017 显示 **group-intercept differences > slope effects** — 我们只约束 slope, 不约束 intercept, 所以 prior 必然与数据驱动的 intercept 冲突. |

**增加 2 条我们没猜到的失败机制**:

(6) **Stage 1c prior 尝试 pin down per-species ordering, 但 Beninca 系统本质是 "shared chaotic attractor"** (9 个物种 Lyapunov 几乎一致). 在 shared attractor 上, per-species rate 是 **state-dependent Jacobian eigenvalue**, 不是常数 intrinsic rate. MTE 给出的是 evolutionary time-averaged rate, 和 mesocosm 观测的 ecological-time rate 不是一回事.

(7) **我们把 prokaryote (Bacteria) + photoautotroph + metazoan heterotroph 混在一个 9-point Pearson 里**. Kremer 2017 明确指出 即使 within phytoplankton, functional-group intercepts 差异达 2 个数量级 (Cyano -1.52, Greens +0.82). 跨门类 line up 本身就是 category error.

---

## 4. Stage 1d 改进方案 — 3 个候选

### 方案 A: 抛弃 MTE, 转向 Klausmeier 化学计量软 sign prior (优先推荐)

**动机**: Beninca mesocosm 的真实内生结构是 **food web 营养盐循环 + predator-prey resonance**, 不是代谢 allometry. Klausmeier stoichiometry (N:P Redfield, phyto C:N:P) 给出的是 **sign / bound prior on pairwise coupling**, 不是 pointwise rate. 这与 "shape, sign, bound" 范式 (Clarke 2025 允许的 prior 形式) 一致.

**Loss 公式**:
```
L_stoich = λ_s · Σ_{i,j ∈ E_known} max(0, -sign(K_ij^prior) · W_ij)^2
```
其中 W_ij 是 GNN edge weight (或 Jacobian-like interaction matrix), K_ij^prior 是 Klausmeier 给出的 **符号** (+1 prey→predator, -1 competition, 等). 只约束 **方向**, 不约束数值.

**实现**: 
- 代码改动: ~30 行 (在 G(x) 上加一个 off-diagonal sign loss, 用 food-web adjacency 作为 mask).
- 边集合 E_known 来自 Beninca 2008 Fig 1a (已经明确给出 trophic links).

**预期 Pearson**:
- Calanoids (+, 因为 food-web sign 会强化 picophyto→calanoid 的 indirect mutualism): +0.02 ~ +0.08
- Bacteria (+, sign prior 会约束 bacteria ↔ ostracods 负 sign, 与 Table 1 的 -0.24*** 一致): +0.05 ~ +0.15
- Harpacticoids (中性, 因为 Stage 1c 已经最好): ±0
- Filam_diatoms (中性或负, 因为 Filam 与其他物种相关性低): ±0

**Fallback**: 如果 sign prior 也崩, 降级到 **only constrain top-2 strongest links** (bacteria-ostracods 负, bacteria-rotifers 正, 都是 Beninca 2008 Table 1 的 *** 级相关), 即稀疏 sign prior.

---

### 方案 B: 保留 MTE 但改为 functional-group-level + intercept-only

**动机**: Kremer 2017 明证 intercept >> slope. Clarke 2025 allows group-difference prior.

**Loss 公式**:
```
group(i) ∈ {Bacteria, Phyto, Pelagic_zoop, Benthic_zoop}
L_group = λ_g · Σ_g Var_{i ∈ g}(log|base|_i) 
       + λ_o · max(0, mean_{g=Phyto} log|base| - mean_{g=Benthic} log|base|)^2
```
只做 **two group-level constraints**:
1. Within-group variance of base rate magnitude 小 (同 group rate 近似).
2. Phyto group mean rate > Benthic group mean rate (这是稳健的 sign-level 生态共识).
不给 per-species 数值.

**实现**: 代码改动 ~40 行.

**预期 Pearson** (对比 Stage 1b 0.132):
- Phyto 组 (Nano, Pico, Filam): 轻微回升 (+0.01 ~ +0.03) 因为 group constraint 较弱
- Bacteria: 不崩 (因为不再给 +0.6 的 outlier target)
- Calanoids/Ostracods: 中性
- Overall: 预期 0.125 ~ 0.140 (回到 Stage 1b 水平或略好)

**Fallback**: 如果 group 划分仍不准, 进一步降级为 "**只保留 Phyto vs Zoop** 的 2-group sign prior", 与 Redfield N:P 的 sink/source 区分对齐.

---

### 方案 C: 抛弃 static prior, 引入 state-dependent local eigenvalue constraint (Rogers 2023 风格)

**动机**: Rogers 2023 明确 rate 是 state-dependent, seasonal. 我们应该约束 **Jacobian 的 local eigenvalue 在合理范围**, 而不是约束 base rate magnitude.

**Loss 公式**:
```
J_i(x_t) = ∂f_visible_i / ∂x_i  (diagonal of local Jacobian)
L_local = λ_l · Σ_{t,i} max(0, |J_i(x_t)| - λ_lyap_max)^2
```
其中 λ_lyap_max = 0.07 /day (Beninca 2008 实测 Lyapunov upper bound) · dt = 0.07×4 = 0.28.

这只约束 **Jacobian eigenvalue 的上界**, 对应 chaotic attractor 的 predictability horizon (15-30 day), 不约束方向或数值.

**实现**: 
- 代码改动: ~60 行 (需要用 autograd 计算 per-timestep Jacobian diagonal, 然后 clip-ReLU).
- 注意: 计算成本上升 (per-batch Jacobian), 但 N=9 还能接受.

**预期 Pearson**:
- 全物种一致性约束 (+ Bacteria, + Calanoids 因为不再被乱 pin), 预期 overall +0.02 ~ +0.05 vs Stage 1b.
- 但可能 regularize 过度 → 需要 λ_l 在 [1e-4, 1e-2] 扫描.

**Fallback**: 如果 Jacobian 计算太慢, 用 finite-difference approximation 或只在每 epoch 采样 10% timestep 计算.

---

### 方案优先级建议

1. **方案 A (Klausmeier sign prior)** — 推荐首选, 因为 (a) 完全跳过 MTE 陷阱, (b) Beninca 2008 Fig 1a 给了明确 food-web prior, (c) Clarke 2025 & Kremer 2017 都支持 "sign/group level" prior, (d) Task 83 已经在 pending 列表.

2. **方案 B (Group-level intercept)** — 次选, 作为 MTE 的 "降维打击" 版本, 保留 MTE 框架但只用 group-level.

3. **方案 C (Local Jacobian bound)** — 探索性, 如果时间允许, 尤其适合 Q2 论文的 "novel state-dependent prior" 叙事.

**避免组合 A+B+C**, 因为会掩盖是哪个机制 work.

---

## 5. 反思 — MTE 本身不该作 prior?

**我的最终立场: MTE 在 Beninca Baltic mesocosm 的 CVHI-Residual 任务上, 不应作为 per-species quantitative prior.**

**论据** (按强到弱):

1. **[决定性]** Clarke 2025 p. 7 明确: WBE model **不适用于 unicells (prokaryotes + protists) 和 diploblastic invertebrates**. Beninca 9 物种里 Bacteria (prokaryote), Pico+Nano+Filam (protists/eukaryotic microalgae, scaling≈isometric per Kremer 2017 α=-0.054) 至少 4 个物种在 MTE 适用域外. **4/9 = 44% 的物种 MTE 不适用**, 这是致命缺陷.

2. **[决定性]** Beninca 2008 Fig 3 实测 **9 物种 Lyapunov 几乎一致** (λ = 0.051-0.066), 说明 mesocosm 的 "rate" 不是 per-species 内禀, 而是 **shared chaotic attractor 上的 state-dependent divergence**. MTE 给的是 evolutionary-scale cross-species rate, 和 ecological-scale 的 attractor divergence 是两回事.

3. **[强]** Rogers 2023 Ecol Lett 证明 plankton rate 是 **seasonal + intermittent + state-dependent**. 常数 target 在根本上不对.

4. **[强]** Kremer 2017 明证 **group-intercept 差异 >> size-slope 效应**. 我们 Stage 1c 只约束 slope 忽略 intercept, 注定失败.

5. **[中]** 对 Baltic mesocosm 的 per-species rate 无直接测量 — Beninca 2008, Beninca 2011, Rogers 2023 都 **没有给 per-species 绝对代谢 rate** — 所以即便想用 MTE, parameter 也无法锚定.

**可以保留 MTE 的场景**: 在更大尺度 (跨群落, 跨 biome, 跨温度梯度) 比较 **functional-group-averaged** rate 时, MTE 仍有用. 但在单个 mesocosm 9-species attractor 内部比较 per-species rate, MTE **既不 applicable 又不 identifiable**.

**应该转向的 prior 方向**:
1. **Food-web sign prior** (Klausmeier, Beninca 2008 Fig 1a): 稳健, 定性, 不可证伪.
2. **Stoichiometry ratio prior** (Redfield N:P, phyto C:N:P): 对 4 个营养盐 + phyto 给出强 mass-balance 约束.
3. **Predator-prey resonance period prior** (Beninca 2011 τ = T/2π): 约束 GNN learn 到的主频率 ≈ 30 day.
4. **Lyapunov eigenvalue bound prior** (Rogers 2023, Beninca 2008 λ≈0.05): 约束 Jacobian spectral radius.

这些都是 **shape / sign / bound** 级别的 prior, 与 Clarke 2025 p. 3 允许的 prior 范式一致, 都不需要 MTE 的 per-species 量化承诺.

---

## 关键文献直接引用 (便于论文写作)

- Beninca, E. et al. 2008. "Chaos in a long-term experiment with a plankton community." *Nature* 451: 822-825. [per-species Lyapunov, Table 1 correlations, food-web Fig 1a]
- Beninca, E., Dakos, V., Van Nes, E.H., Huisman, J., Scheffer, M. 2011. "Resonance of plankton communities with temperature fluctuations." *Am Nat* 178(4): E85-E95. [Eq. 10 τ = T/2π]
- Rogers, T.L., Munch, S.B., Matsuzaki, S.S., Symons, C.C. 2023. "Intermittent instability is widespread in plankton communities." *Ecol Lett* 26: 470-481. [seasonal local eigenvalue, Fig 2, Fig 5 aggregation effect]
- Clarke, A. 2025. "The contribution of metabolic theory to ecology." *Ecol Monogr* 95(3): e70030. [Table 3 per-taxon scaling exponents, p. 3 B₀ unidentifiability, p. 7 WBE model scope]
- Kremer, C.T., Thomas, M.K., Litchman, E. 2017. "Temperature- and size-scaling of phytoplankton population growth rates." *Limnol Oceanogr* 62: 1658-1670. [Table 2 α=-0.054, 95% CI [-0.089, -0.018], functional-group intercept differences]

---

## 绝对路径清单 (相关文件)

- `C:/Users/cuiyanfeng/Desktop/生态模拟/docs/生态论文/MTE+浮游动物/Beninca数据/Chaos_in_a_long_term_experiment_with_a_p.pdf` — Beninca 2008 Nature
- `C:/Users/cuiyanfeng/Desktop/生态模拟/docs/生态论文/MTE+浮游动物/Beninca数据/661902.pdf` — Beninca 2011 Am Nat
- `C:/Users/cuiyanfeng/Desktop/生态模拟/docs/生态论文/MTE+浮游动物/Beninca数据/noaa_49245_DS1.pdf` — Rogers 2023 Ecol Lett
- `C:/Users/cuiyanfeng/Desktop/生态模拟/docs/生态论文/MTE+浮游动物/Ecological Monographs - 2025 - Clarke - The contribution of metabolic theory to ecology.pdf` — Clarke 2025
- `C:/Users/cuiyanfeng/Desktop/生态模拟/docs/生态论文/MTE+浮游动物/Kremer-Temperaturesizescalingphytoplankton-2017.pdf` — Kremer 2017
