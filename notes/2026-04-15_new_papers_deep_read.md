# 新论文深读报告 — 2026-04-15

上下文：Stage 1b (RMSE+aug, no MTE) Pearson=0.132；Stage 1c corrected MTE 已启动。本次对 `docs/生态论文/` 的全部 PDF 进行分桶比对，识别出"前几轮未深读"的条目。

## 新论文清单

对照 2026-04-14 的 related-work-review（只覆盖 Ovaskainen/Tikhonov/Pichler/Trifonova×2）和 overnight_plan（只覆盖 Brown/Glazier/Kremer/Clarke/Klausmeier2008/Beninca），以下 PDF 此前未系统处理：

- `MTE+浮游动物/Phytoplankton_growth_and_stoichiometry_u.pdf` — **Klausmeier, Litchman, Levin 2004 L&O** (Droop + Liebig stoichiometry 原始论文)
- `MTE+浮游动物/paradox_of_the_plankton.pdf` — **Hutchinson 1961** (浮游生物多样性悖论)
- `MTE+浮游动物/400207.pdf` — **Behrenfeld & Bisson 2024 Annu. Rev. Mar. Sci.** (中性理论 + 浮游多样性)
- `MTE+浮游动物/New Phytologist 2010 Price` — **Price et al. 2010** (MTE 在植物的前景与挑战)
- `生态结构等/Methods Ecol Evol 2024 Thorson` — **Thorson et al. 2024 DSEM** (Dynamic Structural Equation Models)
- `生态结构等/rsos.170251.pdf` — **Miele & Matias 2017** (动态生态网络隐藏结构恢复，SBM)
- `生态结构等/noaa_47837_DS1.pdf` — **Munch et al. 2022 MEE** (Recent developments in EDM)
- `生态结构等/Ecological Research 2017 Chang` — **Chang, Ushio, Hsieh 2017** (EDM for beginners)
- `生态结构等/Ecology Letters 2020 McClintock` — **McClintock et al. 2020** (HMM 生态状态动力学)
- `生态结构等/2002.02001v3.pdf` — GNN/图论 (arXiv 2020)
- `GNN/2503.15107v3.pdf` — **Anakok et al. 2025** (GNN interpretability for ecological networks, BVGAE)
- `GNN/2307.03759v3.pdf` — **Jin et al. 2023** (GNN for time series 综述)
- `GNN/1901.00596v4.pdf` — **Wu et al. 2019** (Comprehensive GNN survey)
- `GNN/1-s2.0-S2666651024000032-main.pdf` — Elsevier erratum（无内容价值，**跳过**）
- `残差/2403.02913v2.pdf` — **Sanderse et al. 2024** (SciML for closure models, multiscale)
- `VAE/1907.03907v1.pdf` — **Rubanova et al. 2019** (Latent ODE, 不规则时序)
- `VAE/1605.06432v3.pdf` — **Karl et al. 2017** (Deep Variational Bayes Filters, DVBF)
- `VAE/1810.04152v2.pdf` — **Tucker et al. 2019** (DReG 估计器)
- `VAE/1506.02216v6.pdf` — **Chung et al. 2015** (VRNN, Variational RNN)
- `VAE/1312.6114v11.pdf` — **Kingma & Welling 2014** (VAE 原始)
- `VAE/1509.00519v4.pdf` — **Burda et al. 2016** (IWAE)
- `VAE/2008.12595v4.pdf` — Kingma & Welling 2019 intro to VAEs

重点（对 CVHI 方向性最强）：Klausmeier-Litchman 2004、Thorson 2024 DSEM、Miele & Matias 2017、Munch 2022 EDM、Anakok 2025 GNN 解释性、Rubanova Latent-ODE、Sanderse 2024 closure、Behrenfeld 2024。其余多为 backbone 引文。

## 每篇深读

### Klausmeier, Litchman, Levin 2004 (Droop-Liebig 原版)
核心方程（每物种 3 变量 B、Q_N、Q_P）：
```
dQ_i/dt = V_max,i R_i/(R_i+K_i) − μ(Q)·Q_i
μ(Q)   = μ_∞ · min_i ( 1 − Q_min,i / Q_i )              (Liebig min)
dB/dt  = [μ(Q) − m]·B
dR_i/dt= a(R_in,i − R_i) − V_max,i R_i/(R_i+K_i) · B
```
关键参数（Table 1, Rhee 1974/78）：μ_∞≈1.35 /day；m=0.16 /day；K_N=5.6, K_P=0.2 μmol/L；V_max,P=12.3, V_max,N=341 (×10⁻⁹ μmol/cell/day)；Q_min,N/Q_min,P ≈ 27.7/45.4 ≈ 0.61。
对 CVHI：这是 Stage 2 要用的"真 Klausmeier"，比 klausmeier2008 更直接给出 Beninca 化学计量所需 N:P 比。**可直接把 K_N≫K_P 作为 nutrient→phyto 弹性的先验尺度。**

### Thorson et al. 2024 DSEM (MEE)
**核心思想**：在 ecological context 做 SEM + 时间演化 + latent constructs，能处理 (i) 显式 causal map，(ii) 非独立样本（多物种），(iii) missing values，(iv) 循环因果（feedback loop）。piecewiseSEM 只能无环且无缺失；DSEM 把结构方程嵌进 state-space。
对 CVHI：**最直接的新 baseline 对照**。我们 h·G(x) 本质是 learned causal map with latent construct；可以把 DSEM 作为 CVHI 的"monotone causal" 比较组。它的 `do-calculus` 解读还可以给 counterfactual 损失一个正式支撑。

### Miele & Matias 2017 (R. Soc. Open Sci.)
**核心**：dynamic-SBM（随机块模型）在 snapshot 序列上恢复 latent cluster。每个节点被赋一个随时间演化的隐状态类别，"hidden structure"=块结构随时间漂移。
对 CVHI：灵感 — **把 G(x) 在时间上分块**。目前 G(x) 是一个 MLP，输出随 x 平滑变化；借鉴 DSBM 可以考虑把 G 离散化为 K 个 regime 的 mixture（h 触发 regime switch）。这会给 h 一个更强的 "identifiable" 语义：h 选择 cluster。

### Munch et al. 2022 MEE (EDM review)
**关键区分**三类处理未观测变量的范式：(1) 忽略（process noise 近似），(2) 有参数模型 + state-space，(3) **EDM/attractor reconstruction**，不假设函数形式，用 delay embedding 隐式处理未观测。
对 CVHI：我们是(2)+(3) 混合 (VAE state-space + Takens encoder)。**应该在 paper 里把 CVHI 明确定位为 "EDM-informed parametric hybrid"**。并引用 Deyle et al. 2016、Johnson 2022 (sparse EDM) 作为"单独 EDM 失败在哪"。

### Chang, Ushio, Hsieh 2017 (EDM for beginners)
教学型综述，Simplex/S-map/CCM 流程。对 CVHI：**不作方法贡献**，但 intro 里可以挂一句"EDM 在浮游/ Beninca 类型数据上已广泛应用，CVHI 继承 delay-embedding 想法但用 VAE 参数化"。

### McClintock et al. 2020 Ecol Letters (HMM review)
**核心**：HMM = 离散有限状态隐 + emission distribution。广泛用于 movement、regime、capture-recapture。
对 CVHI：h 是连续的 → 不是 HMM。但这篇给 "regime shift detection via σ(t)" 应用语言一个官方支撑。引用 + 差异化即可。

### Anakok et al. 2025 (GNN 解释性 on ecology)
BVGAE (Bipartite VGAE) + FAIR-BVGAE（去采样 bias）。用 HSIC（Hilbert-Schmidt independence criterion）衡量协变量对 latent connectivity 的影响；可解释 driver effects 在 pollination network 上。
对 CVHI：**方法层面大启发**：
1. **HSIC 检验 h 与 G 的独立性** — 我们的 counterfactual null 其实就是想证 h≠独立于 G。HSIC 比自制 CF margin 更规范。
2. **Sampling bias 处理**：Beninca 数据 chemostat 本身无 sampling bias，但 Portal 有观测偏差 — 可引用为 future work。

### Sanderse et al. 2024 (SciML closure review)
**核心**：Closure problem = "小尺度效应需要被参数化回大尺度方程"；hybrid = physics + NN。分 (a) what to learn (model form), (b) how to learn (objective), (c) embedding physics constraints, (d) 时空离散。关键讨论：稳定性、一致性、收敛性（纯 data-driven 闭合经常数值不稳）。
对 CVHI：**直接的理论锚**。我们的 Δlog x = f_visible + h·G 就是一个闭合模型：f_visible 是"大尺度物理"，h·G 是"小尺度 closure"。应引为：
- Paper framing："residual hidden-state closure for partially observed ecological dynamics"
- **Rollout stability** 的必要性（L1 rollout 有此理论支撑，不是 ad-hoc）
- 未来工作：consistency 分析（h→0 时模型回到 visible-only 的 convergence）

### Rubanova et al. 2019 Latent ODE
RNN+ODE hybrid 处理不规则时序。h(t) 由 neural ODE 演化，观测处 jump 更新。
对 CVHI：我们的 h 目前是按时间网格从 VAE encoder 出的 per-step 样本。**可选升级：把 h 写成 Neural ODE**，能自然处理 Portal 的不规则月采样，也能给 h "slow drift" 一个参数化先验（vs 当前 L3 低频人工先验被证伪）。这是 Stage 3+ 候选。

### Karl et al. 2017 DVBF
显式 locally-linear state-space VAE。对 CVHI：不直接用，但它的 "reparameterize transition" 技巧能用于 h 的时间耦合。

### Behrenfeld & Bisson 2024 (Paradox of plankton, neutral theory)
现代综述：微 niche、chaos、metabolic network dependency、dispersal-sustained disequilibrium 都能解释共存。
对 CVHI：**paper intro 的弹药**。Beninca chaos Lyapunov≈0.05/day 正是他们提到的 "chaotic coexistence" 机制。引用可让 CVHI 在生物学动机上更实。

### Hutchinson 1961 Paradox
必引奠基文献，无新机制信息。

### Price et al. 2010 (MTE-plant)
扩展 MTE 到植物，争议点：b 不是普适 3/4。与 Glazier、Clarke 一致。对 CVHI：加强"b 非普适→用 taxon-specific b"的论证，但不给新数值。

### VAE 家族（Kingma, IWAE, VRNN, DReG, 2019 intro）
backbone 引用。IWAE / DReG 可在方法 ablation 时讨论"为什么不用 importance weighting 提高后验质量"（答案：h 维度低，ELBO 已够）。

### GNN 综述（Wu 2019, Jin 2023 time-series）
intro 引文，无新方法。

---

## 对 CVHI 的具体建议

### A. Stage 2 Klausmeier 从"软 sign"升到"Droop-Liebig soft 公式 hint"
当前计划只用 sign prior。Klausmeier-Litchman 2004 给出完整 Droop-Liebig 公式 → 直接加 **formula hint feature**（像 LV/Holling 一样）：
```python
# 对 phyto→nutrient 边:  holling-like uptake
droop_hint = V_max * R_j / (R_j + K_j)    # 以 B_i 为消耗者
# 对 nutrient→phyto 边:  growth limitation
liebig_hint = 1.0 - Q_min / Q             # 其中 Q 用 x_phyto/x_nutrient 近似
```
位置：`models/cvhi_ncd.py`, `SpeciesGNN_MLP` 边特征拼接处（目前有 LV_hint, Holling_lin_hint, Holling_bi_hint, linear_hint → 追加 droop_hint, liebig_hint 两列）。
λ=0（纯输入特征，不作 loss），预期：phyto hidden Pearson +0.03~0.05。

### B. Stage 1c 的 MTE shape-prior 公式细节（现在就加）
按 overnight_plan 的 taxon-b 表，pen 加在 `f_visible` 对角项的 **rank-correlation** 上：
```python
r_learned_i = df_visible/dx_i evaluated at x=mean            # N 维
r_MTE_i   = log(M_i) * (b_taxon[i] - 1)                      # shape only
loss_MTE = λ * (1 - spearman_corr(r_learned, r_MTE))         # 只保序
λ = 0.02
```
注意：**用 spearman 不用 pearson/MSE**（Clarke 说只能约束 shape；B_0 不可识别）。

### C. 新组件："HSIC counterfactual margin"（源自 Anakok 2025）
现在的 CF null/shuffle 用的是 recon margin；替换/补充为 HSIC 独立性：
```
loss_hsic = − HSIC( h_samples , G_messages )    # maximize dependence
对照 loss_hsic_null = HSIC(h_shuffled, G_messages)  ≈ 0
```
λ=0.01。理论更扎实，审稿人会买账。

### D. 新 baseline 对照：DSEM 和 DSBM
加到 Q2 ablation：
- **A8 = Thorson DSEM**（静态 piecewise + latent construct，在 Beninca 上跑）
- **A9 = Miele-Matias dynamic SBM** on G recovery 任务（G 离散化）
预期 CVHI 显著胜出在 hidden trajectory Pearson；DSEM 在 causal 解释上更好 → 差异化 framing。

### E. Stage 3 候选（若 Stage 2 Pearson 仍 < 0.20）
**Latent ODE h** (Rubanova)：
```python
# 替换 encoder 的 per-step h 采样:
h_0 ~ q(h_0 | X_Takens)                   # VAE 只采 h_0
h_t = ODESolve( f_ODE, h_0, t=[0..T] )    # f_ODE = 小 MLP
```
位置：`models/cvhi_residual.py` 中 h 生成路径。此改动处理 Portal 不规则采样，理论上 h_t 平滑度内生。**不**加人工 L3 prior（已证伪）。预期：Portal Pearson +0.02~0.05。

### F. σ(t) 作 regime-shift detector（从 2026-04-14 review 延续，现在 McClintock 可加引）
encoder 已有 log_sigma；直接做 eval-time 指标：
```python
shift_score(t) = |σ(t) - smooth(σ)(t)|
AUC vs 已知 ecosystem events (Beninca 1997 regime shift 有记录)
```
作为 paper 的 "secondary application claim"，引 McClintock 2020 + Trifonova 2025。

### G. 大 Ablation 补充配置（加到已计划 A0-A7）
- **A10 = +HSIC CF loss**（C 方向）
- **A11 = +Droop/Liebig hints**（A 方向）
- **A12 = Latent-ODE h**（E 方向，若时间够）

### 大图更新
| Stage | 新组件 | 论文来源 | 预期 Pearson 叠加 |
|---|---|---|---|
| 1b (done) | RMSE+aug | — | +0.018 |
| 1c (running) | MTE shape spearman λ=0.02 | Brown/Glazier/Kremer/Clarke/Price | +0.01~0.03 |
| 2 | Droop-Liebig **hints**（非 loss） | Klausmeier-Litchman 2004 | +0.03~0.05 |
| 2.5 | HSIC CF | Anakok 2025 + Gretton | +0.01~0.02（但理论更硬）|
| 3 | Latent-ODE h | Rubanova 2019 | +0.02~0.05（Portal 专） |
| secondary | σ(t) shift detection | McClintock 2020 + Trifonova 2025 | 新 contribution |
| framing | closure-model 定位 | Sanderse 2024 | Paper 卖点 |
| baseline | DSEM + DSBM | Thorson 2024, Miele-Matias 2017 | 差异化 |

**底线**：数值提升主要还是靠 A、E；C、D 主要增强论文可信度。Behrenfeld 2024、Munch 2022、Sanderse 2024 三篇是 framing 重武器，intro/discussion 必挂。
