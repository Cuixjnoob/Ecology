# 痛点分析 + 创新 Brainstorm — 2026-04-15

## 一、已知痛点

### P1. Beninca Pearson 在 0.13 平台 (Phase 2 0.114 → Stage 1b 0.132)
**症状**: 超参数调优 (Optuna 30 trials) 从 0.05→0.11, 生态 prior 再 +0.02. 继续 +0.02 越来越难.
**可能原因**:
- (a) Beninca 系统混沌 Lyapunov ≈ 0.05/day (Beninca 2008), predictability horizon ≈ 20 天 = 5 步 (dt=4天). Model 已逼近信息论上限.
- (b) 9 只有一个是 hidden 的配置, 其它 8 个 visible + 4 nutrient 已经覆盖大部分动力学, 剩余 info 有限.
- (c) Encoder 的 Takens embedding 可能 lag 选择次优.
**可查论文**:
- Sugihara 2012 CCM → 可直接测 Beninca 上的 CCM 检出率, 作为 upper bound 参考
- Munch 2022 EDM review → 复杂混沌系统 hidden recovery 的理论限
- Deyle 2016 → Beninca 数据上 EDM 的报告性能

### P2. Stage 1c MTE shape prior 引起高方差, 部分 species 回退
**症状 (中途数据)**:
- Calanoids: seed 差异大 (0.009 ~ 0.234), mean 0.079 (vs S1b 0.173, -0.09)
- Cyclopoids, Bacteria 等: 稳定但 flat
**可能原因**:
- (a) Bacteria 的 log_r = +2.4 clipped 到 +0.6 还是 outlier, 主导 Pearson 相关
- (b) 静态 rank 与 data-driven dynamics 的 r_i 排序冲突
- (c) 8 个物种 + 1 个异常值, 样本量小, 相关性敏感
**可能解决**:
- 用 Spearman (rank-based), 但需 differentiable proxy
- 移除 Bacteria 独立处理 (它的 b 生态上本就争议, Clarke 2025)
- 降 λ 到 0.005
- 改用 partial constraint: 只约束 phyto vs zoo 两组均值差

### P3. Filam_diatoms 所有 config 都低 (≤0.056)
**症状**: 最差 species, P2 0.056, S1b 0.031 (更差!).
**可能原因**:
- (a) 丝状硅藻动力学缓慢, 长期趋势主导, 短期 dynamics 信噪低
- (b) 在 Beninca 数据中可能出现 "sticky" 行为, 不常变化
**可能解决**:
- Species-specific λ_smooth (大幅放大)
- 慢变量专用 channel (Hier h 的 slow branch 可能专门捕它)
- 加入 low-frequency-only prior (但 L3 已被证伪)
**可查论文**:
- Benincà 2008 supplementary — 可能有 filam 的 Fourier 分析
- McCann 2011 "energy channels" → filam 可能属于慢能量通道

### P4. G_field 对称 (±h 兼容)
**症状**: G_anchor_first 已部分解决, 但仍有 G 的 sign 和 h 的 sign 耦合.
**可能原因**: residual decomposition 本身的歧义性.
**可能解决**:
- MTE-informed G magnitude prior (不是 target, 只是 ordering 先验)
- EMA teacher 自蒸馏 (稳定 sign convention)
- HSIC (Anakok 2025) 约束 h 和 G 的统计依赖性

### P5. Val-based model selection 不可靠
**症状**: Portal 上 ρ(val_recon, Pearson) = -0.66, Beninca 上未测.
**可能原因**: val recon 和 hidden 质量不完全一致 (尤其 chaos).
**可能解决**:
- Top-K ensemble (ρ 负向 = 倒序选)
- Multi-criterion: val_recon + h_var + hf_frac 组合
- Snapshot ensemble (已加)

### P6. GPU 利用率不满 (用户反馈)
**症状**: "GPU 是满的 但是偶尔掉下去 cpu 完全不满"
**可能原因**: 单 seed 小模型 (encoder_d=96), CPU 数据准备偶尔 bottleneck.
**可能解决**:
- 并行多 seed (同时跑 2-3 个 model) — 风险: GPU OOM
- 增大 batch (目前 batch=1, 因为单时序) — 需改架构
- 预加载到 GPU, 消除每 epoch 传输 (已部分做了)

## 二、创新 Brainstorm

### B1. 组件重审
| 组件 | 必要性 | 证据 | 备选 |
|---|---|---|---|
| **f_visible GNN** | 必要 | 无则退化为纯 h drive, 丢 visible structure | 可换 Transformer, 但 GNN 更适合 species graph |
| **G field (per-species h sensitivity)** | 必要 | 无则 h 均匀影响, 丢敏感度差异 | — |
| **G_anchor_first (softplus pin)** | 有效 | 破 ±h 对称, Phase 2 已证实 | EMA self-distill 替代 |
| **Formula hints (LV/Holling)** | 有效 | 无则下降 ~0.02 | 可扩 Klausmeier/Droop hints |
| **MoG posterior (K>1)** | 可选 | Beninca K=1 够 | regime-switch 场景 K>1 |
| **L1 rollout** | 有效 | Sanderse 2024 给理论支撑 (closure consistency) | — |
| **L3 low-freq prior** | **证伪** | 已关闭, h 本含高频 | EMA-teacher 替代 smoothness |
| **MTE G prior (Stage 1)** | **错误** | 放错位置 | 放 f_visible 上 (Stage 1c 也有问题) |
| **MTE shape on f_visible (Stage 1c)** | **尚未定论** | 正在实验中 | Spearman-based / group-wise |
| **RMSE log** | 有效 | S1b +0.018 overall | — |
| **Input dropout aug** | 有效 | S1b 之一 | — |
| **Klausmeier sign prior (Stage 2)** | 待测 | 理论清晰 | Droop-Liebig as edge feature 另一路径 |

### B2. 核心创新点 (Q2 paper cuts)
1. **Residual hidden-state closure framework** (Sanderse 2024 hook)
   - Δlog x = f_visible + h·G 作 closure for unobserved variables
   - Counterfactual necessity as identifiability condition
   - 这是 paper 的**methodological claim**

2. **MTE-informed species graph priors** (生态-ML 融合创新)
   - Pearson corr distance loss 在 f_visible rate ordering
   - Taxon-specific b (Glazier 2005 / Kremer 2017)
   - 这是**ecology prior claim** — 即使 Stage 1c 负结果也是 publishable finding ("strong MTE constraints hurt; only weak shape priors work")

3. **Klausmeier-Droop stoichiometric sign priors** (Stage 2)
   - 弱 soft sign prior 编码 N↔P 耦合方向
   - **生态化学 claim** — 差异化 baseline (vs 纯 ML)

4. **σ(t)-based regime-shift detection** (secondary contribution)
   - encoder 的 posterior σ(t) 直接 surface
   - 关联 McClintock 2020, Trifonova 2025
   - **detection claim** — Beninca 1997 已知 regime shift 作 validation

### B3. 必去掉的 (honest ablation 结果)
- L3 low-frequency prior (已去)
- Hard MTE on G (Stage 1 教训)
- 可能 MTE shape if Stage 1c 确认负 (改 soft hint input only)

### B4. 新组件 (可加)
1. **HSIC counterfactual** (Anakok 2025) — 理论升级 CF loss
2. **Latent-ODE h** (Rubanova 2019) — Portal / 不规则采样天然
3. **Droop-Liebig hints in edge features** — 不作 loss, 只作输入
4. **EMA-teacher self-distillation** — 稳定 sign convention
5. **Trophic-layer-informed GNN depth** — 融合生态结构
6. **MTE-informed edge attention bias** — log(M_j/M_i) 作 bias
7. **Laplacian / phylogeny PE** — 位置编码
8. **SWA / Lookahead / CosineRestart** — Tier 1 优化器

### B5. 哪些是"装饰"? (可能为了 paper 加的但效果存疑)
- MoG posterior (K>1 未见明显收益, K=1 够)
- Residual attention on Takens (未显著)
- Coupling weight (encoder) — 未充分验证
**计划**: Ablation 包括 K=1 vs K=3 对照.

### B6. "为了加而加" vs "必需"
| 组件 | 判定 |
|---|---|
| f_visible MLP backbone | 必需 (vs SoftForms 验证) |
| G field | 必需 (vs 均匀 h) |
| Formula hints | 必需 (vs 纯 MLP) |
| Counterfactual losses | 必需 (vs 纯 recon) |
| KL + energy + smooth | 必需 (VAE 基础) |
| MoG K>1 | 装饰, Beninca K=1 |
| Hier h | 待定 (如 Filam 上有效则必需) |
| L3 HF penalty | 可去 (未稳定见效) |
| MTE shape | 待定 (取决 Stage 1c 结果) |
| Klausmeier sign | 待定 (取决 Stage 2 结果) |

### B7. 创新最高的 3 个 (若时间允许)
1. **Latent-ODE h 取代 VAE-style posterior** — 把 h_t 写成 Neural ODE, Portal 不规则天然适配 (Rubanova 2019). 预期 Portal +0.03-0.05. *SCI Q2 强卖点*.

2. **HSIC counterfactual margin** — 替代 null/shuffle 的 MSE margin, 用独立性检验. 理论更硬, Anakok 2025 预备. 预期 +0.01, 理论上的升级更值.

3. **σ(t) regime-shift detector (secondary)** — 无监督 surface encoder σ(t) 对比 Beninca 1997 regime shift label. 独立 contribution, 论文 "Part B".

---

## 三、执行优先级

**立即做 (今晚)**:
- [x] Stage 1c finishing (contingent 负)
- [ ] Stage 2 Klausmeier
- [ ] 大 Ablation (9 configs × 9 species × 5 seeds)
- [ ] 中期报告 v2

**明日 (醒后决策)**:
- 根据 Ablation 结果选最佳 config
- 实现 Tier 1 ML 升级 (SWA + CosineRestart + DropPath) — 只 +30 min 实现, 再跑 ablation
- 实现 Tier 2 (EMA-teacher + IWAE) — 若 Tier 1 不足
- 决定是否写 Latent-ODE h (Stage 3)

**论文写作阶段**:
- Sanderse 2024 closure-framework framing
- DSEM (Thorson 2024) 作为对照 baseline
- Anakok 2025 HSIC 作为 methodological 升级
- Rubanova 2019 Latent-ODE 作为 extension
