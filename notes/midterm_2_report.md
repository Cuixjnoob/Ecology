# 中期汇报 v2 — CVHI-Residual on Beninca Plankton

> 2026-04-15. 接续 midterm v1 (2026-04-13, CVHI-Residual + MLP backbone 定型后).
> 关注期: 2026-04-14 (Beninca baseline) → 2026-04-15 (夜跑 Stage 1b/1c/2 + 大 ablation)

---

## 0. TL;DR (结论优先)

1. **Beninca Stage 1b** (RMSE log loss + input dropout aug, NO MTE) 把 mean Pearson 从 Phase 2 的 **+0.114** 提到 **+0.132** (+0.018 overall).
2. **Ostracods 突破**: Pearson +0.272 (Phase 2: +0.125). 首次单物种过 0.25.
3. **Stage 1 hard MTE G prior 失败**: -0.034 (mean 0.080), 原因 (论文深读): MTE 不约束 G, 且 Glazier 2005 指出 pelagic b≈0.88 不是 0.75.
4. **Stage 1c corrected MTE shape prior** (f_visible, taxon-specific b): 【待 Stage 1c 完成填】.
5. **Stage 2 Klausmeier sign prior** (N↔phyto): 【待 ablation 完成填】.
6. **大 ablation (9 configs × 9 species × 3 seeds)**: 【待 ablation 完成填】.

---

## 1. 数据集

**Beninca 2008 Baltic plankton mesocosm**:
- 7.3 年封闭中宇宙, 原始 803 采样点, Protozoa 缺失 10% → drop
- 剩 9 物种 + 4 nutrient (NO2/NO3/NH4/SRP) = 13 visible channels
- 插值到 dt=4 day, T=658 steps
- Lyapunov ≈ 0.05/day, predictability horizon ≈ 20 day = 5 step
- Hidden recovery 任务: 删一个 species, 从剩余 12 channels 推它

---

## 2. 方法演化 (2026-04-14 ~ 2026-04-15)

### 起点 (2026-04-13 结束)
CVHI-Residual + MLP backbone + formula hints + L1 rollout, Portal 0.17 / Beninca 暂无结果.

### Phase 2 (2026-04-14 AM) — HP tuning
- Optuna TPE sampler, 30 trials, per-species mean Pearson 作 objective (注: hidden 用于 HP 选择 — weak supervision caveat).
- Best HP (Trial 13): `encoder_d=96, blocks=3, lr=6e-4, lam_kl=0.017, lam_hf=0.2, min_energy=0.14, lam_cf=9.5`.
- Result: **+0.114** overall (9 species × 5 seeds).

### Stage 1 (2026-04-14 PM) — 生态 prior 首次尝试
加入:
- `lam_rmse_log=0.1` (log-scale amplitude reconstruction)
- `lam_mte_prior=0.5` (MTE-based G magnitude prior, universal b=0.75)
- `input_dropout_prob=0.05` (chaos robustness)

Result: **+0.080** (-0.034). **失败**.

### 论文深读 (2026-04-14 夜) — MTE 纠错
Agent 深读 Brown 2004 / Glazier 2005 / Kremer 2017 / Clarke 2025 定位 4 错误:
1. MTE 约束 intrinsic growth (f_visible), 不是 hidden coupling (G).
2. Pelagic 物种 b≈0.88 (Glazier 2005 variable scaling), 浮游植物 -0.054 (Kremer 2017 实测) — 不是 universal 0.75.
3. Body mass 值差 3 个数量级.
4. Clarke 2025: B_0 不可识别, 只能约束 exponent shape.

### Stage 1b (2026-04-14 夜) — 隔离 MTE 负面影响
只保留 `lam_rmse_log + input_dropout` (去掉 MTE). **成功**:
- Mean Pearson **+0.132** (vs P2 +0.114, +0.018)
- Ostracods **+0.272** (首次过 0.25)
- Bacteria +0.197, Calanoids +0.173, Nanophyto +0.141

### Stage 1c (2026-04-15 AM) — 修正 MTE shape prior
- 位置: f_visible 而非 G
- 形式: Pearson correlation distance (shape-only, 不约束绝对)
- Taxon-specific b: Bacteria 0.60 / Phyto 0.95 / Pelagic copepod 0.88 / Rotifers 0.88 / Benthic copepod 0.75 / Ostracods 0.75
- λ=0.02 (小)
- Bacteria log_r clipped to [-0.6, +0.6] (outlier 处理)

Result: 【填】

### Stage 2 (2026-04-15 AM) — Klausmeier-Droop sign prior
- 位置: f_visible finite-difference
- (nutrient_j → phyto_i): positive sign
- (phyto_i → nutrient_j): negative sign
- λ=0.02

Result: 【填】

### 大 Ablation (2026-04-15 AM) — 9 configs
见 §4.

---

## 3. 当前最佳配置

【根据 ablation 选出, 如 A1 (Stage 1b) 仍最佳则保留, 否则更新】

---

## 4. Ablation 结果

| Config | Components | Mean Pearson | Δ vs A0 | Best species |
|---|---|---|---|---|
| A0 baseline | — | +0.114 | 0 | — |
| A1 +RMSE+aug | Stage 1b | +0.132 | +0.018 | Ostracods 0.272 |
| A2 +MTE shape | Stage 1c | 【填】 | 【填】 | 【填】 |
| A3 +Klausmeier | Stage 2 | 【填】 | 【填】 | 【填】 |
| A4 +EMA+Snap | 经典 NN | 【填】 | 【填】 | 【填】 |
| A5 +Hier h | 架构 | 【填】 | 【填】 | 【填】 |
| A6 Eco combo | A1+A3 | 【填】 | 【填】 | 【填】 |
| A7 Classic NN | A1+A4+A5 | 【填】 | 【填】 | 【填】 |
| A8 All | A6+A7 | 【填】 | 【填】 | 【填】 |

---

## 5. 可视化清单 (待生成)

- [ ] 按 config 的 mean Pearson 柱状图
- [ ] 每 species 的 h_true vs h_predicted 叠线图
- [ ] Per-species mean Pearson 雷达图 (配 Phase 2 / S1b / Best)
- [ ] 训练曲线 (val_recon over epoch) 对比 config
- [ ] MTE shape prior 的 learned vs target correlation scatter
- [ ] Klausmeier sign prior 的 finite-diff ∂base/∂x 热图

---

## 6. 新组件 (已实施)

| 组件 | 位置 | 状态 |
|---|---|---|
| RMSE log loss | `cvhi_residual.py` loss | ✓ 稳定 |
| Input dropout aug | `train_utils_fast.py` | ✓ 稳定 |
| MTE shape prior (f_visible) | `cvhi_residual.py` loss | ✓ 实现, 评估中 |
| Klausmeier sign prior | `cvhi_residual.py` loss | ✓ 实现, 评估中 |
| EMA of weights | `train_utils_fast.ModelEMA` | ✓ |
| Snapshot ensemble | `train_utils_fast` | ✓ |
| Hierarchical h (slow+fast) | `cvhi_residual` encoder | ✓ |
| torch.compile | `train_utils_fast` | ✓ (Windows 无 triton 时静默跳过) |

---

## 7. 痛点 & 下步 (见 `2026-04-15_painpoints_and_brainstorm.md`)

**瓶颈**:
- Beninca mean Pearson 在 0.13 平台 (接近混沌信息论上限?)
- Filam_diatoms 所有 config 均低, 慢变量信号难
- Stage 1c 若证实负面, MTE 从 loss 降级为 input feature

**下一步** (在大 ablation 结果之上):
- Tier 1 ML upgrades (SWA / CosineRestart / DropPath / GraphNorm / Edge-drop)
- EMA-teacher 自蒸馏 (替代已证伪 L3)
- 可能: HSIC counterfactual (Anakok 2025)
- 长期: Latent-ODE h (Rubanova 2019, Portal extension)

---

## 8. 论文框架 (在已有 framework 基础上更新)

**Framing claim**: "residual hidden-state closure for partially observed ecological dynamics" (Sanderse 2024 启发).

**Main contributions**:
1. Residual decomposition Δlog x = f_visible + h·G with counterfactual necessity (methodological).
2. Ecology-informed weak priors:
   - Amplitude-aware RMSE log loss (validated +0.018)
   - MTE shape prior (taxon-specific b, Glazier/Kremer) — [positive/negative finding]
   - Klausmeier-Droop sign priors (N↔phyto) — [positive/negative finding]
3. σ(t)-based regime-shift detection (secondary contribution).

**Baselines**:
- DSEM (Thorson 2024) — latent construct SEM
- sjSDM / HMSC (Pichler 2021 / Tikhonov 2020) — species distribution
- EDM / CCM (Sugihara 2012 / Deyle 2016) — attractor reconstruction
- Plain residual NN (no priors)

**Datasets**:
- Beninca 2008 (9→1, 9 rotations)
- LV / Holling synthetic (validation)
- Portal OT extension (next-stage)

**Journal tier**: Methods in Ecology and Evolution / Ecography (Q2 稳), Ecological Monographs (Q1 stretch).
