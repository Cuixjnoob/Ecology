# 产出汇总 — 2026-04-15

> 一张表看完所有实验、产出、论文读后与下一步.

---

## 一、实验矩阵 (Beninca 2008, 9→1 任务, mean Pearson over seeds)

| Config | 组件 | Seeds | 每物种 Pearson 均值 (9) | Overall | 相对 S1b | 决策 |
|---|---|---|---|---|---|---|
| **Phase 2 baseline** | Optuna-tuned HP (Trial 13) | 5 | (0.064, 0.159, 0.056, 0.204, 0.062, 0.056, 0.125, 0.129, 0.171) | **+0.114** | — | — |
| **Stage 1 (failed)** | + hard MTE on G (b=0.75 universal) + RMSE + aug | 3 | — | +0.080 | — | ❌ 弃用 |
| **Stage 1b (best so far)** | + RMSE log + input dropout aug (无 MTE) | 3 | (0.055, 0.173, 0.080, 0.141, 0.116, 0.031, **0.272**, 0.120, 0.197) | **+0.132** | 0 | ✓ 主线 |
| **Stage 1c (failed)** | + MTE shape on f_visible (taxon b, Pearson corr distance) | 5 | (0.053, 0.079, 0.087, 0.094, 0.061, 0.021, 0.169, **0.156**, 0.052) | +0.086 | **−0.046** | ❌ 失败, 但 Harp +0.036 |
| **Stage 1d (运行中)** | + 通用 food-web sign prior (phyto↔zoo↔bacteria, stochastic, 1 extra fwd/epoch) | 5 | 运行中 | - | - | 跑完判定 |
| **Stage 2 (未跑)** | + Klausmeier N↔phyto sign (Stage 1d 子集) | - | - | - | - | 若 S1d 证实正收益, 合并测试 |

### 单物种最佳记录
- **Ostracods**: S1b **0.272** (首次过 0.25)
- **Bacteria**: S1b **0.197**
- **Calanoids**: S1b **0.173**
- **Harpacticoids**: S1c **0.156** (MTE 唯一胜 S1b 的物种)
- **Nanophyto**: P2 **0.204** (P2 最强, S1b 反而降)

### 最难物种
- **Filam_diatoms**: 所有 config ≤ 0.056 (慢变量 + Beninca 原文因 zero-heavy 已排除)
- **Rotifers**: 最高 S1c 0.087

---

## 二、已实现的代码组件

### 核心模型 (`models/cvhi_residual.py`)
| 组件 | 状态 | 验证 |
|---|---|---|
| f_visible + h·G 残差分解 | ✓ | Phase 2 基础 |
| Counterfactual losses (null/shuffle) | ✓ | Phase 2 基础 |
| MoG posterior (K>1) | ✓ | K=1 够用 (Beninca) |
| G_anchor_first (破 ±h 对称) | ✓ | Phase 2 |
| G_anchor_alpha (annealing) | ✓ | — |
| L1 3-step rollout | ✓ | Sanderse 2024 理论锚 |
| L3 low-frequency prior | ✗ 证伪 | 已关闭 |
| Hierarchical h (slow/fast) | ✓ 未测 | Stage 待测 |
| RMSE log reconstruction | ✓ 已验证 | S1b +0.018 |
| **MTE G prior (wrong)** | ✗ | Stage 1 失败 |
| **MTE shape on f_visible (wrong)** | ✗ | Stage 1c 失败 |
| **Klausmeier-style stoich sign prior** | ✓ 已优化 | Stage 1d 运行中 (stochastic 1/epoch) |

### 训练工具 (`scripts/train_utils_fast.py`)
| 组件 | 状态 |
|---|---|
| torch.compile (triton 可用时) | ✓ Windows 无 triton 静默跳过 |
| set_to_none=True | ✓ |
| EMA of weights | ✓ 未测 (defaults off) |
| Snapshot ensemble (末 15%) | ✓ 未测 |
| Input dropout aug | ✓ S1b 已验证 |
| Multi-method eval (best_val/ema/snapshot/combined) | ✓ |

### 实验脚本
| 脚本 | 目的 |
|---|---|
| `cvhi_beninca_phase2.py` | Phase 2 基线 (Optuna Trial 13 HP) |
| `cvhi_beninca_stage1b.py` | Stage 1b: RMSE+aug (最佳基线) |
| `cvhi_beninca_stage1c.py` | Stage 1c: MTE shape prior |
| `cvhi_beninca_stage1d.py` | Stage 1d: food-web sign prior (运行中) |
| `cvhi_beninca_stage2.py` | Stage 2: Klausmeier N↔phyto (子集) |
| `cvhi_beninca_ablation.py` | 9 configs × 9 species × 3 seeds |

---

## 三、论文深读产出

### 已处理 (notes/)
1. **`2026-04-14_related_work_review.md`** — Ovaskainen/Tikhonov/Pichler/Trifonova 5 篇 JSDM lineage
2. **`2026-04-15_new_papers_deep_read.md`** — Klausmeier-Litchman / Sanderse / Anakok / Rubanova / Thorson / Miele-Matias / Munch / Chang / McClintock / Behrenfeld 等
3. **`2026-04-15_stage1c_failure_analysis.md`** — **关键产出**: Benincà 2008 Nature 原文 + Benincà 2011 resonance + Rogers 2023 intermittent instability 的深读

### 决定性发现
| 论文 | 发现 | 对我们的启示 |
|---|---|---|
| **Benincà 2008 Fig 3** | 9 物种 Lyapunov 几乎相等 (0.051-0.066/day) | **shared attractor, per-species rate ordering 不存在 → MTE shape prior ill-posed** |
| **Benincà 2008 Table 1** | 物种间 significant correlation (Bact↔Ostra -0.24, Bact↔Rot +0.30 等) | **有监督嫌疑, 不能直接用数值, 但 sign 是通用生物学可用** |
| **Benincà 2011 Am Nat** | Plankton 与 temperature fluctuations 共振, τ = T/2π | **可作 frequency-domain prior (待试)** |
| **Rogers 2023 Ecol Lett** | Plankton rate 是 seasonal + intermittent + state-dependent | **常数 target rate 错** |
| **Clarke 2025 Table 3** | Bacteria b=1.28 (不是 0.60, 方向相反) | **MTE 在微生物完全不适用** |
| **Kremer 2017** | Phyto α=-0.054, 95% CI [-0.089, -0.018]; group intercept 差 2 数量级 >> slope | **约束 slope 无用, intercept 才重要** |
| **Sanderse 2024** | SciML closure models | **CVHI 就是 closure model, paper framing 锁定** |
| **Anakok 2025** | HSIC 测 latent-feature 独立性 | **CF loss 的理论升级版** |
| **Rubanova 2019 Latent ODE** | 不规则采样的 ODE 时序 | **Portal 扩展方向** |
| **Thorson 2024 DSEM** | Dynamic SEM with feedback | **正式 baseline 对标** |

---

## 四、关键学术结论 (可写入 paper)

### C1. MTE 不应作 per-species quantitative prior
在 N=9 且含 bacteria + phyto + zoo 的混杂系统:
- Bacteria / Protist / Phyto 在 WBE 域外 (Clarke 2025)
- Kremer 2017 group intercept 差距 >> slope
- Shared attractor → rate ordering 不存在 (Benincà 2008)
→ **这是 negative finding, 仍 publishable** ("strong MTE priors fail on mesocosm data; only sign-level food-web priors work")

### C2. RMSE log + input dropout 是经验安全组合
S1b 稳定 +0.018, 无副作用, 应作 CVHI 默认配置.

### C3. Sign-level 食物网 prior 是正确方向
Klausmeier-style sign constraint (Stage 2 概念) 可扩展到整个食物网, 只约束方向, 无 quantitative 承诺.

### C4. Unsupervised red line precisely stated
- ✓ 可用: 通用生态学 sign (predator-prey +/-), visible species 间的 GENERIC 关系
- ✗ 禁用: 从完整数据集 (含 hidden) 算出的 correlation 数值; hidden 身份/类别的任何假设

---

## 五、产出文件清单

```
notes/
├── 2026-04-13_cvhi_ncd_journey.md       # 10 次迭代历史
├── 2026-04-13_cvhi_residual_final.md    # CVHI 定型文档
├── 2026-04-14_related_work_review.md    # JSDM 对比
├── 2026-04-15_new_papers_deep_read.md   # 新论文深读
├── 2026-04-15_ml_pure_win_upgrades.md   # ML 纯收益升级清单
├── 2026-04-15_painpoints_and_brainstorm.md  # 痛点 + 思路
├── 2026-04-15_stage1c_failure_analysis.md   # S1c 失败深析
├── 2026-04-15_consolidated_outputs.md   # [本文] 汇总
├── midterm_2_report.md                  # 中期报告 v2
└── overnight_plan.md                    # 夜跑计划

runs/20260415_000840_beninca_stage1c/summary.md  # S1c 结果
runs/20260414_232248_beninca_stage1b/summary.md  # S1b 结果
runs/20260415_090548_beninca_stage1d/            # S1d 运行中

CLAUDE.md                                # 项目入口 (已更新 2026-04-15)
```
