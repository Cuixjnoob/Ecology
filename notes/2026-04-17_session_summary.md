# Session Summary: 2026-04-16/17 Night

## 一、新论文笔记

### 读了 4 篇 Chaos/混沌方向论文：

1. **Young & Graham 2022** — 延迟坐标嵌入 + DNN 从部分观测恢复混沌吸引子
   - 和我们的 Takens lags 设计直接对应
   - 关键发现：观测维度需 >= 吸引子维度的一半

2. **Racca & Magri 2022** — ESN 预测控制混沌极端事件
   - 启发了 Burst P/R/F 评估框架
   - MFE 的"平坦+burst"模式和 Beninca 高度相似

3. **Ouala 2019 NbedDyn** — 学习增广状态空间 ODE
   - 核心贡献：h 应有自己的动力学（ODE 一致性约束）
   - 我们实现了 LatentDynamicsNet f_h 作为 h 的动力学教练

4. **Trifonova 2015** — 贝叶斯网络 + general/specific 隐变量建模未测量物种
   - 论文定位参考："建模未测量物种组的动力学效应"
   - 和我们的 disentanglement 分析对应

## 二、实验结果汇总

### Beninca 实验

| # | 方法 | Overall Pearson | 判定 |
|---|---|---|---|
| 1 | Nutrient-input-only + alt 5:1 | 0.1446 | 营养盐在 loss 中有用 |
| 2 | **NbedDyn ODE + VAE + alt 5:1** | **0.1620** | **Beninca 最佳** |
| 3 | Point estimate (no VAE) | 0.1380 | VAE 采样有正则化作用 |
| 4 | Per-species h_i | mean=0.070, best_i=0.178 | 信号在但聚合差 |
| 5 | Per-species h + learned agg | learned=0.076, oracle=0.298 | 无监督聚合失败 |

### Disentanglement 分析（关键发现）
- h 的 79% 方差是物种特异的，21% 是共享残差
- 这证明 h 主要编码隐藏物种信号，不是模型误差

### Oracle 0.298 的统计检验
- 12 个自相关通道的 lstsq 拟合，null distribution 均值 = 0.525
- **Oracle 0.298 不显著**，per-species h 的高 oracle 是统计假象

### Huisman 消融（关键发现）

| Config | sp2 | sp4 | sp6 | Overall |
|---|---|---|---|---|
| baseline | 0.586 | 0.525 | 0.291 | **0.467** |
| alt_only | 0.625 | 0.465 | 0.196 | 0.429 |
| **hdyn_only** | **0.687** | **0.584** | **0.305** | **0.525** |
| alt+hdyn | 0.627 | 0.503 | 0.139 | 0.423 |
| alt+hdyn+hf | 0.624 | 0.505 | 0.144 | 0.425 |

**关键结论：**
- **hdyn_only 最佳**（0.525），资源作 input-only 有帮助
- **交替训练在 Huisman 上有害**（0.467 → 0.429），和 Beninca 相反
- **lam_hf 无影响**

## 三、模型代码改动

1. `models/cvhi_residual.py`:
   - 新增 `n_recon_channels` 参数（营养盐/资源 input-only）
   - 新增 `point_estimate` 模式（无 VAE 采样）

2. 新脚本：
   - `cvhi_beninca_nutrient_input.py` — 营养盐 input-only + burst eval
   - `cvhi_beninca_nbeddyn.py` — NbedDyn ODE 一致性
   - `cvhi_beninca_pointest.py` — 点估计（无 VAE）
   - `cvhi_beninca_per_species_h.py` — Per-species h
   - `cvhi_beninca_psh_agg.py` — Per-species h + learned agg
   - `cvhi_huisman_ablation.py` — Huisman 消融
   - `cvhi_huisman_full.py` — Huisman 全物种
   - `disentangle_analysis.py` — 跨 rotation 分离分析
   - `plot_*.py` — 可视化脚本

3. 论文笔记（docs/论文笔记/）：
   - Young2022, Racca2022, Ouala2019 各一篇

## 四、关键未决问题与下一步方向

### Beninca burst 问题分析

核心矛盾：ODE 一致性提升了 Pearson（通过动力学连贯性），但压制了 burst 捕获（burst F 从 0.097 降到 0.086）。

可能的解决方向：
1. **Event-aware ODE consistency**: calm 时强约束 h 的 ODE，burst 时放松
   - 需要检测"何时放松"，可用 phase-space velocity（但 geo-gating 之前无效）
2. **Dual-timescale h**: h = h_slow + h_fast，h_slow 受 ODE 约束，h_fast 自由
   - 模型已有 hierarchical_h 代码（未测试）
3. **Laplace prior 替代 Gaussian**: Laplace 允许稀疏 burst（大多数时刻 h~0，burst 时 h 大）
4. **直接优化 h 而非 encoder**（完全 NbedDyn 风格）: 去掉 encoder，直接把 {h_t} 作为可优化变量

### 交替训练的数据依赖性

交替训练在 Beninca 上有效（+0.03）但在 Huisman 上有害（-0.04）。原因可能是：
- Beninca: f_visible 学不到好的动力学（数据太复杂），h 梯度被淹没 → 交替保护 encoder
- Huisman: f_visible 能学到较好的动力学（模拟数据结构清晰），交替反而破坏联合优化

启示：**训练策略应该自适应**，不是一刀切。

### 论文定位

参照 Trifonova 2015 的 framing：
> "用潜变量建模部分观测生态系统中的未测量动力学影响"
> "建模未测量物种组对可观测群落动力学的效应"

结合我们的 disentanglement 分析（79% species-specific），可以有力论证恢复出的信号主要是隐藏物种。
