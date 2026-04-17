# CLAUDE.md -- 项目入口记忆文件

> 最后更新: **2026-04-16 晚** (交替训练 alt_5_1 成新最佳 0.1595; 营养盐 input-only + burst P/R/F 实验运行中; 新增两篇 Chaos 论文笔记)

---

## 新会话必读清单 (按优先级)

1. **本文 CLAUDE.md 全文** -- 项目入口, 知道整体状态
2. **`notes/2026-04-15_consolidated_outputs.md`** -- 产出汇总: 实验矩阵+论文深读+学术结论
3. **`notes/2026-04-15_stage1c_failure_analysis.md`** -- Beninca 2008/2011 + Rogers 2023 + Clarke 2025 深读
4. **`runs/20260416_150113_beninca_alternating/summary.md`** -- 交替训练结果 (当前最佳)
5. **`docs/论文笔记/Young2022_DeepLearning_DelayCoordinate_PartialObs.md`** -- Chaos 论文1
6. **`docs/论文笔记/Racca2022_DataDriven_ExtremeEvents_ChaoticFlow.md`** -- Chaos 论文2

---

## 一、项目一句话

**部分观测混沌生态动力学中的隐藏物种推断**: 从 N 个 visible 物种时序恢复 1 个 hidden 物种时序, **严格无 hidden 监督** (训练中绝不用 hidden_true), 目标发 SCI Q2/Q3.

**用户**: 16 岁学生, Q2 (MEE/Ecography/Ecol Informatics) 为主目标.

---

## 二、核心方法: Eco-GNRD (formerly CVHI-Residual)

### 架构
```
Posterior Encoder (GNN + Takens delay embedding, lags=1,2,4,8)
    | q(h|X) = N(mu, sigma^2)
Dynamics:  log(x_{t+1}/x_t) = f_visible(x_t) + h_t * G(x_t)
    |        (两个 Species-GNN, MLP backbone + formula hints)
    v
Loss: recon + rollout(3-step) + 反事实(null, shuffle) + KL + sparsity
```

### 训练策略: **交替训练 5:1** (当前最佳)
- Phase A (5 ep): 训练 f_visible + G_field, 冻结 encoder
- Phase B (1 ep): 冻结 f_visible + G_field, 训练 encoder
- 动机: 联合训练时 h 梯度极小 (hidden 只解释 <0.2% visible), 交替优化直接给 encoder 残差信号
- Warmup 20% epochs 后开始交替

### 关键组件
- **MLP backbone + formula hints**: 每条边 message 由 MLP 从 [x_i, x_j, s_i, s_j, LV_hint, Holling hints] 计算
- **L1 3-step rollout**: 多步 teacher-forced 自洽
- **h * G(x) 残差分解**: h=0 时贡献为 0, 硬约束消除架空
- **Counterfactual losses** (null/shuffle): h 必要性 + 时序结构约束
- **RMSE log + input dropout aug**: S1b 验证的纯 ML 改进
- **G_anchor_first + alpha annealing**: 破 +/-h 对称

### 已证伪/已弃用
| 方向 | 为什么 |
|---|---|
| Anchor from Linear Sparse+EM | 违反无监督红线 |
| L3 低频先验 | 真 hidden 本身有高频 |
| MoG posterior K>1 | Beninca K=1 够 |
| Hard MTE on G (Stage 1) | 位置错+数值错 |
| MTE shape on f_visible (Stage 1c) | Bacteria b 方向反, shared attractor |
| Food-web sign prior (Stage 1d) | Overall net negative |
| MSP (Masked Species Prediction) | 不如纯 alt_5_1 |
| Geometry-aware smooth gating | 无效 |

---

## 三、红线 (绝不触碰)

1. **训练中绝不用 hidden_true**: 不作监督目标/anchor/pseudo-label
2. **不引入外部协变量** (降雨/NDVI 等)
3. **任务严格 n->1**, 单 hidden
4. **节点必须是物种** (GNN 语义保留)
5. **不能假设 hidden 身份/类别**
6. **不能用含 hidden 的统计量** (Beninca Table 1 correlations 禁用数值, sign 可用)

---

## 四、当前数据集 + 结果

### 主数据集: Beninca 2008 Baltic plankton mesocosm
- 9 物种 + 4 营养盐 = 13 channels, T=658 (dt=4day)
- Lyapunov = 0.05/day, predictability horizon = 20 day = 5 step
- 9->1 rotation: 轮流把每个物种作 hidden
- Beninca 2008 Fig 3: 9 物种 Lyapunov **几乎相等** -> **shared chaotic attractor**

### Beninca 实验矩阵 (Overall mean Pearson, 3 seeds)

| Config | 组件 | Overall | 判定 |
|---|---|---|---|
| Phase 2 baseline | Optuna HP | +0.114 | 起点 |
| Stage 1b | RMSE log + input dropout | +0.132 | 之前最佳 |
| alt_3_1 | 交替 3:1 | +0.1325 | 略好于 joint |
| **alt_5_1** | **交替 5:1** | **+0.1595** | **当前最佳** |
| pretrain+alt | 预训练150ep + 交替 3:1 | +0.1403 | 不如 alt_5_1 |
| Stage 1c | MTE shape prior | +0.086 | FAIL |
| Stage 1d | Food-web sign prior | +0.111 | FAIL |
| MSP+alt | Masked Species Prediction | +0.134 | 不如 alt_5_1 |
| Geo gating | Phase-space velocity gating | +0.132 | 无效 |

### 单物种最高记录 (alt_5_1)
| Species | Pearson | 说明 |
|---|---|---|
| **Ostracods** | **+0.305** | alt_5_1, 历史最佳 |
| **Calanoids** | **+0.248** | alt_5_1 |
| **Harpacticoids** | **+0.233** | alt_5_1 |
| Nanophyto | +0.143 | alt_5_1 |
| Picophyto | +0.145 | alt_5_1 |
| Bacteria | +0.119 | alt_5_1 (pretrain+alt 有 0.172) |
| Cyclopoids | +0.095 | alt_5_1 (pretrain+alt 有 0.122) |
| Rotifers | +0.089 | alt_5_1 |
| Filam_diatoms | +0.060 | alt_5_1 (最难, 原论文已排除) |

### 进行中实验 (2026-04-16)
- **营养盐 input-only + burst P/R/F**: `scripts/cvhi_beninca_nutrient_input.py`
  - 营养盐不参与 recon loss, 仅作 encoder/dynamics 输入
  - 新增 Burst Precision/Recall/F-score 评估 (参照 Racca & Magri 2022)
  - 使用 alt_5_1 训练策略

---

## 五、关键学术结论

### C1. Eco-prior 在混沌多物种系统中 net negative
- 3 次独立 eco prior 全部 overall 输于无 prior baseline
- 原因: Shared attractor + per-species rate ordering 不成立

### C2. 交替训练 5:1 是纯 ML 最大改进
- 从 0.132 -> 0.1595 (+0.0275)
- 解决 h 梯度消失问题: 联合训练时 h 只解释 <0.2% visible

### C3. MTE 在 Baltic mesocosm 不适用
- Bacteria b=1.28 (Clarke 2025), 不是 0.60
- Phyto alpha=-0.054, group intercept >> slope (Kremer 2017)

### C4. Burst 捕获是核心瓶颈 (新发现 2026-04-16)
- Beninca 数据 85-98% 平坦, burst 仅 2-15%
- Pearson 被平坦段稀释, burst 信号被淹没
- 参照 Racca & Magri 2022: 引入 Burst P/R/F 作为补充评估指标
- Burst F-score 可量化"模型学到趋势但抓不住突变"

---

## 六、新增论文笔记 (2026-04-16 Chaos 方向)

### Young & Graham 2022 (arXiv:2211.11061)
- 延迟坐标嵌入 + DNN 从部分观测恢复混沌吸引子
- 关键发现: 观测维度需 >= d_M/2 (吸引子维度的一半)
- 与我们的 Takens delay embedding 设计直接对应
- 笔记: `docs/论文笔记/Young2022_DeepLearning_DelayCoordinate_PartialObs.md`

### Racca & Magri 2022 (arXiv:2204.11682)
- ESN 预测/控制混沌剪切流中的极端事件 (burst)
- MFE 系统的"平坦+burst"模式与 Beninca 数据高度相似
- Precision/Recall/F-score 评估极端事件, 能提前 5 Lyapunov 时间预测
- 启发: burst 二分类评估框架 (已实现) + ESN 作 baseline 候选
- 笔记: `docs/论文笔记/Racca2022_DataDriven_ExtremeEvents_ChaoticFlow.md`

---

## 七、项目主要文件

### 代码
| 文件 | 角色 |
|---|---|
| `models/cvhi_residual.py` | EcoGNRD (Eco-GNRD) 主类; **含 n_recon_channels 参数** (2026-04-16 新增, 支持营养盐 input-only). 保留 `CVHI_Residual` 别名向后兼容 |
| `models/cvhi_ncd.py` | GNN backbone: MLP/SoftForms, MultiLayerSpeciesGNN, encoder |
| `scripts/train_utils_fast.py` | 训练工具: torch.compile + EMA + Snapshot ensemble |
| `scripts/load_beninca.py` | Beninca 数据 loader (9 species + 4 nutrients, dt=4 interp) |
| `scripts/cvhi_beninca_alternating.py` | **交替训练** (baseline/3:1/5:1/pretrain) |
| `scripts/cvhi_beninca_nutrient_input.py` | **最新**: 营养盐 input-only + burst P/R/F + alt_5_1 |
| `scripts/cvhi_beninca_stage1b.py` | Stage 1b: RMSE+aug |
| `scripts/cvhi_beninca_msp.py` | MSP + 交替训练 (不如 alt_5_1) |
| `scripts/cvhi_geo_gating.py` | Geometry-aware gating (无效) |

### 论文笔记 (2026-04-16 新增)
| 文件 | 内容 |
|---|---|
| `docs/论文笔记/Young2022_*.md` | 延迟坐标 + DNN 恢复混沌吸引子 |
| `docs/论文笔记/Racca2022_*.md` | ESN 预测控制混沌极端事件 |

### 关键 Runs
| Run | 结果 |
|---|---|
| `runs/20260416_150113_beninca_alternating/` | **alt_5_1 = 0.1595** (当前最佳) |
| `runs/20260416_161413_beninca_msp/` | MSP: alt_5_1 仍最佳 |
| `runs/20260416_172306_beninca_geo_gating/` | Geo gating 无效 |
| `runs/20260416_211305_beninca_nutrient_input/` | 营养盐 input-only (运行中) |

---

## 八、技术细节须知

### 环境
- Windows 11, Python, PyTorch
- **no Triton on Windows** (torch.compile 静默跳过)
- GPU 单张, batch=1 (单时序)
- 单 seed ~26s (交替训练), 3 seeds x 9 species = 27 runs ~12 min

### 常用命令
```bash
# 交替训练 (当前最佳)
python -u -m scripts.cvhi_beninca_alternating 2>&1 | tee logs/alternating.log

# 营养盐 input-only + burst eval (最新)
python -u -m scripts.cvhi_beninca_nutrient_input 2>&1 | tee logs/nutrient_input.log

# 编译检查
python -m py_compile models/cvhi_residual.py scripts/train_utils_fast.py
```

### 编码坑
- Windows GBK -> print() 用 ASCII, 写文件用 encoding="utf-8"
- print() encoding error 会阻断后续 IO (S1c 曾因此丢失 artifacts)

---

## 九、当前进展与下一步 (2026-04-16)

### 已完成
1. alt_5_1 确认为最佳训练策略 (+0.1595)
2. MSP, geo gating 等变体均不如 alt_5_1
3. 两篇 Chaos 论文 (Young 2022 延迟坐标, Racca 2022 极端事件) 深读并写入笔记
4. 模型新增 `n_recon_channels` 参数, 支持营养盐 input-only
5. 新增 Burst P/R/F 评估函数 (参照 Racca & Magri)

### 运行中
- 营养盐 input-only + alt_5_1 + burst eval (~12 min)

### 待决策
| 方案 | 目的 |
|---|---|
| 根据 burst P/R/F 结果优化 burst 捕获 | 核心瓶颈 |
| MI ceiling 实验 | 证明 Pearson 接近上限 |
| ESN baseline (Racca 启发) | 简单 baseline 对比 |
| Latent ODE h 替代 VAE-style | 更强的 h 时序建模 |

---

## 十、修改原则 (给所有未来 session)

1. 先读 CLAUDE.md 全文
2. 每次改动单一组件, 立即多 seed 验证
3. **红线绝不触碰**: 训练中不出现 hidden_true
4. 交替训练 5:1 是默认训练策略
5. Pearson 为主要指标, burst F-score 为补充
6. Windows 编码: print ASCII, 写文件 utf-8
7. 用户 16 岁, Q2/Q3, 时间预算有限
