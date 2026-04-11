# 代码目录与文件说明

> 最后更新：2026-04-11

## 顶层结构

```
生态模拟/
├── CLAUDE.md                           # 项目入口记忆文件（Claude Code 必读）
├── codex_iteration_log.md              # Codex 4 轮迭代实验日志（假设→变更→指标→keep/revert）
├── eco_dynamics_codex_spec_zh.md       # 原始 Codex 项目规范
├── eco_dynamics_implementation_brief_zh.md  # 实现简报
├── takens_embedding_spec_zh.md         # Takens 嵌入设计文档
├── requirements.txt                    # Python 依赖
├── run_all.sh                          # 一键运行/状态/同步脚本
├── git_sync.sh                         # Git 同步脚本
│
├── configs/                            # YAML 配置文件
├── data/                               # 数据生成与预处理
├── models/                             # 模型定义
├── train/                              # 训练器
├── losses/                             # 损失函数（部分在 trainer 中内联）
├── eval/                               # 评估工具
├── scripts/                            # 实验入口脚本
├── viz/                                # 可视化工具
├── notes/                              # 本目录，项目文档
├── docs/                               # 研究描述文书
├── runs/                               # 实验输出（每次运行一个子目录）
└── results/                            # 旧版结果快照（已弃用，勿与 runs/ 混淆）
```

---

## configs/ — 配置文件

```
configs/
├── partial_lv_mvp.yaml                 # 【主线】原版生产配置（40 epochs, 无 v2 keys）
├── partial_lv_mvp_v2_mechanism.yaml    # 【主线】v2 机制分离配置（50 epochs, 含 multiscale/local_variance/residual_energy）
├── base.yaml                           # 旧主线（图模型 forecast）基础配置
├── baseline_obs_graph.yaml             # 旧 baseline: observed-only graph
├── baseline_obs_mlp.yaml              # 旧 baseline: observed-only MLP
├── baseline_persistence.yaml           # 旧 baseline: persistence
├── ablation_no_delay.yaml              # 旧消融: 无 delay embedding
├── ablation_no_hidden.yaml             # 旧消融: 无 hidden nodes
├── ablation_one_step_only.yaml         # 旧消融: 仅 one-step
├── hidden_inference_experiment.yaml    # B 线: hidden-only 推断配置
└── single_lv_hidden_experiment.yaml    # B 线: 单物种 LV + hidden 实验
```

**注意**：`partial_lv_mvp.yaml` 不含 v2 新增的 `lambda_multiscale` / `lambda_local_variance` / `lambda_residual_energy` 键。代码通过 `.get(key, 0.0)` 兼容，运行时这些损失权重为 0。

---

## models/ — 模型定义

```
models/
├── __init__.py
├── partial_lv_recovery_model.py        # 【主线核心】PartialLVRecoveryModel（364 行）
│   四路分工：LV drift + residual (curriculum) + hidden_fast + noise
│   环境 OU 过程：env + τ_env × (target - env)
│   含 Takens delay encoding + GRU context + rollout memory
│
├── full_model.py                       # 旧主线 EcoDynamicsModel（273 行）
│   图模型：GNN message passing + LogGrowthDecoder
│   含 latent recurrence, species embeddings
│   当前主线未使用，但代码保留完好
│
├── hidden_inference_model.py           # B 线 HiddenSpeciesInferenceModel（136 行）
│   GNN-based hidden 推断模型
│
├── encoders.py                         # 公共编码器：MLP, ObservedDelayEncoder, GlobalContextEncoder, LatentNodeEncoder
├── decoder.py                          # LogGrowthDecoder（旧主线用）
├── gnn.py                              # DenseMessagePassingStack（旧主线用）
└── graph_builder.py                    # DenseGraphBuilder（旧主线用）
```

---

## train/ — 训练器

```
train/
├── __init__.py
├── partial_lv_mvp_trainer.py           # 【主线核心】PartialLVMVPTrainer（899 行）
│   包含：17 项损失计算、噪声退火、full-context train step、
│   validation_metrics、hidden recovery 评估、forecast_case
│   curriculum：residual_curriculum_progress = min(1.0, epoch / (total_epochs × 0.6))
│
├── trainer.py                          # 旧主线 Trainer（433 行）
│   包含 set_random_seed、create_data_loaders、save_json 等公共工具
│   以及旧图模型的完整训练循环
│
├── hidden_inference_trainer.py         # B 线 hidden 推断训练器
├── train_one_step.py                   # 旧 one-step 训练
└── train_rollout.py                    # 旧 rollout 训练
```

---

## data/ — 数据生成与预处理

```
data/
├── __init__.py
├── partial_lv_mvp.py                   # 【主线核心】合成生态系统生成器（517 行）
│   generate_partial_lv_mvp_system()
│   5 visible + 1 hidden + 1 environment + 1 pulse
│   离散 LV (Ricker) + 数据质量筛选
│
├── dataset.py                          # TimeSeriesBundle + build_windowed_datasets
├── lv_simulator.py                     # 通用 LV 模拟器（旧主线用）
├── transforms.py                       # 标准化变换
└── window_sampler.py                   # 滑窗采样器
```

---

## losses/ — 损失函数

```
losses/
├── __init__.py
├── ecological.py                       # 生态约束损失：range_penalty, smoothness_penalty,
│                                         sparsity_penalty, metabolic_prior_loss,
│                                         direct_latent_balance_penalty
│                                         注意：主线大部分损失在 partial_lv_mvp_trainer.py 中内联
├── prediction.py                       # one_step_loss（旧主线用）
└── rollout.py                          # rollout_loss（旧主线用）
```

---

## eval/ — 评估工具

```
eval/
├── __init__.py
├── metrics.py                          # compute_rollout_metrics：RMSE, MAE, Pearson, Spearman, stability
├── baselines.py                        # PersistenceBaseline, ObservedMLPBaseline
└── rollout_eval.py                     # 旧主线 rollout 评估逻辑
```

**注意**：主线评估主要在 `PartialLVMVPTrainer` 内部完成（`_validation_metrics`, `evaluate_full_context`, `recover_hidden_on_split`, `forecast_case`），`eval/` 下的模块更多服务于旧主线。

---

## scripts/ — 实验入口

```
scripts/
├── __init__.py
├── run_partial_lv_mvp.py               # 【主线入口】（905 行）完整实验流程：
│                                         数据生成 → 噪声扫描 → 正式训练 → 评估 → 绘图 → 保存
│
├── run_train.py                        # 旧主线训练入口
├── run_eval.py                         # 旧主线评估入口
├── run_pipeline.py                     # 旧主线 pipeline 入口
├── run_ablation.py                     # 旧消融实验
├── run_hidden_inference_experiment.py  # B 线 hidden 推断实验
└── run_single_experiment.py            # B 线单物种实验
```

---

## viz/ — 可视化

```
viz/
├── __init__.py
├── chinese_experiment.py               # 中文实验可视化
├── graph_viz.py                        # 图结构可视化
├── heatmaps.py                         # 热力图
└── trajectories.py                     # 轨迹可视化
```

**注意**：主线绘图逻辑在 `scripts/run_partial_lv_mvp.py` 中内联（6 张图），`viz/` 下的模块更多服务于旧主线。

---

## runs/ — 实验输出

每次 `run_partial_lv_mvp.py` 运行会创建 `runs/{timestamp}_{experiment_name}/` 目录。

典型结构：
```
runs/20260411_115901_partial_lv_lv_guided_stochastic_refined/
├── README.md                           # 自动生成的实验说明
└── results/
    ├── summary.json                    # 完整指标 + 诊断 + 噪声配置
    ├── data_snapshot.npz               # 数据快照
    ├── fig1_true_trajectories.png
    ├── fig2_hidden_test_overlay.png
    ├── fig3_visible_rollout_compare.png
    ├── fig4_visible_fullcontext_compare.png
    ├── fig5_training_metrics.png
    └── fig6_diagnostics.png
```

当前共 26 个 run 目录。详见 `notes/experiment_status.md`。

---

## results/ — 旧版结果

```
results/
├── data_snapshot.npz                   # 旧版数据快照
├── metrics_deep.json                   # 旧版深层模型指标
├── metrics_shallow.json                # 旧版浅层模型指标
├── summary.json                        # 旧版汇总
└── fig*.png                            # 旧版图表
```

**⚠️ 此目录为旧 pipeline 输出，与当前主线无关，勿混淆。**
