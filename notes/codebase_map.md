# 代码目录与文件说明

> 最后更新：2026-04-12（精简清理后）

## 顶层结构

```
生态模拟/
├── CLAUDE.md                           # 项目入口记忆文件（AI 必读）
├── codex_iteration_log.md              # Codex 4 轮迭代实验日志
├── README.md                           # 项目简介
├── requirements.txt                    # Python 依赖
│
├── configs/                            # YAML 配置文件（2 个）
├── data/                               # 数据生成与预处理（4 个文件）
├── models/                             # 模型定义（2 个文件）
├── train/                              # 训练器 + 工具函数（2 个文件）
├── scripts/                            # 实验入口脚本（1 个文件）
├── notes/                              # 项目文档（5 个文件）
├── docs/                               # 研究描述文书
└── runs/                               # 保留的 3 个里程碑实验
```

---

## configs/ — 配置文件

| 文件 | 用途 |
|------|------|
| `partial_lv_mvp.yaml` | 原版生产配置（40 epochs, 无 v2 keys） |
| `partial_lv_mvp_v2_mechanism.yaml` | v2 机制分离配置（50 epochs, 含 multiscale/local_variance/residual_energy） |

**注意**：原版配置不含 `lambda_multiscale` 等 v2 新增键。代码通过 `.get(key, 0.0)` 兼容。

---

## models/ — 模型定义

| 文件 | 行数 | 用途 |
|------|------|------|
| `partial_lv_recovery_model.py` | ~380 | **核心模型**：4-way rollout (LV drift + residual + hidden_fast + noise) + OU 环境 |
| `encoders.py` | ~130 | 基础网络构件：MLP（GELU 激活）等 |

---

## train/ — 训练器

| 文件 | 行数 | 用途 |
|------|------|------|
| `partial_lv_mvp_trainer.py` | ~900 | **核心训练器**：17 项损失、噪声退火、full-context 训练、评估 |
| `utils.py` | ~65 | 工具函数：`set_random_seed` / `create_data_loaders` / `save_json` / `resolve_device` |

---

## data/ — 数据生成与预处理

| 文件 | 行数 | 用途 |
|------|------|------|
| `partial_lv_mvp.py` | ~520 | **合成生态系统生成器**：5 visible + 1 hidden + 1 env + 1 pulse |
| `dataset.py` | ~300 | `TimeSeriesBundle` 容器 + `build_windowed_datasets` 窗口采样 |
| `transforms.py` | ~70 | `LogZScoreTransform` 标准化 |

---

## scripts/ — 实验入口

| 文件 | 行数 | 用途 |
|------|------|------|
| `run_partial_lv_mvp.py` | ~910 | **唯一入口**：数据生成 → 噪声扫描 → 训练 → 评估 → 绘图 → 保存 |

---

## runs/ — 保留的实验结果

仅保留 3 个里程碑 run：

| Run | 意义 |
|-----|------|
| `20260411_004141_partial_lv_hidden_environment_compare` | 首次成功的 hidden+env 实验 |
| `20260411_115901_partial_lv_lv_guided_stochastic_refined` | **当前 best run**（回退至此） |
| `20260411_193943_partial_lv_lv_guided_stochastic_refined` | 最新实验结果 |

每个 run 包含 `README.md` + `results/`（summary.json + data_snapshot.npz + fig1~fig6.png）

---

## 已删除的旧文件（2026-04-12 清理）

以下文件/目录已删除，仅在 git 历史中保留：

- **旧模型**：`models/full_model.py`, `gnn.py`, `graph_builder.py`, `decoder.py`, `hidden_inference_model.py`
- **旧训练**：`train/trainer.py`（工具函数已提取至 `train/utils.py`）, `hidden_inference_trainer.py`, `train_one_step.py`, `train_rollout.py`
- **旧脚本**：`scripts/run_train.py`, `run_eval.py`, `run_pipeline.py`, `run_ablation.py`, `run_hidden_inference_experiment.py`, `run_single_experiment.py`
- **旧损失/评估**：`losses/`（整个目录）、`eval/`（整个目录）
- **旧数据**：`data/lv_simulator.py`, `data/window_sampler.py`（函数已内联至 dataset.py）
- **旧可视化**：`viz/`（整个目录，主线绘图在 run_partial_lv_mvp.py 中内联）
- **旧配置**：`base.yaml`, `baseline_*.yaml`, `ablation_*.yaml` 等 9 个
- **旧文档**：`eco_dynamics_codex_spec_zh.md`, `eco_dynamics_implementation_brief_zh.md`, `takens_embedding_spec_zh.md`
- **旧结果**：`results/` 目录
- **旧运行**：约 20 个 run 目录（空目录 + pipeline + B-line + 早期/中间实验）
