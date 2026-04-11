# 实验状态

> 最后更新：2026-04-11  
> 数据来源：`runs/` 目录、`summary.json` 文件、`codex_iteration_log.md`

---

## 总览

仓库中共 26 个实验运行目录，归属三条实验线。当前主线是 **C 线（partial-observation 联合 recovery）**。

---

## 实验线分类

### A 线：旧图模型 forecast
**相关脚本**：`run_train.py` / `run_pipeline.py` / `run_ablation.py`

| 目录 | summary | 状态 |
|------|---------|------|
| `pipeline/` | 无 | 历史线 |
| `pipeline_quick_verify/` | 无 | 历史线 |
| `pipeline_full_verify/` | 无 | 历史线 |

**说明**：使用 `EcoDynamicsModel`（GNN + message passing），以 visible forecast 为主目标。与当前主线不共享模型或训练器。已弃用。

---

### B 线：hidden-only recovery
**相关脚本**：`run_hidden_inference_experiment.py` / `run_single_experiment.py`

| 目录 | summary | 状态 |
|------|---------|------|
| `hidden_inference_verify/` | 无（有 `metrics.json` + `best.pt`）| 阶段验证 |
| `hidden_inference_verify_v2/` | 无 | 阶段验证 |
| `hidden_inference_verify_v3/` | 无 | 阶段验证 |
| `single_lv_hidden_experiment_verify/` | 无 | 阶段验证 |
| `single_lv_hidden_experiment_verify_v2/` | 无 | 阶段验证 |

**说明**：使用 `HiddenSpeciesInferenceModel`（GNN-based），专注 hidden species 推断。提供了"hidden 可以被恢复"的早期验证，但不处理 visible future 和 environment disentanglement。

---

### C 线：partial-observation 联合 recovery（当前主线）
**相关脚本**：`run_partial_lv_mvp.py`

#### C1. 早期 context 实验

| 目录 | summary | README | 关键指标 |
|------|---------|--------|----------|
| `20260411_001734_partial_lv_mvp_context` | ✅ | ❌ | 最早的 mvp 尝试 |
| `20260411_001808_partial_lv_mvp_context` | ✅ | ✅ | — |
| `20260411_001843_partial_lv_mvp_context` | ✅ | ✅ | — |

**说明**：初期探索，验证 delay encoding + hidden head 基础架构。

#### C2. hidden/environment 对比实验

| 目录 | summary | README |
|------|---------|--------|
| `20260411_004141_partial_lv_hidden_environment_compare` | ✅ | ✅ |

**说明**：引入 environment latent 的关键对比实验。对比了 Model A（hidden-only）vs Model B（hidden + environment）：
- Model A：hidden RMSE 0.066, Pearson 0.988，但 env correlation 为 0
- Model B：hidden RMSE 0.100, Pearson 0.983，env correlation 0.912（过强纠缠）
- **结论**：environment latent 有必要，但需要 disentanglement 约束

#### C3. LV-guided stochastic 实验

| 目录 | summary | README |
|------|---------|--------|
| `20260411_012137_partial_lv_lv_guided_stochastic` | ✅ | ✅ |

**说明**：引入 LV soft guidance + stochastic forward 的关键转折点。在此之前模型是纯神经网络 rollout，之后变为 LV drift + residual + noise 三路分工。

#### C4. refined 主线迭代（13 个运行）

##### Pre-Codex-iteration（早期 refined，由 Codex 自动调优产生）

| 目录 | summary | README | 说明 |
|------|---------|--------|------|
| `20260411_014827_…_refined` | ✅ | ✅ | 首个 refined run |
| `20260411_031108_…_refined` | ✅ | ✅ | 噪声/解耦修正 |
| `20260411_041311_…_refined` | ❌ | ❌ | 不确定，可能是中间调试 |
| `20260411_041650_…_refined` | ❌ | ❌ | 不确定，可能是中间调试 |
| `20260411_115546_…_refined` | ❌ | ❌ | 不确定，可能是中间调试 |
| `20260411_115901_…_refined` | ✅ | ✅ | **当前接受的 best run** |

##### Codex 4 轮迭代（详见 `codex_iteration_log.md`）

| 目录 | 对应迭代 | summary | 决策 | 要点 |
|------|----------|---------|------|------|
| `20260411_181324_…_refined` | — | ❌ | 不确定 | 可能是预编译/调试 |
| `20260411_181359_…_refined` | — | ❌ | 不确定 | 同上 |
| `20260411_181512_…_refined` | Iteration 1 | ✅ | **revert** | 硬限 residual budget → hidden 崩溃 |
| `20260411_181816_…_refined` | Iteration 2 | ✅ | **revert** | 多 cut-point full-context → 全面恶化 |
| `20260411_182236_…_refined` | Iteration 3 | ✅ | **partial keep** | structured channel → visible 大幅改善但 hidden 崩溃 |
| `20260411_182835_…_refined` | Iteration 4 | ✅ | **revert** | 约束 hidden-visible path → amplitude 崩溃 |
| `20260411_193943_…_refined` | 迭代后最终 | ✅ | 回退至 115901 后的确认运行 | visible RMSE 1.20（比 best 差，确认回退正确） |

---

## 当前接受的 best run

**`runs/20260411_115901_partial_lv_lv_guided_stochastic_refined`**

来源：`codex_iteration_log.md` 在 Iteration 4 结束后明确记录：
> "Keep 20260411_115901_partial_lv_lv_guided_stochastic_refined as the accepted best run."

| 指标 | 值 |
|------|-----|
| sliding visible RMSE | 0.794 |
| full-context visible RMSE | 0.807 |
| hidden RMSE | 0.166 |
| hidden Pearson | 0.902 |
| amplitude collapse | 0.058 |
| hidden/env correlation | 0.099 |
| LV/residual ratio mean | 1.645 |
| residual dominates fraction | 0.871 |

---

## 实验推荐阅读顺序

如果第一次接手仓库，建议按此顺序阅读 summary.json：

1. **`20260411_004141_…_compare`** → 理解为什么引入 environment latent
2. **`20260411_012137_…_stochastic`** → 理解为什么加入 LV-guided stochastic
3. **`20260411_115901_…_refined`** → 当前 best run，理解 refined 主线
4. **`codex_iteration_log.md`** → 理解 4 轮迭代的失败与教训

---

## 混乱点与不确定项

1. 顶层 `results/` 保留旧 pipeline 输出，容易误认为当前主线
2. `041311` / `041650` / `115546` / `181324` / `181359` 五个目录无 summary.json，身份不确定（可能是调试中断或编译测试）
3. 仓库中没有集中式的 best_run 声明文件，best run 信息来自 `codex_iteration_log.md` 的文字记录
4. `193943` 是迭代后的确认运行，指标比 `115901` 差，说明代码在回退后重新运行时结果有随机性差异
