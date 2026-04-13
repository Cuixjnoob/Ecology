# 下一步计划

> 最后更新：2026-04-12（方法论转向后重排优先级）
> 核心导向：**hidden recovery-centric**，不再追求 visible future prediction

---

## 优先级 1：验证 interaction matrix 恢复质量

**做什么**：对比 A/C 两个 run 的 interaction_pred（从 `data_snapshot.npz`），统计 hidden 边的符号准确率和幅值误差。

**为什么排第一**：2×2 实验表明 LV 先验对 hidden recovery 影响很小。如果 interaction matrix 也恢复得不好，LV 先验就失去了唯一的实际价值，可以考虑彻底去掉。反之，如果 LV 数据上 matrix 准确但非线性数据上不准，那是可解释性成立的证据。

**预计耗时**：~10 分钟（只是分析已有 run 的数据）

---

## 优先级 2：动力学恢复评估

**做什么**：新增一个评估函数：从训练段末尾取 hidden_initial，做 rollout 几十步，对比 rollout 出的 hidden 轨迹和真值。

**为什么重要**：验证 hidden 是否具有"动力学一致性"，而不只是 encoder 统计映射。这是区分"好模型"和"查表模型"的关键测试。

**预计耗时**：~1 小时（代码改动 + 4 个 run 重跑）

---

## 优先级 3：多 seed 实验

**做什么**：用 seed=43,44,45,46,47 跑 2×2，统计差异的显著性。

**为什么重要**：当前 Pearson 0.987 vs 0.989 的差异需要统计显著性。

---

## 已完成的方向（降级到归档）

### ✅ 用新 val_score 跑基线实验

**做什么**：用 v1 配置 + 新的 hidden-centric val_score 跑完整 40 epochs 实验。

**为什么排第一**：
- 方法论转向后，需要建立新的基线数据
- 旧 best run (115901) 用的是 visible-centric val_score，模型选择标准不同
- 新 val_score 可能选出不同的 best epoch，hidden 指标可能更好

**预计耗时**：~20-30 分钟

**依赖**：无

**命令**：`python -m scripts.run_partial_lv_mvp --config configs/partial_lv_mvp.yaml`

---

## 优先级 2：消融实验 — hidden_fast_network 的独立贡献

**做什么**：关闭 hidden_fast_network（设 hidden_fast_scale=0），对比 hidden recovery 变化。

**为什么排第二**：
- 直接验证架构组件对核心目标的贡献
- 如果无贡献 → 简化架构；有贡献 → 确认价值

**预计耗时**：~20 分钟

---

## 优先级 3：消融实验 — OU 环境 vs 直接预测

**做什么**：将环境更新从 OU 改为直接神经网络预测。

**为什么重要**：验证 OU 过程对 hidden/env disentanglement 的贡献。

---

## 优先级 4：visible-only baseline 对比

**做什么**：训练无 hidden/env 的模型，测量"加入 hidden 后对已知数据重构的额外解释力"。

**为什么重要**：项目核心主张"额外解释力"的直接检验。

**注意**：方法论转向后，对比方式变为"with-hidden vs without-hidden 在已知数据重构上的差异"，不再是"预测未来的差异"。

---

## 优先级 5：多种子验证

**做什么**：用 seed=43,44,45,46,47 跑 5 次，统计 hidden Pearson 的均值和标准差。

**为什么重要**：当前 hidden Pearson 0.90 只在 seed=42 上验证过。

---

## 暂缓的方向

| 方向 | 原因 |
|------|------|
| v2 配置实验 | 主要改善 visible，不再是核心目标 |
| Structured visible driver | 只在追求 visible 改善时有意义 |
| 真实数据接入 | 合成数据验证未完成 |
| 多 hidden species | 架构复杂度过高 |
| Particle rollout 优化 | 非当前瓶颈 |
