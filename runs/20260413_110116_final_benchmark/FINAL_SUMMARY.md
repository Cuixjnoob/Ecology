# 最终实验总结

## 核心发现

1. **稀疏约束的线性 baseline + EM 是最佳方法**，远超各种 GNN 架构。
2. **GNN 在 partial observation 下 identifiability 崩溃**（hidden 塌缩为常数）。
3. **稀疏性先验在 LV 和非线性（Holling）数据上都有效**。

## 方法对比 (严格无 hidden 监督)

| 方法 | LV Pearson | LV RMSE | Holling Pearson | Holling RMSE |
|------|-----------|---------|-----------------|-------------|
| Linear Sparse + EM | 0.9773 | 0.0620 | 0.8762 | 0.2851 |
| SINDy Library + EM | 0.5310 | 0.2476 | 0.2803 | 0.5678 |
| HNSR Hybrid | 0.0633 | 0.2915 | 0.1299 | 0.5878 |
| SparseHybridGNN | 0.1042 | 0.2905 | N/A | N/A |
| LinearSeededGNN | 0.1387 | 0.2894 | 0.2240 | 0.5765 |

## 研究叙事

### 关键科学发现

**在部分观测生态动力学中，hidden 物种的 identifiability 要求 baseline 受到严格的稀疏约束。** 高容量 deep GNN 反而会因吸收 hidden signal 到 baseline 中而导致 identifiability 崩溃。稀疏 LV 先验（L1 on A）即使在非 LV 数据（Holling）上也有效。

### 论文价值

- **反直觉结论**: 简单线性方法 > 深度 GNN
- **理论意义**: 展示了 partial observation identifiability 的 capacity-sparsity tradeoff
- **实用意义**: 方法简单、快速、可解释、易推广
