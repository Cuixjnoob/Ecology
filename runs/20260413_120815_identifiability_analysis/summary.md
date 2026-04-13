# Identifiability Analysis: Partial Observation Hidden Recovery

## 核心问题

在 partial observation 下，hidden 是否有唯一解？
如果多个 (hidden, dynamics) 组合都能 fit visible，hidden 就是 non-identifiable。

## 实验

每个数据集跑 60 次 Linear Sparse + EM，
变化 λ ∈ [0.3, 0.5, 0.7, 1.0], seed ∈ [100, 100+15)。

## 结果

### LV

- 恢复次数: 60
- 平均 Pearson vs true: 0.9806 ± 0.0039
- 平均 Pairwise |corr|: 0.9982 ± 0.0015
- Identifiability index: 1.018

### Holling

- 恢复次数: 60
- 平均 Pearson vs true: 0.6193 ± 0.1191
- 平均 Pairwise |corr|: 0.9620 ± 0.0412
- Identifiability index: 1.553

## 解读

- **idx > 1.0**: Recoveries 之间一致性很高，hidden **可辨识**
  说明所有方法都恢复出相同的 hidden (up to scale)
- **idx ≈ 1.0**: Recoveries 一致性 ≈ recovery 准确性
  可能是中等可辨识
- **idx < 0.8**: 多解性明显，hidden **non-identifiable**
  不同方法给出不同的 hidden，都能 'fit' visible

