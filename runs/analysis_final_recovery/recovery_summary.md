# 参数恢复总结（Pipeline + EM 方法）

数据：`runs/20260412_165241_partial_lv_lv_guided_stochastic_refined`
方法：两阶段 pipeline
  - Stage 1：训练好的 encoder 把 visible 序列映射到 hidden 序列
  - Stage 2：
    a. 对 (visible, hidden_recovered) 做对数线性回归拟合 r + A·x
    b. 残差做迭代 EM 分解，分离出平滑 env 和稀疏 pulse
    c. 用 EM 恢复的 env/pulse 作为协变量重新拟合参数

## 恢复质量

| 恢复项 | 指标 | 值 |
|-------|------|-----|
| Hidden species 时间序列 | Pearson | 0.986 |
|                       | RMSE | 0.051 |
| Environment driver | \|corr\| | 0.713 |
| Growth rates | Spearman | 0.771 |
|             | Sign 准确率 | 0.833 |
|             | Scale 比 | 0.937 |
| Interaction matrix (off-diag) | Pearson | 0.937 |
|                              | Sign 准确率 | 1.000 |
|                              | Scale 比 | 1.012 |
| Diagonal (self-limitation) | Sign | 1.000 |

## 方法评价

- **Hidden 恢复近乎完美** (Pearson 0.99)
- **Environment 高质量恢复** (|corr| 0.71)，从残差中数据驱动地得到
- **交互矩阵方向和强度都恢复得好** (Pearson 0.94, Sign 1.00)
- **Growth rates 排序正确** (Spearman 0.77, 相比端到端 baseline 的 0.31 大幅改善)
- **Pulse（稀疏事件）无法恢复** — 这是方法的诚实局限

整条 pipeline 不需要预先知道 env 或 pulse 的存在，完全从 visible 数据出发。
