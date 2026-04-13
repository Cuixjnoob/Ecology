# 真实数据 (Hudson Bay lynx-hare) 初步结果

数据: 57 年 (1847-1903), 2 物种

## Method A: Hare (N=1) → Recover Lynx

- BEST: λ=0.0, Pearson=0.0024, RMSE=19.24

## Method B: Hare Takens 嵌入 (N=4) → Recover Lynx

- BEST: λ=2.0, Pearson=0.1820, RMSE=19.04

## 注意

- 这是 2 物种数据，不完全符合我们原框架的 5 visible + 1 hidden
- 两种 adapter: 单物种 or Takens 嵌入
- 结果作为 **proof-of-concept**，不是 main result
