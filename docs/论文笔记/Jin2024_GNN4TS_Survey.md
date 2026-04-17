# [Jin et al. 2024] A Survey on Graph Neural Networks for Time Series: Forecasting, Classification, Imputation, and Anomaly Detection

## 基本信息
- **作者**: Ming Jin, Huan Yee Koh, Qingsong Wen, Daniele Zambon, Cesare Alippi, Geoffrey I. Webb, Irwin King, Shirui Pan
- **期刊/来源**: IEEE Transactions (arXiv:2307.03759), 项目页面: https://github.com/KimMeen/Awesome-GNN4TS
- **年份**: 2024 (v3更新于2024年8月)
- **机构**: Griffith University, Monash University, Squirrel AI, USI, CUHK

## 研究问题
如何系统地综述GNN在时间序列分析中的应用？论文涵盖四大任务维度：预测(Forecasting)、分类(Classification)、异常检测(Anomaly Detection)和插补(Imputation)。这是首篇全面覆盖GNN4TS所有主流任务的综述。

## 核心方法/架构

### 时空图定义
- **属性图**: `G = (A, X)`，A为邻接矩阵，X为节点特征矩阵
- **时空图**: `G = {G_1, G_2, ..., G_T}`，每个 `G_t = (A_t, X_t)`
- 图结构可以固定或随时间演化

### GNN核心操作
```
a_i^(k) = AGGREGATE^(k)({h_j^(k-1) : v_j ∈ N(v_i)})
h_i^(k) = COMBINE^(k)(h_i^(k-1), a_i^(k))
```

### 图结构获取策略

**启发式方法：**
1. **空间邻近性**: `A_{i,j} = 1/d_{ij}`（基于地理距离）
2. **成对连接性**: `A_{i,j} = 1` 若直接连接（如道路网络）
3. **成对相似性**: `A_{i,j} = (x_i^T x_j) / (||x_i|| ||x_j||)`（余弦相似度/Pearson相关/DTW）
4. **函数依赖**: Granger因果关系、转移熵等

**学习方法：**
- 基于嵌入: `A_{i,j} = ReLU(Θ_i^T Θ_j)`
- 基于注意力: 通过注意力分数定义A
- 稀疏化: ReLU激活或top-k选择

### 四大任务的统一框架

**1. 时间序列预测 (Forecasting)**
```
θ*, ϕ* = arg min_{θ,ϕ} L_F(p_ϕ(f_θ(X_{t-T:t}, A_{t-T:t})), Y)
```
- 单步预测 vs 多步预测
- 短期预测 vs 长期预测
- 确定性 vs 概率性

**2. 异常检测 (Anomaly Detection)**
- 在正常数据上训练重建/预测模型
- 异常时模型无法最小化差异 → 检测到异常
- 阈值设置是关键超参数

**3. 插补 (Imputation)**
```
θ*, ϕ* = arg min_{θ,ϕ} L_I(p_ϕ(f_θ(X̃_{t-T:t}, A_{t-T:t})), X_{t-T:t})
```
- 样本内插补 vs 样本外插补
- 确定性(如GRIN) vs 概率性(如PriSTI)

**4. 分类 (Classification)**
```
θ*, ϕ* = arg min_{θ,ϕ} L_C(p_ϕ(f_θ(X, A)), Y)
```
- Series-as-Graph: 序列变图，图分类
- Series-as-Node: 序列作节点，节点分类

### 方法论分类 (STGNN三维度)

**空间模块：**
- 谱GNN: ChebConv多项式近似
- 空间GNN: 消息传递或图扩散
- 混合方法

**时间模块：**
- 时域: 循环(RNN/GRU)、卷积(TCN)、注意力(Transformer)
- 频域: 傅里叶变换

**整体架构：**
- 离散因子化: 空间和时间独立处理 (如STGCN)
- 离散耦合: 空间和时间交织 (如DCRNN)
- 连续因子化: 部分使用神经ODE (如STGODE)
- 连续耦合: 完全使用神经ODE (如MTGODE)

### 代表性方法总结
| 方法 | 年份 | 空间模块 | 时间模块 | 是否需要预定义图 |
|------|------|---------|---------|----------------|
| DCRNN | 2018 | 图扩散 | GRU | 是 |
| STGCN | 2018 | ChebConv | 门控时间卷积 | 是 |
| Graph WaveNet | 2019 | 图扩散 | TCN | 可选(可学习) |
| MTGNN | 2020 | 消息传递 | TCN | 否(学习) |
| AGCRN | 2020 | GCN变体 | GRU | 否(学习) |
| STGODE | 2021 | 消息传递 | ODE+TCN | 是 |
| MTGODE | 2022 | 消息传递 | ODE | 否(学习) |

## 关键发现与结论
1. GNN能显式建模变量间(inter-variable)和时间间(inter-temporal)的关系，这是传统方法无法做到的
2. 学习图结构(而非预定义)可以发现数据驱动的隐含关系，可能比启发式方法更有效
3. 连续架构(基于Neural ODE)可以更好地刻画长程时空依赖
4. 插补任务中GNN可以利用空间邻居信息填补缺失值——这对不完整观测数据非常重要
5. 未来方向：可扩展性、可解释性、因果推理、预训练模型

## 重要公式

**STGCN时空块：**
```
空间: X̂_t = SPATIAL(X_t, A_t) = ChebConv(X_t, A)
时间: X̂ = TEMPORAL(X̂_t) = GLU(Conv1D(X̂_t))
```

**Graph WaveNet自适应图学习：**
```
A_{adaptive} = softmax(ReLU(E_1 · E_2^T))
```
其中 E_1, E_2 是可学习的节点嵌入矩阵。

**GRIN插补 (图循环插补网络)：**
利用空间GNN和GRU的组合，在有缺失值的时间序列上进行端到端训练。

## 与我们项目的关联

**这是与我们项目关联最紧密的综述论文：**

1. **Hidden Recovery就是一个"插补+预测"联合任务**：
   - 隐藏物种的种群动态可以视为"缺失时间序列"
   - 可观测物种提供的空间(交互)信息可以帮助推断隐藏物种
   - 论文中的GRIN等插补方法直接启发我们的模型设计

2. **图结构学习至关重要**：
   - 在生态系统中，物种间的真实交互关系通常未知
   - 学习型图结构(如Graph WaveNet的自适应邻接矩阵)可以让CVHI自动发现物种间的隐含关系
   - Pearson相关/Granger因果等启发式方法可作为基线

3. **STGNN架构选择**：
   - 我们的CVHI可以采用"离散耦合"架构(类似DCRNN)
   - 空间模块建模物种间交互，时间模块建模种群动态

4. **评估指标**：
   - 论文中的预测任务使用RMSE和MAE
   - 但我们的项目优先关注Pearson相关(趋势一致性)，这与论文的标准有所不同

5. **异常检测的启示**：
   - 论文中的异常检测方法可用于检测生态系统中的突变事件(如物种灭绝、入侵)

## 一句话总结
GNN时间序列分析的最全面综述，涵盖预测/分类/插补/异常检测四大任务，其中的STGNN架构设计和插补方法论直接指导了我们CVHI模型对隐藏物种动态的推断。
