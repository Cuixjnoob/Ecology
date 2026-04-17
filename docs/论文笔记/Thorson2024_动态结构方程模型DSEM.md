# [Thorson et al. 2024] Dynamic Structural Equation Models Synthesize Ecosystem Dynamics Constrained by Ecological Mechanisms

## 基本信息
- **作者**: James T. Thorson, Alexander G. Andrews III, Timothy E. Essington, Scott I. Large
- **期刊**: Methods in Ecology and Evolution, 15: 744-755
- **年份**: 2024
- **类型**: 研究论文

## 研究问题
如何将结构方程模型（SEM）扩展到时间序列分析中，同时容纳同步效应和滞后效应，并处理缺失数据和非正态分布？现有方法（VAR、SEM、DFA）各有局限，DSEM能否统一这些方法？

## 核心方法
### 动态结构方程模型（DSEM）
- **核心创新**: 将SEM的路径系数扩展到包含时间滞后，构建非可分（nonseparable）精度矩阵
- **数学框架**: 
  - 观测模型: y_{t,j} ~ f_j(μ_{t,j}, θ_j)
  - 线性预测器: g_j(μ_{t,j}) = α_j + x_{t,j}
  - GMRF: vec(X) ~ MVN(0, Q)，其中Q是TJ×TJ的稀疏精度矩阵
  - 精度矩阵: Q_joint = (I-Γ_joint)^T V^{-1} (I-Γ_joint)

### "箭头-滞后"符号
- 扩展SEM的箭头符号，如 x1→x2 (1) 表示变量x1对x2有1个时间步的滞后效应
- 允许任意组合同步效应和滞后效应

### 统一现有模型
- **VAR（向量自回归）**: 仅有滞后交互
- **SEM**: 仅有同步交互
- **DFA（动态因子分析）**: 潜变量随机游走+同步负荷
- **DSEM**: 以上皆为其嵌套子模型

### 案例研究
1. **模拟**: 基于Isle Royale狼-驼鹿的双物种VAR，展示DSEM在缺失数据下优于传统动态线性模型
2. **营养级联**: 加州洋流中海星→海胆→海带的级联效应，海獭同步正效应
3. **气候影响**: 北极海冰减少→冷水栖息地减少→桡足类捕食减少→阿拉斯加鳕鱼早期存活受阻

### R包: `dsem`
- 在CRAN上可用，使用"箭头-滞后"语法指定模型
- 基于TMB的GLMM框架，计算效率高

## 关键发现与结论
1. DSEM可以灵活指定同步和滞后因果效应，统一了生态学中常用的多种时间序列模型
2. 处理缺失数据的能力优于传统方法
3. 允许研究者基于生态学专业知识构建因果图，然后用数据拟合
4. 可以预测干预（pulse/press实验）的级联效应（通过Leontief矩阵）
5. 计算效率足够高，可嵌入更大的综合种群模型

## 与我们项目的关联
- **因果建模baseline**: DSEM是一种结合生态机制约束的因果时间序列方法，可作为我们GNN方法的对比baseline
- **缺失数据处理**: DSEM天然处理缺失数据的能力与我们hidden species场景直接相关
- **网络结构**: DSEM的路径系数矩阵Γ类似于我们GNN学习的邻接矩阵
- **关键差异**: DSEM假设线性关系，而生态系统本质非线性；DSEM需要先验指定因果图，而我们希望从数据中学习
- **Leontief矩阵**: 预测干预传播效应的工具，与我们评估hidden species对其他物种影响的目标相关

## 一句话总结
Thorson 2024提出了动态结构方程模型DSEM，统一了VAR、SEM、DFA等时间序列方法，可灵活指定同步和滞后因果效应并处理缺失数据，是我们项目在因果生态建模方面的重要baseline。
