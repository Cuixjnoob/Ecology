# [Zhou et al. 2020] Graph Neural Networks: A Review of Methods and Applications

## 基本信息
- **作者**: Jie Zhou, Ganqu Cui, Shengding Hu, Zhengyan Zhang, Cheng Yang, Zhiyuan Liu, Lifeng Wang, Changcheng Li, Maosong Sun
- **期刊/来源**: AI Open (Elsevier), Volume 1, 2020
- **年份**: 2020
- **机构**: 清华大学, 北京邮电大学, 腾讯

## 研究问题
如何系统性地综述图神经网络(GNN)的方法和应用？论文从设计者的角度出发，提出了一个通用的GNN设计流水线(design pipeline)，并讨论了各个组件的变体，系统性地分类了GNN的应用场景。

## 核心方法/架构

### 通用设计流水线
论文提出GNN模型设计的四个步骤：
1. **发现图结构** — 区分结构化场景(分子、物理系统、知识图谱)和非结构化场景(文本、图像)
2. **确定图类型和规模** — 有向/无向、同构/异构、静态/动态
3. **设计损失函数** — 节点级、边级、图级任务；监督/半监督/无监督设置
4. **构建计算模块** — 传播模块、采样模块、池化模块

### 计算模块详解

**传播模块 - 卷积算子：**
- **谱方法(Spectral)**：基于图信号处理，核心公式为 `g_w * x = U g_w U^T x`，代表方法包括Spectral Network、ChebNet、GCN
- **空间方法(Spatial)**：直接在图拓扑上定义卷积，代表方法包括GraphSAGE、GAT、MPNN
- **注意力机制**：GAT通过注意力权重为不同邻居分配不同权重

**传播模块 - 循环算子：**
- GNN(原始版本)使用不动点迭代
- GGNN使用GRU门控机制
- Tree-LSTM和Graph LSTM扩展到图结构

**跳跃连接(Skip Connection)：**
- Highway GCN、JKN、DeepGCNs等解决过平滑问题

**采样模块：**
- 节点采样(GraphSAGE)、层采样(FastGCN)、子图采样(ClusterGCN)

**池化模块：**
- 直接池化：max/mean/sum/attention
- 层次池化：DiffPool(可学习层次聚类)、gPool、SAGPool

### 损失函数
- 节点分类：交叉熵损失
- 链接预测：重构损失
- 图分类：图级交叉熵

## 关键发现与结论
1. GNN的核心思想是通过消息传递(message passing)在节点间传播信息，聚合特征和拓扑信息
2. 谱方法有坚实的理论基础但难以泛化到不同结构的图；空间方法更灵活但缺乏理论保障
3. GCN是谱方法和空间方法的桥梁(K=1的ChebNet近似)
4. 提出四个开放问题：模型深度、可扩展性、异质性、动态性

## 重要公式

**GCN核心公式：**
```
H = D̃^(-1/2) Ã D̃^(-1/2) X W
```
其中 `Ã = A + I_N`，`D̃_{ii} = Σ_j Ã_{ij}`

**GAT注意力权重：**
```
α_{vu} = exp(LeakyReLU(a^T[Wh_v || Wh_u])) / Σ_{k∈N_v} exp(LeakyReLU(a^T[Wh_v || Wh_k]))
```

**MPNN消息传递框架：**
```
m_{v}^{t+1} = Σ_{u∈N_v} M_t(h_v^t, h_u^t, e_{vu})
h_{v}^{t+1} = U_t(h_v^t, m_{v}^{t+1})
```

**Graph Network (GN) Block更新规则：**
```
e_{k}^{t+1} = φ^e(e_k^t, h_{r_k}^t, h_{s_k}^t, u^t)
h_{v}^{t+1} = φ^h(ē_v^{t+1}, h_v^t, u^t)
u^{t+1} = φ^u(ē^{t+1}, h̄^{t+1}, u^t)
```

## 与我们项目的关联
本文是GNN领域的经典综述，为我们的CVHI模型设计提供了系统性的指导框架：
1. **设计流水线直接适用**：我们的生态网络属于"结构化场景"，图类型为动态异构图(物种作为节点，相互作用作为边，且随时间变化)
2. **空间方法更适合我们**：由于生态网络可能在不同数据集间结构不同，谱方法的非泛化性限制了应用；GraphSAGE和GAT的归纳学习(inductive learning)能力对我们处理不同生态系统更有价值
3. **消息传递框架**：MPNN框架可以自然地建模species之间的信息传递(如捕食关系、竞争关系)，这与我们的hidden recovery任务中推断隐藏物种的影响完全吻合
4. **池化模块**：在图级任务中，DiffPool等层次池化方法可能帮助我们从物种级特征聚合到群落级特征
5. **动态图处理**：论文提到的动态图方法对我们建模时间序列中的生态网络变化具有参考价值

## 一句话总结
GNN方法与应用的系统性综述，提出通用设计流水线，涵盖谱方法/空间方法/注意力机制等核心计算模块，为我们设计CVHI模型的GNN架构提供了全面的方法论参考。
