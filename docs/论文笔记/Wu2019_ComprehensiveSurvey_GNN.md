# [Wu et al. 2019] A Comprehensive Survey on Graph Neural Networks

## 基本信息
- **作者**: Zonghan Wu, Shirui Pan, Fengwen Chen, Guodong Long, Chengqi Zhang, Philip S. Yu
- **期刊/来源**: IEEE Transactions on Neural Networks and Learning Systems (TNNLS), arXiv:1901.00596
- **年份**: 2019 (arXiv), 发表于IEEE TNNLS
- **机构**: University of Technology Sydney, Monash University, University of Illinois at Chicago

## 研究问题
如何对图神经网络(GNN)进行全面系统的综述？论文提出了一种新的分类法(taxonomy)，将GNN分为四大类，并讨论了各类方法的核心思想、代表模型及应用。

## 核心方法/架构

### 四大分类

**1. 循环图神经网络 (RecGNNs)**
- 核心思想：使用相同参数递归更新节点状态直至收敛
- GNN*(Scarselli 2009)：使用收缩映射的不动点迭代
  - `h_v^(t) = Σ_{u∈N(v)} f(x_v, x_e, x_u, h_u^(t-1))`
- GGNN(Li et al. 2015)：使用GRU替代收缩映射，固定步数
  - `h_v^(t) = GRU(h_v^(t-1), Σ_{u∈N(v)} W h_u^(t-1))`
- SSE：使用加权平均实现随机异步更新

**2. 卷积图神经网络 (ConvGNNs)**
分为谱方法和空间方法：

*谱方法核心公式：*
```
x *_G g_θ = U g_θ U^T x   (基本谱卷积)
```

- Spectral CNN: `g_θ = Θ(k)` 可学习对角矩阵，O(n³)
- ChebNet: 用Chebyshev多项式近似，K阶局部化
  - `x *_G g_θ = Σ_{i=0}^K θ_i T_i(L̃) x`
- GCN: K=1的ChebNet简化
  - `H = f(Ā X Θ)`，其中 `Ā = D̃^{-1/2} Ã D̃^{-1/2}`
- CayleyNet: 使用Cayley多项式，捕获窄频带

*空间方法核心公式：*

- MPNN框架: `h_v^(k) = U_k(h_v^(k-1), Σ_{u∈N(v)} M_k(h_v^(k-1), h_u^(k-1), x_e))`
- GraphSAGE: `h_v^(k) = σ(W^(k) · f_k(h_v^(k-1), {h_u^(k-1), ∀u ∈ S_{N(v)}}))`
- GAT: `h_v^(k) = σ(Σ_{u∈N(v)∪v} α_vu^(k) W^(k) h_u^(k-1))`
  - 注意力权重: `α_vu = softmax(g(a^T[Wh_v || Wh_u]))`
- GIN: `h_v^(k) = MLP((1+ε^(k))h_v^(k-1) + Σ_{u∈N(v)} h_u^(k-1))`

**3. 图自编码器 (GAEs)**
- 网络嵌入：编码器用GCN层，解码器重建邻接矩阵
- 图生成：逐步生成节点和边，或一次性输出整个图

**4. 时空图神经网络 (STGNNs)**
- 同时考虑空间依赖和时间依赖
- 结合图卷积(空间)与RNN/CNN(时间)

### 训练效率改进
| 方法 | 时间复杂度 | 内存复杂度 | 特点 |
|------|-----------|-----------|------|
| GCN (全批次) | O(KNF²+KMF) | O(KNF+MF) | 基线 |
| GraphSAGE | O(N·r^K·F²) | O(N·r^K·F) | 内存换时间 |
| FastGCN | O(K·s·F²+K·N·F) | O(K·s·F) | 层采样 |
| Cluster-GCN | O(KNF²+KMF) | O(K·b·F) | 子图采样，最低内存 |

## 关键发现与结论
1. RecGNNs和ConvGNNs的核心区别：RecGNN各层共享参数，ConvGNN各层使用不同参数
2. GCN同时可以从谱和空间两个角度理解——它是谱方法和空间方法的桥梁
3. 空间方法由于效率高、灵活性强、泛化能力好，近年来发展更快
4. GIN证明了之前的MPNN方法无法区分某些不同的图结构
5. 提出四个未来方向：模型深度、可扩展性权衡、异质性、动态性

## 重要公式

**GCN完整公式：**
```
H = f(Ā X Θ)
Ā = D̃^{-1/2} Ã D̃^{-1/2}
Ã = A + I_n
```

**GAT注意力权重：**
```
α_vu^(k) = softmax(g(a^T [W^(k) h_v^(k-1) || W^(k) h_u^(k-1)]))
```

**DCNN扩散卷积：**
```
H^(k) = f(W^(k) ⊙ P^k X)
```
其中 P = D^{-1}A 是转移概率矩阵。

**时空图定义：**
```
G^(t) = (V, E, X^(t))，X^(t) ∈ R^{n×d}
```

## 与我们项目的关联

1. **四分类法对CVHI设计的启发**：
   - 我们的模型属于"ConvGNN + STGNN"的混合类型
   - 空间维度：物种间交互(ConvGNN建模)
   - 时间维度：种群动态变化(RNN/TCN建模)

2. **GAE与隐变量推断**：
   - 图自编码器(GAE)的思想可以用于学习隐藏物种的潜在表示
   - 编码器：从部分观测的生态网络学习潜在嵌入
   - 解码器：从潜在嵌入重建完整的物种交互网络(包括隐藏物种)

3. **时空图的直接对应**：
   - 论文定义的时空图 `G^(t) = (V, E, X^(t))` 完美对应我们的生态时间序列数据
   - 节点属性 `X^(t)` 是时刻t各物种的种群大小

4. **训练效率**：
   - 如果生态网络规模较大(多物种系统)，Cluster-GCN等采样策略可以帮助训练

5. **GIN的表达能力**：
   - GIN具有最强的图区分能力，可能帮助我们的模型区分不同的生态网络结构

## 一句话总结
GNN领域最全面的综述之一，提出RecGNN/ConvGNN/GAE/STGNN四分类法，系统比较了各类方法的优劣，其中STGNN和GAE的思想对我们建模生态时间序列和推断隐藏物种直接适用。
