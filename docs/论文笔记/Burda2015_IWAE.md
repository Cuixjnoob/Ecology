# [Burda et al. 2015] Importance Weighted Autoencoders

## 基本信息
- **作者**: Yuri Burda, Roger Grosse, Ruslan Salakhutdinov
- **机构**: University of Toronto
- **年份**: 2015 (arXiv: 1509.00519)
- **发表**: ICLR 2016
- **领域**: 深度生成模型、变分推断、重要性采样

## 研究问题
标准VAE对后验分布做了较强假设（近似阶乘分解、参数可由神经网络从观测回归得到），这限制了模型的表达能力。VAE目标函数会惩罚那些不能很好解释数据的近似后验样本，导致学到的表示过于简化，无法充分利用网络的建模能力。如何在保持VAE架构的同时，获得更紧的对数似然下界？

## 核心方法/架构

### Importance Weighted Autoencoder (IWAE)
IWAE使用与VAE相同的网络架构（生成网络+识别网络），但采用基于重要性加权的更紧下界进行训练。

**k-样本重要性加权下界**：
$$\mathcal{L}_k(x) = \mathbb{E}_{h_1,...,h_k \sim q(h|x)}\left[\log \frac{1}{k}\sum_{i=1}^{k}\frac{p(x, h_i)}{q(h_i|x)}\right]$$

其中 h_1,...,h_k 从识别模型独立采样，p(x,h_i)/q(h_i|x) 为未归一化的重要性权重 w_i。

**关键性质**：
1. k=1时退化为标准VAE的ELBO
2. 对所有k：log p(x) >= L_{k+1} >= L_k（单调性）
3. 当 p(h,x)/q(h|x) 有界时，k趋近无穷时 L_k 趋近 log p(x)

**多层随机隐变量架构**：
- 生成过程通过层级祖先采样：p(x|theta) = sum_{h1,...,hL} p(hL)p(hL-1|hL)...p(x|h1)
- 识别模型：q(h|x) = q(h1|x)q(h2|h1)...q(hL|hL-1)
- 先验 p(hL) 为零均值单位方差高斯
- 各层条件分布为对角高斯，参数由前馈神经网络计算

**训练过程**：
- 使用重参数化技巧推导梯度：h_l(epsilon_l, h_{l-1}, theta) = Sigma^{1/2} * epsilon_l + mu
- 梯度更新基于归一化重要性权重的加权平均：sum_i w_hat_i * nabla_theta log w_i
- 计算开销与k线性增长，可通过GPU并行化处理

### 活跃潜变量维度分析
- 提出"活跃维度"度量：A_u = Cov_x[E_{u~q(u|x)}[u]]，阈值A_u > 10^{-2}
- 发现VAE和IWAE学到的活跃维度远低于总维度
- IWAE在k>1时学到更多活跃维度
- 这是目标函数驱动的（非优化问题），因为将VAE模型切换到IWAE目标会增加活跃维度，反之减少

## 关键发现与结论
1. **更好的生成性能**：IWAE在MNIST（-82.90 nats，2层k=50）和Omniglot（-103.38 nats）上取得当时最优的对数似然
2. **更丰富的潜空间**：IWAE比VAE学到更多的活跃潜变量维度，说明其潜空间表示更丰富
3. **k的影响不对称**：增加k对IWAE有显著改善，但对VAE仅有轻微帮助
4. **潜维度失活是目标函数驱动的**：不是优化困难导致，而是VAE目标函数本身倾向于关闭不必要的维度
5. **双层模型**优于单层模型，但第二层仅使用少量维度（<10个活跃）

## 重要公式

**IWAE下界**：
$$\mathcal{L}_k(x) = \mathbb{E}_{h_1,...,h_k \sim q(h|x)}\left[\log \frac{1}{k}\sum_{i=1}^{k}\frac{p(x, h_i)}{q(h_i|x)}\right]$$

**单调性定理**：
$$\log p(x) \geq \mathcal{L}_{k+1} \geq \mathcal{L}_k$$

**梯度估计器**：
$$\nabla_\theta \mathcal{L}_k \approx \sum_{i=1}^{k}\tilde{w}_i \nabla_\theta \log w(x, h(\epsilon_i, x, \theta), \theta)$$

其中 $\tilde{w}_i = w_i / \sum_j w_j$ 为归一化重要性权重。

**活跃维度度量**：
$$A_u = \text{Cov}_x\left[\mathbb{E}_{u \sim q(u|x)}[u]\right]$$

## 与我们项目的关联
IWAE的思想可以直接改进CVHI的训练：
- **更紧的ELBO**：在推断hidden species后验 q(h|X) 时，使用IWAE的多样本重要性加权下界可以获得比标准VAE更紧的对数似然估计，从而学到更准确的后验
- **解决潜维度失活问题**：生态系统中hidden species可能有多个需要推断的状态维度。VAE可能关闭某些维度，而IWAE可以保留更多活跃维度，更完整地表征hidden species的状态
- **后验灵活性**：IWAE放松了对后验的阶乘分解假设，这对于建模hidden species之间存在复杂依赖关系的场景很重要
- **评估与对比**：即使CVHI主框架使用标准VAE目标训练，也可以用IWAE下界（L_5000）作为更准确的模型评估指标

## 一句话总结
IWAE通过多样本重要性加权构造了比标准VAE更紧的对数似然下界，学到更丰富的潜空间表示，为提升VAE后验推断的灵活性和准确性提供了实用且理论严谨的方法。
