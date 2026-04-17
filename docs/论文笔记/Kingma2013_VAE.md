# [Kingma & Welling 2013] Auto-Encoding Variational Bayes

## 基本信息
- **作者**: Diederik P. Kingma, Max Welling
- **机构**: Universiteit van Amsterdam
- **年份**: 2013 (arXiv: 1312.6114)
- **发表**: ICLR 2014
- **领域**: 深度生成模型、变分推断

## 研究问题
如何在具有连续潜变量且后验分布不可解析（intractable）的有向概率模型中，进行高效的近似推断和学习？传统的变分贝叶斯方法（如均场近似）要求对近似后验的期望有解析解，在一般情况下同样不可解。同时，对于大规模数据集，基于采样的方法（如MCMC）计算代价过高。

## 核心方法/架构

### 1. SGVB估计器与AEVB算法
论文提出了两大核心贡献：

**Reparameterization Trick（重参数化技巧）**：
- 核心思想：将随机变量 z ~ q_phi(z|x) 重写为确定性变换 z = g_phi(epsilon, x)，其中 epsilon ~ p(epsilon) 为辅助噪声变量
- 对于高斯情形：z = mu + sigma * epsilon，epsilon ~ N(0, I)
- 这使得对变分下界的蒙特卡洛估计可以对参数 phi 求导

**ELBO推导**：
- 边际似然分解：log p_theta(x) = KL(q_phi(z|x) || p_theta(z|x)) + L(theta, phi; x)
- 变分下界 (ELBO)：L = E_{q_phi(z|x)}[-log q_phi(z|x) + log p_theta(x,z)]
- 等价形式：L = -KL(q_phi(z|x) || p_theta(z)) + E_{q_phi(z|x)}[log p_theta(x|z)]
- 第一项为KL正则化项，第二项为重构误差的期望

**Encoder/Decoder架构**：
- **Encoder（识别模型）**: q_phi(z|x) = N(z; mu(x), sigma^2(x)I)，mu和sigma由神经网络输出
- **Decoder（生成模型）**: p_theta(x|z)，由MLP参数化（实值数据用高斯分布，二值数据用Bernoulli分布）
- **先验**: p(z) = N(0, I)，标准高斯先验

**KL散度解析解**（高斯情形）：
- 当先验和近似后验均为高斯时，KL散度有解析形式
- 仅需对重构误差进行采样估计，降低了估计方差

### 2. 训练算法
- 使用小批量SGD，minibatch大小M=100
- 每个数据点仅需L=1个采样即可
- 联合优化生成模型参数theta和推断模型参数phi

## 关键发现与结论
1. AEVB在MNIST和Frey Face数据集上均优于Wake-Sleep算法，收敛更快且达到更好的解
2. 增加潜变量维度不会导致过拟合，这归因于变分下界的正则化效果
3. 低维潜空间（如2D）可用于数据可视化，学到的流形结构有意义
4. 当minibatch足够大时，每个数据点仅需1个采样（L=1）即可有效训练

## 重要公式

**ELBO（变分下界）**：
$$\log p_\theta(x^{(i)}) \geq \mathcal{L}(\theta, \phi; x^{(i)}) = -D_{KL}(q_\phi(z|x^{(i)}) \| p_\theta(z)) + \mathbb{E}_{q_\phi(z|x^{(i)})}[\log p_\theta(x^{(i)}|z)]$$

**Reparameterization Trick**：
$$\tilde{z} = g_\phi(\epsilon, x) = \mu + \sigma \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

**SGVB估计器（Version B）**：
$$\tilde{\mathcal{L}}^B(\theta, \phi; x^{(i)}) = -D_{KL}(q_\phi(z|x^{(i)}) \| p_\theta(z)) + \frac{1}{L}\sum_{l=1}^{L}\log p_\theta(x^{(i)}|z^{(i,l)})$$

**高斯KL散度解析解**：
$$-D_{KL}(q\|p) = \frac{1}{2}\sum_{j=1}^{J}(1 + \log(\sigma_j^2) - \mu_j^2 - \sigma_j^2)$$

## 与我们项目的关联
我们的CVHI框架直接建立在VAE的理论基础之上：
- **后验推断 q(h|X)**：CVHI使用VAE的encoder结构来推断hidden species的后验分布 q(h|X)，其中X为可观测物种数据，h为隐藏物种状态。这与本文中 q_phi(z|x) 的角色完全一致
- **Reparameterization trick**：在训练CVHI时，需要通过重参数化技巧使得对hidden species后验的采样过程可微分，从而实现端到端的梯度反传
- **ELBO优化**：CVHI的训练目标同样基于ELBO，平衡重构精度（Pearson相关性）和潜变量分布的正则化（KL散度）
- **先验设计**：本文使用标准高斯先验 p(z)=N(0,I)，而CVHI可以设计更具生态学意义的先验（如考虑物种间相互作用的图结构先验），这正是GNN模块发挥作用之处

## 一句话总结
VAE通过重参数化技巧和摊销推断，实现了对连续潜变量模型的高效变分学习，是CVHI中推断隐藏物种后验分布的理论基石。
