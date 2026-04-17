# [Tucker et al. 2018] Doubly Reparameterized Gradient Estimators for Monte Carlo Objectives

## 基本信息
- **作者**: George Tucker, Dieterich Lawson, Shixiang Gu, Chris J. Maddison
- **机构**: Google Brain, New York University, University of Oxford, DeepMind
- **年份**: 2018 (arXiv: 1810.04152)
- **发表**: ICLR 2019
- **领域**: 变分推断、梯度估计、深度生成模型

## 研究问题
IWAE (Burda et al. 2015) 提供了比标准ELBO更紧的多样本变分下界，但存在一个反直觉的问题：随着样本数K的增加，推断网络梯度估计器的信噪比(SNR)趋近于零（Rainforth et al. 2018），导致推断网络训练质量下降。Roeder et al. (2017) 提出了一个改进的梯度估计器（STL），但无法证明其无偏性。本文的目标是设计一个无偏、低方差、计算高效的梯度估计器，解决IWAE随K增大的SNR退化问题。

## 核心方法/架构

### 问题分析
IWAE下界对推断网络参数phi的总导数可分解为：
$$\nabla_\phi \mathcal{L}_K = \mathbb{E}_{\epsilon_{1:K}}\left[\sum_{i=1}^{K}\frac{w_i}{\sum_j w_j}\left(-\frac{\partial}{\partial \phi}\log q_\phi(z_i|x) + \frac{\partial \log w_i}{\partial z_i}\frac{dz_i}{d\phi}\right)\right]$$

第一项（含 partial/partial phi log q）对梯度方差贡献显著。当K=1时此项期望为零，但K>1时不为零。

### DReG（Doubly Reparameterized Gradient）核心思路
对第一项应用**第二次重参数化技巧**。利用REINFORCE梯度与重参数化梯度的等价性：
$$\mathbb{E}_{q_\phi(z|x)}\left[f(z)\frac{\partial}{\partial \phi}\log q_\phi(z|x)\right] = \mathbb{E}_\epsilon\left[\frac{\partial f(z)}{\partial z}\frac{\partial z(\epsilon, \phi)}{\partial \phi}\right]$$

### IWAE-DReG估计器
经过推导和项的消去，得到简洁的最终形式：
$$\nabla_\phi \mathcal{L}_K = \mathbb{E}_{\epsilon_{1:K}}\left[\sum_{i=1}^{K}\left(\frac{w_i}{\sum_j w_j}\right)^2 \frac{\partial \log w_i}{\partial z_i}\frac{\partial z_i}{\partial \phi}\right]$$

关键性质：
- **无偏**：是IWAE下界梯度的无偏估计
- **SNR随K增大而改善**：O(sqrt(K))，与生成网络梯度一致
- **最优后验时方差为零**：当q(z|x) = p(z|x)时估计器精确为零
- **计算成本不变**：与标准IWAE梯度估计器相同

### 推广到其他方法

**RWS-DReG（Reweighted Wake-Sleep）**：
$$\text{Wake update} = \mathbb{E}_{\epsilon_{1:K}}\left[\sum_{i=1}^{K}\left(\frac{w_i^2}{(\sum_j w_j)^2} - \frac{w_i}{\sum_j w_j}\right)\frac{\partial \log w_i}{\partial z_i}\frac{\partial z_i}{\partial \phi}\right]$$

**DReG(alpha) -- 凸组合**：
$$\mathbb{E}\left[\sum_{i=1}^{K}\left(\alpha\frac{w_i}{\sum_j w_j} + (1-2\alpha)\frac{w_i^2}{(\sum_j w_j)^2}\right)\frac{\partial \log w_i}{\partial z_i}\frac{\partial z_i}{\partial \phi}\right]$$
- alpha=0: IWAE-DReG
- alpha=0.5: STL (Roeder et al. 2017)
- alpha=1: RWS-DReG

**JVI-DReG（Jackknife Variational Inference）**：
- JVI是K样本和K-1样本IWAE估计器的线性组合，可对各项分别使用DReG

## 关键发现与结论
1. **STL估计器有偏**：本文证明Roeder et al. (2017)的STL估计器在K>1时有偏，但偏差可通过第二次重参数化高效估计并消除
2. **SNR改善**：IWAE-DReG的SNR随K增大而增大（O(sqrt(K))），彻底解决了标准IWAE梯度的SNR退化问题
3. **方差显著降低**：在MNIST和Omniglot的生成建模任务中，DReG估计器在IWAE、RWS、JVI三个目标上均显著降低了梯度方差
4. **性能提升**：降低的方差转化为更好的测试对数似然，K=64时IWAE-DReG在MNIST上达到约-86.5 nats
5. **任务依赖性**：在生成建模任务中RWS-DReG表现最好，但在结构化预测任务中IWAE-DReG更稳定（RWS在后期训练中可能不稳定）
6. **通用性**：DReG是计算高效的无偏即插即用梯度估计器，可替代标准估计器

## 重要公式

**IWAE下界**：
$$\mathcal{L}_K = \mathbb{E}_{z_{1:K}}\left[\log\left(\frac{1}{K}\sum_{i=1}^{K}\frac{p_\theta(x, z_i)}{q_\phi(z_i|x)}\right)\right]$$

**IWAE-DReG梯度估计器**：
$$\nabla_\phi \mathcal{L}_K = \mathbb{E}_{\epsilon_{1:K}}\left[\sum_{i=1}^{K}\left(\frac{w_i}{\sum_j w_j}\right)^2 \frac{\partial \log w_i}{\partial z_i}\frac{\partial z_i}{\partial \phi}\right]$$

**REINFORCE-Reparameterization等价性**：
$$\mathbb{E}_{q_\phi(z|x)}\left[f(z)\frac{\partial}{\partial \phi}\log q_\phi(z|x)\right] = \mathbb{E}_\epsilon\left[\frac{\partial f(z)}{\partial z}\frac{\partial z(\epsilon, \phi)}{\partial \phi}\right]$$

## 与我们项目的关联
DReG为CVHI的训练优化提供了实用工具：
- **IWAE训练的改进**：如果CVHI采用IWAE目标训练（多样本下界），DReG估计器可以避免推断网络梯度的SNR退化问题，确保编码器 q(h|X) 在使用大K时仍能有效学习
- **稳定训练**：生态数据通常噪声大且样本有限，降低梯度方差对于训练稳定性至关重要。DReG估计器的低方差特性可以改善CVHI在小数据集上的训练
- **即插即用**：DReG不改变模型架构，仅改变梯度计算方式，可以直接应用于CVHI的现有VAE框架
- **权衡选择**：DReG(alpha)提供了IWAE-DReG和RWS-DReG的连续权衡，可根据hidden species recovery任务的特点选择最优alpha

## 一句话总结
DReG通过对IWAE梯度进行第二次重参数化，构造了无偏、低方差且计算高效的梯度估计器，解决了多样本变分下界中推断网络梯度信噪比退化的核心问题。
