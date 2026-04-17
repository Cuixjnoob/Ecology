# [Girin et al. 2021] Dynamical Variational Autoencoders: A Comprehensive Review

## 基本信息
- **作者**: Laurent Girin, Simon Leglaive, Xiaoyu Bie, Julien Diard, Thomas Hueber, Xavier Alameda-Pineda
- **机构**: Univ. Grenoble Alpes, CNRS, GIPSA-lab; CentraleSupelec; Inria
- **年份**: 2020 (arXiv: 2008.12595)
- **发表**: Foundations and Trends in Machine Learning, Vol. 15, No. 1-2, pp 1-175 (2021)
- **领域**: 深度生成模型综述、动态变分自编码器、序列建模

## 研究问题
标准VAE独立处理输入数据，不考虑序列中数据向量之间的时间依赖性。近年来一系列工作将VAE扩展到序列数据处理，但这些工作使用不同的记号和表述方式，缺乏统一的框架和系统比较。本综述旨在：
1. 建立统一的DVAE模型类定义
2. 用一致的记号详细介绍七种代表性DVAE模型
3. 将这些模型与经典的时序模型（RNN、状态空间模型）联系
4. 提供实验基准比较

## 核心方法/架构

### DVAE的统一框架

#### 分类体系
论文建立了从概率生成模型到DVAE的层次分类：
- 概率生成模型 -> 显式PDF(prescribed) vs 隐式(implicit)
- 显式 -> 贝叶斯网络(BN) -> 动态贝叶斯网络(DBN)
- 深度化 -> 深度BN -> VAE
- 时序扩展 -> 深度动态BN(DDBN) -> DVAE

#### 生成模型的通用形式
DVAE的生成模型包含：
- **潜变量 z_{1:T}**：低维表示
- **观测变量 x_{1:T}**：高维序列数据
- **确定性状态**（可选）：RNN隐状态等

依赖结构的关键选择：
1. 潜变量之间是否有时间依赖？（马尔可夫转移 vs 独立）
2. 观测是否仅依赖当前潜变量？（状态空间假设 vs 自回归）
3. 是否使用确定性循环状态？

#### 推断模型
近似后验 q(z_{1:T}|x_{1:T}) 的不同分解方式：
- 滤波型：q(z_t|x_{1:t})
- 平滑型：q(z_t|x_{1:T})
- 条件独立型 vs 序列依赖型

#### 变分下界(VLB)
DVAE的训练目标是最大化：
$$\text{VLB} = \mathbb{E}_{q(z_{1:T}|x_{1:T})}\left[\log \frac{p_\theta(x_{1:T}, z_{1:T})}{q_\phi(z_{1:T}|x_{1:T})}\right]$$

### 七种DVAE模型详解

#### 1. Deep Kalman Filter (DKF)
- 生成模型满足状态空间假设
- 潜变量有马尔可夫转移
- 识别模型使用双向RNN进行平滑推断

#### 2. Kalman Variational Autoencoder (KVAE)
- 结合线性高斯状态空间模型和VAE
- 潜空间分为两层：低层由VAE编码/解码，高层遵循线性动态

#### 3. STOchastic Recurrent Networks (STORN)
- 在RNN中引入独立的潜随机变量序列
- 潜变量先验在时间步间独立
- 可视为VRNN-I的特例

#### 4. Variational Recurrent Neural Networks (VRNN)
- 每个时间步包含一个以RNN隐状态为条件的VAE
- 关键创新：条件先验 p(z_t|h_{t-1}) 引入时间依赖
- 详见单独的VRNN论文笔记

#### 5. Stochastic Recurrent Neural Networks (SRNN)
- 使用分离的确定性和随机路径
- 在推断时使用反向RNN进行平滑
- 比VRNN具有更灵活的推断模型

#### 6. Recurrent Variational Autoencoders (RVAE)
- 基于状态空间模型假设
- 引入线性潜动态约束
- 类似KVAE但处理方式不同

#### 7. Disentangled Sequential Autoencoders (DSAE)
- 将潜空间分解为静态因子（序列级）和动态因子（帧级）
- 适合语音等具有说话人特征和内容变化的数据

### 自回归DVAE的额外分类
- **teacher-forced**: 训练时使用真实观测，测试时使用生成的观测
- **full generative**: 训练和测试时都使用生成的观测

## 关键发现与结论
1. **统一视角**：所有七种DVAE模型可在统一框架下理解，关键差异在于依赖结构的选择
2. **确定性vs随机性路径的权衡**：使用确定性RNN路径（如VRNN、SRNN）可以增强表达能力，但可能导致潜变量不被充分利用
3. **状态空间假设的重要性**：强制执行马尔可夫假设（如DKF、DVBF）有助于获得可解释的完整信息潜空间
4. **KL消失问题**：当解码器足够强大时，模型可能忽略潜变量（KL趋近零），这是DVAE面临的普遍挑战
5. **语音实验基准**：在语音分析-再合成任务上，SRNN和VRNN表现最好
6. **3D人体运动**：在运动数据上，各模型性能差异较小

## 重要公式

**DVAE统一VLB**：
$$\text{VLB} = \mathbb{E}_{q_\phi(z_{1:T}|x_{1:T})}\left[\sum_{t=1}^{T}\log p_\theta(x_t|z_t, \text{pa}(x_t)) - \text{KL}(q_\phi(z_t|\text{pa}_q(z_t)) \| p_\theta(z_t|\text{pa}_p(z_t)))\right]$$

其中 pa(.) 表示各模型中不同的父节点依赖。

**马尔可夫转移**：
$$p_\theta(z_t|z_{t-1}) \text{ (状态空间模型)}$$

**条件先验（VRNN型）**：
$$p(z_t|h_{t-1}) \text{ 其中 } h_t = f(x_t, z_t, h_{t-1})$$

**独立先验（STORN型）**：
$$p(z_t) = \mathcal{N}(0, I) \text{ (时间步间独立)}$$

## 与我们项目的关联
这篇综述对CVHI的架构设计有全面的指导意义：
- **架构选择指南**：综述系统比较了七种DVAE架构的优劣，可帮助我们为CVHI选择最合适的时序VAE变体。考虑到hidden species recovery需要完整信息的潜空间，DKF/DVBF类的状态空间模型可能更合适
- **依赖结构设计**：CVHI需要决定hidden species的时序依赖结构——是马尔可夫转移 p(h_t|h_{t-1}) 还是通过RNN隐状态间接建模？综述的分类框架直接适用
- **KL消失问题**：如果CVHI的GNN解码器足够强大，可能导致VAE忽略hidden species的潜变量。综述讨论的各种缓解策略（KL退火、free bits等）可直接借鉴
- **推断模型选择**：滤波（仅用过去数据）vs 平滑（用全序列）的选择与生态预测场景直接相关——实时监测需要滤波，历史分析可用平滑
- **编码器-解码器统一视角**：将CVHI的GNN编码器和VAE解码器纳入DVAE的统一框架中理解

## 一句话总结
本综述建立了动态VAE (DVAE) 的统一理论框架，系统比较了七种代表性模型在生成结构、推断模型和训练目标上的异同，为设计时序潜变量模型提供了全面的参考和实验基准。
