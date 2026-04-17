# [Chung et al. 2015] A Recurrent Latent Variable Model for Sequential Data

## 基本信息
- **作者**: Junyoung Chung, Kyle Kastner, Laurent Dinh, Kratarth Goel, Aaron Courville, Yoshua Bengio
- **机构**: Universite de Montreal
- **年份**: 2015 (arXiv: 1506.02216)
- **发表**: NeurIPS 2015
- **领域**: 序列生成模型、变分推断、循环神经网络

## 研究问题
如何将潜在随机变量引入RNN的隐状态，以更好地建模高度结构化序列数据（如自然语音）中的复杂变异性？标准RNN的转移函数完全确定性，唯一的随机性来源仅在条件输出概率模型中，这对于建模强依赖性、高变异性的序列数据是不充分的。

## 核心方法/架构

### VRNN（Variational Recurrent Neural Network）
VRNN在每个时间步包含一个VAE，且这些VAE以RNN隐状态 h_{t-1} 为条件：

**1. 条件先验（Conditional Prior）**：
- 先验不再是标准高斯，而是依赖于RNN隐状态的条件分布
- z_t ~ N(mu_{0,t}, diag(sigma^2_{0,t}))，其中 [mu_{0,t}, sigma_{0,t}] = phi^prior(h_{t-1})
- 这是与STORN等方法的关键区别：引入了潜变量之间的时间依赖性

**2. 生成分布**：
- x_t | z_t ~ N(mu_{x,t}, diag(sigma^2_{x,t}))
- [mu_{x,t}, sigma_{x,t}] = phi^dec(phi^z(z_t), h_{t-1})
- 既依赖于潜变量z_t，也依赖于RNN隐状态h_{t-1}

**3. RNN更新**：
- h_t = f(phi^x(x_t), phi^z(z_t), h_{t-1})
- 隐状态h_t是x_{<=t}和z_{<=t}的函数

**4. 近似后验**：
- z_t | x_t ~ N(mu_{z,t}, diag(sigma^2_{z,t}))
- [mu_{z,t}, sigma_{z,t}] = phi^enc(phi^x(x_t), h_{t-1})
- 编码和解码通过RNN隐状态h_{t-1}相耦合

**5. 联合分布分解**：
- 生成: p(x_{<=T}, z_{<=T}) = prod_t p(x_t|z_{<=t}, x_{<t}) p(z_t|x_{<t}, z_{<t})
- 推断: q(z_{<=T}|x_{<=T}) = prod_t q(z_t|x_{<=t}, z_{<t})

### 训练目标
时间步逐步的变分下界：
$$\mathbb{E}_{q(z_{\leq T}|x_{\leq T})}\left[\sum_{t=1}^{T}\left(-\text{KL}(q(z_t|x_{\leq t}, z_{<t}) \| p(z_t|x_{<t}, z_{<t})) + \log p(x_t|z_{\leq t}, x_{<t})\right)\right]$$

### 模型变体
- **VRNN-Gauss**: 使用简单高斯输出
- **VRNN-GMM**: 使用高斯混合模型输出
- **VRNN-I**: 不使用条件先验（先验在时间步间独立），STORN可视为其特例

## 关键发现与结论
1. 在四个语音数据集和一个手写数据集上，VRNN模型的对数似然显著优于标准RNN
2. VRNN-Gauss（简单高斯输出）即可取得很好效果，而RNN-Gauss则生成接近纯噪声的样本
3. 带条件先验的VRNN优于不带条件先验的VRNN-I，证明了潜变量间时间依赖性的重要性
4. VRNN生成的语音波形噪声更少，手写样本风格更一致
5. KL散度在波形转变处增大，说明潜变量能捕捉模态转换

## 重要公式

**条件先验**：
$$z_t \sim \mathcal{N}(\mu_{0,t}, \text{diag}(\sigma^2_{0,t})), \quad [\mu_{0,t}, \sigma_{0,t}] = \varphi^{\text{prior}}_\tau(h_{t-1})$$

**时间步变分下界**：
$$\mathcal{L} = \mathbb{E}_{q}\left[\sum_{t=1}^{T}\left(-\text{KL}(q(z_t|x_{\leq t}, z_{<t}) \| p(z_t|x_{<t}, z_{<t})) + \log p(x_t|z_{\leq t}, x_{<t})\right)\right]$$

**RNN状态更新**：
$$h_t = f_\theta(\varphi^x_\tau(x_t), \varphi^z_\tau(z_t), h_{t-1})$$

## 与我们项目的关联
VRNN为CVHI处理时间序列数据提供了重要参考：
- **时序建模**：生态系统中物种的恢复过程是时间序列问题。VRNN展示了如何在每个时间步引入潜变量来捕捉复杂变异性，这与CVHI在每个时间步推断hidden species状态 h_t 的需求一致
- **条件先验**：VRNN的条件先验 p(z_t|h_{t-1}) 启发我们设计CVHI中hidden species的时序先验——前一时刻的生态状态应影响当前时刻hidden species的先验分布
- **RNN + VAE融合架构**：CVHI可以借鉴VRNN的架构，将GNN编码的空间信息与RNN编码的时间信息结合，在每个时间步进行hidden species的变分推断
- **与DVAE综述的联系**：VRNN是Dynamical VAE家族的重要成员（见Girin et al. 2021综述）

## 一句话总结
VRNN通过在RNN每个时间步嵌入条件VAE（特别是引入依赖于RNN隐状态的条件先验），有效建模了序列数据中的高层随机变异性，为时序潜变量模型奠定了重要基础。
