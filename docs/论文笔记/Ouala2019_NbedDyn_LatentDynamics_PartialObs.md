# Ouala et al. (2019) -- Learning Latent Dynamics for Partially-Observed Chaotic Systems (NbedDyn)

**arXiv:** 1907.02452v1
**作者:** Said Ouala, Duong Nguyen, Lucas Drumetz, Bertrand Chapron, Ananda Pascual, Fabrice Collard, Lucile Gaultier, Ronan Fablet
**机构:** IMT Atlantique / Ifremer / IMEDEA / OceanDataLab
**日期:** 2019-07-04

---

## 1. 核心问题

动力系统部分观测时，观测空间中不存在光滑 ODE 能描述观测的时间演化（因为映射不是一对一的）。Takens 定理保证了延迟嵌入的存在性，但需要手动选择 lag 和维度。

**NbedDyn 的核心创新：不用延迟嵌入，而是直接学习增广状态空间 + ODE。**

## 2. 方法框架

### 2.1 增广状态空间

$$X_t = [x_t^T, y_t^T]^T$$

- $x_t \in \mathbb{R}^n$: 观测状态
- $y_t \in \mathbb{R}^{d_E - n}$: 未观测潜变量（需要学习）
- $d_E$: 增广空间维度

### 2.2 状态空间模型

$$\dot{X}_t = f_\theta(X_t), \quad x_t = G(X_t)$$

- $f_\theta$: 神经网络参数化的 ODE
- $G$: 观测算子（对 $x_t$ 分量取恒等）
- 数值积分: 4阶 Runge-Kutta

### 2.3 训练目标（Eq.6 -- 核心公式）

$$\min_\theta \min_{\{y_t\}} \sum_{t=1}^T \|x_t - G(\Phi_{\theta,t}(X_{t-1}))\|^2 + \lambda\|X_t - \Phi_{\theta,t}(X_{t-1})\|^2$$

- 第一项：一步预测误差（观测空间）
- 第二项：ODE 一致性约束（增广空间）
- $\lambda$：权重参数

**关键：同时优化 ODE 参数 $\theta$ 和所有潜变量 $\{y_t\}$。**

### 2.4 预测阶段

给定新观测序列，通过变分优化（Eq.7）推断初始潜变量 $\hat{y}_T$，然后用训练好的 ODE 做预测。初始化策略：从训练集找最相似的轨迹段。

## 3. 实验结果

### 3.1 Lorenz-63（标量观测 $x_1$）

| 模型 | 1步 RMSE | 4步 RMSE | Lyapunov |
|---|---|---|---|
| Analog (best) | 1.2e-4 | 1.04e-3 | 0.84 |
| Sparse Regression | 1.85e-3 | 2.56e-3 | NaN |
| **Latent-ODE** | **0.0801** | **0.520** | **NaN** |
| **NbedDyn dE=6** | **6.8e-6** | **6.5e-5** | **0.87** |

- NbedDyn 比次优模型好 **一个数量级**
- 追踪长达 **9 Lyapunov 时间**（Fig.5）
- Latent-ODE（encoder-based）效果差很多

### 3.2 Sea Level Anomaly（真实数据）

| 模型 | 1天 RMSE | 2天 RMSE | 4天 RMSE |
|---|---|---|---|
| Latent-ODE | 0.025 | 0.032 | 0.048 |
| Analog | 0.046 | 0.062 | 0.073 |
| **NbedDyn** | **0.002** | **0.007** | **0.027** |

- 1天预测: 相对改善 90%+

### 3.3 维度分析（Appendix A）

通过分析学到的 Jacobian 特征值模量：
- Lorenz dE=6 时，只有 3 个特征值非零 -> 模型自动发现系统维度=3
- SLA dE=60 时，50 个特征值非零

## 4. 与 Takens 定理的关系（Section 5）

NbedDyn 的预测流程隐式定义了一个延迟嵌入：

$$\psi(\{x_t\}_{t_0:T}) = \arg\min_{X_T} \min_{\{X_t\}_{t<T}} (\text{forecasting + ODE consistency})$$

这个嵌入是通过优化获得的，而非手动选择 lag/dim。

## 5. 与 CVHI-Residual 的对比

| 方面 | NbedDyn | CVHI-Residual |
|---|---|---|
| **潜变量推断** | 直接优化 $y_t$ | Encoder $q(h|x)$ |
| **潜变量维度** | 多维 $y \in \mathbb{R}^{d_E-n}$ | 标量 $h \in \mathbb{R}$ |
| **动力学** | 单一 ODE $f_\theta(X)$ | 分离 $f_{vis}(x) + h \cdot G(x)$ |
| **潜变量约束** | ODE 一致性 | KL + smoothness + counterfactual |
| **训练** | 直接优化 | VAE-style |
| **适用场景** | 通用部分观测 | 专门用于 hidden species recovery |

## 6. 对我们项目的启发

### 6.1 ODE 一致性损失（已实现）

NbedDyn Eq.6 的第二项 $\lambda\|X_t - \Phi_{\theta,t}(X_{t-1})\|^2$ 可以适配到 CVHI：

$$\mathcal{L}_{h\_ode} = \text{MSE}(h_{encoder}(t), \ h(t-1) + f_h(h(t-1), x(t-1)))$$

这比现有的 smoothness prior 更强大：它学到 h 的**实际动力学**。

### 6.2 NbedDyn 的 Latent-ODE 对比数据很有价值

- Latent-ODE（Rubanova 2019，encoder-based）在 Lorenz-63 上 RMSE = 0.0801
- NbedDyn（直接优化 + ODE 一致性）RMSE = 6.8e-6

这暗示 **encoder-based 推断 < 直接优化 + ODE 约束**。我们的 CVHI 也是 encoder-based，加入 ODE 一致性可能弥补差距。

### 6.3 潜在的进一步改进

- **多维 h**: 当前 h 是标量。如果 hidden species 的动力学需要多维表示，可以扩展。
- **直接优化 h**: 训练时不只用 encoder，也直接优化 $\{h_t\}$（如 NbedDyn）。但这可能和 VAE 框架冲突。
- **Jacobian 维度分析**: 用 Appendix A 的方法分析学到的增广状态的有效维度。
