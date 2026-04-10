# 部分观测生态动力学：实现规范

## 0. 文档目的

本文档给出一个**最小化但技术上自洽**的模型规范，用于处理部分观测生态动力系统中的隐藏效应推断问题。文档面向直接实现，默认读者为编码代理或研究工程实现者。目标**不是**证明隐藏物种的精确数量或精确身份，而是在部分观测条件下学习一个潜在生态机制，使模型在多步动力学一致性上得到实质提升。

实现范围被有意限制在我们已经讨论过的模块内：

1. 延迟嵌入 / Takens 风格时间重构；
2. 生态残差通道；
3. 观测节点图骨架；
4. 潜在隐藏效应节点；
5. 基于 GNN 的交互更新；
6. 从潜在图状态到动力学增量的解码器；
7. 生态约束项，尤其是轻量级代谢先验；
8. 以**正向模拟 rollout** 为中心的评估方案。

除非数值稳定性确有必要，不应额外引入新的大模块。

---

## 1. 问题定义

### 1.1 观测数据

设生态系统的完整状态为

\[
\mathbf{z}_t = [\mathbf{x}_t, \mathbf{h}_t],
\]

其中：

- \(\mathbf{x}_t \in \mathbb{R}^{N_o}\)：可观测物种的丰度或其变换后的表示；
- \(\mathbf{h}_t\)：不可观测的生态状态，训练时不可得。

实际可用的数据仅为观测轨迹

\[
\{\mathbf{x}_1, \mathbf{x}_2, \dots, \mathbf{x}_T\}。
\]

可选地，可加入环境协变量：

\[
\mathbf{u}_t \in \mathbb{R}^{d_u},
\]

但基线实现必须在 \(\mathbf{u}_t\) 缺失时依然有效。

### 1.2 学习目标

学习一个模型

\[
\mathcal{F}_\theta
\]

使其在给定过去的观测历史后，可以预测下一步观测动力学，并在多步 rollout 下保持稳定：

\[
\hat{\mathbf{x}}_{t+1} = \mathbf{x}_t + \Delta \hat{\mathbf{x}}_t,
\]

其中

\[
\Delta \hat{\mathbf{x}}_t = \mathcal{F}_\theta(\text{截至 } t \text{ 的历史})。
\]

模型必须能够将未解释的部分动力学归因到一个潜在生态通道，而不是将全部效应都强行压入显式的观测—观测相互作用中。

### 1.3 科学解释目标

模型输出应支持如下解释框架：

- 仅靠观测物种之间的直接耦合，无法充分刻画系统动力学；
- 引入潜在生态通道有助于闭合动力学；
- 对潜在通道的验证，主要依据是**正向动力学模拟改进**，而非直接声称获得了生物学上的唯一可识别性。

---

## 2. 建模立场

该模型是一个**具有生态结构约束的动力学替代模型**，而不是对完整未观测生态系统的严格机制重建。

模型作出如下承诺：

1. **观测物种保持为显式节点。** 不将其完全压缩为一个黑箱序列编码器输出。
2. **未观测影响以潜在状态表示。** 该潜在状态被解释为聚合的隐藏生态效应，而不必等同于一个字面意义上的隐藏物种。
3. **时间结构显式建模。** 使用延迟嵌入，使当前推断依赖于一个短时历史，而不是单一时刻快照。
4. **评估以动力学为主，而非纯逐点拟合。** 多步 rollout 是一级指标。
5. **生态先验是软约束，不是硬方程。** 主体模型是数据驱动的，生态约束仅将其正则到更合理的生态区间。

非目标包括：

- 证明隐藏物种的形式可观测性；
- 精确恢复真实的交互作用图；
- 构造过大的通用架构；
- 对任意生态 ODE 建立完整逆问题求解器。

---

## 3. 状态表示

### 3.1 观测预处理

输入轨迹应先变换到数值稳定的表示空间。可选方案如下：

- 若原始丰度已经尺度适中、严格为正且动态范围可控，则可直接使用原始值；
- 对计数型或重尾丰度，使用 \(\log(1+x)\) 变换；
- 在对数变换后执行 z-score 标准化，且均值与方差仅使用训练集统计量计算。

推荐基线为：

\[
\tilde{x}_{i,t} = \frac{\log(1 + x_{i,t}) - \mu_i}{\sigma_i + \epsilon}。
\]

所有报告指标在可行时都应同时给出：

- 变换空间下的误差；
- 反变换回原始丰度空间后的误差。

### 3.2 延迟嵌入

对每个观测物种 \(i\)，构造长度为 \(L\) 的延迟向量：

\[
\mathbf{d}_{i,t} = [\tilde{x}_{i,t}, \tilde{x}_{i,t-\tau}, \tilde{x}_{i,t-2\tau}, \dots, \tilde{x}_{i,t-(L-1)\tau}]。
\]

推荐基线：

- \(\tau = 1\)；
- \(L \in \{4, 6, 8\}\)，默认取 6。

这里不声称实现了严格意义上的 Takens 重构定理。其用途是操作性的：

- 编码短时记忆动力学；
- 在观测轨迹中暴露潜在隐藏状态留下的投影痕迹。

### 3.3 全局上下文向量

构造全局上下文向量：

\[
\mathbf{g}_t = [\text{flatten}(\mathbf{D}_t), \mathbf{u}_t],
\]

其中 \(\mathbf{D}_t\) 为所有观测物种延迟向量的集合。

该全局上下文用于：

- 初始化潜在隐藏效应节点；
- 可选地参数化时变图权重。

---

## 4. 生态残差通道

### 4.1 作用

生态残差通道用于表示：下一步动力学中，那一部分**无法仅由观测—观测直接作用解释**的成分。

它不能被解释为“证明存在一个隐藏物种”。它是一个有结构的模型部件，用来吸收未解析的生态影响。

### 4.2 操作性定义

先定义一个仅使用观测信息的基线预测器：

\[
\Delta \hat{\mathbf{x}}^{\text{obs}}_t = f_{\text{obs}}(\mathbf{D}_t)。
\]

然后在分析中定义残差目标：

\[
\mathbf{r}^{\text{eco}}_t = \Delta \mathbf{x}_t - \Delta \hat{\mathbf{x}}^{\text{obs}}_t,
\]

其中

\[
\Delta \mathbf{x}_t = \mathbf{x}_{t+1} - \mathbf{x}_t。
\]

在最终模型中，该残差不作为监督真值输入，而是通过结构设计为其分配一条潜在通道去拟合这类残差结构。

### 4.3 为什么必须显式保留该通道

该通道承担三个角色：

1. **诊断角色**：可量化仅观测模型何处失败；
2. **架构角色**：为潜在隐藏效应节点提供结构动机；
3. **解释角色**：支持“动力学改进来自未解析生态影响建模，而非单纯加大模型容量”的论述。

---

## 5. 图构建

### 5.1 节点集合

在每个时刻 \(t\)，构造如下图：

- \(N_o\) 个观测物种节点；
- \(N_h\) 个潜在隐藏效应节点。

最小基线：

- \(N_h = 1\)。

可选扩展（用于后续消融）：

- \(N_h \in \{2, 4\}\)。

### 5.2 观测节点特征

对观测节点 \(i\)：

\[
\mathbf{v}^{(o)}_{i,t} = \phi_o(\mathbf{d}_{i,t}),
\]

其中 \(\phi_o\) 是一个小型 MLP。

推荐维度：

- 输入维度：\(L\)；
- 隐层维度：32 或 64；
- 输出节点嵌入维度 \(d\)：64。

### 5.3 潜在隐藏节点特征

从全局上下文初始化潜在节点 \(k\)：

\[
\mathbf{v}^{(h)}_{k,t} = \phi_h(\mathbf{g}_t) + \mathbf{e}_k,
\]

其中 \(\mathbf{e}_k\) 为可学习的潜在节点身份嵌入。

若 \(N_h = 1\)，则可简化为：

\[
\mathbf{v}^{(h)}_{t} = \phi_h(\mathbf{g}_t)。
\]

### 5.4 边类型

使用带类型的有向边：

1. observed \(\rightarrow\) observed；
2. observed \(\rightarrow\) hidden；
3. hidden \(\rightarrow\) observed；
4. hidden \(\rightarrow\) hidden。

当 \(N_h=1\) 时，最小基线可省略 hidden-hidden 边。

### 5.5 边参数化

对每条有向边 \(i \rightarrow j\)，定义边特征：

\[
\mathbf{e}_{ij,t} = \phi_e([\mathbf{v}_{i,t}, \mathbf{v}_{j,t}, \mathbf{g}_t])。
\]

为控制实现复杂度，采用稠密图加可学习边门控：

\[
\alpha_{ij,t} = \sigma(w_{ij,t}),
\]

或

\[
\alpha_{ij,t} = \sigma(\psi([\mathbf{v}_{i,t}, \mathbf{v}_{j,t}, \mathbf{g}_t]))。
\]

随后消息项由 \(\alpha_{ij,t}\) 加权。

稀疏阈值化仅用于分析阶段；在早期训练中不建议使用硬阈值。

---

## 6. 核心动力学模块

### 6.1 消息传递更新

在时刻 \(t\)，执行一层或多层消息传递。

对每条边 \(i \rightarrow j\)：

\[
\mathbf{m}_{ij,t} = \alpha_{ij,t} \cdot \psi_m([\mathbf{v}_{i,t}, \mathbf{v}_{j,t}, \mathbf{e}_{ij,t}])。
\]

聚合进入节点 \(j\) 的消息：

\[
\mathbf{m}_{j,t} = \sum_{i \neq j} \mathbf{m}_{ij,t}。
\]

然后更新节点状态：

\[
\mathbf{v}'_{j,t} = \psi_u([\mathbf{v}_{j,t}, \mathbf{m}_{j,t}, \mathbf{g}_t])。
\]

推荐基线：

- 2 层 message passing；
- 节点更新使用残差连接；
- LayerNorm 可选；对于短轨迹，不推荐 BatchNorm。

### 6.2 潜在隐藏节点动力学的解释

潜在节点没有显式监督。它的训练信号仅来自：

- 减少一步预测误差；
- 减少 rollout 误差；
- 满足生态正则项。

因此其语义是**功能性的**：

- 存储未解析的生态记忆；
- 中介那些无法由观测节点直接耦合解释的延迟或间接作用；
- 作为聚合的隐藏生态机制。

### 6.3 为什么使用图而不是纯序列模型

纯序列模型也可以预测时间序列，但图表述保留了以下结构性收益：

- 物种级表示；
- 有向交互作用分析；
- 观测—观测与潜在中介效应的分离；
- 边可视化与生态解释能力。

因此，图不是单纯的建模便利，而是科学论述的一部分。

---

## 7. 解码器：从图状态到动力学量

### 7.1 输出目标

优先预测下一步**增量**而非绝对状态：

\[
\Delta \hat{x}_{i,t} = f_{\text{dec}}(\mathbf{v}'_{i,t}, \mathbf{v}'_{h,t}, \mathbf{g}_t)。
\]

然后

\[
\hat{x}_{i,t+1} = x_{i,t} + \Delta \hat{x}_{i,t}。
\]

这样做的理由是：生态动力学更自然地表达为速率或增量。

### 7.2 解码器结构

对观测节点 \(i\)，使用：

\[
\Delta \hat{x}_{i,t} = \psi_d([\mathbf{v}'_{i,t}, \bar{\mathbf{v}}^{(h)}_t, \mathbf{c}_t])，
\]

其中：

- \(\bar{\mathbf{v}}^{(h)}_t\)：潜在隐藏节点状态的拼接或平均；
- \(\mathbf{c}_t\)：从 \(\mathbf{g}_t\) 提取的紧凑上下文向量。

推荐输出头：

- 2 层 MLP；
- 隐层维度 64；
- 线性输出层。

### 7.3 可选的分解输出

在分析阶段，可选地将增量拆分为：

\[
\Delta \hat{x}_{i,t} = \Delta \hat{x}^{\text{direct}}_{i,t} + \Delta \hat{x}^{\text{latent}}_{i,t}。
\]

该分解有利于可视化与解释，但并非第一版稳定实现的必要条件。

---

## 8. 生态先验与约束项

生态约束必须作为软正则项使用，且在训练早期不能压过主预测目标。

### 8.1 范围与稳定性护栏

为抑制 rollout 期间的数值爆炸，当预测下一时刻状态超出合理范围时施加平滑惩罚。

若使用变换空间训练，则可在变换空间约束；若 rollout 过程中反变换回原始空间，则可在原始丰度空间约束。

例如：

\[
\mathcal{L}_{\text{range}} = \sum_{t,i} \text{ReLU}(\hat{x}_{i,t+1}-x^{\max}_i)^2 + \text{ReLU}(x^{\min}_i-\hat{x}_{i,t+1})^2。
\]

### 8.2 平滑性 / 有界加速度

若采样时间分辨率较细，可对过强的二阶波动进行惩罚：

\[
\mathcal{L}_{\text{smooth}} = \sum_t \|\Delta \hat{\mathbf{x}}_{t+1} - \Delta \hat{\mathbf{x}}_t\|_2^2。
\]

若真实系统具有强振荡或混沌性质，则该项应谨慎使用。

### 8.3 轻量级代谢先验

若数据集中存在可信的物种性状信息（如体重、温度代理量、营养级指示变量等），可加入一个受代谢生态学启发的软先验。代谢理论将代谢率视为连接个体过程与生态过程的基础生物学速率，常见尺度关系写作 \(B \propto M^{3/4} e^{-E/kT}\)。fileciteturn2file0 fileciteturn2file1

该先验必须保持为可选项，因为很多生态数据集并不具备足够可信的性状协变量。

若性状向量 \(\mathbf{q}_i\) 可用，则定义预测活动尺度：

\[
\rho_i = \psi_{\text{meta}}(\mathbf{q}_i),
\]

并令预测增量幅度与该尺度相关：

\[
\mathcal{L}_{\text{meta}} = \sum_i \left( s_i - \text{stopgrad}(\tilde{\rho}_i) \right)^2,
\]

其中 \(s_i\) 可取训练时间范围上该物种平均绝对预测增量，或潜在通道贡献强度等经验统计量。

除非数据本身足以支撑，否则**不要**将完整代谢方程硬编码进主状态转移。

### 8.4 相互作用稀疏性

为避免图始终为稠密且不可解释的完全连接结构，可对边门控施加弱稀疏惩罚：

\[
\mathcal{L}_{\text{sparse}} = \sum_{t,i,j} \alpha_{ij,t}。
\]

该项必须较弱。若早期训练就施加强稀疏，会明显损害优化。

---

## 9. 损失函数

总损失定义为：

\[
\mathcal{L} = \lambda_1 \mathcal{L}_{\text{1step}} + \lambda_2 \mathcal{L}_{\text{rollout}} + \lambda_3 \mathcal{L}_{\text{range}} + \lambda_4 \mathcal{L}_{\text{sparse}} + \lambda_5 \mathcal{L}_{\text{meta}} + \lambda_6 \mathcal{L}_{\text{smooth}}。
\]

### 9.1 一步预测损失

\[
\mathcal{L}_{\text{1step}} = \sum_t \|\hat{\mathbf{x}}_{t+1} - \mathbf{x}_{t+1}\|_2^2。
\]

若序列存在尖峰或异常脉冲，优先考虑 Huber loss，而非纯 MSE。

### 9.2 Rollout 损失

该项是与主评估一致的核心训练项。

设 rollout 视野为 \(H\)。从一个真实观测窗口终点 \(t\) 出发，自回归模拟：

\[
\hat{\mathbf{x}}_{t+1}, \hat{\mathbf{x}}_{t+2}, \dots, \hat{\mathbf{x}}_{t+H}。
\]

定义：

\[
\mathcal{L}_{\text{rollout}} = \sum_{h=1}^{H} w_h \|\hat{\mathbf{x}}_{t+h} - \mathbf{x}_{t+h}\|_2^2。
\]

推荐权重策略：

- 若 \(H\) 较短，可取均匀权重；
- 或令 \(w_h\) 轻微递增，以强调长程漂移控制。

推荐 curriculum：

- 初始 \(H=1\) 或 2；
- 随训练逐步增加到 5、10、20，具体取决于数据长度与数值稳定性。

### 9.3 建议的基线权重

可先使用如下实践性设定：

- \(\lambda_1 = 1.0\)
- \(\lambda_2 = 0.5\)
- \(\lambda_3 = 0.05\)
- \(\lambda_4 = 1e-4\)
- \(\lambda_5 = 0.0\)（若无性状数据）
- \(\lambda_6 = 0.01\)

然后根据验证集上的 rollout 指标而非一步指标单独调参。

---

## 10. 训练协议

### 10.1 数据划分

必须使用按时间顺序的切分，而不是在时间上随机打乱。

推荐：

- train：前 60%–70%；
- validation：中间 15%–20%；
- test：最后 15%–20%。

标准化统计量和窗口构造均不得引入未来信息泄漏。

### 10.2 滑动窗口样本

每个训练样本应包含：

- 历史窗口长度 \(W\)；
- 目标 rollout 视野 \(H\)。

例如：

- 输入索引：\([t-W+1, \dots, t]\)
- 目标索引：\([t+1, \dots, t+H]\)

推荐基线：

- \(W = 12\) 或 16；
- 初始 \(H = 5\)。

### 10.3 Teacher forcing 计划

不要一开始就进行长程自由 rollout 训练。

推荐分阶段训练：

1. **阶段 A**：仅一步 teacher-forced 训练；
2. **阶段 B**：短 rollout，部分 teacher forcing；
3. **阶段 C**：目标视野上的自由 rollout。

该训练顺序对数值稳定性非常重要。

### 10.4 优化设置

推荐基线：

- optimizer：AdamW；
- 初始学习率：1e-3；
- weight decay：1e-4；
- gradient clipping：1.0；
- batch size：在显存允许下尽可能大，通常为 16–64 个窗口。

早停应依据验证集上的 rollout 指标，而不能只看一步预测指标。

---

## 11. 前向传播规范

给定一个以 \(t\) 为终点的窗口，前向过程如下：

1. 预处理观测状态；
2. 为所有观测物种构造延迟向量；
3. 构造观测节点嵌入；
4. 从全局上下文构造潜在隐藏节点嵌入；
5. 计算带类型的边门控与边特征；
6. 执行消息传递层；
7. 对每个观测节点解码得到增量；
8. 更新状态得到 \(\hat{\mathbf{x}}_{t+1}\)；
9. 若为 rollout 模式，将预测值写回历史缓冲区并重复。

该流程同时定义训练期 rollout 与推理期 rollout，不应再额外引入独立模拟引擎。

---

## 12. 正向模拟评估（主评估）

这是整个项目中最重要的评估部分。

### 12.1 评估理由

一个模型即便在一步预测上表现很好，也可能并未学到真实动力学；它可能只是利用局部插值能力，却在自治模拟时迅速漂移。

因此，主评估必须是：

> 用一段真实历史初始化模型，然后令模型自回归向前演化，并将生成轨迹与真实未来轨迹比较。

这直接检验所学习到的潜在生态机制是否真的改善了动力学闭合。

### 12.2 评估流程

对每个测试窗口终点 \(t\)：

1. 提供真实历史 \([t-W+1, \dots, t]\)；
2. 生成 \(\hat{\mathbf{x}}_{t+1}\)；
3. 将预测反馈为下一步输入；
4. 持续模拟到视野 \(H_{eval}\)。

应在多个视野上评估，例如：

- 短期：5；
- 中期：10；
- 长期：20 或数据允许的最大安全视野。

### 12.3 Rollout 指标

至少计算以下指标：

1. **rollout RMSE**
   \[
   \text{RMSE}(H) = \sqrt{\frac{1}{HN_o}\sum_{h=1}^{H}\|\hat{\mathbf{x}}_{t+h}-\mathbf{x}_{t+h}\|_2^2}
   \]
2. **rollout MAE**
3. **逐物种 Pearson / Spearman 相关系数**（在 rollout 轨迹上计算）
4. **轨迹漂移得分**
   - 即视野内累积绝对误差；
5. **稳定性失败率**
   - 即产生 NaN、数值爆炸、或反变换后出现不可能负丰度的 rollout 比例。

在有意义时可选：

6. **相空间距离**（低维投影中）；
7. **转折点准确率**（峰值、崩塌点、增量符号变化等）。

### 12.4 对比基线集合

至少与以下三个基线比较：

1. **Persistence baseline**
   \[
   \hat{\mathbf{x}}_{t+1} = \mathbf{x}_t
   \]
2. **仅观测 MLP/RNN 基线**
   - 无图、无潜在隐藏节点；
3. **仅观测图基线**
   - 只有观测节点图，没有隐藏节点。

然后与完整模型比较：

4. **观测图 + 潜在隐藏效应节点**。

这一比较是必需的。如果完整模型在 rollout 上无法优于仅观测图基线，则关于隐藏通道的论述会非常薄弱。

### 12.5 主成功标准

主结果的表述不应是“隐藏节点在生物学上真实存在”。

而应表述为：

- 引入潜在生态通道后，rollout 保真度有实质提升；
- 该提升在多个视野和随机种子下持续存在；
- 该提升不能仅用“模型容量增加”这一平凡因素解释。

---

## 13. 消融实验计划

执行以下消融。

### A1. 去掉潜在隐藏节点

目的：检验潜在生态通道是否必要。

### A2. 去掉延迟嵌入

仅使用当前状态，不使用延迟向量。

目的：检验短记忆重构是否重要。

### A3. 去掉生态残差动机 / 去掉潜在中介路径

在实践中可实现为：保持参数量近似不变，但阻断 hidden-to-observed 边。

目的：检验真正起作用的是“潜在中介通道”，而非一般性的附加容量。

### A4. 去掉 rollout loss

只做一步训练。

目的：检验 rollout-aware 训练是否必要。

### A5. 去掉生态正则项

目的：检验其是否改善数值稳定性与生态合理性。

### A6. 改变潜在隐藏节点数量

取 \(N_h = 1, 2, 4\)。

目的：检验一个聚合隐藏效应节点是否已足以支持当前数据。

---

## 14. 可视化要求

项目必须输出可视化结果。至少包括以下内容。

### V1. 真值轨迹 vs rollout 轨迹

对选定测试窗口和选定物种，绘制：

- 横轴：时间；
- 纵轴：丰度或变换后的丰度；
- 曲线：真实轨迹、基线模型 rollout、完整模型 rollout。

这是最核心的定性图。

### V2. 误差随视野增长曲线

绘制各模型 rollout RMSE 随 horizon 增长的曲线。

该图直接体现漂移行为。

### V3. 逐物种 rollout 误差热图

- 行：物种；
- 列：rollout horizon；
- 单元值：平均绝对误差。

该图可揭示哪些物种真正受益于潜在通道。

### V4. 边门控热图 / 平均交互矩阵

对 \(\alpha_{ij,t}\) 在时间上取平均。

应分别展示：

- observed-observed 区块；
- hidden-observed 区块。

该图用于支持结构解释，但不能过度宣称其为真实生物学网络。

### V5. 潜在活动强度 vs 生态残差代理

若分析中构造了仅观测模型残差代理，则比较：

- 潜在节点范数或潜在中介贡献强度；
- 残差幅度。

目的：展示潜在通道恰恰在观测模型失败时变得活跃。

### V6. 相图比较（可选但推荐）

对 2–3 个关键物种，在低维投影状态空间中比较：

- 真实轨迹；
- 预测轨迹。

当系统存在振荡或循环动力学时，该图很有价值。

---

## 15. 不确定性与鲁棒性

### 15.1 多随机种子

至少使用 3 个随机种子训练。

对 rollout 指标报告均值与标准差。

### 15.2 MC dropout 或小型集成

若可行，可在推理时启用 dropout，或训练一个小型 ensemble。

将预测方差作为 rollout 期间的置信度诊断。

### 15.3 不确定性的解释

不确定性应被解释为：在部分可观测条件下模型的预测信心；而不是“某个真实隐藏物种存在的后验概率”。

---

## 16. 失败模式与诊断规则

### 16.1 一步预测好，但 rollout 差

解释：

- 模型只学到了局部插值，没有学到稳定动力学；
- rollout loss 权重可能过低，或时间状态表示不足。

处理：

- 逐步增加 rollout 训练视野；
- 强化 range penalty；
- 若存在明显过拟合，则适当减小模型容量。

### 16.2 潜在隐藏节点未被使用

症状：

- hidden-to-observed 门控接近零；
- 与仅观测图模型相比没有明显优势。

解释：

- 数据本身可能不足以支撑潜在通道；
- 架构可能过于容易让 observed-only 路径独占解释权。

处理：

- 检查仅观测模型残差；
- 降低直接路径容量；
- 使用结构更明确的潜在中介解码器。

### 16.3 潜在节点吸收一切，图变得不可解释

症状：

- hidden-to-observed 通道主导全部预测；
- observed-observed 边几乎塌缩。

处理：

- 增加轻度稀疏或平衡正则；
- 可选地加入分解惩罚，使 direct 与 latent 两个通道都保持活跃。

### 16.4 Rollout 数值爆炸

处理：

- 对预测增量做 clipping；
- 在变换空间中建模；
- 降低学习率；
- 增大 range penalty；
- 在训练早期缩短 rollout horizon。

---

## 17. 建议的代码结构

```text
project/
  configs/
    base.yaml
    ablation_no_hidden.yaml
    ablation_no_delay.yaml
    ablation_one_step_only.yaml
  data/
    dataset.py
    transforms.py
    window_sampler.py
  models/
    encoders.py          # delay encoder, global encoder
    graph_builder.py     # node/edge construction
    gnn.py               # message passing core
    decoder.py           # increment decoder
    full_model.py        # end-to-end wrapper
  losses/
    prediction.py
    rollout.py
    ecological.py
  train/
    trainer.py
    train_one_step.py
    train_rollout.py
  eval/
    metrics.py
    rollout_eval.py
    baselines.py
  viz/
    trajectories.py
    heatmaps.py
    graph_viz.py
  scripts/
    run_train.py
    run_eval.py
    run_ablation.py
```

---

## 18. 最小伪代码

```python
for batch in train_loader:
    history, future = batch

    # history: [B, W, N_obs]
    # future:  [B, H, N_obs]

    x_hist = preprocess(history)
    x_future = preprocess(future)

    total_loss = 0.0
    sim_buffer = x_hist.clone()

    for h in range(H_train):
        delay_features = build_delay_features(sim_buffer)
        node_obs = obs_encoder(delay_features)
        global_ctx = build_global_context(delay_features)
        node_latent = latent_encoder(global_ctx)

        graph_state = build_graph(node_obs, node_latent, global_ctx)
        graph_state = gnn(graph_state)

        dx_pred = decoder(graph_state, global_ctx)
        x_next_pred = sim_buffer[:, -1, :] + dx_pred

        total_loss += one_step_loss(x_next_pred, x_future[:, h, :])
        total_loss += ecological_regularizers(...)

        if use_rollout_loss:
            total_loss += rollout_weight[h] * rollout_loss(x_next_pred, x_future[:, h, :])

        sim_buffer = append_prediction(sim_buffer, x_next_pred)

    optimizer.zero_grad()
    total_loss.backward()
    clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
```

---

## 19. 推荐实验顺序

1. 先实现 persistence baseline；
2. 再实现仅观测 MLP baseline；
3. 再实现仅观测图 baseline；
4. 再实现完整图 + 1 个潜在隐藏节点；
5. 验证一步训练可以稳定工作；
6. 开启 rollout 训练；
7. # 生成 rollout 可视化
8. 执行消融实验；
9. 可选测试多个潜在节点数；
10. 仅在以上都完成后，再考虑更强的生态先验。

该顺序非常重要。不要从一个大而复杂的架构开始。

---

## 20. 报告标准

实现输出至少应包括：

1. validation / test 上的一步预测指标；
2. 多个 rollout horizon 下的 rollout 指标；
3. 基线比较表；
4. 消融实验表；
5. 必要可视化；
6. 对潜在通道使用情况的简要解释；
7. 对模型局限性的明确说明。

### 20.1 论断边界

允许的论断：

> 在部分观测条件下，潜在生态通道能够提高正向模拟质量，并捕捉到仅观测模型无法覆盖的结构性残差动力学。

未经更强证据，不允许的论断：

> 模型已经识别出了真实隐藏物种及其精确生态作用网络。

---

## 21. 局限性

1. 潜在节点是聚合隐藏效应表示，并不保证与真实隐藏物种一一对应；
2. 延迟嵌入在现实噪声与有限数据下只是操作性构造，不构成完整状态重建证明；
3. 学到的边权是功能性交互权重，不能自动视为因果生态作用；
4. 良好的 rollout 不意味着唯一可识别性；
5. 代谢先验在缺乏可信性状变量时，只能作为弱生态偏置项。

---

## 22. 最终实现目标

一个成功的第一版应满足以下全部条件：

1. 在按时间顺序切分的窗口上训练稳定；
2. 在多步 rollout 上优于 persistence 与 observed-only 基线；
3. 输出可解释的图结构与潜在通道可视化；
4. 去掉潜在通道后，rollout 质量出现明确下降；
5. 避免明显非生态的数值伪影。

对当前阶段而言，这样已经足够。第一版不要尝试解决完整的隐藏物种可识别性问题。

---

## 23. 关于生态先验设计的参考说明

上文的代谢先验仅被用作**软生态正则**的动机来源。代谢理论将代谢视为连接个体尺度过程与生态动力学的基础速率，标准尺度关系强调体重与温度效应。fileciteturn2file0 fileciteturn2file1

本文档对该理论的使用是刻意保守的：它只作为将模型偏向生态合理区间的方向性偏置，而不被当作学习转移算子的完整支配方程。
