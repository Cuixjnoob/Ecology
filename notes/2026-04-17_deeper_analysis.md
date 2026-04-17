# 深层分析：方法的根本瓶颈在哪里

## 一、数据梳理

### Huisman vs Beninca 的差异不只是"混沌强度"

| | Huisman | Beninca |
|---|---|---|
| T | 1001 | 658 |
| dt | 2 day | 4 day |
| 物种数 | 6 | 9 |
| 隐藏因素 | 只有 1 个缺失物种 | 缺失物种 + 环境 + 测量噪声 |
| 动力学形式 | 精确已知 (chemostat ODE) | 未知 |
| 噪声 | 无 | 有（测量误差、插值） |
| burst 特征 | 周期性振荡 | 间歇性、不规则 |
| Overall Pearson | 0.48-0.53 | 0.16 |

**关键差异不是混沌强度，而是"隐藏因素的纯度"。** Huisman 里 h 只需要编码 1 个物种。Beninca 里 h 要同时编码缺失物种 + 模型误差 + 环境。

### disentanglement 分析给出的数字

- 79% species-specific, 21% shared
- 但 shared 中可能还包含"和某些物种相关的环境效应"
- 真正的 species signal 可能在 60-70%

## 二、交替训练为什么在 Huisman 上有害

消融结果：
- Huisman baseline 0.467 → alt_only 0.429 (-0.038)
- Beninca baseline 0.130 → alt_only 0.160 (+0.030)

**假说**：交替训练通过"保护 encoder 不被 f_visible 主导"来提升 h 的信号。

- Huisman: f_visible 能学到很好的动力学（数据是纯净的 ODE），h 的梯度不会被淹没。交替反而打断了联合优化。
- Beninca: f_visible 学不好（数据有噪声、真实动力学未知），h 的梯度极小（<0.2%），交替保护了 encoder。

**推论**：交替训练是一个"信号增强"技巧，只在 h 梯度极弱时有用。如果能直接解决梯度弱的问题（比如更好的 loss 设计），交替训练可能不需要。

## 三、h_dyn (ODE consistency) 为什么始终有用

消融结果：
- Huisman: baseline 0.467 → hdyn_only 0.525 (+0.058)
- Beninca: alt_5_1 0.160 → NbedDyn 0.162 (+0.002)

Huisman 上效果更明显。原因：
- h_dyn 压缩了 h 的有效自由度，从 T 个独立值降到"初始条件 + 动力学参数"
- 这防止 h 过拟合到训练集的特定噪声模式
- Huisman 无噪声，h_dyn 直接帮助 h 学到物种的真实动力学
- Beninca 有噪声，h_dyn 的正则化效果被噪声稀释

## 四、"平滑 vs burst"的更深层分析

### 为什么 smoothness 和 burst 对立？

在 Beninca 数据上：
- burst 占 2-15% 的时间步
- 在这些时间步，visible 的 log-ratio 很大（|actual| >> mean）
- h 在这些时间步应该也很大
- 但 smoothness 惩罚 ||d^2 h / dt^2||^2，任何快速变化都被惩罚

**核心矛盾不是 smoothness 本身，而是 smoothness 对所有时间步一视同仁。**

### 为什么之前的 event weighting 没用？

event weighting 修改的是 recon loss 的权重，不是 smoothness 的权重：
```
recon_loss = mean(w(t) * (pred - actual)^2)
```
w(t) 在 burst 时更大，强制预测在 burst 时更准。但 smoothness 没有放松：
```
smooth_loss = mean(||d^2 h||^2)  # 对所有 t 一样
```
所以即使 recon loss 强调了 burst，h 仍然被拉平。

### 真正需要的是什么？

**时变的 smoothness**：
```
smooth_loss = mean(gate(t) * ||d^2 h||^2)
```
gate(t) 在 calm 时 = 1（正常 smoothness），在 burst 时 = 0（完全放松）。

gate(t) 从哪来？从 visible 的相空间速度：
```
v(t) = ||actual_log_ratio(t)||
gate(t) = 1 - sigmoid(alpha * (v(t) - threshold))
```

但之前的 geo-gating 已经试过类似思路，无效。为什么？

**可能的原因**：geo-gating 改的是 smoothness 权重，但 ODE consistency 也在压制 burst。如果 ODE consistency 比 smoothness 更强，只改 smoothness 不够。

**解法**：smoothness 和 ODE consistency **都**要时变：
```
smooth_loss = mean(gate(t) * ||d^2 h||^2)
ode_loss = mean(gate(t) * ||h(t) - h_pred(t)||^2)
```

## 五、最有希望的改进方案

### 方案 A: 双时间尺度 h（推荐先试）

```
h = h_slow + h_fast

h_slow: ODE consistent + very smooth → 编码趋势
h_fast: sparse (L1) + weak smooth → 编码 burst

pred_i = f_vis_i(x) + (h_slow + h_fast) * G_i(x)
```

loss 分工：
- h_slow 的 smoothness = 0.2, ODE consistency = 0.5
- h_fast 的 smoothness = 0.002, L1 sparsity = 0.1, 无 ODE
- h_fast 的 energy loss 用 max(0, min_energy - var(h_fast)) 防止完全为零

**为什么这比时变 smoothness 更好？**
- 不需要"检测 burst"这个不确定的步骤
- h_slow 和 h_fast 自然分工：optimizer 会把平坦段的信号放进 h_slow（因为 ODE 奖励连贯性），把 burst 放进 h_fast（因为 h_fast 的 L1 允许偶发的大值）
- 两个分量分别评估：Pearson(h_slow, hidden) + Pearson(h_slow+h_fast, hidden) + burst_F(h_slow+h_fast, hidden)

### 方案 B: Laplace prior 替代 Gaussian（更简单）

当前 KL 用 N(0, sigma^2) 先验。换成 Laplace(0, b) 先验：
- Laplace 的 log PDF = -|h|/b，对应 L1 惩罚
- 大部分时刻 h 被拉向 0（平坦段），但偶尔的大值受的惩罚比 Gaussian 小得多
- Gaussian 对 |h|=3 的惩罚 ∝ 9，Laplace 只 ∝ 3

实现：把 KL 换成 L1：
```python
kl_laplace = (h.abs() / b).mean()   # 替代高斯 KL
```

这是最小代码改动（1 行），可能有意外效果。

### 方案 C: 直接优化 h + burst-aware recon（最激进）

完全去掉 encoder。h_t 是可学习参数：
```python
h = nn.Parameter(torch.zeros(T))   # 直接优化
```

配合：
- ODE consistency（f_h 预测 h 的动力学）
- burst-aware recon loss（burst 时刻权重大）
- L1 sparsity（大部分时刻 h~0）

这是 NbedDyn 的原始方案。优势是没有 encoder 的信息瓶颈。
劣势是 T=658 个自由参数 → 需要很强的正则化。

## 六、实验优先级

1. **方案 A（双时间尺度）** — hierarchical_h=True，代码已有，改动最小
2. **方案 B（Laplace prior）** — 1 行代码改动
3. **方案 C（直接优化 h）** — 需要较大改动

目标：在 Beninca 上同时达到 Pearson >= 0.16 AND burst F >= 0.12。
