# 核心矛盾：如何同时保证平滑和捕获高峰

## 问题定义

Beninca 数据特征：
- 85-98% 时间平坦（h 应该平滑，接近零）
- 2-15% 时间是 burst（h 必须快速大幅变化）
- 混沌动力学，burst 不是随机的——有前兆

当前所有正则化（smoothness, ODE consistency, KL）都在做同一件事：**压制 h 的快速变化**。这对平坦段是对的，但对 burst 是致命的。

实验证据：
- NbedDyn ODE consistency: Pearson 0.162 (提升) 但 burst F 0.086 (降低)
- Smoothness prior: 让 h 更平滑 → 趋势更好 → Pearson 高 → burst 被抹平

## 三种可能的解法

### 解法 1: 状态依赖的正则化强度

**核心思想**: 平坦时强正则化，burst 时弱正则化。

怎么知道什么时候该放松？**visible 的相空间速度**就是信号。

```
v(t) = ||d log x / dt||    # 可观测物种的变化速度
当 v(t) 大时 → 系统在快速变化 → 放松 h 的 smoothness
当 v(t) 小时 → 系统平稳 → 加强 h 的 smoothness
```

具体实现：
```python
speed = ||actual_log_ratio(t)||   # 已有，不需要额外计算
gate = sigmoid(alpha * (speed - threshold))  # 0=calm, 1=fast
lam_smooth_t = lam_smooth * (1 - beta * gate)  # fast时减小smoothness
```

**之前试过 geo-gating，无效**。但之前的实现是全局固定 threshold。改进：
- threshold 可学习
- 不只作用于 smoothness，也作用于 ODE consistency
- 同时在 burst 时刻增大 recon loss 的权重（event weighting 的复归）

### 解法 2: 双时间尺度 h

**核心思想**: h = h_slow + h_fast。两个分量分别正则化。

```
h_slow: 强 ODE consistency + 强 smoothness → 捕获长期趋势
h_fast: 弱 smoothness + 稀疏约束 (L1/Laplace) → 只在 burst 时非零
```

h_slow 负责 Pearson（趋势匹配），h_fast 负责 burst F-score（高峰捕获）。

实现：模型已有 `hierarchical_h=True`（slow + fast channel），但从未测试。
修改：
- slow channel: lam_smooth * 10, ODE consistency
- fast channel: lam_smooth * 0.1, L1 sparsity (大部分时刻为零)

**理论根据**: Rogers 2023 明确指出 plankton instability 是 intermittent + seasonal。slow 对应 seasonal，fast 对应 intermittent。

### 解法 3: NbedDyn 直接优化 + burst-aware loss

**核心思想**: 不用 encoder。直接优化 {h_t} 作为自由参数（NbedDyn 原始方案）。

优势：
- 没有 encoder 的信息瓶颈
- h_t 可以直接跟踪 burst 而不经过平滑的 encoder attention
- ODE consistency 仍然提供正则化

配合 burst-aware recon loss：
```python
w(t) = 1 + alpha * (|actual(t)| / mean(|actual|))^2   # burst时刻权重大
loss = mean(w(t) * (pred(t) - actual(t))^2)
```

这样 burst 时刻的重建误差被放大，优化器被迫让 h 在那些时刻有大的值。

**风险**: 直接优化 {h_t} 需要 T 个自由参数（T~658），容易过拟合。ODE consistency 是关键正则化。

## 推荐方案

**先试解法 2（双时间尺度）**，因为：
1. 代码已有（hierarchical_h），改动最小
2. 两个分量分工明确，可分别评估
3. 不需要"检测 burst"这种不确定的步骤
4. 和现有的 NbedDyn ODE consistency 自然兼容（只对 slow 加 ODE）

如果解法 2 不 work，再试解法 3（直接优化 h）。

## 具体实验设计

```
Config A: hierarchical_h=True, slow smooth=0.2, fast smooth=0.002, fast L1=0.1
Config B: hierarchical_h=True, slow ODE=0.5, fast ODE=0.0 (fast 无动力学约束)
Config C: hierarchical_h=True + event weighting on recon loss
```

每个 config 在 Beninca 上报 Pearson + burst F-score，和 NbedDyn scalar h (0.162, burst F 0.086) 对比。

目标：Pearson >= 0.16 AND burst F >= 0.12（两个都不降）。
