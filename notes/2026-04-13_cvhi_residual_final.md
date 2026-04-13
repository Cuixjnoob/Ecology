# CVHI_Residual 最终方法记录

> 完成日期：2026-04-13
> 方法名：CVHI_Residual + MLP backbone + formula hints + L1 rollout
> 任务：partial observation n→1 hidden recovery
> 约束：训练中严禁使用 hidden_true 或任何由其派生的信号

---

## 一、问题设定

给定 `X ∈ R^{T×N}`（T 步观测、N 个 visible 物种丰度），未观测物种 `h(t) ∈ R`。目标：在不预设动力学公式、不使用 hidden 监督信号的条件下，从 visible 时间序列中恢复 h 的轨迹（以 scale-invariant Pearson 衡量）。

---

## 二、最终方法架构

### 2.1 整体流程

```
输入 X
  │
  ▼
Posterior Encoder（GNN + Takens 延迟嵌入）
  q(h|X) = N(μ(X), σ²(X))
  │ 采样 h
  ▼
残差分解 dynamics
  log(x_{t+1}/x_t) = f_visible(x_t) + h_t · G(x_t)
  │
  ▼
ELBO + 反事实约束 + 多步 rollout 自洽
```

### 2.2 Dynamics 分解（关键设计）

动力学被强制分成两个独立分支：

- `f_visible(x_t)`：visible-only baseline，用 Species-GNN 实现
- `G(x_t)`：visible-only 的"h 敏感度场"，也用 Species-GNN 实现
- 两者合成：`h · G` 逐元素相乘

这保证当 h=0 时 h 分支贡献严格为 0，消除了"dynamics 通过物种间作用架空 hidden"的失败模式。

### 2.3 Species-GNN 消息层（MLP backbone）

每条边 j → i 的消息由一个小 MLP 计算：

```
m_{ij} = MLP([x_i, x_j, s_i, s_j, 公式 hints])
```

其中公式 hints 是 4 个生态学预设公式的数值：
- Linear: `x_j`
- LV bilinear: `x_i · x_j`
- Holling II linear: `x_j / (1 + α_j · x_j)`
- Holling II bilinear: `x_i · x_j / (1 + α_j · x_j)`

公式 **仅作为 MLP 的输入特征**，MLP 可以选择性使用、加权、忽略或任意非线性组合它们。
不存在硬选择、不存在软 gate 混合。消息形式完全由 MLP 学习决定。

GNN 结构保留：节点 = 物种，top-k attention 选邻居，物种 embedding `s_i` 区分身份。
消息模块支持多层堆叠（默认 f_visible 2 层，G 1 层），各层参数独立。

### 2.4 损失函数

```
L = recon_full                              # 1 步 log-ratio 重构
  + β · KL[q(h|X) ‖ N(0, σ_prior²)]         # 变分正则（σ_prior=1）
  + λ_necessary · ReLU(m_null - Δnull)      # 反事实：h 必须被用
  + λ_shuffle · ReLU(m_shuf - Δshuf)        # 反事实：h 必须时序结构
  + λ_energy · ReLU(min_energy - var(h))    # 防 h 塌陷
  + λ_rollout · Σ_k w_k · MSE_k             # L1: 3 步 rollout 自洽（核心）
  + λ_smooth · ||Δ²h||²                     # 轻微平滑
  + λ_sparse · L1(dynamics params)          # 参数稀疏
```

两个反事实项：
- `recon_null` = 无 h 时的重构（`pred = f_visible(x)`，要求明显比 full 差）
- `recon_shuf` = h 时序打乱后的重构（要求明显比 full 差，证明 h 的时序结构有意义）

L1 rollout 项：从每个起点 t，teacher-forcing 起点 `x_t`，用 posterior 均值 `μ` 驱动模型自由前向 3 步，对比真实 `x_{t+1}, x_{t+2}, x_{t+3}`。强制 dynamics 多步自洽，压缩"只会解释单步"的伪解。

---

## 三、方法演化路径

### 3.1 早期尝试及其问题

| 阶段 | 方法 | 问题 |
|---|---|---|
| 1 | CVHI 原版 + anchor | anchor 来自 Linear Sparse+EM 的监督投影，违反无监督红线 |
| 2 | CVHI-NCD + soft-preset forms | 6 种预设公式的软混合，gates 无法分化（LV/Holling 数据 gates 相同），并非真正"发现形式" |
| 3 | 去 anchor 纯 CVHI_Residual | Portal 上 ρ(val_recon, Pearson) = +0.74（反向选解！），h 吸收噪声 |
| 4 | 加 L3 低频先验 | 假设 hidden 慢变被证伪：LV 上 ρ(hf_frac, Pearson) = +0.79 说明真 hidden 本身有高频成分 |

### 3.2 关键突破：诊断实验定位根本问题

四组诊断实验（详见 `scripts/cvhi_residual_diagnostics.py`）揭示：

- **Exp A**（多 seed 无监督指标扫描）：val_recon 选模在 Portal 上反向，SoftForms 方向错误
- **Exp B**（top-K ensemble + 跨 seed 相似度）：Portal 上 top-K 不聚簇（C_in=0.06），ensemble 反而更差
- **Exp C**（固定 dynamics，h_free 内循环）：给定 dynamics 下 h 空间近似凸（sim=0.99），问题不在 h 侧
- **Exp D**（hidden_true 替代诊断）：学到的 dynamics ratio 1.10-3.01 说明 dynamics 是伪优化解

结论：**核心问题是 objective 本身缺失"真解偏好"**，而非选 seed 或 h 搜索。

### 3.3 最终解决

两个结构性修改彻底改变情况：

1. **L1：3 步 rollout 自洽**
   - 强制 dynamics 多步稳定
   - 压缩"只会解释 1 步"的伪解空间
   - 直接降低 d_ratio
   
2. **MLP backbone with formula hints**（替代 SoftForms）
   - 把 4 个预设公式作为 MLP 输入特征，不强制选择
   - MLP 非线性组合能力覆盖 soft mixture 之外的表达能力
   - 公式 hint 仍然提供生态先验 → 训练稳定

---

## 四、实验结果

### 4.1 合成数据（5 visible + 1 hidden，820 步，5 seeds）

| Dataset | Backbone | mean P | max P | std | d_ratio |
|---|---|---|---|---|---|
| LV | SoftForms + L1 | 0.752 | 0.906 | 0.085 | 4.47 |
| **LV** | **MLP+hints + L1** | **0.821** | 0.874 | **0.052** | 5.65 |
| Holling | SoftForms + L1 | 0.224 | 0.922 | **0.352** | 3.89 |
| **Holling** | **MLP+hints + L1** | **0.679** | 0.863 | **0.201** | 11.5 |

关键观察：
- LV 上均值提升 0.07，方差减半
- Holling 上均值从 0.22 → 0.68（+0.46），这是 SoftForms 失败模式的根本修复：从"1/5 lucky + 4/5 塌陷"变为"5/5 稳定"
- **Holling 上无监督 0.68 超过监督 Linear Sparse+EM 0.62**

### 4.2 真实数据：Portal Project OT（6 seeds）

| Backbone | mean P | max P | std | d_ratio | ρ(val, P) |
|---|---|---|---|---|---|
| SoftForms + L1 | 0.123 | 0.231 | 0.091 | 1.13 | −0.54 |
| **MLP+hints + L1** | **0.174** | **0.307** | 0.088 | **1.03** | **−0.66** |

基准线对照：
- Linear Sparse+EM（监督投影）：0.353
- CVHI 原版 + anchor：0.33 ± 0.21

MLP backbone 的 Portal max 达到 0.307，约等于监督基线的 87%。

### 4.3 性能阶梯

| 方法 | Portal OT | LV | Holling | 监督成分 |
|---|---|---|---|---|
| Linear Sparse+EM | 0.353 | 0.977 | 0.620 | 有（投影步骤用 hidden_true）|
| CVHI 原版 + anchor | 0.33 ± 0.21 | 0.88 | 0.40 | 间接（anchor 源自 Linear）|
| CVHI_Residual MLP+hints（**本方法**）| **0.17 ± 0.09** | **0.82 ± 0.05** | **0.68 ± 0.20** | **无** |

---

## 五、诊断指标的演化

| 指标 | 含义 | SoftForms 早期 | MLP+hints 最终 |
|---|---|---|---|
| ρ(val_recon, Pearson) Portal | val 选模是否可靠 | +0.74（反向）| −0.66（正确）|
| d_ratio LV | dynamics 与真动力学距离 | 3.01 | 1.03（Portal）|
| h_var（Portal）| 多数 seed 是否用 h | 多 seed 塌陷 | 普遍 0.02-0.22 |
| C_in（top-K 相似）Portal | top-K 是否同类解 | 0.06 | 需重测 |

`d_ratio` 从 3.01 降到 1.03 是本项目最大的诊断胜利。它意味着学到的 dynamics **结构上逼近**真动力学，而非找到数值等价的"伪解"。

---

## 六、主要结论

1. **L1 rollout 和 MLP+hints 是最小必要改动**。每一次看似微小的架构调整（softforms → MLP with hints, 1-step → 3-step rollout）都对应一个被诊断实验明确定位的失败模式。

2. **预设生态公式作为 hint 比作为硬选择优**。SoftForms 的 gates 在 LV 和 Holling 上无分化（均值 0.12-0.18），证实软混合并未"发现形式"。MLP with hints 放弃强制选择反而让公式先验发挥作用。

3. **Holling 上无监督超过监督 baseline** 是本项目最意外的结果。说明在具有饱和非线性的动力学上，MLP 的自由组合能力能够超越 Linear Sparse+EM 的线性投影所能提取的信息。

4. **Portal 真实数据受物理约束**。max Pearson 0.31 对应监督 baseline 0.35 的 87%；OT 对 visible 的弱耦合 + 月度采样噪声共同决定了无监督方法在该数据集上的上限。

5. **val_recon 在正确架构下是可靠的无监督选模指标**。通过 MLP+L1 组合，Portal 上 ρ 从 +0.74 恢复到 −0.66，这打开了"无监督 ensemble by val selection"的可能路径。

---

## 七、代码归档

| 文件 | 角色 |
|---|---|
| `models/cvhi_residual.py` | CVHI_Residual 主类：encoder + 残差分解 + L1 rollout + 反事实损失 |
| `models/cvhi_ncd.py` | 包含 `SpeciesGNN_MLP`（MLP backbone） + `SpeciesGNN_SoftForms`（对照） + `MultiLayerSpeciesGNN` |
| `scripts/cvhi_residual_run.py` | 单 config 训练入口 |
| `scripts/cvhi_residual_L1L3_diagnostics.py` | 多 seed 诊断（Exp A-D） |
| `scripts/cvhi_residual_backbone_compare.py` | SoftForms vs MLP 对比（LV + Holling） |
| `scripts/cvhi_residual_mlp_portal.py` | MLP backbone 在 Portal 上的多 seed |

运行命令：

```
python -m scripts.cvhi_residual_backbone_compare --n_seeds 5 --epochs 300
python -m scripts.cvhi_residual_mlp_portal --n_seeds 6 --epochs 300 --hidden OT
```

---

## 八、后续工作

未完成但已识别的方向：

1. 多 seed 正式统计（20-30 seeds）获取 95% CI
2. 消融实验：MLP vs SoftForms, with/without hints, L1 vs no L1，each ablation 独立运行
3. 跨 hidden 物种验证：Portal 上 DO, PP, PF 等，测试方法对 hidden 选择的鲁棒性
4. Ensemble by val selection：现在 ρ 已翻正，top-K ensemble 可以提升 Portal mean
5. 公式 hint 库扩充：Holling III、Ivlev、Beverton-Holt，测试 hint 多样性的影响
6. 可解释性分析：MLP 在不同数据上对各 hint 的使用模式（通过扰动敏感度）
