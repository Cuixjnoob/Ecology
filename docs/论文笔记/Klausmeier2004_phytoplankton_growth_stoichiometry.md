# [Klausmeier et al. 2004] Phytoplankton Growth and Stoichiometry under Multiple Nutrient Limitation

## 基本信息
- **作者**: Christopher A. Klausmeier, Elena Litchman, Simon A. Levin
- **期刊/来源**: Limnology and Oceanography, 49(4, part 2), pp. 1463-1470
- **年份**: 2004

## 研究问题
浮游植物在多营养盐限制下的生长和化学计量如何决定？灵活化学计量模型能否统一解释经典恒化器实验(Rhee 1978; Goldman et al. 1979)中的结果？营养盐耗竭动态的规律是什么？

## 核心方法
- 构建多营养盐灵活化学计量数学模型：结合恒化器营养盐供给、Michaelis-Menten摄取动力学和Droop生长方程，通过Liebig最小定律耦合
- 模型分析分为两个阶段：指数生长期(exponential phase)和平衡期(equilibrium phase)
- 数值求解和解析近似与Scenedesmus实验数据定量对比
- 参数基于Rhee (1974, 1978)的实测值

## 关键发现与结论
1. **两阶段动态**: 浮游植物生长明确分为指数增长和趋近平衡两个阶段，各有不同的化学计量行为
2. **指数增长期**: 
   - 化学计量由摄取速率比决定：Q₁/Q₂ = f₁(R₁)/f₂(R₂)
   - 在最优摄取假设下，Q_N/Q_P ≈ Q_min,N/Q_min,P(结构性比率)
   - 浮游植物"吃什么就是什么"(are what they eat)
3. **平衡期**: 
   - 化学计量由限制性营养盐决定
   - 非限制性营养盐的配额取决于生长率
   - 化学计量在N限制(低N:P)和P限制(高N:P)下分化
4. **与经典实验的对比**: 模型定量再现了Rhee(固定稀释率变化N:P供给)和Goldman(固定供给变化稀释率)的实验结果
5. **营养盐耗竭**: 灵活化学计量模型预测两种营养盐可同时被降到低浓度(与固定化学计量模型预测不同)
6. **模型局限**: 在极端N:P供给比时拟合不佳，需要配额对摄取的负反馈

## 重要公式/概念
- **完整模型** (公式1):
  - dR_i/dt = a(R_in,i - R_i) - f_i(R_i)B
  - dQ_i/dt = f_i(R_i) - μ_∞·min(1-Q_min,1/Q_1, 1-Q_min,2/Q_2)·Q_i
  - dB/dt = μ_∞·min(...)·B - mB
- **Droop方程**: μ = μ_∞(1 - Q_min/Q)
- **Michaelis-Menten摄取**: f_i(R_i) = V_max,i·R_i/(R_i + K_i)
- **最优摄取假设**: V_max,1/V_max,2 = Q_min,1/Q_min,2
- **R*竞争理论**: 平衡时限制性资源浓度 R* = K·m·Q_min/(μ_∞·V_max - m·Q_min - V_max)
- **Redfield比**: C:N:P = 106:16:1的平均浮游植物化学计量

## 与我们项目的关联
- **动力学方程**: 本文的Droop-Liebig-Michaelis-Menten模型可直接作为我们浮游植物生长动力学的基础方程
- **参数化**: 提供了Scenedesmus的完整参数集(表1)，可作为模型参数化的参考
- **多营养盐动态**: 多营养盐限制产生的非线性动态与我们关注的混沌生态动力学相关
- **Hidden recovery**: 营养盐浓度的变化模式(两阶段耗竭)可作为推断未观测浮游植物状态的线索
- **化学计量作为状态指标**: 浮游植物N:P比反映其生长条件(指数增长 vs. 营养限制)，可为GNN提供物理约束

## 一句话总结
Klausmeier等人用灵活化学计量数学模型(Droop-Liebig)定量再现了浮游植物在多营养盐限制下的生长两阶段动态，揭示了指数增长期(化学计量由摄取率决定)和平衡期(化学计量由限制性营养盐决定)的本质差异。
