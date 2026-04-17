# 夜跑计划 (2026-04-14 night → 2026-04-15 morning)

## 已读论文综合 (Agent 深读结果)

### MTE 正确应用

**关键纠错** (来自 Brown / Glazier / Kremer / Clarke):

| 物种类型 | Glazier b | r 标度 M^(b-1) | 备注 |
|---|---|---|---|
| Bacteria (unicell, pelagic) | 0.60 | M^(-0.40) | Clarke: 实际可能 ≥1.0 (superlinear) |
| Phytoplankton (Pico/Nano/Filam) | **0.88** | M^(-0.12) | **Kremer 2017 实测 -0.054, 甚至更弱** |
| Copepods (Calan/Cyclop/Harp, pelagic) | **0.88** | M^(-0.12) | 非 0.75! Pelagic boost |
| Rotifers (small pelagic) | 0.88 | M^(-0.12) | 同上 |
| Ostracods | 0.75 | M^(-0.25) | 非 pelagic |

**我 Stage 1 用的 universal -0.25 是对 pelagic 物种的 5 倍过强!**

### 应用位置 (Clarke 明确说)

✓ **f_visible diagonal (intrinsic growth)**: 弱 shape prior  
✗ **G (hidden coupling)**: MTE 不说什么 → 不该约束  
✗ **绝对 rate**: B_0 不可识别  
✓ **Only exponent / shape**: OK  
✓ **Body mass as feature**: let model learn  

### Beninca chaos 特性

- Lyapunov ≈ 0.05 /day, 1/λ ≈ 15-25 days
- h 的 smoothness 不应 > 15 day scale

### Klausmeier Droop (Stage 2)

- Phyto + Nutrient 耦合结构:
  ```
  μ_i = μ_∞,i · min_j (1 - Q_min,ij/Q_ij)  # Liebig min
  Q_i̇ = V_max,i R_i/(R_i+K_i) - μ_i Q_i
  ```
- 实施: nutrient → phyto 正耦合 (soft sign prior)
- phyto → nutrient 负耦合 (消耗)

---

## 夜跑 Timeline (估计 7-8h 总)

```
[00:00 - 00:30]   Stage 1c 设计 + 实施
[00:30 - 01:30]   Stage 1c 运行 (45 min) + 分析
[01:30 - 02:30]   Stage 2 Klausmeier 设计 + 实施 + 启动
[02:30 - 03:30]   Stage 2 运行 + 分析
[03:30 - 05:30]   大 ablation 启动 (8 configs × 9 species × 5 seeds)
[05:30 - 07:00]   Ablation 运行 (2h compute)
[07:00 - 08:00]   Report 写作 + 痛点分析 + brainstorm
```

---

## Stage 1c 设计 (corrected MTE)

### 修改
1. **Body mass 作 species feature**: 初始化 species_emb[:, 0] 为 log10(M)
2. **Weak MTE shape prior**: 
   - 在 f_visible baseline 上 (不 在 G)
   - 只约束方向 (correlation sign), 不约束绝对
   - λ = 0.02 (小)
3. **Taxon-specific b** (用 Glazier + Kremer 值):
   - Bacteria: b=0.6
   - Phyto 3 种: b=0.88 → but Kremer says -0.054, 用保守 -0.05
   - Zoopl 5 种: b=0.88 (pelagic) → -0.12
4. **保留 RMSE + aug** (Stage 1b 证实有效)
5. **不动 G**

### 预期 Pearson
```
Ostracods (已 0.272): 可能保持
Bacteria (0.197):     可能持平
Nanophyto (0.141):    可能回到 0.20+
Overall:              0.14-0.18 (vs Stage 1b 0.132)
```

---

## Stage 2 设计 (Klausmeier)

### 修改
1. 加 soft sign prior on f_visible 的 nutrient-phyto 耦合:
   - d(log Pico) / d(NO3 level) > 0
   - d(log NO3) / d(Pico level) < 0
2. 只约束 phytoplankton (Pico, Nano, Filam) 和 nutrients (NO2, NO3, NH4, SRP)
3. 弱 prior (λ ≈ 0.02)

### 预期 Pearson
```
Phyto 物种:      可能 +0.03-0.05 (NO3/SRP 是 phyto 的关键)
Nutrient hidden: 若测, 可能 +0.05
Overall:         +0.02-0.04 叠加
```

---

## 大 Ablation 设计

### Configs
- A0: Phase 2 baseline (无新组件)
- A1: +RMSE+aug (Stage 1b, 已跑)
- A2: +MTE corrected (Stage 1c)
- A3: +Klausmeier (Stage 2)
- A4: +EMA + Snapshot
- A5: +Hierarchical h
- A6: A1+A2+A3 (生态组合)
- A7: All (ecology + classical + architectural)

### 规模
- 8 configs × 9 species × 5 seeds = 360 runs
- ~30s/run = ~180 min = 3 hours

### 分析
- 每 component 边际贡献
- 最佳组合
- Per-species diff
