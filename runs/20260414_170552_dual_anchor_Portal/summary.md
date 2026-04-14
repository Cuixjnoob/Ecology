# Dual-Anchor + val-select 实验

- dataset: Portal
- seeds: 1  epochs: 30

## 主表

| method | mean P | std | notes |
|---|---|---|---|
| pos_only (单向 +1) | +0.1852 | nan | 上次 SOTA |
| neg_only (单向 -1) | +0.1813 | nan | 对照 |
| **chosen (dual+val_recon)** | **+0.1852** | nan | 无监督选 |
| oracle (pick by Pearson) | +0.1852 | nan | 作弊上限 |

Chose +1 in 1/1 seeds.

## Per-seed

| seed | P_+1 | val_+1 | P_-1 | val_-1 | chose | P_chosen |
|---|---|---|---|---|---|---|
| 42 | +0.185 | 8.7058 | +0.181 | 10.1168 | +1 | +0.185 |
