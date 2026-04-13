# CVHI_Residual 诊断实验 (Exp A-D)

seeds: [42, 123, 456, 789, 2024, 31415, 27182, 65537], epochs=300, top_k=3

## Portal OT

### Exp A: Multi-seed全指标

| seed | Pearson | val_recon | train_recon | m_null | m_shuf | h_var |
|---|---|---|---|---|---|---|
| 42 | +0.1483 | 0.1473 | 0.1322 | +0.0022 | +0.0002 | 0.007 |
| 123 | +0.0571 | 0.1406 | 0.1225 | +0.0100 | +0.0141 | 0.166 |
| 456 | +0.2520 | 0.1458 | 0.1318 | +0.0034 | +0.0023 | 0.080 |
| 789 | +0.0870 | 0.1377 | 0.1219 | +0.0097 | +0.0131 | 0.144 |
| 2024 | +0.0879 | 0.1451 | 0.1255 | +0.0055 | +0.0065 | 0.129 |
| 31415 | +0.2673 | 0.1454 | 0.1289 | +0.0036 | +0.0049 | 0.091 |
| 27182 | +0.1278 | 0.1450 | 0.1294 | +0.0020 | +0.0027 | 0.069 |
| 65537 | +0.2772 | 0.1455 | 0.1304 | +0.0029 | +0.0075 | 0.066 |

无监督指标与 |Pearson| 的 Spearman ρ:
- val_recon:   +0.738  (lower val → higher P = good)
- m_null:      -0.595
- m_shuf:      -0.476
- h_var:       -0.714

### Exp B: Top-3 val-selection + ensemble

- top-K 内 h_mean 两两 Pearson: C_in = +0.063
- bot-K 内 h_mean 两两 Pearson: C_out = -0.193
- top-K ensemble Pearson: +0.1139
- top-K mean Pearson: +0.0907

### Exp C: H-step prototype

- seed (used for H-step): best val seed
- 4 个 init 的 h_final 两两 Pearson 平均: +0.990
- seed 原 Pearson: +0.0870
- H-step 跨 init 最佳 Pearson: +0.2012

### Exp D: hidden_true 替代诊断

- recon_null (无 h):       0.1316
- recon_encoder:           0.1209
- recon_true (替换):       0.1335
- ratio recon_true/recon_enc: 1.104
## Synthetic LV

### Exp A: Multi-seed全指标

| seed | Pearson | val_recon | train_recon | m_null | m_shuf | h_var |
|---|---|---|---|---|---|---|
| 42 | +0.5279 | 0.0424 | 0.0301 | +0.0264 | +0.0370 | 0.405 |
| 123 | +0.9152 | 0.0469 | 0.0341 | +0.0290 | +0.0415 | 0.466 |
| 456 | +0.5772 | 0.0397 | 0.0290 | +0.0250 | +0.0341 | 0.416 |
| 789 | +0.8815 | 0.0440 | 0.0309 | +0.0250 | +0.0364 | 0.399 |
| 2024 | +0.5216 | 0.0459 | 0.0312 | +0.0240 | +0.0330 | 0.373 |
| 31415 | +0.6123 | 0.0412 | 0.0285 | +0.0271 | +0.0375 | 0.415 |
| 27182 | +0.8698 | 0.0624 | 0.0441 | +0.0144 | +0.0182 | 0.278 |
| 65537 | +0.6070 | 0.0408 | 0.0294 | +0.0300 | +0.0454 | 0.434 |

无监督指标与 |Pearson| 的 Spearman ρ:
- val_recon:   +0.405  (lower val → higher P = good)
- m_null:      +0.262
- m_shuf:      +0.262
- h_var:       +0.238

### Exp B: Top-3 val-selection + ensemble

- top-K 内 h_mean 两两 Pearson: C_in = +0.980
- bot-K 内 h_mean 两两 Pearson: C_out = -0.044
- top-K ensemble Pearson: +0.6029
- top-K mean Pearson: +0.5988

### Exp C: H-step prototype

- seed (used for H-step): best val seed
- 4 个 init 的 h_final 两两 Pearson 平均: +0.996
- seed 原 Pearson: +0.5772
- H-step 跨 init 最佳 Pearson: +0.5995

### Exp D: hidden_true 替代诊断

- recon_null (无 h):       0.0540
- recon_encoder:           0.0279
- recon_true (替换):       0.0840
- ratio recon_true/recon_enc: 3.010

## 决策树

按 Exp A/B/C/D 结果综合判定是否需要上方向 2.
