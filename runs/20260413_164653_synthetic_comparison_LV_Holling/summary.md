# CVHI_Residual (无预设公式) 在 LV vs Holling 合成数据上对比

seeds: [42, 123, 456, 789, 2024], epochs=300, config=L1only_K3

## 汇总

| Dataset | mean P | max P | std P | mean d_ratio |
|---|---|---|---|---|
| LV | +0.7519 | +0.9059 | 0.0849 | 4.472 |
| Holling | +0.2238 | +0.9222 | 0.3517 | 3.890 |

## 各 seed 详细

### LV

| seed | Pearson | d_ratio | val_recon | h_var |
|---|---|---|---|---|
| 42 | +0.9059 | 1.170 | 0.0264 | 0.911 |
| 123 | +0.7382 | 2.023 | 0.0238 | 0.940 |
| 456 | +0.6756 | 6.034 | 0.0209 | 0.908 |
| 789 | +0.7660 | 6.953 | 0.0206 | 0.921 |
| 2024 | +0.6737 | 6.178 | 0.0217 | 0.858 |

### Holling

| seed | Pearson | d_ratio | val_recon | h_var |
|---|---|---|---|---|
| 42 | +0.1249 | 4.756 | 0.1771 | 3.604 |
| 123 | +0.0017 | 5.045 | 0.1923 | 3.371 |
| 456 | +0.0507 | 4.600 | 0.1990 | 3.515 |
| 789 | +0.9222 | 1.253 | 0.1994 | 3.573 |
| 2024 | +0.0194 | 3.798 | 0.1807 | 3.392 |

## Form selection analysis (per layer, averaged across seeds)

### LV

**f_visible**:

| form | mean effective | std |
|---|---|---|
| layer1_LV_bilin | 0.0166 | 0.0012 |
| layer1_Linear | 0.0166 | 0.0017 |
| layer1_HollingII_lin | 0.0163 | 0.0017 |
| layer0_Linear | 0.0154 | 0.0018 |
| layer0_LV_bilin | 0.0152 | 0.0013 |
| layer0_HollingII_lin | 0.0151 | 0.0021 |
| layer1_HollingII_bilin | 0.0139 | 0.0010 |
| layer0_HollingII_bilin | 0.0135 | 0.0015 |
| layer1_FreeNN | 0.0001 | 0.0001 |
| layer0_FreeNN | 0.0001 | 0.0001 |

**G_field**:

| form | mean effective | std |
|---|---|---|
| layer0_Linear | 0.0116 | 0.0005 |
| layer0_HollingII_lin | 0.0108 | 0.0006 |
| layer0_LV_bilin | 0.0107 | 0.0010 |
| layer0_HollingII_bilin | 0.0099 | 0.0009 |
| layer0_FreeNN | 0.0000 | 0.0000 |

### Holling

**f_visible**:

| form | mean effective | std |
|---|---|---|
| layer0_HollingII_lin | 0.0178 | 0.0011 |
| layer1_HollingII_lin | 0.0177 | 0.0021 |
| layer0_Linear | 0.0177 | 0.0010 |
| layer1_Linear | 0.0177 | 0.0018 |
| layer0_LV_bilin | 0.0176 | 0.0008 |
| layer1_LV_bilin | 0.0168 | 0.0008 |
| layer0_HollingII_bilin | 0.0156 | 0.0008 |
| layer1_HollingII_bilin | 0.0150 | 0.0019 |
| layer1_FreeNN | 0.0001 | 0.0001 |
| layer0_FreeNN | 0.0001 | 0.0001 |

**G_field**:

| form | mean effective | std |
|---|---|---|
| layer0_HollingII_lin | 0.0152 | 0.0014 |
| layer0_Linear | 0.0145 | 0.0009 |
| layer0_LV_bilin | 0.0140 | 0.0009 |
| layer0_HollingII_bilin | 0.0140 | 0.0009 |
| layer0_FreeNN | 0.0000 | 0.0000 |

