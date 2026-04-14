# 完整逐组件 Ablation 实验报告

- seeds: 20
- epochs: 300
- 总训练次数: 300

## 主结果表

| Ablation | Dataset | mean P | std | 95% CI | max | d_ratio |
|---|---|---|---|---|---|---|
| full | LV | +0.7273 | 0.2163 | [+0.626, +0.829] | +0.918 | 6.452 |
| full | Holling | +0.7380 | 0.1897 | [+0.649, +0.827] | +0.955 | 13.329 |
| full | Portal | +0.1398 | 0.0870 | [+0.099, +0.181] | +0.307 | 1.055 |
| no_hints | LV | +0.7900 | 0.1968 | [+0.698, +0.882] | +0.927 | 5.431 |
| no_hints | Holling | +0.7368 | 0.2074 | [+0.640, +0.834] | +0.926 | 16.498 |
| no_hints | Portal | +0.1024 | 0.0702 | [+0.070, +0.135] | +0.301 | 1.043 |
| no_rollout | LV | +0.7351 | 0.2002 | [+0.641, +0.829] | +0.912 | 3.263 |
| no_rollout | Holling | +0.7363 | 0.1912 | [+0.647, +0.826] | +0.941 | 10.272 |
| no_rollout | Portal | +0.1230 | 0.0822 | [+0.085, +0.161] | +0.276 | 1.029 |
| no_residual | LV | +0.4189 | 0.2283 | [+0.312, +0.526] | +0.792 | 9.753 |
| no_residual | Holling | +0.5757 | 0.2087 | [+0.478, +0.673] | +0.942 | 5.291 |
| no_residual | Portal | +0.1534 | 0.0794 | [+0.116, +0.191] | +0.296 | 1.261 |
| no_cf | LV | +0.5920 | 0.2945 | [+0.454, +0.730] | +0.923 | 3.163 |
| no_cf | Holling | +0.6654 | 0.2153 | [+0.565, +0.766] | +0.912 | 9.974 |
| no_cf | Portal | +0.0474 | 0.0311 | [+0.033, +0.062] | +0.136 | 1.002 |

## Δ vs full baseline

| Ablation | Dataset | Δ mean P | Δ d_ratio |
|---|---|---|---|
| no_hints | LV | +0.0627 | -1.020 |
| no_hints | Holling | -0.0011 | +3.169 |
| no_hints | Portal | -0.0374 | -0.012 |
| no_rollout | LV | +0.0078 | -3.188 |
| no_rollout | Holling | -0.0017 | -3.058 |
| no_rollout | Portal | -0.0168 | -0.027 |
| no_residual | LV | -0.3084 | +3.301 |
| no_residual | Holling | -0.1623 | -8.038 |
| no_residual | Portal | +0.0136 | +0.206 |
| no_cf | LV | -0.1353 | -3.288 |
| no_cf | Holling | -0.0725 | -3.355 |
| no_cf | Portal | -0.0924 | -0.054 |

## 配置说明

- **full**: {'use_formula_hints': True, 'use_G_field': True, 'use_rollout': True, 'use_cf': True}
- **no_hints**: {'use_formula_hints': False, 'use_G_field': True, 'use_rollout': True, 'use_cf': True}
- **no_rollout**: {'use_formula_hints': True, 'use_G_field': True, 'use_rollout': False, 'use_cf': True}
- **no_residual**: {'use_formula_hints': True, 'use_G_field': False, 'use_rollout': True, 'use_cf': True}
- **no_cf**: {'use_formula_hints': True, 'use_G_field': True, 'use_rollout': True, 'use_cf': False}
