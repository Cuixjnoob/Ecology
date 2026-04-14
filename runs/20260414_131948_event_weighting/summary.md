# Event Weighting (Exp1) 结果

- seeds: 5  epochs: 300

| config | dataset | mean P | std | 95% CI | max | d_ratio |
|---|---|---|---|---|---|---|
| uniform | LV | +0.8053 | 0.0747 | [+0.713, +0.898] | +0.873 | 5.232 |
| uniform | Holling | +0.6585 | 0.2314 | [+0.371, +0.946] | +0.849 | 11.233 |
| uniform | Portal | +0.1391 | 0.1371 | [-0.031, +0.309] | +0.304 | 1.021 |
| event_a1 | LV | +0.7918 | 0.1374 | [+0.621, +0.962] | +0.892 | 5.185 |
| event_a1 | Holling | +0.6739 | 0.2274 | [+0.392, +0.956] | +0.858 | 11.811 |
| event_a1 | Portal | +0.1918 | 0.1165 | [+0.047, +0.336] | +0.351 | 1.020 |
| event_a2 | LV | +0.8583 | 0.0754 | [+0.765, +0.952] | +0.914 | 5.055 |
| event_a2 | Holling | +0.6697 | 0.2269 | [+0.388, +0.951] | +0.873 | 11.803 |
| event_a2 | Portal | +0.0959 | 0.0526 | [+0.031, +0.161] | +0.166 | 1.099 |

## Δ vs uniform

| config | dataset | Δ mean P |
|---|---|---|
| event_a1 | LV | -0.0135 |
| event_a1 | Holling | +0.0154 |
| event_a1 | Portal | +0.0527 |
| event_a2 | LV | +0.0530 |
| event_a2 | Holling | +0.0112 |
| event_a2 | Portal | -0.0431 |
