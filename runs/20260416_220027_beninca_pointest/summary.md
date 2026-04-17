# Point Estimate + NbedDyn + Alt 5:1

No VAE, no KL, no sampling. h = encoder output.
Seeds: [42, 123, 456], Epochs: 500, Alt: 5:1, lam_h_ode=0.5

| Species | Pearson | Burst_F | Margin |
|---|---|---|---|
| Cyclopoids | +0.076 | 0.091 | -0.0635 |
| Calanoids | +0.237 | 0.086 | -0.0076 |
| Rotifers | +0.074 | 0.136 | -0.1731 |
| Nanophyto | +0.181 | 0.121 | -0.0000 |
| Picophyto | +0.137 | 0.066 | -0.0569 |
| Filam_diatoms | +0.048 | 0.045 | -0.0105 |
| Ostracods | +0.211 | 0.101 | -0.0328 |
| Harpacticoids | +0.172 | 0.106 | -0.0028 |
| Bacteria | +0.106 | 0.081 | 0.0014 |

**Overall**: Pearson=+0.1380, Burst_F=0.093

Ref: NbedDyn(VAE)=+0.1620, alt_5_1=+0.1595
