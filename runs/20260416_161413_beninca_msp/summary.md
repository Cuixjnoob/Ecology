# MSP + Alternating training experiment

Seeds: [42, 123, 456], Epochs: 500

| Config | alt | pa | pb | mask | n_mask |
|---|---|---|---|---|---|
| baseline | False | 0 | 0 | False | 0 |
| alt_5_1 | True | 5 | 1 | False | 0 |
| msp_joint | False | 0 | 0 | True | 1 |
| msp_alt | True | 5 | 1 | True | 1 |
| msp_alt_2 | True | 5 | 1 | True | 2 |

| Species | baseline | alt_5_1 | msp_joint | msp_alt | msp_alt_2 |
|---|---|---|---|---|---|
| Cyclopoids | +0.061 | +0.095 | +0.076 | +0.117 | +0.069 |
| Calanoids | +0.178 | +0.248 | +0.028 | +0.159 | +0.150 |
| Rotifers | +0.083 | +0.089 | +0.021 | +0.101 | +0.101 |
| Nanophyto | +0.124 | +0.143 | +0.065 | +0.065 | +0.097 |
| Picophyto | +0.121 | +0.145 | +0.054 | +0.125 | +0.144 |
| Filam_diatoms | +0.081 | +0.060 | +0.062 | +0.036 | +0.082 |
| Ostracods | +0.223 | +0.305 | +0.141 | +0.308 | +0.297 |
| Harpacticoids | +0.195 | +0.233 | +0.061 | +0.126 | +0.157 |
| Bacteria | +0.105 | +0.119 | +0.060 | +0.170 | +0.157 |

**Overall**: baseline=+0.1300, alt_5_1=+0.1595, msp_joint=+0.0630, msp_alt=+0.1340, msp_alt_2=+0.1394

Best: **alt_5_1** = +0.1595
