# Alternating training experiment

Seeds: [42, 123, 456], Epochs: 500

| Config | pretrain | phase_A | phase_B |
|---|---|---|---|
| baseline | 0 | 0 | 0 |
| alt_3_1 | 0 | 3 | 1 |
| alt_5_1 | 0 | 5 | 1 |
| pretrain+alt | 150 | 3 | 1 |

| Species | baseline | alt_3_1 | alt_5_1 | pretrain+alt |
|---|---|---|---|---|
| Cyclopoids | +0.061 | +0.079 | +0.095 | +0.122 |
| Calanoids | +0.178 | +0.166 | +0.248 | +0.253 |
| Rotifers | +0.083 | +0.060 | +0.089 | +0.047 |
| Nanophyto | +0.124 | +0.082 | +0.143 | +0.121 |
| Picophyto | +0.121 | +0.126 | +0.145 | +0.117 |
| Filam_diatoms | +0.081 | +0.066 | +0.060 | +0.072 |
| Ostracods | +0.223 | +0.288 | +0.305 | +0.188 |
| Harpacticoids | +0.195 | +0.203 | +0.233 | +0.170 |
| Bacteria | +0.105 | +0.123 | +0.119 | +0.172 |

**Overall**: baseline=+0.1300, alt_3_1=+0.1325, alt_5_1=+0.1595, pretrain+alt=+0.1403

**Avg margin**: baseline=-0.0006, alt_3_1=0.0028, alt_5_1=0.0046, pretrain+alt=0.0019

Best: **alt_5_1** = +0.1595
