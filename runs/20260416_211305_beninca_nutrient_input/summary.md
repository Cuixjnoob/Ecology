# Alternating 5:1 + Nutrients-as-Input + Burst P/R/F

Seeds: [42, 123, 456], Epochs: 500, Alt: 5:1
Recon loss: species-only (8 visible species), nutrients input-only
Burst threshold: top 10% |d log x|

| Species | Pearson | Burst_F | Burst_P | Burst_R | Margin |
|---|---|---|---|---|---|
| Cyclopoids | +0.087 | 0.066 | 0.066 | 0.066 | -0.0109 |
| Calanoids | +0.159 | 0.106 | 0.106 | 0.106 | -0.0006 |
| Rotifers | +0.071 | 0.121 | 0.121 | 0.121 | 0.0074 |
| Nanophyto | +0.097 | 0.111 | 0.111 | 0.111 | 0.0088 |
| Picophyto | +0.156 | 0.066 | 0.066 | 0.066 | 0.0042 |
| Filam_diatoms | +0.069 | 0.076 | 0.076 | 0.076 | 0.0051 |
| Ostracods | +0.351 | 0.126 | 0.126 | 0.126 | -0.0094 |
| Harpacticoids | +0.150 | 0.096 | 0.096 | 0.096 | 0.0081 |
| Bacteria | +0.161 | 0.106 | 0.106 | 0.106 | 0.0166 |

**Overall**: Pearson=+0.1446, Burst_F=0.097, Burst_P=0.097, Burst_R=0.097

Ref: S1b Pearson = +0.132
