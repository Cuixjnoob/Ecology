Dear Prof. Ushio and Prof. Masuda,

Thank you very much for the detailed ecological insights and for sharing the extended dataset (2002–2024). We are truly grateful for your generosity and support.

Below, we summarize our experimental results on the Maizuru Bay dataset and would greatly appreciate your ecological perspective on the recovery patterns we observed.

## Method overview

We developed an unsupervised method (Eco-GNRD) to infer hidden species dynamics from partially observed ecological time series. The setup is: given N−1 observed species, can we recover the temporal dynamics of the 1 held-out ("hidden") species, **without ever using the hidden species' data during training**? We evaluate recovery quality using the Pearson correlation between the inferred and true hidden time series on a held-out validation period (last 25% of the time series).

## Results on Maizuru Bay (Ushio et al. 2018 dataset)

We tested all 15 dominant species as the hidden target (10 random seeds each). Results are compared with an LSTM baseline and a supervised ceiling (Ridge regression with access to the hidden species during training):

| Species | Eco-GNRD (ours) | LSTM | Supervised Ridge |
|---|---|---|---|
| Pseudolabrus sieboldi | **+0.462** | +0.267 | +0.663 |
| Girella punctata | **+0.355** | +0.343 | +0.148 |
| Halichoeres tenuispinis | **+0.349** | +0.194 | +0.286 |
| Pterogobius zonoleucus | **+0.333** | -0.134 | +0.491 |
| Parajulis poecilepterus | **+0.333** | +0.040 | +0.595 |
| Trachurus japonicus | +0.299 | +0.244 | +0.555 |
| Plotosus japonicus | +0.275 | -0.010 | +0.265 |
| Sebastes inermis | +0.263 | +0.151 | +0.300 |
| Rudarius ercodes | +0.172 | +0.195 | +0.214 |
| Chaenogobius gulosus | +0.096 | +0.038 | -0.180 |
| Siganus fuscescens | +0.096 | -0.009 | +0.115 |
| Sphyraena pinguis | +0.054 | -0.010 | +0.047 |
| Engraulis japonicus | +0.022 | -0.058 | -0.072 |
| Aurelia sp. | +0.019 | +0.223 | -0.071 |
| Tridentiger trigonocephalus | -0.003 | +0.048 | -0.011 |
| **Overall mean** | **+0.208** | +0.101 | +0.223 |

## Key observations and questions

We noticed a clear pattern: **species that are ecologically well-coupled to the local community tend to be more recoverable**, while migratory or weakly-interacting species are difficult to recover. Specifically:

### 1. High-recovery species (Pearson > 0.3)

Pseudolabrus sieboldi (+0.462), Girella punctata (+0.355), Halichoeres tenuispinis (+0.349), Pterogobius zonoleucus (+0.333), and Parajulis poecilepterus (+0.333) are all recovered well.

**Question**: Are these species generally considered to be resident reef-associated fish with strong local biotic interactions (e.g., competition, predation) within the Maizuru Bay community? Would you consider them to form a relatively closed sub-community?

### 2. Low-recovery species (Pearson < 0.1)

As you noted, Engraulis japonicus (+0.022) is migratory and shows essentially no recovery — consistent with the absence of local interactions in Ushio et al. (2018) Figure 1. Similarly, Tridentiger trigonocephalus (-0.003) and Sphyraena pinguis (+0.054) show near-zero recovery.

**Question**: Are Tridentiger trigonocephalus and Sphyraena pinguis also species whose dynamics are primarily driven by factors external to the local community (e.g., migration, recruitment from outside the bay, or environmental forcing)?

### 3. Temperature sensitivity

In a separate experiment, we compared model performance with and without water temperature as input:

| Species | With temperature | Without temperature | Difference |
|---|---|---|---|
| Trachurus japonicus | +0.541 | +0.299 | -0.242 |
| Pseudolabrus sieboldi | +0.472 | +0.462 | -0.010 |
| Halichoeres tenuispinis | +0.558 | +0.349 | -0.209 |

Trachurus japonicus shows a large drop when temperature is removed, suggesting its dynamics are substantially driven by environmental forcing. Pseudolabrus sieboldi is nearly unaffected, suggesting its dynamics are primarily governed by local species interactions.

**Question**: Does this pattern align with your ecological understanding? Are there other species in the community that you would expect to be particularly temperature-sensitive or environmentally driven?

### 4. General ecological interpretation

Our method assumes that the hidden species exerts its influence through local species interactions (competition, predation, etc.). The recovery quality therefore reflects the strength and specificity of a species' coupling to the rest of the observed community.

**Question**: From your long-term observation experience, do you have a general sense of which species in Maizuru Bay are most tightly coupled to the local community versus most influenced by external processes? Any such ecological insight would be invaluable for validating and interpreting our method.

## Thank you

We are very excited about the extended 2002–2024 dataset and plan to test our method on it as well. Thank you again for your time and expertise — any feedback on the ecological plausibility of these results would be extremely helpful for our manuscript.

Best regards,
Xingji Cui
