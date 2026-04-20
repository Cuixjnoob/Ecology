Dear Prof. Hilker,

Thank you very much for your reply. Your point about system-dependence is exactly what we are observing, and I would like to share some concrete results that may interest you.

## What we do

We developed an unsupervised method that, given N-1 observed species time series, attempts to infer the dynamical influence of 1 unobserved ("hidden") species. The model decomposes the dynamics as:

$$\log(x_{t+1,k}/x_{t,k}) = f_{\text{visible},k}(x_t) + h_t \cdot G_k(x_t)$$

where $f_{\text{visible}}$ captures observable species interactions, $h_t$ is a latent scalar (inferred hidden influence), and $G_k$ is a per-species sensitivity field indicating how strongly the hidden species affects each visible species. Crucially, $h_t$ is never supervised — the model learns it purely from the visible dynamics.

## Results on simulated data (Huisman & Weissing 1999)

On the 6-species resource competition system (known ground truth), we can:

1. **Recover hidden species trajectories** with Pearson correlation up to +0.64 on held-out validation data (mean +0.42 across all 6 species in leave-one-out rotation, 10 seeds).

2. **Recover the interaction structure**: The learned sensitivity field $G_k$ matches the true resource-mediated competition pattern. When we compare the learned $G$ with the effective competition coefficients derived from the consumption matrix $C$ and half-saturation matrix $K$, we obtain Spearman correlations of +0.55 to +1.00 (4/6 species significant at p < 0.02). This is something black-box methods like Latent ODE cannot provide.

3. **Counterfactual necessity**: Removing the hidden variable ($h=0$) causes prediction accuracy to collapse from +0.42 to +0.02, confirming that $h$ captures dynamically necessary information, not noise.

## What do we actually recover on real data?

This is where your remark about "relative strengths of ongoing processes" becomes very relevant. We tested on three real datasets:

- **Beninca et al. 2008** (Baltic mesocosm, 9 plankton species): mean Pearson +0.15
- **Maizuru Bay fish community** (Ushio et al. 2018, 15 species): mean Pearson +0.21
- **Blasius et al. 2020** (chemostat predator-prey, 9 experiments): mean Pearson +0.26

However, a cross-species specificity analysis revealed an important nuance: the recovered signal $h_t$ is often not uniquely attributable to the specific hidden species. Instead, it captures a mixture of:

- **Species-specific interaction effects** (dominant in closed systems like Huisman and the chemostat)
- **Shared environmental drivers** (dominant in open field systems like Maizuru, where temperature drives most species)

In the Maizuru dataset, we performed a temperature ablation: running the model with and without sea surface temperature as input. This cleanly separates environmentally-driven from interaction-driven recovery:

| Species | With temperature | Without temperature | Δ | Interpretation |
|---|---|---|---|---|
| Trachurus japonicus | +0.541 | +0.299 | −0.242 | Temperature-driven |
| Halichoeres tenuispinis | +0.558 | +0.349 | −0.209 | Temperature-driven |
| Parajulis poecilepterus | +0.609 | +0.333 | −0.276 | Temperature-driven |
| **Pseudolabrus sieboldi** | **+0.472** | **+0.462** | **−0.010** | **Interaction-driven** |
| **Girella punctata** | **+0.318** | **+0.355** | **+0.037** | **Interaction-driven** |
| Overall (15 species) | +0.277 | +0.208 | −0.069 | |

Species like *Pseudolabrus sieboldi* and *Girella punctata* show virtually no change when temperature is removed, indicating their recovery is driven by local species interactions rather than environmental forcing. In contrast, *Trachurus japonicus* and *Parajulis poecilepterus* lose much of their recovery without temperature, suggesting their dynamics are primarily environmentally driven.

Prof. Ushio (the original dataset author) independently confirmed that *Pseudolabrus sieboldi* is a temperature-insensitive reef-resident fish, while *Engraulis japonicus* (Pearson +0.02, near zero) is a migratory species with no significant local biotic interactions.

## Our interpretation

We believe the method recovers **the missing dynamical influence in the observed community** — which, depending on the system, may be:

- In closed/controlled systems: primarily the hidden species' direct effect (interaction-driven)
- In open systems: a mixture of hidden species effects and shared environmental forcing

The recoverability appears to depend on what you called "the relative strengths of the ongoing processes": when local species interactions dominate over environmental forcing, recovery is more species-specific; when environmental forcing dominates, the recovered signal is confounded.

We would be very grateful for your perspective on this interpretation. In particular:

1. Do you think the distinction between "closed-system recovery" (interaction-driven, more identifiable) and "open-system recovery" (environmentally confounded) is ecologically meaningful, or is it an oversimplification?

2. From a theoretical standpoint, are there conditions under which one could formally guarantee identifiability of a hidden species from partial observations of a coupled dynamical system?

Thank you again for your time and insight.

Best regards,
Xingji Cui
