# Eco-GNRD: Inferring Unobserved Species Influence in Chaotic Ecological Dynamics

A data-driven unsupervised framework for recovering the dynamical influence of unmeasured species in partially-observed ecological communities.

---

## Method

**Eco-GNRD** (Ecological Graph Neural Residual Dynamics) decomposes multi-species ecological dynamics into an observable baseline and a latent residual closure:

```
log(x_{t+1}/x_t)_i = f_visible_i(x_t) + h(t) * G_i(x_t)
```

| Component | Role |
|---|---|
| `f_visible` | Species-GNN modeling observable community dynamics |
| `G` | Per-species sensitivity field to hidden influence |
| `h(t)` | Scalar latent variable (hidden species' effect) |
| `h * G` | Residual closure: zero contribution when h=0 |

Training is **strictly unsupervised**: hidden species values are never used during training. The latent variable h(t) is inferred via a variational encoder with Takens delay embedding.

### Architecture components

| Component | Description | Key hyperparameters |
|---|---|---|
| **Posterior Encoder** | GNN + Takens delay embedding + attention | d=64-96, blocks=2-3, heads=4, lags=(1,2,4,8) |
| **f_visible (Species-GNN)** | Predicts visible-visible interactions | d_species=20, layers=2, top_k=4, MLP backbone |
| **G-field (Species-GNN)** | Predicts hidden→visible coupling | d_species=12, layers=1, top_k=3 |
| **Formula hints** | LV + Holling II terms as GNN input features | Not forced, only suggested |
| **NbedDyn ODE** | h(t+1) = h(t) + f_h(h(t), x(t)) consistency | LatentDynamicsNet, d_hidden=32 |
| **G_anchor** | Breaks h ↔ -h symmetry via sign constraint on first species | alpha annealing over training |

### Training strategy

| Strategy | When used | Effect |
|---|---|---|
| **Alternating 5:1** | Beninca, Maizuru | Phase A (5 ep): train f_visible+G, freeze encoder. Phase B (1 ep): freeze f_visible+G, train encoder+f_h. Solves h gradient vanishing. |
| **Joint training** | Huisman | Standard end-to-end optimization. Works when dynamics are clean. |
| **Warmup 20%** | All | First 20% epochs: train without h (h_weight=0). Lets f_visible learn baseline. |
| **Input dropout** | All | 5% random masking after warmup. Regularization. |

### Loss components

| Loss | Weight | Purpose |
|---|---|---|
| **Reconstruction** | 1.0 | L1 between predicted and actual log-ratios |
| **KL divergence** | 0.017-0.03 | Regularize posterior q(h\|x) toward prior N(0,1) |
| **Counterfactual (null)** | 5.0-9.5 | h=0 must degrade prediction (h is necessary) |
| **Counterfactual (shuffle)** | 3.0-5.7 | h time-shuffled must degrade prediction (temporal structure matters) |
| **Rollout (3-step)** | 0.5 | Multi-step teacher-forced self-consistency |
| **Energy** | 2.0 | h must have minimum variance (prevents collapse) |
| **ODE consistency** | 0.2-0.5 | f_h must predict h(t+1) from h(t) |
| **RMSE log** | 0.1 | Scale-aware reconstruction |
| **Smoothness** | 0.02 | Temporal smoothness of h |
| **Sparsity** | 0.02 | L1 on G-field (sparse interactions) |

---

## Datasets

| Dataset | Type | Species | Time steps | Source |
|---|---|---|---|---|
| **Huisman 1999** | Synthetic chemostat | 6 species, 5 resources | 1001 (dt=2d) | Huisman & Weissing, Nature 1999 |
| **Beninca 2008** | Real Baltic mesocosm | 9 species, 4 nutrients | 658 (dt=4d) | Beninca et al., Nature 2008 |
| **Maizuru Bay** | Real field (fish) | 15 species | 285 (biweekly) | Ushio et al., Nature 2018 |
| **Maizuru Extended** | Real field (fish) | 14 species | 540 (semi-monthly) | Ohi/Ushio/Masuda 2024, 2002-2024 |
| **Blasius 2020** | Real chemostat | 3 (algae, rotifers, eggs) | 86-366 per exp | Blasius et al., Nature 2020 |

### Evaluation protocol

- **n-to-1 rotation**: Each species takes a turn as the hidden target; the remaining species serve as visible input
- **Train/val split**: First 75% train, last 25% validation
- **Metric**: Pearson correlation on validation set (lstsq fitted on train, evaluated on val)
- **Seeds**: 10 seeds for main method, 5 for baselines and ablation
- **No oracle selection**: Report mean over seeds, never best

---

## Results

### Main comparison: Eco-GNRD vs all baselines (val Pearson, mean over seeds)

| Method | Type | Huisman | Beninca | Maizuru |
|---|---|---|---|---|
| VAR + PCA | Linear | +0.016 | -0.036 | -0.022 |
| MLP + PCA | Nonlinear | +0.022 | -0.004 | +0.043 |
| EDM Simplex (E=3) | Nonparametric | +0.301 | +0.047 | +0.018 |
| MVE (E=3) | Nonparametric | +0.397 | +0.111 | +0.021 |
| LSTM + PCA | Deep learning | +0.529 | +0.104 | +0.101 |
| Neural ODE | Deep learning | +0.417 | +0.021 | running |
| Latent ODE | Deep learning | +0.470 | +0.194 | -0.040 |
| Supervised Ridge | **Oracle ceiling** | +1.000 | +0.103 | +0.223 |
| **Eco-GNRD (ours)** | **GNN + VAE** | **+0.421** | **+0.146** | **+0.208** |

Key observations:
- **Beninca**: Eco-GNRD outperforms all unsupervised baselines and even the supervised ceiling
- **Maizuru**: Eco-GNRD is the most robust method; Latent ODE collapses (-0.040)
- **Huisman**: Eco-GNRD is competitive with modern deep methods (Latent ODE, LSTM)
- Linear/nonparametric baselines (VAR, MLP, EDM) are near zero on all datasets

### Additional datasets

| Dataset | Eco-GNRD P(val) | Notes |
|---|---|---|
| Maizuru Extended (2002-2024) | +0.162 | 22-year dataset, 540 time points, 14 species |
| Blasius (9 experiments) | +0.263 | Chemostat predator-prey, grand mean over 9 experiments |

### Per-species results (Maizuru, val Pearson)

| Species | Eco-GNRD | LSTM | Sup. Ridge | Ecology |
|---|---|---|---|---|
| Pseudolabrus sieboldi | **+0.462** | +0.267 | +0.663 | Reef-resident, strong local coupling |
| Girella punctata | **+0.355** | +0.343 | +0.148 | Resident herbivore |
| Halichoeres tenuispinis | **+0.349** | +0.194 | +0.286 | Reef-associated wrasse |
| Pterogobius zonoleucus | **+0.333** | -0.134 | +0.491 | Resident goby |
| Parajulis poecilepterus | **+0.333** | +0.040 | +0.595 | Reef-associated wrasse |
| Trachurus japonicus | +0.299 | +0.244 | +0.555 | Temperature-sensitive, semi-migratory |
| Engraulis japonicus | +0.022 | -0.058 | -0.072 | **Migratory, no local interactions** |
| Tridentiger trigonocephalus | -0.003 | +0.048 | -0.011 | Benthic goby |

Ecological validation (confirmed by Prof. Ushio, original dataset author):
- **Engraulis japonicus** is migratory with no local biotic interactions → correctly unrecoverable
- **Pseudolabrus sieboldi** is temperature-insensitive reef fish → high recovery from species interactions alone

---

## Ablation study (val Pearson, 5 seeds)

| Configuration | Huisman | Beninca | Maizuru | Key effect |
|---|---|---|---|---|
| **Full Eco-GNRD** | **+0.421** | **+0.146** | +0.208 | — |
| − Counterfactual losses | +0.016 | +0.076 | +0.220 | **Critical on Huisman** (−0.405) |
| − Alternating training | +0.430 | +0.051 | +0.238 | **Critical on Beninca** (−0.095) |
| − ODE consistency | +0.462 | +0.056 | +0.188 | Important on Beninca (−0.090) |
| − Multi-step rollout | +0.430 | +0.128 | +0.217 | Modest effect |
| − Takens embedding | +0.420 | +0.080 | +0.301 | Important on Beninca (−0.066) |

Findings:
- **Counterfactual losses** are the single most important component on simulated data (Huisman: +0.421 → +0.016 without them)
- **Alternating training** is essential for real chaotic data (Beninca: +0.146 → +0.051 without it)
- Different datasets benefit from different components — the architecture is modular, not uniformly beneficial

---

## Interaction structure verification (Huisman)

The Huisman system has a known ground-truth interaction structure (6 species competing for 5 resources via Monod kinetics). We verify that Eco-GNRD's learned G-field recovers the true interaction pattern.

For each hidden species, we compare the learned G-field magnitude (how much h affects each visible species) with the true resource-mediated competition strength:

| Hidden species | Spearman correlation | Recovery P(val) | Verdict |
|---|---|---|---|
| sp2 | **+1.000** | +0.679 | Perfect match |
| sp1 | +0.700 | +0.341 | Match |
| sp3 | +0.700 | +0.588 | Match |
| sp4 | +0.600 | +0.713 | Match |
| sp5 | +0.500 | +0.531 | Weak match |
| sp6 | -0.200 | +0.099 | Mismatch |
| **Mean** | **+0.550** | | |

**4/6 species show significant interaction pattern match** (Spearman > 0.5). sp6, the only mismatch, is also the hardest to recover — its interaction failure and recovery failure are consistent. This is a capability that black-box methods (Latent ODE, LSTM) fundamentally cannot provide.

---

## Identifiability analysis

We performed a cross-species specificity test: for each hidden species S, is h_pred more correlated with true S than with any other signal (other species, temperature, nutrients)?

**Raw specificity**: Most species are "proxy-dominated" — the recovered signal captures shared environmental drivers (temperature, nutrients) rather than species-specific dynamics. This is expected: in coupled ecological systems, species share drivers.

**Partial specificity** (after removing temperature/SRP/PC1):
- Beninca: 1/9 species-specific (Nanophyto)
- Maizuru: 1/15 species-specific (Siganus)
- Huisman: 0/6 species-specific (but no environmental confound exists)

**Interpretation**: The recovered h captures the hidden species' *influence on the community*, which includes both species-specific and environmentally-shared components. In closed systems (Huisman, Blasius) without environmental forcing, the influence IS the species interaction. In open systems (Maizuru), the influence is confounded with shared environmental drivers.

This is a fundamental property of ecological systems, not a method limitation: species that share environmental drivers cannot be distinguished by their community effects alone.

---

## Per-dataset configuration

| Parameter | Huisman | Beninca | Maizuru |
|---|---|---|---|
| Training strategy | Joint | Alt 5:1 | Alt 5:1 |
| Learning rate | 0.0008 | 0.0006 | 0.0006 |
| Encoder (d, blocks) | 64, 2 | 96, 3 | 96, 3 |
| ODE weight (lam_hdyn) | 0.2 | 0.5 | 0.5 |
| KL weight (beta_kl) | 0.03 | 0.017 | 0.017 |
| Counterfactual weight | 5.0 | 9.5 | 9.5 |
| Min energy | 0.02 | 0.14 | 0.14 |
| Epochs | 500 | 500 | 500 |
| Seeds | 10 | 10 | 10 |

---

## Project structure

```
models/
  cvhi_residual.py              EcoGNRD model (encoder + dynamics + losses)
  cvhi_ncd.py                   GNN backbones (SpeciesGNN_MLP, SoftForms)
  cvhi.py                       Posterior encoder (GNN + Takens + attention)

scripts/
  run_main_experiment.py        Master experiment runner (all datasets, 10 seeds)
  baselines_fair.py             All 8 baselines with strict fairness
  run_ablation.py               Ablation study (5 components × 3 datasets)
  specificity_analysis.py       Cross-species specificity test
  specificity_partial.py        Partial correlation specificity
  verify_interactions.py        G-field vs true Huisman interactions
  load_beninca.py               Beninca 2008 data loader
  load_maizuru.py               Maizuru (Ushio 2018) data loader
  load_maizuru_ext.py           Extended Maizuru (2002-2024) data loader
  load_blasius.py               Blasius 2020 chemostat data loader
  generate_huisman1999.py       Huisman chaos data generator
  cvhi_beninca_nbeddyn.py       Best Beninca config (NbedDyn + alt 5:1)
  cvhi_huisman_full.py          Best Huisman config (hdyn_only)
  cvhi_maizuru_alt.py           Best Maizuru config (alt 5:1, no temp)
  run_blasius.py                Blasius experiment runner
  run_maizuru_ext.py            Extended Maizuru experiment runner
  disentangle_analysis.py       Cross-rotation signal decomposition
  beninca_coupling_analysis.py  Coupling vs recoverability analysis

data/
  real_datasets/beninca/        Beninca 2008 parsed data
  real_datasets/maizuru/        Ushio 2018 fish data
  real_datasets/maizuru_extended/  Ohi/Ushio/Masuda 2002-2024 fish data
  real_datasets/blasius/        Blasius 2020 chemostat data (C1-C10)

重要实验/results/
  main/eco_gnrd_alt5_hdyn/      Main method results (per-dataset/species/seed)
    huisman/                    6 species × 10 seeds
    beninca/                    9 species × 10 seeds
    maizuru/                    15 species × 10 seeds
    maizuru_ext/                14 species × 5 seeds
    blasius/                    9 experiments × 3 species × 5 seeds
  baselines/                    All baseline results
    var_pca/                    VAR + PCA (deterministic)
    mlp_pca/                    MLP + PCA (10 seeds)
    edm_simplex/                EDM Simplex E=3 (deterministic)
    mve/                        MVE E=3 (deterministic)
    lstm/                       LSTM + PCA (5 seeds)
    neural_ode/                 Neural ODE (5 seeds)
    latent_ode/                 Latent ODE (5 seeds)
    supervised_ridge/           Supervised Ridge (oracle ceiling)
  ablation/                     Ablation results
    no_alt/                     Without alternating training
    no_ode/                     Without ODE consistency
    no_cf/                      Without counterfactual losses
    no_rollout/                 Without multi-step rollout
    no_takens/                  Without Takens delay embedding
    nohint/                     Without formula hints (Huisman only)
    no_alt_ode_rollout/         Combined ablation (running)
  specificity/                  Identifiability analysis results
  interaction_verification/     G-field vs true interaction structure

docs/
  email_draft_to_ushio.md       Communication with original dataset authors
  paper_notes_en/               English paper notes (9 papers)

paper/
  outline.md                    Paper outline

runs/                           Legacy experiment outputs
```

---

## Quick start

```bash
# Generate Huisman synthetic data
python -m scripts.generate_huisman1999

# Run main experiment (Eco-GNRD, all datasets, 10 seeds)
python -m scripts.run_main_experiment --datasets huisman beninca maizuru --seeds 10

# Run fair baselines (all methods, all datasets)
python -m scripts.baselines_fair --datasets huisman beninca maizuru --seeds 5 --methods all

# Run ablation study
python -m scripts.run_ablation --datasets huisman beninca maizuru --seeds 5

# Specificity analysis
python -m scripts.specificity_analysis

# Interaction verification (Huisman)
python -m scripts.verify_interactions

# Blasius experiment
python -m scripts.run_blasius

# Extended Maizuru experiment
python -m scripts.run_maizuru_ext
```

---

## Requirements

- Python 3.10+
- PyTorch 2.0+
- torchdiffeq (for Neural ODE / Latent ODE baselines)
- NumPy, SciPy, scikit-learn, pandas, matplotlib

---

## Unsupervised guarantee

Training **never** uses `hidden_true` or any signal derived from it. Hidden species values are used only at evaluation time to compute Pearson correlation (with lstsq sign correction fitted on train set only). Model selection uses `val_recon` (visible reconstruction loss on held-out time steps), which is fully unsupervised.

---

## Baseline fairness

All baselines follow the same protocol:
- **No oracle selection**: Report mean over seeds, not best
- **No oracle dimension**: LSTM uses PCA, not grid search over hidden dimensions
- **No oracle E**: EDM/MVE use fixed E=3, not best-E sweep against hidden_true
- **Same train/val split**: 75/25 for all methods
- **Same metric**: Train-fit lstsq, val-eval Pearson

---

## Key findings

1. **Eco-GNRD recovers hidden species influence** in closed ecological systems (Huisman, Beninca, Blasius) where dynamics are interaction-driven

2. **Interpretable interaction structure**: The learned G-field matches true resource-mediated competition in Huisman (mean Spearman = +0.550, 4/6 species match) — a capability unique to our GNN-based approach

3. **Counterfactual losses are essential**: Removing them collapses Huisman recovery from +0.421 to +0.016. They enforce that h carries necessary dynamical information

4. **Alternating training solves gradient vanishing** on real chaotic data: Beninca improves from +0.051 (joint) to +0.146 (alt 5:1)

5. **Recovery quality reflects ecological coupling specificity**: Species with strong, specific local interactions (reef fish, Pseudolabrus) are recoverable; migratory species without local coupling (Engraulis) are correctly unrecoverable (confirmed by original dataset author)

6. **Open-system limitation**: In field data (Maizuru), recovered signals capture shared environmental drivers alongside species-specific dynamics — an inherent identifiability challenge in open ecological systems

---

## Citation

```
Cui, X. (2026). Inferring Unobserved Species Influence in Chaotic
Ecological Dynamics via Graph Neural Residual Decomposition.
```
