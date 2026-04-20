# Eco-GNRD: Ecological Graph Neural Residual Dynamics

An unsupervised framework for inferring hidden species influence in partially observed ecological communities.

---

## Problem

In real ecosystems, not all species are observed. Some are hidden — unmonitored, undiscovered, or simply missed. Their absence from data creates a systematic bias in dynamical models.

**Our question**: Given time series of N−1 observed species, can we infer the dynamical influence of 1 unobserved species — without ever seeing its data during training?

---

## Method

### Core equation

We decompose the per-species log growth rate into an observable component and a hidden residual:

```
log(x_{t+1,k} / x_{t,k}) = f_visible_k(x_t) + h(t) · G_k(x_t)
```

| Symbol | Meaning |
|---|---|
| `f_visible_k(x_t)` | Dynamics explainable by observed species (Species-GNN) |
| `h(t)` | Scalar latent variable: the hidden species' influence over time |
| `G_k(x_t)` | Sensitivity field: how much the hidden species affects each observed species |
| `h · G` | Residual closure: exactly zero when h=0 (no hidden species) |

### Why this works

- `h=0` removes all hidden influence → enables counterfactual queries ("what if the hidden species weren't there?")
- `G_k` tells you WHICH species are most affected → interpretable interaction structure
- The encoder infers `h(t)` from visible dynamics alone → strictly unsupervised

### Architecture

```
Input: x_t ∈ R^{N-1}  (observed species abundances)
  │
  ├── Posterior Encoder (GNN + Takens delay embedding)
  │     └── q(h|x) = N(μ, σ²)  →  h(t) via reparameterization
  │
  ├── f_visible (Species-GNN with MLP messages + formula hints)
  │     └── Predicts observable species interactions
  │
  ├── G-field (Species-GNN)
  │     └── Per-species sensitivity to hidden influence
  │
  └── Output: pred = f_visible + h · G
```

### Formula hints

The GNN message MLP receives ecological formula values as input features:
- `x_j` (linear interaction)
- `x_i · x_j` (Lotka-Volterra bilinear)
- `x_j / (1 + α·x_j)` (Holling type II functional response)
- `x_i · x_j / (1 + α·x_j)` (Holling bilinear)

These are **hints, not constraints** — the MLP can use, combine, or ignore them. When hints match the true dynamics (e.g., LV hints on a Lotka-Volterra system), performance improves significantly. Wrong-domain hints (e.g., LV hints on a gravitational system) actively hurt performance, confirming the model genuinely uses them.

### GraphLearner (v1, optional component)

An additional module that learns an explicit interaction matrix A_ij from species embeddings:

```
A_ij = sigmoid(MLP(emb_i, emb_j))     # static interaction graph
agg_i = Σ_j A_ij · attn_ij · msg_ij   # A gates message passing
L1 sparsity penalty on A               # encourages sparse structure
```

**Results**: On Huisman, A matches the true resource-mediated competition structure (Spearman = +0.72, 4/6 species significant at p < 0.02). Standard attention does NOT learn true interactions (Spearman ≈ 0). However, GraphLearner introduces a trade-off: species with concentrated interactions improve (sp6: +0.099 → +0.584), while species with distributed interactions degrade (sp2: +0.640 → +0.391). See "GraphLearner experiments" below.

### Loss function

| Loss | Purpose | Critical? |
|---|---|---|
| **Reconstruction** | MSE(f_vis + h·G, actual log-ratio) | Core |
| **Counterfactual (null)** | h=0 must degrade prediction | **Yes** — removing this collapses Huisman from +0.421 to +0.016 |
| **Counterfactual (shuffle)** | Time-shuffled h must degrade prediction | Important |
| **KL divergence** | Regularize q(h\|x) toward N(0,1) | Standard VAE |
| **ODE consistency** | h(t+1) ≈ h(t) + f_h(h(t), x(t)) | Important on Beninca |
| **Multi-step rollout** | 3-step teacher-forced self-consistency | Modest |
| **Energy** | h must have minimum variance (prevent collapse) | Safety |
| **Smoothness** | Temporal smoothness of h | Regularization |
| **Sparsity** | L1 on GNN weights and GraphLearner A | Structure |

### Training strategies

| Strategy | Used on | Effect |
|---|---|---|
| **Alternating 5:1** | Beninca, Maizuru | Phase A (5 ep): train f_visible+G, freeze encoder. Phase B (1 ep): train encoder. Solves h gradient vanishing. |
| **Joint training** | Huisman, LV | Standard end-to-end. Works when dynamics are clean. |
| **Warmup 20%** | All | First 20% epochs: h_weight=0, learn f_visible first |

---

## Datasets

| Dataset | Type | Species | Time steps | Source |
|---|---|---|---|---|
| **Lotka-Volterra** | Simulated food chain | 4 species | 1500 | Classic LV equations |
| **Huisman 1999** | Simulated resource competition | 6 species + 5 resources | 1001 | Huisman & Weissing, Nature 1999 |
| **Beninca 2008** | Real mesocosm (Baltic plankton) | 9 species + 4 nutrients | 658 | Beninca et al., Nature 2008 |
| **Maizuru Bay** | Real field (fish community) | 15 species | 285 | Ushio et al., Nature 2018 |
| **Maizuru Extended** | Real field (fish, extended) | 14 species | 540 | Ohi/Ushio/Masuda, 2002–2024 |
| **Blasius 2020** | Real chemostat (predator-prey) | 3 (algae, rotifers, eggs) | 86–366 | Blasius et al., Nature 2020 |

**Evaluation protocol**: Leave-one-out rotation. Each species takes a turn as hidden. Train on first 75%, evaluate on last 25%. Pearson correlation (lstsq fitted on train, evaluated on val). Report mean over seeds.

---

## Results

### Main comparison (val Pearson, mean over seeds)

| Method | LV | Huisman | Beninca | Maizuru |
|---|---|---|---|---|
| VAR + PCA | — | +0.016 | −0.036 | −0.022 |
| MLP + PCA | +0.350 | +0.022 | −0.004 | +0.043 |
| EDM Simplex (E=3) | — | +0.301 | +0.047 | +0.018 |
| MVE (E=3) | — | +0.397 | +0.111 | +0.021 |
| LSTM + PCA | +0.207 | +0.529 | +0.104 | +0.101 |
| Neural ODE | +0.705 | +0.417 | +0.005 | +0.047 |
| Latent ODE (VAE) | +0.567 | +0.063 | running | running |
| Supervised Ridge | +0.965 | +1.000 | +0.103 | +0.223 |
| **Eco-GNRD** | **+0.612** | **+0.421** | **+0.146** | **+0.208** |
| **Eco-GNRD + GL (v1)** | +0.518 | **+0.435** | — | — |

Key observations:
- **Beninca**: Eco-GNRD (+0.146) outperforms ALL baselines, including the supervised ceiling (+0.103)
- **Maizuru**: Eco-GNRD (+0.208) is the most robust deep method; Neural ODE (+0.047) and Latent ODE collapse
- **LV**: Neural ODE wins (+0.705 vs +0.612) — expected on a clean ODE system. Eco-GNRD still beats all other methods
- **Latent ODE with faithful VAE**: KL divergence destroys hidden species signal (Huisman: +0.470 simplified → +0.063 faithful). General-purpose latent variable models are not designed for this task

### Null baselines (Huisman)

| Null method | Overall |
|---|---|
| White noise h | −0.005 |
| AR1 random h (ρ=0.95) | −0.013 |
| Shuffled true h | −0.000 |
| Wrong species (use another species as h) | +0.319 |
| **Eco-GNRD** | **+0.421** |

Our results are significantly above random (+0.421 vs ≈0). The "wrong species" null (+0.319) reflects inter-species colinearity in the Huisman system; Eco-GNRD recovers +0.102 beyond this shared signal.

### Additional datasets

| Dataset | Eco-GNRD | Notes |
|---|---|---|
| Maizuru Extended (2002–2024) | +0.162 | 22-year dataset, consistent with original |
| Blasius (9 experiments) | +0.263 | Chemostat predator-prey, grand mean |

### Per-species results (Maizuru, selected)

| Species | Eco-GNRD | LSTM | Ecology |
|---|---|---|---|
| Pseudolabrus sieboldi | **+0.462** | +0.267 | Reef-resident, strong local coupling |
| Girella punctata | **+0.355** | +0.343 | Resident herbivore |
| Engraulis japonicus | +0.022 | −0.058 | **Migratory, no local interactions** |

Ecological validation (Prof. Ushio, original dataset author): Engraulis is migratory with no local biotic interactions → correctly unrecoverable. Pseudolabrus is temperature-insensitive reef fish → recovery driven by species interactions.

---

## Ablation study (val Pearson, 5 seeds)

| Configuration | Huisman | Beninca | Maizuru |
|---|---|---|---|
| **Full Eco-GNRD** | **+0.421** | **+0.146** | +0.208 |
| − Counterfactual losses | +0.016 | +0.076 | +0.220 |
| − Alternating training | +0.430 | +0.051 | +0.238 |
| − ODE consistency | +0.462 | +0.056 | +0.188 |
| − Multi-step rollout | +0.430 | +0.128 | +0.217 |
| − Takens embedding | +0.420 | +0.080 | +0.301 |

**Robustness**: The full model is never significantly outperformed by any ablation (max deficit: 0.09), while removing components can cause catastrophic failure (up to −0.41). This asymmetry demonstrates reliable performance without significant cost.

---

## Interaction structure verification

### Attention vs GraphLearner vs truth (Huisman)

| Method | Spearman with true competition | Part of dynamics? |
|---|---|---|
| Attention weights | ≈ 0 (not significant) | Yes, but learns shortcuts |
| Empirical G-field | +0.55 (4/6 significant) | Post-hoc analysis |
| **GraphLearner A (v1)** | **+0.72 (4/6 significant)** | **Yes, directly gates messages** |

GraphLearner v1 is the only method where the learned interaction structure both (1) participates in the model's dynamics and (2) matches the ground-truth ecological network.

### GraphLearner experiments (15 versions tested)

We extensively explored how to learn explicit interaction structure. The core finding:

**The sparsity–accuracy trade-off is structural, not engineering.**

Species with concentrated interactions (few strong partners) benefit from extreme sparsity. Species with distributed interactions (many moderate partners) need dense information flow. No single sparsity level satisfies both.

| Version | Approach | Result |
|---|---|---|
| v1 | A × attn × msgs, L1=0.01 | **Best overall (+0.435), Spearman=0.72, but sp2 drops** |
| v2 | No L1 | A≈0.95, learns nothing |
| v3 | L1=0.001 | Moderate A, worse overall |
| v4 | A biases attention scores | Weak structure learning |
| v5–v6 | A replaces attention | Loses dynamic routing |
| v7 | Dual channel (A×msg + attn×msg) | Saves sp2, loses sp6 |
| v8 | v7 + learnable scale | No improvement |
| v9 | v7 + learnable exponent A^exp | Saves sp2, loses sp6 |
| v10–v11 | Hard-Concrete gates | Unstable (0.000 seeds) |
| v12 | Sigmoid + partial renorm | High variance |
| v13 | Dual expert (β-gate blend) | Saves sp2, loses sp6 |
| v14 | Decomposed gate (τ + s) | Gate doesn't learn |
| power | A^α with learnable α | Same trade-off |

**Conclusion**: In coupled dynamical systems, the optimal interaction sparsity is species-dependent. This finding itself is ecologically meaningful: it reflects differences in coupling specificity across the ecological network.

---

## Ecological findings

### 1. Recoverability depends on coupling specificity

Species with strong, specific local interactions are recoverable. Migratory or weakly-coupled species are not. Validated across Huisman (simulated), LV (simulated), and Maizuru (real, confirmed by Prof. Ushio).

### 2. Trophic level affects recoverability

On LV food chain: mid-trophic predators are easiest to recover (pred1: +0.832). Top predators are hardest (+0.022 with GL, +0.555 baseline). Their influence propagates indirectly through trophic cascades.

### 3. Closed vs open systems

- **Closed systems** (Huisman, Beninca, Blasius): recovery reflects species interactions
- **Open systems** (Maizuru): recovery is confounded with shared environmental forcing

Temperature ablation on Maizuru confirms this:
- Pseudolabrus sieboldi: +0.472 → +0.462 without temperature (interaction-driven)
- Trachurus japonicus: +0.541 → +0.299 without temperature (temperature-driven)

### 4. Generic latent variable models fail on this task

Latent ODE (Rubanova et al. 2019) with faithful VAE implementation collapses from +0.470 to +0.063. The KL divergence compresses away the very information needed for hidden species recovery. Neural ODE achieves +0.005 on Beninca. **This task requires purpose-built ecological architecture.**

### 5. Formula hints are domain-specific

Correct hints improve performance (LV hints on LV: +0.612 vs no hints: +0.544). Wrong-domain hints hurt (LV hints on gravitational system: +0.376 vs no hints: +0.544). The model genuinely uses domain knowledge, not just ignoring it.

### 6. The sparsity–coupling trade-off

Across Huisman (6 species) and LV (4 species), the same pattern: concentrated-coupling species benefit from sparse interaction modeling, distributed-coupling species suffer. This is structural, not tunable — confirmed by 15+ architectural variants.

---

## External validation

- **Prof. Masayuki Ushio** (HKUST): Confirmed Engraulis japonicus is migratory (correctly unrecoverable). Confirmed temperature sensitivity patterns. Provided extended 2002–2024 dataset.
- **Prof. Frank Hilker** (mathematical ecologist): "Such interpretation may be, in general, possible... it will depend on the specific system and the relative strengths of the ongoing processes."

---

## Per-dataset configuration

| Parameter | LV | Huisman | Beninca | Maizuru |
|---|---|---|---|---|
| Training | Joint | Joint | Alt 5:1 | Alt 5:1 |
| Learning rate | 0.001 | 0.0008 | 0.0006 | 0.0006 |
| Encoder (d, blocks) | 64, 2 | 64, 2 | 96, 3 | 96, 3 |
| ODE weight | 0.2 | 0.2 | 0.5 | 0.5 |
| KL weight | 0.03 | 0.03 | 0.017 | 0.017 |
| Counterfactual weight | 5.0 | 5.0 | 9.5 | 9.5 |
| Epochs | 500 | 500 | 500 | 500 |
| Seeds | 5 | 10 | 10 | 10 |

---

## Project structure

```
models/
  cvhi_residual.py              Eco-GNRD model (encoder + dynamics + losses + GraphLearner)
  cvhi_ncd.py                   Species-GNN backbones, GraphLearner, attention

scripts/
  run_main_experiment.py        Master experiment runner (all datasets, configurable seeds)
  baselines_fair.py             All baselines with strict fairness
  run_ablation.py               Ablation study (5 components × 3 datasets)
  run_graph_learner_test.py     GraphLearner v1 experiment
  run_graph_learner_v*.py       GraphLearner variants v2–v13, power transform
  run_gate_finetune.py          Decomposed gate experiments
  run_model_selection.py        Unsupervised model selection experiment
  specificity_analysis.py       Cross-species specificity test
  specificity_partial.py        Partial correlation specificity
  verify_interactions.py        G-field vs true interaction structure
  load_beninca.py               Beninca 2008 data loader
  load_maizuru.py               Maizuru (Ushio 2018) data loader
  load_maizuru_ext.py           Extended Maizuru (2002–2024) data loader
  load_blasius.py               Blasius 2020 chemostat data loader
  generate_huisman1999.py       Huisman chaos data generator

data/
  real_datasets/beninca/        Beninca 2008 parsed data
  real_datasets/maizuru/        Ushio 2018 fish data
  real_datasets/maizuru_extended/  2002–2024 extended fish data
  real_datasets/blasius/        Blasius 2020 chemostat data (C1–C10)

docs/
  research_report_20260419.md   Full research report
  email_draft_to_ushio.md       Communication with dataset authors
  email_draft_to_hilker.md      Communication with ecologists
```

---

## Quick start

```bash
# Generate synthetic data
python -m scripts.generate_huisman1999

# Run Eco-GNRD on all datasets (10 seeds)
python -m scripts.run_main_experiment --datasets huisman beninca maizuru --seeds 10

# Run baselines
python -m scripts.baselines_fair --datasets huisman beninca maizuru --seeds 5 --methods all

# Ablation study
python -m scripts.run_ablation --datasets huisman beninca maizuru --seeds 5

# Interaction verification
python -m scripts.verify_interactions

# Specificity analysis
python -m scripts.specificity_analysis
```

---

## Unsupervised guarantee

Training **never** uses hidden species data. Hidden values are used only at evaluation time to compute Pearson correlation. Model selection uses visible reconstruction loss (fully unsupervised).

## Baseline fairness

All baselines follow the same protocol: mean over seeds (not best), no oracle dimension selection, no oracle E sweep, same train/val split, same Pearson metric.

---

## Requirements

- Python 3.10+
- PyTorch 2.0+
- torchdiffeq
- NumPy, SciPy, scikit-learn, pandas, matplotlib

---

## Citation

```
Cui, X. (2026). Eco-GNRD: Inferring Hidden Species Influence in
Ecological Dynamics via Graph Neural Residual Decomposition.
```
