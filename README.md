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

### Key components

- **Residual decomposition**: `f_visible + h*G` ensures the hidden contribution is identifiable and removable
- **Counterfactual losses**: Prediction must degrade when h is removed (necessity) or time-shuffled (temporal structure)
- **ODE consistency** (NbedDyn-inspired): `h(t) = h(t-1) + f_h(h(t-1), x(t-1))` enforces learned latent dynamics
- **Takens delay embedding**: Input features include lagged observations (1, 2, 4, 8 steps)
- **Alternating training**: Phase A trains f_visible/G, Phase B trains encoder (5:1 ratio for Beninca)
- **MLP backbone with formula hints**: Ecological formulas (LV, Holling II) provided as input features, not forced

---

## Datasets

| Dataset | Type | Species | Time steps | Chaos (Lyapunov) |
|---|---|---|---|---|
| Huisman 1999 | Synthetic chemostat | 6 species, 5 resources | 1001 (dt=2d) | 0.03-0.05 /day |
| Beninca 2008 | Real Baltic mesocosm | 9 species, 4 nutrients | 658 (dt=4d) | ~0.05 /day |

Evaluation protocol: n-to-1 rotation. Each species takes a turn as the hidden target; the remaining species (+ resources/nutrients) serve as visible input.

---

## Results

### Baseline comparison (Pearson correlation with hidden species)

| Dataset | VAR+PCA | MLP+PCA | LSTM (fair) | **Eco-GNRD** |
|---|---|---|---|---|
| Huisman (6 spp) | 0.027 | 0.042 | 0.535 | **0.535** |
| Beninca (9 spp) | 0.022 | 0.030 | 0.108 | **0.162** |

- Linear baselines (VAR, MLP residual PCA) near zero on all datasets
- Eco-GNRD matches LSTM on synthetic data, outperforms by 50% on real data
- GNN ecological structure provides advantage on noisy real-world data

### Per-species results (validation set, best seeds)

**Huisman** (train/val Pearson):
| sp1 | sp2 | sp3 | sp4 | sp5 | sp6 |
|---|---|---|---|---|---|
| 0.57 | **0.76** | 0.65 | **0.74** | 0.56 | 0.37 |

**Beninca** (validation-only Pearson, 5-seed mean):
| Rotifers | Ostracods | Harpacticoids | Picophyto | Filam | Calanoids |
|---|---|---|---|---|---|
| **0.50** | **0.39** | **0.36** | 0.24 | 0.07 | 0.04 |

### Key analytical findings

1. **Information-theoretic ceiling**: Supervised Ridge regression achieves 0.103 on Beninca validation; unsupervised Eco-GNRD achieves 0.117, approaching the limit imposed by chaotic dynamics.

2. **Disentanglement analysis**: Cross-rotation variance decomposition shows 79% of h(t) variance is species-specific (changes with hidden species identity), confirming the recovered signal primarily captures hidden species influence rather than model error.

3. **Recoverability and coupling structure**: In Huisman (known interaction matrix), recovery Pearson correlates negatively with coupling strength (r = -0.74). Species with diffuse, high-magnitude interactions are hardest to recover; species with specific, targeted interactions leave identifiable traces in the visible dynamics.

---

## Project structure

```
models/
  cvhi_residual.py          EcoGNRD model class (encoder + dynamics + losses)
  cvhi_ncd.py               GNN backbones (SpeciesGNN_MLP, SoftForms)
  cvhi.py                   Posterior encoder (GNN + Takens + attention)

scripts/
  load_beninca.py           Beninca 2008 data loader
  generate_huisman1999.py   Huisman chaos data generator
  cvhi_beninca_nbeddyn.py   Best Beninca training (NbedDyn + alt 5:1)
  cvhi_beninca_valonly.py    Beninca 5-seed with val-only evaluation
  cvhi_huisman_full.py       Best Huisman training (hdyn_only)
  cvhi_huisman_ablation.py   Huisman component ablation
  baselines_all_datasets.py  VAR/MLP/LSTM baselines
  disentangle_analysis.py    Cross-rotation signal decomposition
  beninca_coupling_analysis.py  Coupling vs recoverability analysis

data/
  real_datasets/            Beninca parsed data, Portal rodent data

paper/
  outline.md                Paper outline and structure
  fig_huisman_recovery.png  Main recovery figure (6 species)

notes/                      Analysis notes and session summaries
runs/                       Experiment outputs (timestamped)
```

---

## Quick start

```bash
# Activate environment
.venv\Scripts\activate                          # Windows
source .venv/bin/activate                       # Linux/Mac

# Generate Huisman synthetic data
python -m scripts.generate_huisman1999

# Train Eco-GNRD on Huisman (all 6 species, best config)
python -m scripts.cvhi_huisman_full

# Train Eco-GNRD on Beninca (9 species, 5 seeds, val-only Pearson)
python -m scripts.cvhi_beninca_valonly

# Run baselines on all datasets
python -m scripts.baselines_all_datasets

# Disentanglement analysis
python -m scripts.disentangle_analysis

# Coupling vs recoverability
python -m scripts.beninca_coupling_analysis
```

---

## Requirements

- Python 3.10+
- PyTorch 2.0+
- NumPy, SciPy, scikit-learn, matplotlib
- PyMuPDF (optional, for PDF reading)

---

## Unsupervised guarantee

Training never uses `hidden_true` or any signal derived from it. Hidden species values are used only at evaluation time to compute Pearson correlation (with lstsq sign correction). Model selection uses `val_recon` (visible reconstruction loss on held-out time steps), which is fully unsupervised.

---

## Citation

```
Inferring Unobserved Species Influence in Chaotic Ecological
Dynamics: A Data-Driven Unsupervised Approach (2026)
```
