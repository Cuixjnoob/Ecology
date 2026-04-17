# Racca & Magri (2022) -- Data-driven Prediction and Control of Extreme Events in a Chaotic Flow

## Paper Info
- **Authors**: Alberto Racca (Cambridge), Luca Magri (Imperial College London / Cambridge / Turing Institute)
- **Year**: 2022
- **arXiv**: 2204.11682v1 (submitted to Physical Review Fluids)

---

## Core Problem

How to predict extreme events (sudden, dramatic state changes) in chaotic systems from data alone, and further control/suppress these events? Extreme events are ubiquitous in fluid mechanics (turbulent bursts, thermoacoustic oscillations), oceanography (rogue waves), and atmospheric science (blocking events). They are rare heavy-tailed events with sudden onset and destructive impact.

## Physical System: MFE Chaotic Shear Flow

- 9-dimensional ODE system (Galerkin projection of Navier-Stokes)
- Simulates qualitative behavior of planar shear flow
- At Re=400: chaotic transients, eventually converging to laminar solution
- **Extreme events**: Intermittent large bursts of kinetic energy k(t) = 0.5 * sum(a_i^2)
- Event threshold: k(t) >= k_e = 0.1
- Physical process: laminar structure -> slow laminarization -> sudden collapse into vortices -> energy burst
- **Pattern: mostly quiescent + intermittent bursts** -- highly similar to Beninca data!
- Leading Lyapunov exponent: Lambda = 0.0163; 1 Lyapunov time (LT) ~ 61 time units

## Method: Echo State Networks (ESN)

### Architecture (Reservoir Computing)
- Input layer -> high-dimensional reservoir (random fixed connections) -> linear output layer
- Reservoir dynamics: r(t+1) = tanh(W_in * [u_in; b_in] + W * r(t))
- Output: u_p(t+1) = [r(t+1); 1]^T * W_out
- Only output weights W_out are trained (ridge regression) -- no backpropagation needed
- Extremely low computational cost

### Key Innovation: Recycle Validation (RV)
- Traditional single-shot validation (SSV): one validation set, not representative for chaotic data
- RV: closed-loop validation on multiple time windows within training data
- Performance gain equivalent to adding hundreds to thousands of neurons
- Computational cost same as SSV (train only once)

## Experimental Results

### Extreme Event Prediction (Precision / Recall / F-score)

| Prediction Time (PT) | Precision | Recall | F-score |
|:---:|:---:|:---:|:---:|
| 1 LT | ~0.98 | ~0.99 | ~0.98 |
| 2 LT | ~0.90 | ~0.90 | ~0.90 |
| 3 LT | ~0.80 | ~0.80 | ~0.80 |
| 5 LT | ~0.60 | ~0.60 | ~0.60 |

ESN can predict extreme events up to 5 Lyapunov times ahead!

### Statistical Extrapolation
- ESN-generated PDF tails match true distribution far better than training data statistics
- Small reservoirs (500-1000 neurons) already capture long-term statistics
- Evaluated with Kantorovich (Wasserstein) distance + Mean Logarithmic Error

### Extreme Event Control
- Strategy: ESN predicts event 1 LT ahead -> increase Re to 2000 for 1.5 LT -> reduces forcing/dissipation -> prevents burst
- Result: PDF tail reduced by ~1 order of magnitude

## Relevance to Eco-GNRD

### Striking Structural Similarity

| Racca's MFE system | Our Beninca plankton |
|:---:|:---:|
| 85%+ time "laminar" quiescent | 85-98% time flat |
| Intermittent "turbulent bursts" | Intermittent bursts (2-15%) |
| Kinetic energy spikes | Species abundance spikes |
| Heavy-tailed distribution | Similar rare-event distribution |
| Chaotic dynamics | Confirmed chaotic system |

Their "extreme event prediction" is essentially our "burst detection/prediction" problem!

### Methodological Borrowings

- **ESN as baseline**: Extremely simple to train (linear regression only), data-efficient -- ideal baseline for Beninca short time series (~2300 days). Compare with GNN to verify GNN's added value
- **Imbalanced classification framing for bursts**: Reframe burst prediction as binary classification; use Precision/Recall/F-score instead of only Pearson/RMSE. Directly addresses "model learns trends but misses bursts"
- **Statistical extrapolation**: Models trained on short time series can extrapolate long-term statistical properties -- valuable for ecological time series (always too short)
- **Recycle Validation**: Validation strategy for chaotic time series using open/closed-loop dual modes; directly applicable to our model selection

### Evaluation Framework Improvements (Implemented)
- **Burst Precision/Recall/F-score**: Set threshold (e.g., mean + 2*sigma), binary classification evaluation
- **Prediction Horizon for Bursts (PHburst)**: How far ahead can the model accurately predict bursts
- **Wasserstein distance**: Overall distance between predicted and true PDFs
- **Mean Logarithmic Error**: Emphasizes tail (burst region) prediction accuracy
