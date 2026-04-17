# Ouala et al. (2019) -- Learning Latent Dynamics for Partially-Observed Chaotic Systems (NbedDyn)

## Paper Info
- **Authors**: Said Ouala, Duong Nguyen, Lucas Drumetz, Bertrand Chapron, Ananda Pascual, Fabrice Collard, Lucile Gaultier, Ronan Fablet
- **Institutions**: IMT Atlantique / Ifremer / IMEDEA / OceanDataLab
- **Year**: 2019
- **arXiv**: 1907.02452v1

---

## Core Problem

When a dynamical system is only partially observed, no smooth ODE exists in the observation space to describe the temporal evolution (because the mapping is not one-to-one). Takens' theorem guarantees delay embeddings exist but requires manual selection of lag and dimension. NbedDyn's key innovation: instead of delay embedding, directly learn an augmented state space + ODE.

## Method Framework

### Augmented State Space
- X_t = [x_t; y_t] where x_t is the observed state and y_t is the unobserved latent variable
- d_E = total augmented dimension

### State-Space Model
- ODE: dX/dt = f_theta(X_t), parameterized by neural network
- Observation: x_t = G(X_t), identity on observed components
- Integration: 4th-order Runge-Kutta

### Training Objective (Core)
- min over theta and {y_t}: sum of ||x_t - G(Phi(X_{t-1}))||^2 + lambda * ||X_t - Phi(X_{t-1})||^2
- First term: one-step prediction error (observation space)
- Second term: ODE consistency constraint (augmented space)
- Key: jointly optimizes ODE parameters theta AND all latent variables {y_t}

### Prediction Phase
- Given new observations, infer initial latent state via variational optimization
- Initialize from most similar trajectory segment in training set

## Experimental Results

### Lorenz-63 (scalar observation x_1 only)

| Model | 1-step RMSE | 4-step RMSE | Lyapunov |
|---|---|---|---|
| Analog (best) | 1.2e-4 | 1.04e-3 | 0.84 |
| Sparse Regression | 1.85e-3 | 2.56e-3 | NaN |
| Latent-ODE (encoder) | 0.0801 | 0.520 | NaN |
| **NbedDyn (d_E=6)** | **6.8e-6** | **6.5e-5** | **0.87** |

- NbedDyn outperforms next-best by an order of magnitude
- Tracks up to 9 Lyapunov times
- Latent-ODE (encoder-based, Rubanova 2019) performs much worse

### Sea Level Anomaly (real data)
- 1-day RMSE: NbedDyn 0.002 vs. Latent-ODE 0.025 vs. Analog 0.046
- 90%+ relative improvement at 1-day horizon

### Dimension Analysis
- Jacobian eigenvalue analysis reveals the model automatically discovers the true system dimension (3 for Lorenz from d_E=6)

## Connection to Takens' Theorem

NbedDyn's prediction pipeline implicitly defines a delay embedding through optimization rather than manual lag/dimension selection. The variational inference of latent states is functionally equivalent to finding an embedding.

## Relevance to Eco-GNRD

### Direct Comparisons with Our Architecture

| Aspect | NbedDyn | Eco-GNRD |
|---|---|---|
| Latent inference | Direct optimization of y_t | Encoder q(h|x) |
| Latent dimension | Multi-dimensional | Scalar h |
| Dynamics | Single ODE f(X) | Separated f_vis(x) + h*G(x) |
| Latent constraint | ODE consistency | KL + smoothness + counterfactual |
| Training | Direct optimization | VAE-style |

### Key Insights for Our Project

- **ODE consistency loss**: The lambda*||X_t - Phi(X_{t-1})||^2 term can be adapted to CVHI as L_h_ode = MSE(h_encoder(t), h(t-1) + f_h(h(t-1), x(t-1)))
- **Encoder-based vs. direct optimization**: Latent-ODE (encoder) RMSE = 0.0801 vs. NbedDyn (direct) RMSE = 6.8e-6 -- suggests adding ODE consistency to our encoder-based approach could help
- **Multi-dimensional h**: If hidden species dynamics need richer representation, extending from scalar to multi-dimensional h is worth exploring
- **Jacobian dimension analysis**: Can diagnose effective dimension of learned augmented states
