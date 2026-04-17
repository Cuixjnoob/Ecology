# Young & Graham (2022) -- Deep Learning Delay Coordinate Dynamics for Chaotic Attractors from Partial Observable Data

## Paper Info
- **Authors**: Charles D. Young, Michael D. Graham
- **Institution**: University of Wisconsin-Madison
- **Year**: 2022
- **arXiv**: 2211.11061v1 (submitted to Chaos)

---

## Core Problem

How to predict full chaotic dynamics when only scalar or partial state observations are available? Takens' theorem guarantees delay coordinate embeddings are diffeomorphic to the attractor, but learning this mapping for chaotic/highly nonlinear systems is extremely challenging.

## Method Framework

### Delay Coordinate Embedding + Deep Neural Networks

Given partial observation u_d(t) in R^{d_p}, construct the delay embedding vector:

y(t) = [u_d(t), u_d(t-tau), u_d(t-2*tau), ..., u_d(t-(m-1)*tau)]

where m = number of delays, tau = delay interval, embedding dimension = d_p * m.

### Three Learning Components

1. **Time Integration Map G**: Either discrete time stepper (DTS) y(t+tau) = G(y(t)) or continuous Neural ODE dy/dt = g(y)
2. **Reconstruction Map F**: Maps from delay embedding back to full state: u_hat(t) = F(y(t)); requires complete-state training data; supervised learning
3. **Full prediction pipeline**: Partial observations -> delay embedding -> time evolution G -> state reconstruction F -> full prediction

### Network Architecture
- Deep fully-connected networks (5 layers, width 100-500)
- Residual connections
- Adam optimizer + MSE loss
- NODE uses torchdiffeq solver

## Experimental Results

### Lorenz System (scalar observation, d_p=1)
- Delay embedding m=6, tau=0.1
- Both DTS and NODE accurately track ~2-3 Lyapunov times
- Excellent attractor reconstruction (very low KL divergence)
- Conclusion: scalar observation suffices to reconstruct the 3D Lorenz attractor

### Kuramoto-Sivashinsky Equation (high-dimensional attractor)

**L=22 (manifold dimension d_M ~ 8):**

| d_p | m | d_p*m | Tracking | Attractor reconstruction |
|:---:|:---:|:---:|:---:|:---:|
| 1 | 16 | 16 | Poor (<0.25 LT) | Failed |
| 2 | 8 | 16 | Poor (<0.25 LT) | Failed |
| 4 | 4 | 16 | Good (>1 LT) | Good |
| 8 | 2 | 16 | Excellent (>2 LT) | Excellent |

**Critical finding**: Even with the same total embedding dimension (d_p*m=16), performance dramatically improves when d_p >= d_M/2. Very sparse observations (d_p < d_M/2) fail even with many delays.

### Evaluation Metrics
- Short-term tracking: normalized error between predicted and true trajectories
- Long-term dynamics: autocorrelation function (ACF) matching
- Attractor statistics: KL divergence of joint PDFs

## Relevance to Eco-GNRD

### Direct Correspondence to Hidden Species Recovery
- Their partial observation = our observing only a subset of species
- Their reconstruction map F = our recovering hidden species from visible ones
- Their time integration G = our predicting ecological dynamics evolution

### Embedding Dimension Guidance
- **Key practical rule: observation dimension d_p must be at least d_M/2 (half the attractor dimension)**
- For Beninca plankton (~6-8 species, attractor dimension likely 4-8): need to observe at least 2-4 species for effective recovery
- Observing only 1 species may require prohibitively many delay steps and still fail

### Architecture Improvements for Eco-GNRD
- **Delay embedding as input**: Use delay vectors [u(t), u(t-tau), ...] as GNN node features (already implemented in our Takens delay embedding with lags=1,2,4,8)
- **Separate reconstruction + dynamics**: Reference their dual-map design (G + F) for separating "time evolution" and "state reconstruction"
- **Neural ODE for irregular sampling**: For continuous-time ecological data with uneven sampling, NODE may be more natural

### Experimental Design
- Systematically test different observation ratios (2/6, 3/6, 4/6 observed species)
- Test effect of number of delay steps
- Use KL divergence to evaluate attractor quality of recovered dynamics
