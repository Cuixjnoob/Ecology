# Rogers et al. (2023) -- Intermittent Instability Is Widespread in Plankton Communities

## Paper Info
- **Authors**: Tanya L. Rogers, Stephan B. Munch, Shin-ichiro S. Matsuzaki, Celia C. Symons
- **Journal**: Ecology Letters, Vol. 26, pp. 470-481
- **Year**: 2023
- **DOI**: 10.1111/ele.14168

---

## Background

Prior ecological chaos studies (including Beninca 2008) focused on global instability (positive global Lyapunov exponent over long timescales). A neglected question is time-varying local instability: at a given moment, do small perturbations grow or shrink in the short term?

**Intermittent instability**: Populations at the "edge of chaos" can alternate between locally stable and unstable periods. Even globally stable (non-chaotic) time series can exhibit episodes of local instability. This directly affects ecological prediction windows.

## Data & Methods

### Data
- Monthly plankton time series from 17 lakes + 4 marine stations (global distribution)
- Three taxonomic resolutions: Species level (154 series), Functional group level (48 series), Trophic level (41 series)

### Analytical Pipeline
1. **Delay embedding + S-map**: Reconstruct state space via Takens embedding; use S-map (locally weighted linear regression) to fit dynamics
2. **Jacobian reconstruction**: Extract time-dependent Jacobian matrix J(x_t) from S-map coefficients
3. **Global Lyapunov exponent**: LE = (1/T) * ln|Lambda_1(product of J(x_t))| ; LE > 0.01 = chaotic
4. **Local eigenvalues**: ln|lambda_1(J(x_t))| at each time point; >0 = locally unstable
5. **Variance Expansion Ratio (VER)**: trace(J*J^T); measures total variance amplification at each step
6. **Seasonality analysis**: Penalized regression on sine/cosine basis functions; peak power at 11.5-12.5 months = "seasonal"

## Key Findings

### Chaos Prevalence by Taxonomic Resolution

| Resolution | Chaotic | Total series |
|---|---|---|
| Species | **52%** (80/154) | 154 |
| Functional group | **42%** (20/48) | 48 |
| Trophic level | **7%** (3/41) | 41 |

Chaos is far more prevalent at fine taxonomic resolution; aggregation to trophic level drastically reduces chaos to 7%.

### Intermittent Instability Is Pervasive
- Among non-chaotic series: 46-58% show intermittent local instability
- Average proportion of positive local eigenvalues: 21-50% depending on resolution
- Even non-chaotic series can spend substantial time in locally unstable regimes

### Strong Seasonality in Local Instability
- 50-56% of species/functional group series show annual periodicity in local eigenvalues
- **Maximum instability occurs in spring** (across most species with positive eigenvalues + seasonality)
- Maximum instability months tend to coincide with or follow maximum growth rate months
- Maximum instability precedes maximum abundance

### VER Predicts Forecast Error
- VER positively correlates with step-ahead prediction error (higher VER = less predictable)
- Relationship strongest at species level, weakens with aggregation

### Taxonomic Aggregation Improves Stability
- Increasing aggregation: variability (CV) decreases, predictability (R^2) increases, chaos prevalence drops sharply
- Consistent with Huisman & Weissing (1999): individual species chaotically fluctuate but total biomass is nearly constant

### Geographic Patterns
- Higher LE at low-temperature/high-latitude stations (functional group p=0.004; trophic level p=0.087)
- Stronger relative seasonality of local eigenvalues at low-temperature stations
- Consistent with models predicting that seasonality increases chaos probability

## Key Equations

- Delay embedding: x_t = f(x_{t-tau}, x_{t-2*tau}, ..., x_{t-E*tau})
- Lyapunov exponent: LE = (1/T) * ln|Lambda_1(prod J(x_t))|
- Variance Expansion Ratio: VER = trace(J(x_t) * J(x_t)^T)

## Relevance to Eco-GNRD

### Time-Varying Prediction Difficulty
- Local stability varies with time (especially seasonally). Hidden species recovery should be easier during locally stable periods (fall/winter) and harder during unstable periods (spring)
- Suggests evaluating model performance stratified by season or local stability regime
- VER could serve as an uncertainty quantification tool: assign larger uncertainty to recovery results in high-VER periods

### Taxonomic Resolution Effects
- Recovering functional-group-level hidden dynamics may be more feasible than species-level
- If the goal is recovering total phytoplankton or total zooplankton dynamics, success rates should be much higher than for single species
- Guides experimental design: test recovery at different aggregation levels

### Theoretical Framework Integration
- Completes a theoretical chain with Beninca 2008 and 2011:
  - **Beninca 2008**: Food webs can generate chaos (constant conditions, pure intrinsic dynamics)
  - **Beninca 2011**: Environmental noise can induce chaos-like fluctuations through resonance even in stable systems
  - **Rogers 2023**: In nature, intermittent instability is pervasive, varies seasonally and with taxonomic resolution
- Implication for Eco-GNRD: we need a framework that adapts to time-varying stability -- relying more on priors/constraints during unstable periods and trusting data-driven predictions during stable periods

### Methodological Tools
- S-map + local Jacobian eigenvalue analysis can serve as a diagnostic tool for our project: compute local Lyapunov exponents for Beninca data, identify stable vs. unstable periods, compare model recovery error against local instability
