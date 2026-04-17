# Trifonova et al. (2015) -- Spatio-temporal Bayesian Network Models with Latent Variables for Revealing Trophic Dynamics and Functional Networks in Fisheries Ecology

## Paper Info
- **Authors**: Neda Trifonova, Andrew Kenny, David Maxwell, Daniel Duplisea, Jose Fernandes, Allan Tucker
- **Journal**: Ecological Informatics, Vol. 30, pp. 142-158
- **Year**: 2015

---

## Core Question

Can Bayesian networks with latent variables reveal trophic-level dynamics in the North Sea ecosystem? Can latent variables effectively represent unobserved species groups (e.g., zooplankton) and improve biomass predictions?

## Data

- IBTS (International Bottom Trawl Survey) data from 7 spatial regions of the North Sea (1983-2010)
- Variables: pelagic fish (P), small piscivorous fish (SP), large piscivorous fish and apex predators (LP) biomass, plus temperature, primary production, and fishing catch

## Methods

Five model architectures compared:
1. **ARHMM**: Autoregressive Hidden Markov Model
2. **DBN**: Dynamic Bayesian Network
3. **SDBN**: DBN with spatial autocorrelation
4. **HDBN**: DBN with two latent variables
5. **HSDBN**: DBN with two latent variables + spatial autocorrelation (proposed model)

### Dual Latent Variable Design
- **General latent variable**: Discrete; captures overall regime changes across trophic groups (regime shifts)
- **Specific latent variable**: Continuous; represents unobserved zooplankton biomass

### Structure Learning
- Hill-climbing algorithm to learn BN structure from data
- Expert knowledge constraints incorporated
- Spatial nodes: neighborhood average biomass as parent nodes to correct spatial autocorrelation

### Inference
- EM algorithm for latent variable state and parameter learning
- BIC score for model selection: BIC = logP(Theta) + logP(Theta|D) - 0.5*k*log(n)

## Key Findings

- **HSDBN performs best**: The model with both latent variables and spatial autocorrelation achieves the most accurate biomass predictions
- **General latent variable**: Successfully captures regime shifts -- overall changes in biomass variance across trophic groups
- **Specific latent variable**: Successfully represents unobserved zooplankton dynamics; validated against independent zooplankton observation data
- **Functional networks**: Learned network structures reveal spatially and temporally heterogeneous trophic relationships (predator-prey links)
- **Spatial heterogeneity**: Different spatial regions have different driving factors (temperature, fisheries, primary production) and their effects on species vary across regions
- **Prediction accuracy**: Model performance varies by spatial region and species group

## Key Concepts

- Bayesian network: p(x) = product of p(x_i | pa_i); factorizes joint probability via conditional independence
- Dynamic BN: Nodes represent variables at different time slices; temporal connections model dynamics
- EM algorithm: Iteratively learns latent variable states and parameters

## Relevance to Eco-GNRD

### Directly Relevant as a Methodological Precursor
- This paper is a key precursor to our "hidden species recovery" work -- using latent variables to represent unobserved ecological groups
- The dual latent variable architecture (general + specific) is directly transferable:
  - General HV captures system-level regime shifts
  - Specific HV recovers missing species
  - This matches our project goals closely

### Validation Strategy
- Using independent zooplankton data to validate the physical meaning of the specific latent variable provides a methodological template for validating our hidden recovery results

### BN vs. GNN
- Both use graph structure to model species interactions, but GNN handles continuous dynamics more flexibly
- BN's discrete general HV for regime detection could complement GNN's continuous dynamics modeling

### Additional Real-World Dataset
- North Sea fisheries data could serve as an additional validation platform for our methods (similar to Portal data)

### Spatial Modeling
- Spatial autocorrelation correction ideas are extensible to GNN graph structure design
