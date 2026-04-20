# GraphLearner Improvement Brief

## Problem Statement

I have a GNN-based ecological dynamics model. The f_visible component uses **attention + MLP messages** to model species interactions:

```
agg_i = Σ_j attn_ij × msg_ij(x_i, x_j)
```

**Attention does NOT learn ecologically correct interaction structure** (verified: Spearman ≈ 0 against known ground truth on Huisman system). It learns computational shortcuts instead.

I added a **GraphLearner** — a small MLP that learns a static interaction matrix A_ij from species embeddings:

```
A_ij = sigmoid(MLP(emb_i, emb_j))    # N×N, values in [0,1]
```

With L1 sparsity penalty (0.01 × A.mean()), A is pushed toward sparse structure.

## What works: v1 (multiplicative gating)

```python
agg_i = Σ_j  A_ij × attn_ij × msg_ij
```

Results on Huisman (6-species resource competition, ground truth known):

| Species | v1 (with GL) | Baseline (no GL) | Diff |
|---|---|---|---|
| sp1 | +0.272 | +0.313 | -0.041 |
| sp2 | +0.391 | **+0.640** | **-0.249** |
| sp3 | **+0.674** | +0.450 | +0.224 |
| sp4 | +0.443 | +0.591 | -0.148 |
| sp5 | +0.248 | +0.431 | -0.183 |
| sp6 | **+0.584** | +0.099 | **+0.485** |
| **Overall** | **+0.435** | +0.421 | +0.014 |

**Interaction structure**: Spearman = +0.72 (mean), 4/6 species significant (p < 0.02)

**v1 is the ONLY version** where the model learns correct ecological interactions. This is critical — it proves the model's dynamical mechanism is ecologically correct, not just that outputs happen to correlate.

## The problem: sp2 trade-off

v1's A converges to ~0.04 (extreme sparsity due to L1). This means 96% of messages are suppressed.

- **sp6 benefits**: sp6 has strong but diffuse coupling. Extreme sparsity forces focus on strongest edges → signal emerges from noise. sp6 jumps from +0.099 to +0.584.
- **sp2 suffers**: sp2 has moderate, distributed interactions. Extreme sparsity cuts off useful information flow. sp2 drops from +0.640 to +0.391.

## What I've tried (8 alternative versions, all failed)

| Version | Idea | Result | Why it failed |
|---|---|---|---|
| v2 | Remove L1 penalty | A stays at 0.95, learns nothing | No gradient pressure to create structure |
| v3 | Reduce L1 to 0.001 | A ≈ 0.57, moderate structure | sp2 recovered (+0.631) but sp6 lost (+0.102) |
| v4 | A as attention score bias: scores += log(A) | A ≈ 0.22, weak structure | Spearman low (+0.18), worse than v1 |
| v5 | A replaces attention entirely, L1=0.001 | A ≈ 0.95, learns nothing | No pressure without L1 |
| v6 | A replaces attention, L1=0.01 | A ≈ 0.56, some structure | +0.368 overall, worse than baseline |
| v7 | Dual channel additive: A×msgs + attn×msgs | sp2 OK (+0.609), sp6 bad (+0.097) | Structure channel too weak (A=0.04 → tiny contribution) |
| v8 | v7 + learnable scale: scale×A×msgs + attn×msgs | Early results bad | Scale didn't help enough |
| v9 | v7 + learnable exponent: A^exp × msgs + attn×msgs | sp2 OK (+0.598), sp6 bad (+0.054) | Exponent → 0 for sp6 edges, losing sparsity benefit |

## Core tension

- **Extreme sparsity (A ≈ 0.04)**: Learns correct structure (Spearman=0.72), helps sp6 (+0.485), hurts sp2 (-0.249)
- **Moderate/no sparsity (A > 0.3)**: Preserves sp2, but sp6 gets no benefit and structure matching degrades

No version found that simultaneously achieves:
1. sp6 improvement (needs extreme sparsity)
2. sp2 preservation (needs moderate/no sparsity)  
3. Correct interaction structure (needs A to be meaningfully structured)

## Constraints

- **Strictly unsupervised**: hidden species data never used in training
- A must be **part of the dynamics** (not post-hoc), so it can claim to reflect learned dynamical structure
- The rest of the model (encoder, G-field, ODE consistency, counterfactual losses) should not be changed
- Only f_visible's message aggregation is being modified

## System details

- N = 10 (5 visible species + 5 resources) per experiment
- Species embeddings: 20-dim learnable vectors
- GraphLearner embeddings: 16-dim separate learnable vectors
- MLP messages: [x_i, x_j, emb_i, emb_j, 4 formula hints] → 32-hidden → 1 scalar
- Attention: Q/K from [x, emb], top-k=4 sparse softmax
- Training: 500 epochs, AdamW, cosine LR schedule

## What I need

A modification to the GraphLearner or the way A interacts with attention/messages that:
1. Preserves v1's correct interaction structure learning (Spearman > 0.5)
2. Preserves v1's sp6 improvement (+0.5 or better)
3. Does NOT collapse sp2 (keep sp2 > +0.55)

Or argue convincingly why this is impossible and v1's trade-off is fundamental.
