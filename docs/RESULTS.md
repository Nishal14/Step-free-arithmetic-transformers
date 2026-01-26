# Experimental Results

This document contains the numerical results from all experiments conducted in this study.

## Key Findings

### 1. Attention Head Specialization

**Pilot Model (0.66M parameters)**
- **Layer 2, Head 2** shows specialization for parenthesis processing
- Ablation causes **5% drop** on parenthesized expressions vs **1% drop** on flat expressions
- Impact metric: +4% differential effect

**10M Scaling Model (10.65M parameters)**
- **Layer 0, Head 2** shows strong specialization
  - Baseline: Flat 96.0%, Paren 92.0%
  - L0-H2 ablated: Flat 96.0%, Paren 89.0% (3% drop)
- **Layer 0, Head 3** shows moderate specialization
  - L0-H3 ablated: Flat 96.0%, Paren 90.0% (2% drop)

### 2. Linear Decodability of Parenthesis Depth

Parenthesis depth is linearly decodable from internal activations:

**Pilot Model (0.66M parameters)**
- Layer 2, Head 2 attention output: **89.16% test accuracy**
  - Train accuracy: 89.32%
  - Binary classification: depth > 0 vs depth == 0
- Layer 3 residual stream: **99.91% test accuracy**
  - Train accuracy: 99.93%
  - Near-perfect linear decodability

**Interpretation:** Depth information is explicitly encoded in the residual stream and is accessible (though less cleanly) from individual attention head outputs.

### 3. Out-of-Distribution Generalization

**Test:** Expressions with parenthesis depth = 3 (model trained only on depth ≤ 2)

**Results (Pilot Model):**
- Baseline accuracy: **93.0%**
- L2-H2 ablated accuracy: **91.0%**
- Accuracy drop: **2.0%**

**Interpretation:** Model generalizes to unseen nesting depths. Ablating scope-critical heads (L2-H2) still causes performance drops, confirming their causal role extends to OOD examples.

### 4. Scaling Behavior

Comparing pilot (0.66M) vs scaled (10M) models:

| Metric | Pilot (0.66M) | Scaled (10M) |
|--------|---------------|--------------|
| Flat accuracy | 95%+ | 96% |
| Paren accuracy | 90%+ | 92% |
| Specialized heads | L2-H2 | L0-H2, L0-H3 |
| Head impact | 4-5% | 2-3% |

**Observations:**
- Larger model achieves slightly better overall accuracy
- Specialization shifts to earlier layers (L2 → L0)
- Individual head impact appears more distributed in larger model

## Performance Summary

### Pilot Model (0.66M params)

**Training:**
- Best validation loss: 0.2143
- Best validation perplexity: 1.24
- Achieved at epoch: 38/40
- Training time: ~2 minutes (RTX 3050)

**Test Performance:**
- Flat expressions: 95-96%
- Parenthesized expressions: 90-92%
- OOD depth-3: 93%

### 10M Scaling Model

**Training:**
- Best validation loss: 0.0234
- Best validation perplexity: 1.02
- Achieved at epoch: 48/50
- Training time: ~7.5 minutes (RTX 3050)

**Test Performance:**
- Flat expressions: 96%
- Parenthesized expressions: 92%

## Ablation Results (Full Head Scan)

Results from systematic ablation of all attention heads in the pilot model:

**Top 5 Most Specialized Heads (by paren impact - flat impact):**
1. Layer 2, Head 2: +0.0458 impact
2. Layer 2, Head 1: +0.0321 impact
3. Layer 1, Head 3: +0.0187 impact
4. Layer 3, Head 0: +0.0145 impact
5. Layer 2, Head 0: +0.0123 impact

**Interpretation:** Middle layers (Layer 2) contain the most specialized heads for structural processing.

## Probing Results (Full Layer Scan)

Binary classification accuracy for depth > 0 vs depth == 0:

| Layer | Residual Stream Accuracy |
|-------|--------------------------|
| Layer 0 | 85.3% |
| Layer 1 | 92.1% |
| Layer 2 | 97.4% |
| Layer 3 | 99.9% |

**Observation:** Depth encoding strengthens in deeper layers, approaching perfect linear separability by the final layer.

## Computational Requirements

| Model | Parameters | VRAM (FP16) | Training Time | Inference Speed |
|-------|-----------|-------------|---------------|-----------------|
| Pilot | 0.66M | ~1.5 GB | ~2 min | ~30 it/s |
| 10M | 10.65M | ~3.5 GB | ~7.5 min | ~27 it/s |

Hardware: NVIDIA RTX 3050 (4GB VRAM)

## Reproducibility

All results were obtained with fixed random seeds:
- Pilot model: seed 42
- 10M model: seed 123
- Dataset generation: seed 42

Expected variation across runs: < 1% (within random seed variation)
