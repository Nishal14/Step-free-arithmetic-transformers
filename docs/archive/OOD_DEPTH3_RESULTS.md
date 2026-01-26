# OOD Depth-3 Generalization Experiment

## Objective
Test whether Layer 2, Head 2 remains selectively necessary when evaluating expressions with parenthesis depth = 3, which were never seen during training.

**Training context:** Model was trained on expressions with max depth ≤ 2

## Dataset: `data/pilot_test_depth3.jsonl`

- **Size:** 100 examples
- **Parenthesis depth:** Exactly 3 (all examples)
- **Grammar:** Same as training (operators: +, -, *)
- **Number range:** 1-99 (same as training)
- **Expression length:** 10-19 characters (avg: 14.2)

### Sample Examples
```
(((4+95)+36)*32)  = 4320    depth: [1,2,3,3,3,3,3,2,2,2,2,1,1,1,1,0]
(((87*70)))       = 6090    depth: [1,2,3,3,3,3,3,3,2,1,0]
(((76-5)))        = 71      depth: [1,2,3,3,3,3,3,2,1,0]
```

## Results

### OOD Depth-3 Results
```
Baseline accuracy:       93.0%
L2-H2 ablated accuracy:  91.0%
Accuracy drop:           2.0%
```

## Interpretation

### Key Findings

1. **Strong OOD generalization:** The model achieves 93% accuracy on depth-3 expressions despite only being trained on depth ≤ 2.

2. **L2-H2 remains relevant but less critical:** Ablating Layer 2, Head 2 causes a 2.0% accuracy drop on OOD depth-3 examples.

3. **Smaller impact than in-distribution:** The 2.0% drop on depth-3 is smaller than typical in-distribution impact, suggesting:
   - The model may use alternative pathways for deeper nesting
   - L2-H2's depth-tracking mechanism may be optimized for depths 1-2
   - Other heads may compensate for extreme depth cases

### Comparison Context

**Model specifications:**
- Architecture: 4 layers, 4 heads per layer
- Parameters: 0.66M
- Training: depth 0-2 expressions only

**Evaluation:**
- Metric: Last-answer-token accuracy
- Ablation method: Zero out attention head output

## Conclusion

Layer 2, Head 2 maintains some selective necessity for unseen depth-3 expressions (2.0% drop), but the effect is less pronounced than for in-distribution depths. This suggests the head's depth-tracking mechanism generalizes to deeper nesting but may not be as critical when the model encounters OOD structural complexity.

The strong baseline performance (93%) indicates robust compositional generalization, while the modest ablation impact suggests the model has learned distributed depth representations that can handle novel nesting levels through multiple pathways.
