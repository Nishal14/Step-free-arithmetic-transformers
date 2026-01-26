# Experimental Methodology

This document describes the three main experiments conducted in this study.

## 1. Attention Head Ablation

### Objective
Identify which attention heads are causally necessary for processing parenthesized arithmetic expressions.

### Method
For each attention head (layer, head) in the model:
1. Zero out the head's output during forward pass
2. Evaluate on two test sets:
   - **Flat test set**: Expressions without parentheses (e.g., `3+5*2`)
   - **Paren test set**: Expressions with parentheses (e.g., `(3+5)*2`)
3. Compare accuracy drops on each test set

### Metrics
- **Flat accuracy**: Accuracy on flat expressions with head ablated
- **Paren accuracy**: Accuracy on parenthesized expressions with head ablated
- **Impact**: `(baseline_paren - ablated_paren) - (baseline_flat - ablated_flat)`

**Interpretation:** Heads with high positive impact are specialized for parenthesis processing.

### Implementation
```python
def set_ablation(model, layer_idx, head_id):
    """Ablate a specific attention head by setting ablate_head attribute."""
    model.blocks[layer_idx].attn.ablate_head = head_id
```

The model's attention mechanism checks for this attribute and zeros the corresponding head output:
```python
if hasattr(self, 'ablate_head') and self.ablate_head is not None:
    attn_output[:, :, self.ablate_head, :] = 0
```

### Running the Experiment

**Full head scan:**
```bash
python eval/eval_pilot_ablation.py \
    --checkpoint runs/pilot/checkpoint_best.pt \
    --device cuda
```

**Targeted ablation:**
```bash
python eval/eval_pilot_ablation.py \
    --checkpoint runs/pilot/checkpoint_best.pt \
    --device cuda \
    --ablate-layer 2 \
    --ablate-head 2
```

## 2. Linear Probing

### Objective
Test whether parenthesis depth is linearly decodable from internal model activations.

### Method
1. Extract features from specific model locations:
   - **Attention head outputs**: `model.blocks[layer].attn.last_head_output`
   - **Residual stream outputs**: After each transformer block
2. Label each token position with binary depth: 0 if `depth == 0`, else 1
3. Train logistic regression classifier: `features → depth label`
4. Evaluate on held-out test set

### Locations Probed

**Primary target:**
- Layer 2, Head 2 output (identified as specialized via ablation)

**Secondary targets:**
- Residual stream after each layer (0, 1, 2, 3)

### Classification Task
- **Binary classification**: depth > 0 vs depth == 0
- Simpler than multi-class, tests if depth information is present at all
- Balanced classes in the training set

### Feature Extraction
For attention head outputs:
```python
head_output = model.blocks[layer_idx].attn.last_head_output
features = head_output[0, :, head_idx, :].cpu().numpy()
```

For residual streams (via forward hook):
```python
def residual_hook(module, input, output):
    residual_activations['output'] = output.detach()

model.blocks[layer_idx].register_forward_hook(residual_hook)
```

### Token-Character Alignment
The model operates on tokens (with `<BOS>`, `<EOS>`), but depth labels are character-aligned. Mapping:
```
Position in model output = character index + 1 (accounting for <BOS>)
```

### Running the Experiment

```bash
python probe/probe_depth.py \
    --checkpoint runs/pilot/checkpoint_best.pt \
    --layer 2 \
    --head 2 \
    --device cuda
```

For residual stream probing, the script automatically scans all layers.

## 3. Out-of-Distribution Generalization

### Objective
Test whether the model generalizes to parenthesis depths unseen during training.

### Setup
- **Training data**: Maximum depth = 2
- **Test data**: All expressions have depth = 3 exactly
- **Examples**:
  - `(((1+2)))`
  - `((3+4)*5)`
  - `(((2*3)+4)*5)`

### Method
1. Generate 100 expressions with exactly depth = 3
2. Verify all have `max(depth) == 3`
3. Evaluate baseline accuracy
4. Ablate Layer 2, Head 2 and re-evaluate
5. Measure accuracy drop

### Hypothesis
If L2-H2 is genuinely responsible for parenthesis processing (not just memorizing training patterns), ablating it should hurt performance even on OOD depths.

### Dataset Generation

```bash
python src/data/generate_depth3_dataset.py
```

Output: `data/pilot_test_depth3.jsonl`

### Running the Experiment

```bash
python eval/eval_ood_depth3.py \
    --checkpoint runs/pilot/checkpoint_best.pt \
    --device cuda
```

## Experimental Controls

### Randomization
- All experiments use fixed random seeds (42 for pilot, 123 for 10M)
- Dataset generation uses seed 42
- Ensures reproducibility

### Test Set Separation
- Test sets are completely held-out during training
- No overlap between train/val/test splits
- Flat vs paren test sets are disjoint

### Baseline Comparisons
All ablation experiments report:
1. Baseline (no ablation) performance
2. Ablated performance
3. Difference (drop)

### Multiple Seeds
For robustness, experiments should ideally be repeated with multiple seeds. Current results use single seeds but are reproducible.

## Limitations

1. **Single seed**: Results are for one random initialization (though reproducible)
2. **Small test sets**: 100 examples per test set (sufficient for these accuracies, but not large-scale)
3. **Binary depth classification**: Probing uses binary labels, not full depth values
4. **Ablation is zero-ablation**: Other intervention methods (e.g., mean ablation, random ablation) not tested
5. **Static analysis**: No analysis of attention patterns or activation magnitudes

## Future Extensions

Potential follow-up experiments:
- Multi-seed analysis for statistical significance
- Multi-class depth probing (classify exact depth: 0, 1, 2, 3)
- Attention pattern visualization
- Activation patching experiments
- Cross-model comparison (pilot vs 10M)
- Scaling to larger models (50M+)
