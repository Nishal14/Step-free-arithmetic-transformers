# Activation Patching Guide

## Overview

The activation patching script (`patch_pilot.py`) tests **causal sufficiency** of Layer 2, Head 2 (L2-H2) for handling parenthesized expressions.

## What is Activation Patching?

Activation patching is a mechanistic interpretability technique that tests whether a specific component's activation is **sufficient** to fix incorrect predictions.

**Procedure**:
1. Run a **correct** example → save L2-H2 activation
2. Run an **incorrect** example → replace its L2-H2 activation with the saved one
3. Check if the prediction becomes correct

**If patching fixes the prediction**: L2-H2 contains information causally sufficient for the task.

## Quick Start

### Prerequisites

- Trained pilot model checkpoint
- Test dataset: `data/pilot_test_paren.jsonl`

### Running the Experiment

**Windows:**
```bash
patch_pilot.bat
```

**Direct Python:**
```bash
.venv/Scripts/python.exe patch_pilot.py \
  --checkpoint runs/pilot_test/checkpoint_best.pt \
  --device cuda
```

### Command-Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--checkpoint` | Required | Path to model checkpoint |
| `--device` | `cuda` | Device (`cuda` or `cpu`) |
| `--test-path` | `data/pilot_test_paren.jsonl` | Test dataset |

## How It Works

### Target

- **Layer**: 2
- **Head**: 2
- **Module**: `model.blocks[2].attn`
- **Tensor**: `last_head_output` (shape: batch, seq_len, num_heads, head_dim)

### Patching Mechanism

The model already supports activation patching via:

```python
attn = model.blocks[2].attn
attn.patch_head_index = 2  # Which head to patch
attn.patch_head_output = saved_activation  # Shape: (1, seq_len, head_dim)
```

When these attributes are set, the forward pass replaces L2-H2's output with the saved activation.

### Experiments

#### 1. Main Patching Test

1. Find one correctly predicted example
2. Find one incorrectly predicted example (preferably same structure)
3. Run correct example, save L2-H2 activation
4. Run incorrect example WITHOUT patch → prediction wrong
5. Run incorrect example WITH patch → prediction fixed?

#### 2. Control: Different Head (L2-H1)

- Patch with L2-H1 instead of L2-H2
- **Expected**: No fix (verifies specificity to L2-H2)

#### 3. Control: Wrong → Wrong

- Patch with activation from another incorrect example
- **Expected**: No fix (verifies that "correctness" is needed, not just any activation)

### Sequence Length Handling

Examples with different sequence lengths are handled by:
1. **Preferred**: Finding pairs with matching sequence lengths
2. **Fallback**: Truncating the saved activation to match the target length

## Output Format

```
============================================================
Activation Patching Experiment
============================================================
Target: Layer 2, Head 2
Checkpoint: runs/pilot_test/checkpoint_best.pt
Device: cpu

Model loaded: 0.66M parameters

Finding examples...
Good example: 41-((65+69)-69) = -24 (seq_len: 22)
Bad example:  (91*63)-(72*78) = 117 (seq_len: 22)

============================================================
EXPERIMENT 1: Patch L2-H2 from correct to incorrect example
============================================================
Good example prediction: 2 (true: 2) [OK]
Bad example (no patch): 6 (true: 2) [WRONG]
Bad example (patched):  6 (true: 2) [No change]

============================================================
CONTROL 1: Patch different head (L2-H1 instead of L2-H2)
============================================================
Bad example (L2-H1 patch): 6 (true: 2) [No fix - expected]

============================================================
CONTROL 2: Patch from wrong->wrong example
============================================================
Could not find another incorrect example for control

============================================================
SUMMARY
============================================================
Target: Layer 2, Head 2
Good example: 41-((65+69)-69) = -24
Bad example:  (91*63)-(72*78) = 117

Results:
  Unpatched:        6 (wrong)
  Patched (L2-H2):  6 (wrong)
  Control (L2-H1):  6 (wrong)

[NO EFFECT] Patching did not fix the prediction
  (Model may need more training, or head is not causally sufficient)

============================================================
```

## Interpreting Results

### Success Case

```
Results:
  Unpatched:        7 (wrong)
  Patched (L2-H2):  3 (CORRECT)
  Control (L2-H1):  7 (wrong)

[SUCCESS] L2-H2 patching fixes the prediction
[SUCCESS] Specificity: L2-H1 does not fix it
```

**Interpretation**: L2-H2 is causally sufficient for parenthesis handling.

### Partial Success

```
Results:
  Unpatched:        7 (wrong)
  Patched (L2-H2):  3 (CORRECT)
  Control (L2-H1):  3 (CORRECT)

[PARTIAL] L2-H2 fixes it, but control also worked
```

**Interpretation**: Multiple heads can fix the prediction (redundancy or confounding).

### No Effect (Current Results)

```
Results:
  Unpatched:        6 (wrong)
  Patched (L2-H2):  6 (wrong)
  Control (L2-H1):  6 (wrong)

[NO EFFECT] Patching did not fix the prediction
```

**Interpretation**:
- Model hasn't learned the task sufficiently (current checkpoint: 5 epochs, ~4% accuracy)
- OR head is necessary but not sufficient
- OR need different example pairs

## Current Checkpoint Results

**Checkpoint**: `runs/pilot_test/checkpoint_best.pt` (5 epochs)

- **Good example found**: `41-((65+69)-69) = -24` ✓
- **Bad example found**: `(91*63)-(72*78) = 117` ✓
- **Patching effect**: None
- **Expected**: Model needs more training

## Requirements for Meaningful Results

### 1. Well-Trained Model

Train for 50+ epochs with good accuracy:

```bash
# Edit configs/pilot.yaml: set num_epochs: 50
.venv/Scripts/python.exe train_pilot.py \
  --config configs/pilot.yaml \
  --output-dir runs/pilot_full \
  --device cuda
```

**Target metrics**:
- Flat accuracy: >70%
- Paren accuracy: >60%
- Model learns parenthesis precedence

### 2. Prior Ablation Analysis

Run ablation evaluation to identify specialized heads:

```bash
.venv/Scripts/python.exe eval_pilot_ablation.py \
  --checkpoint runs/pilot_full/checkpoint_best.pt \
  --device cuda
```

Look for heads with **high positive impact** (impact > 0.05) on parenthesized expressions.

### 3. Update Target

If ablation shows a different head is specialized (e.g., L3-H2), edit `patch_pilot.py`:

```python
# Change these lines throughout:
good_state = model.blocks[3].attn.last_head_output[:, :, 2, :].clone()  # Layer 3
attn = model.blocks[3].attn  # Layer 3
```

## Implementation Details

### Example Selection Strategy

1. **First pass**: Collect all correct and incorrect examples
2. **Second pass**: Find pairs with matching sequence lengths
3. **Fallback**: Use first available if no exact match

### Activation Extraction

```python
# Run correct example
outputs = model(good_example["input_ids"].to(device))

# Extract L2-H2 activation after forward pass
# last_head_output shape: (batch, seq_len, num_heads, head_dim)
good_state = model.blocks[2].attn.last_head_output[:, :, 2, :].clone()
# good_state shape: (1, seq_len, head_dim)
```

### Patching Application

```python
# Set patching attributes
attn = model.blocks[2].attn
attn.patch_head_index = 2
attn.patch_head_output = good_state

# Forward pass will use patched activation
outputs = model(bad_example["input_ids"].to(device))

# Clear patch immediately
attn.patch_head_index = None
attn.patch_head_output = None
```

### Accuracy Metric

- **Final token only**: Checks if the last predicted token matches the last label
- **Strict**: No partial credit
- **Simple**: Fast evaluation for pilot experiments

## Limitations

1. **Simplified metric**: Only checks final token, not full result string
2. **Single example pairs**: No statistical significance testing
3. **No position-wise patching**: Patches entire sequence at once
4. **No cross-layer patching**: Only tests L2-H2
5. **Manual target selection**: No automated head selection

## Possible Extensions

### 1. Position-Wise Patching

Patch only specific positions (e.g., within parentheses):

```python
# Patch only positions 5-10
patch_state = torch.zeros_like(good_state)
patch_state[:, 5:10, :] = good_state[:, 5:10, :]
```

### 2. Multi-Head Patching

Test if multiple heads together are sufficient:

```python
# Patch L2-H2 and L2-H3 together
for head_idx in [2, 3]:
    attn.patch_head_index = head_idx
    attn.patch_head_output = saved_states[head_idx]
```

### 3. Automated Head Selection

Use ablation results to automatically test top-k specialized heads:

```python
# Read ablation results
specialized_heads = [(2, 2), (3, 1), (1, 3)]  # (layer, head)

for layer, head in specialized_heads:
    # Run patching experiment
    ...
```

### 4. Statistical Testing

Run patching on multiple example pairs:

```python
successes = 0
for good, bad in example_pairs:
    if patch_fixes(good, bad, layer=2, head=2):
        successes += 1

print(f"Success rate: {successes / len(example_pairs)}")
```

## Troubleshooting

### "Could not find a correctly predicted example"

- Model accuracy too low (train longer)
- Check test set has diverse examples

### "Could not find another incorrect example for control"

- Not enough incorrect examples with matching sequence length
- Expected with well-trained models (high accuracy)
- Can skip this control if necessary

### "Sequence lengths differ"

- Script handles automatically by truncating
- Prefer finding matching lengths for cleaner results

### "No effect from patching"

**Expected if**:
- Model undertrained (<10 epochs, <50% accuracy)
- Head is necessary but not sufficient
- Need different example pairs

**Solutions**:
1. Train model longer (50+ epochs)
2. Run ablation to confirm head specialization
3. Try different example pairs
4. Test different layers/heads

---

**Status**: ✅ Implementation complete
**Date**: January 24, 2026
**Target**: Layer 2, Head 2
**Current results**: No effect (model undertrained)
**Next steps**: Train to convergence, re-run patching
