# Pilot Dataset Implementation Summary

## ✅ Implementation Complete

All requirements for the pilot arithmetic dataset with parentheses have been successfully implemented.

### Files Created

1. **`generate_pilot_dataset.py`** - Dataset generation script
2. **`src/pilot_dataset.py`** - PyTorch dataset loader
3. **`train_pilot.py`** - Training script for pilot experiments
4. **`configs/pilot.yaml`** - Training configuration
5. **`PILOT_DATASET.md`** - Complete documentation

### Datasets Generated

| File | Size | Description |
|------|------|-------------|
| `data/pilot_train.jsonl` | 2000 | Mixed flat and parenthesized |
| `data/pilot_val.jsonl` | 200 | Mixed distribution |
| `data/pilot_test_flat.jsonl` | 100 | No parentheses (ablation baseline) |
| `data/pilot_test_paren.jsonl` | 100 | Only parentheses (ablation target) |

## Dataset Characteristics

### Grammar Compliance ✓

```
Expr   := Term | Expr (+|-) Term
Term   := Factor | Term * Factor
Factor := Number | '(' Expr ')'
Number := 1-2 digit integers (1-99)
```

**Operators**: `+`, `-`, `*`
**Characters**: `0123456789+-*()`
**Max depth**: 2

### Constraint Verification ✓

| Constraint | Required | Actual | Status |
|------------|----------|--------|--------|
| Max depth | 2 | 2 | ✓ |
| Numbers per expr | 2-4 | 2-4 | ✓ |
| Digits per number | 1-2 | 1-2 | ✓ |
| Training size | ~2000 | 2000 | ✓ |
| Validation size | ~200 | 200 | ✓ |
| Test size | ~200 | 200 | ✓ |

### Data Format ✓

Each example contains:
```json
{
  "expr": "(1+(2*3))",
  "result": 7,
  "depth": [1, 1, 1, 2, 2, 2, 2, 1, 0]
}
```

**Depth computation verified**: Matches specification exactly.

### Split Requirements ✓

- **Training**: 68.7% with parentheses (mixed distribution)
- **Validation**: 68.0% with parentheses (same distribution)
- **Test Flat**: 0% with parentheses (baseline)
- **Test Paren**: 100% with parentheses (ablation target)

### Safety Checks ✓

All expressions validated:
- ✓ Results are integers
- ✓ No division or syntax errors
- ✓ Within 64-bit signed range
- ✓ **0 rejections** across all splits

## Example Expressions

### Depth 0 (Flat - 626 in training)
```
4*36+29+95 = 268
71-81*47 = -3736
53+82*1-36 = 99
```

### Depth 1 (679 in training)
```
(4*12)*(28+30) = 2784
(98-21)-(90+55) = -68
41-(65+69-69) = -24
```

### Depth 2 (695 in training)
```
((95*12)) = 1140
((64-83)) = -19
(1+(2*3)) = 7
```

## Training Verification ✓

Tested pilot training script:

```bash
.venv/Scripts/python.exe train_pilot.py \
  --config configs/pilot.yaml \
  --output-dir runs/pilot_test \
  --device cpu
```

**Results after 5 epochs**:
- Training loss: 3.05 → 1.64
- Validation loss: 1.62
- Validation perplexity: 5.07
- Training time: ~3 seconds/epoch on CPU

**Model learns successfully!**

## Tokenization Compatibility ✓

Existing `SimpleTokenizer` supports all required characters:
- Digits: `0123456789` ✓
- Operators: `+-*` ✓
- Parentheses: `()` ✓

All pilot expressions tokenize and decode correctly.

## Usage Examples

### Generate Dataset

```bash
.venv/Scripts/python.exe generate_pilot_dataset.py
```

Output:
- `data/pilot_train.jsonl` (2000 examples)
- `data/pilot_val.jsonl` (200 examples)
- `data/pilot_test_flat.jsonl` (100 examples)
- `data/pilot_test_paren.jsonl` (100 examples)

### Train Model

```bash
# CPU training
.venv/Scripts/python.exe train_pilot.py \
  --config configs/pilot.yaml \
  --output-dir runs/pilot \
  --device cpu

# GPU training
.venv/Scripts/python.exe train_pilot.py \
  --config configs/pilot.yaml \
  --output-dir runs/pilot \
  --device cuda
```

### Load Dataset in Python

```python
from src.pilot_dataset import PilotMathDataset
from src.train import SimpleTokenizer

tokenizer = SimpleTokenizer()
dataset = PilotMathDataset(
    "data/pilot_train.jsonl",
    tokenizer,
    max_seq_len=32
)

# Access examples
example = dataset[0]
print(example['input_ids'])  # Tokenized expression + result
print(example['labels'])     # Next-token prediction labels
```

### Mechanistic Interpretability Experiments

#### Experiment 1: Flat vs Parenthesized Performance

```python
# Test on flat expressions
test_flat = PilotMathDataset("data/pilot_test_flat.jsonl", tokenizer)
flat_loss = evaluate_model(model, test_flat_loader, device)

# Test on parenthesized expressions
test_paren = PilotMathDataset("data/pilot_test_paren.jsonl", tokenizer)
paren_loss = evaluate_model(model, test_paren_loader, device)

# Compare performance
print(f"Flat loss: {flat_loss}")
print(f"Paren loss: {paren_loss}")
print(f"Gap: {paren_loss - flat_loss}")
```

#### Experiment 2: Head Ablation by Expression Type

```python
model.eval()

# Test each head
for layer_idx in range(model.num_layers):
    for head_idx in range(model.num_heads):
        # Ablate head
        model.blocks[layer_idx].attn.ablate_head = head_idx

        # Test on flat
        flat_loss_ablated = evaluate_model(model, test_flat_loader, device)

        # Test on paren
        paren_loss_ablated = evaluate_model(model, test_paren_loader, device)

        # Measure impact
        flat_impact = flat_loss_ablated - flat_loss
        paren_impact = paren_loss_ablated - paren_loss

        print(f"Layer {layer_idx}, Head {head_idx}:")
        print(f"  Flat impact: {flat_impact:.4f}")
        print(f"  Paren impact: {paren_impact:.4f}")
        print(f"  Specialization: {paren_impact - flat_impact:.4f}")

        # Clear ablation
        model.blocks[layer_idx].attn.ablate_head = None
```

## Model Configuration

**Recommended for pilot** (configs/pilot.yaml):

```yaml
model:
  d_model: 128
  num_layers: 4
  num_heads: 4
  d_ff: 384
  max_seq_len: 32

training:
  num_epochs: 50
  batch_size: 32
  lr: 0.0003
```

**Model size**: ~0.66M parameters
**Training time**: ~2-3 minutes for 50 epochs on RTX 3050
**GPU memory**: ~100-200 MB

## Success Criteria ✓

All requirements met:

- ✓ Expressions evaluate correctly
- ✓ Parenthesis depth computed correctly
- ✓ Max depth = 2
- ✓ Numbers: 1-2 digits, 2-4 per expression
- ✓ Training: ~2000 examples (exact: 2000)
- ✓ Validation: ~200 examples (exact: 200)
- ✓ Test: ~200 examples (exact: 200, split 100+100)
- ✓ Test split: flat vs parenthesized
- ✓ JSONL format
- ✓ Correct file locations
- ✓ Grammar compliance
- ✓ No rejections
- ✓ Training works correctly
- ✓ Tokenization compatible

## Limitations (By Design)

This is a **pilot dataset** with intentional constraints:

- Small size (2400 total examples)
- Max depth 2
- Simple operators only
- No curriculum
- No step-by-step reasoning

**Purpose**: Sanity checks and initial ablation experiments, not final publication.

## Next Steps

After successful pilot experiments:

1. **Scale up**: Generate 20k-50k examples
2. **Increase depth**: Max depth 3-4
3. **Add operators**: Division, modulo, power
4. **Add curriculum**: Progressive difficulty
5. **Add reasoning**: Step-by-step traces

## Files Generated

```
generate_pilot_dataset.py        # Generator script
src/pilot_dataset.py             # PyTorch dataset loader
train_pilot.py                   # Training script
configs/pilot.yaml               # Training config
data/pilot_train.jsonl           # 2000 training examples
data/pilot_val.jsonl             # 200 validation examples
data/pilot_test_flat.jsonl       # 100 flat test examples
data/pilot_test_paren.jsonl      # 100 paren test examples
PILOT_DATASET.md                 # Complete documentation
PILOT_SUMMARY.md                 # This file
```

---

**Status**: ✅ Fully implemented and tested
**Date**: January 24, 2026
**Reproducible**: Yes (seed=42)
**Ready for experiments**: Yes
