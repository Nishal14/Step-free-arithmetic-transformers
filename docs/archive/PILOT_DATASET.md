# Pilot Dataset: Arithmetic with Parentheses

## Overview

Small-scale pilot dataset for mechanistic interpretability experiments on arithmetic expressions with parentheses.

**Purpose**: Sanity checks and initial head ablation experiments, not final research dataset.

## Dataset Specification

### Grammar

```
Expr   := Term | Expr (+|-) Term
Term   := Factor | Term * Factor
Factor := Number | '(' Expr ')'
Number := 1-2 digit integers (no leading zeros)
```

**Operators**: `+`, `-`, `*` (no division)
**Characters**: `0123456789+-*()`
**No spaces in expressions**

### Constraints

| Parameter | Value |
|-----------|-------|
| Max parenthesis depth | 2 |
| Numbers per expression | 2-4 |
| Digits per number | 1-2 |
| Training set | 2000 |
| Validation set | 200 |
| Test set (flat) | 100 |
| Test set (paren) | 100 |

## Dataset Splits

### Files Generated

```
data/pilot_train.jsonl      # 2000 examples, mixed
data/pilot_val.jsonl         # 200 examples, mixed
data/pilot_test_flat.jsonl   # 100 examples, no parentheses
data/pilot_test_paren.jsonl  # 100 examples, with parentheses
```

### Split Statistics

**Training Set** (2000 examples):
- With parentheses: 1374 (68.7%)
- Max depth: 2
- Expression length: 3-17 characters (avg: 10.0)

**Validation Set** (200 examples):
- With parentheses: 136 (68.0%)
- Max depth: 2
- Expression length: 4-17 characters (avg: 10.1)

**Test Flat** (100 examples):
- With parentheses: 0 (0.0%)
- Max depth: 0
- Expression length: 4-11 characters (avg: 7.8)

**Test Paren** (100 examples):
- With parentheses: 100 (100.0%)
- Max depth: 2
- Expression length: 5-17 characters (avg: 11.2)

## Data Format

Each example is a JSON object with:

```json
{
  "expr": "(1+(2*3))",
  "result": 7,
  "depth": [1, 1, 1, 2, 2, 2, 2, 1, 0]
}
```

### Fields

- **`expr`**: Expression string (no spaces)
- **`result`**: Integer result of evaluation
- **`depth`**: List of parenthesis depth per character

### Depth Computation

Depth is computed character by character:

1. Start with `depth = 0`
2. For `(`: increment depth, then record
3. For `)`: decrement depth, then record
4. For other characters: record current depth

**Example**: `(1+(2*3))`
```
Position:  0  1  2  3  4  5  6  7  8
Character: (  1  +  (  2  *  3  )  )
Depth:     1  1  1  2  2  2  2  1  0
```

## Example Expressions

### Depth 0 (Flat)
```
4*36+29+95 = 268
71-81*47 = -3736
53+82*1-36 = 99
```

### Depth 1
```
(4*12)*(28+30) = 2784
(98-21)-(90+55) = -68
41-(65+69-69) = -24
```

### Depth 2
```
((95*12)) = 1140
((64-83)) = -19
(1+(2*3)) = 7
```

## Generation Details

### Safety Checks

Expressions are validated:
- Result must be an integer
- No division or syntax errors
- Absolute value < 2^63 (64-bit signed range)

### Rejection Rate

- Training: 0 rejections
- Validation: 0 rejections
- Test (flat): 0 rejections
- Test (paren): 0 rejections

**All expressions are valid!**

## Usage for Mechanistic Interpretability

### Experiment 1: Flat vs Parenthesized

Compare model performance on:
- `pilot_test_flat.jsonl` (no parentheses)
- `pilot_test_paren.jsonl` (with parentheses)

**Expected**: Model may struggle with parentheses if undertrained.

### Experiment 2: Head Ablation by Depth

1. Train model on `pilot_train.jsonl`
2. For each attention head:
   - Ablate head (set outputs to zero)
   - Test on flat vs parenthesized splits
3. Identify heads specialized for parenthesis handling

### Experiment 3: Activation Patching

Patch activations from:
- Flat expression → parenthesized expression
- Depth 1 → depth 2 expression

Measure impact on performance.

## Training Configuration

Recommended config for pilot experiments:

```yaml
model:
  d_model: 128
  num_layers: 4
  num_heads: 4
  d_ff: 384
  max_seq_len: 32  # Max expression length ~17

data:
  train_path: "data/pilot_train.jsonl"
  val_path: "data/pilot_val.jsonl"
  max_seq_len: 32
  use_steps: false  # Final answer only

training:
  num_epochs: 50
  batch_size: 32
  eval_interval: 5
```

## Tokenization

Use the existing `SimpleTokenizer` from `src/train.py`:

```python
from src.train import SimpleTokenizer

tokenizer = SimpleTokenizer()
expr = "(1+2)*3"
tokens = tokenizer.encode(expr)  # [BOS, (, 1, +, 2, ), *, 3, EOS]
```

The tokenizer already supports all required characters:
- Digits: `0123456789`
- Operators: `+-*`
- Parentheses: `()`

## Depth-Aware Analysis

The `depth` field enables:

1. **Per-depth accuracy**: How well does the model handle depth 0 vs 1 vs 2?
2. **Depth-specific attention**: Do certain heads activate more at higher depths?
3. **Depth generalization**: Train on depth ≤1, test on depth 2

## Generation Script

To regenerate or modify the dataset:

```bash
.venv/Scripts/python.exe generate_pilot_dataset.py
```

The script is deterministic (seed=42) for reproducibility.

## Limitations

This is a **pilot dataset** with intentional constraints:

- Small size (~2400 total examples)
- Max depth 2 (shallow nesting)
- Simple operators only
- No unary minus or division
- No curriculum or difficulty progression

**Not suitable for final publication** - use this for initial experiments only.

## Next Steps

After pilot experiments succeed:

1. Generate larger dataset (20k-50k examples)
2. Increase max depth to 3-4
3. Add more operators (division, modulo)
4. Include curriculum (easy → hard)
5. Add step-by-step reasoning traces

## Files

- **`generate_pilot_dataset.py`** - Generation script
- **`data/pilot_*.jsonl`** - Dataset files
- **`PILOT_DATASET.md`** - This documentation

---

**Status**: ✅ Dataset generated and verified
**Date**: January 24, 2026
**Seed**: 42 (reproducible)
