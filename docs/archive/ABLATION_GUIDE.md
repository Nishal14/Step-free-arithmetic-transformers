# Ablation Evaluation Guide

## Overview

The ablation evaluation script (`eval_pilot_ablation.py`) tests whether individual attention heads are specialized for processing parenthesized expressions vs flat expressions.

## Quick Start

### Prerequisites

1. Trained pilot model checkpoint (e.g., `runs/pilot/checkpoint_best.pt`)
2. Test datasets generated:
   - `data/pilot_test_flat.jsonl` (no parentheses)
   - `data/pilot_test_paren.jsonl` (only parentheses)

### Running Ablation Evaluation

**Windows:**
```bash
eval_ablation.bat
```

**Git Bash:**
```bash
bash eval_ablation.sh
```

**Direct Python:**
```bash
.venv/Scripts/python.exe eval_pilot_ablation.py \
  --checkpoint runs/pilot/checkpoint_best.pt \
  --device cuda \
  --batch-size 32
```

## How It Works

### 1. Baseline Evaluation

First, the script evaluates the model without any ablation:
- **Flat accuracy**: Percentage of correct predictions on flat expressions
- **Paren accuracy**: Percentage of correct predictions on parenthesized expressions
- **Gap**: Difference between paren and flat accuracy

### 2. Head Ablation

For each attention head (16 heads total in the pilot model: 4 layers × 4 heads):
1. Zero out that head's output: `block.attn.ablate_head = head_id`
2. Re-evaluate on both test sets
3. Measure the impact of ablation

### 3. Specialization Metric

**Impact** = (paren_baseline - paren_ablated) - (flat_baseline - flat_ablated)

- **Positive impact**: Head is more important for parenthesized expressions
- **Negative impact**: Head is more important for flat expressions
- **Zero impact**: Head affects both equally (or not at all)

## Accuracy Metric

The script checks if the model correctly predicts **all result tokens** (the numbers after "=").

Example:
- Input: `<BOS> 1 + 2 = <labels>`
- Labels: `1 + 2 = 3 <EOS>`
- Evaluation: Check if predicted tokens match `3` exactly

For multi-digit results (e.g., `99` or `-787`), all digits must be correct.

## Interpreting Results

### Expected Results (Well-Trained Model)

With a well-trained model (50+ epochs, >70% accuracy), you should see:

1. **Baseline Gap**: Paren accuracy might be slightly lower than flat (parentheses add complexity)
2. **Head Specialization**: Some heads show higher impact on paren expressions
3. **Layer Patterns**: Different layers specialize in different aspects:
   - Early layers: Token/operator recognition
   - Middle layers: Parenthesis matching, precedence
   - Late layers: Arithmetic computation

### Current Results (Pilot Checkpoint)

The provided checkpoint (`runs/pilot_test/checkpoint_best.pt`) was trained for only **5 epochs** as a verification test:

- **Flat accuracy**: 0%
- **Paren accuracy**: 1%
- **All heads show zero impact**

This is expected - the model hasn't learned the task yet.

## Training a Better Model

To get meaningful ablation results, train for more epochs:

```bash
# Train for 50 epochs (recommended)
.venv/Scripts/python.exe train_pilot.py \
  --config configs/pilot.yaml \
  --output-dir runs/pilot_full \
  --device cuda \
  --seed 42
```

Edit `configs/pilot.yaml` to increase epochs:
```yaml
training:
  num_epochs: 50  # Increase from 5
```

Expected training time:
- **CPU**: ~2-3 minutes for 50 epochs
- **GPU (RTX 3050)**: ~30-45 seconds for 50 epochs

## Output Format

### Console Output

```
============================================================
BASELINE (No Ablation)
============================================================
Flat accuracy:   0.7200 (72.0%)
Paren accuracy:  0.6500 (65.0%)
Gap (paren-flat): -0.0700

============================================================
HEAD ABLATION EXPERIMENTS
============================================================
Layer    Head   Flat Acc     Paren Acc    Gap          Impact
------------------------------------------------------------
L0     H0    0.7100       0.6400       -0.0700       +0.0000
L0     H1    0.7200       0.6200       -0.0500       -0.0300
L0     H2    0.7100       0.6500       -0.0600       +0.0100
...

============================================================
SUMMARY
============================================================

Top 5 heads by specialization (paren impact - flat impact):
Rank   Layer    Head   Impact       Paren Acc
--------------------------------------------------
1      L2       H3     +0.0800       0.5700
2      L1       H2     +0.0500       0.6000
3      L3       H1     +0.0400       0.6100
...
```

### Key Metrics

- **Flat Acc**: Accuracy on flat expressions after ablating this head
- **Paren Acc**: Accuracy on paren expressions after ablating this head
- **Gap**: Paren Acc - Flat Acc (positive means paren is better)
- **Impact**: Specialization score (positive means specialized for paren)

### Interpreting Specialization

**High positive impact** (e.g., +0.08):
- Head is critical for parenthesized expressions
- Ablating it hurts paren accuracy more than flat accuracy
- Likely handles parenthesis matching or precedence

**Near-zero impact** (e.g., ±0.01):
- Head contributes equally to both types
- May handle general arithmetic or token prediction

**Negative impact** (rare):
- Head might be detrimental to paren expressions
- Could indicate interference or redundancy

## Advanced Usage

### Custom Checkpoint

```bash
.venv/Scripts/python.exe eval_pilot_ablation.py \
  --checkpoint path/to/your/checkpoint.pt \
  --device cuda
```

### CPU Evaluation

```bash
.venv/Scripts/python.exe eval_pilot_ablation.py \
  --checkpoint runs/pilot/checkpoint_best.pt \
  --device cpu
```

### Different Batch Size

```bash
.venv/Scripts/python.exe eval_pilot_ablation.py \
  --checkpoint runs/pilot/checkpoint_best.pt \
  --device cuda \
  --batch-size 16  # Reduce if GPU memory limited
```

## Files Created

- `eval_pilot_ablation.py` - Main evaluation script
- `eval_ablation.bat` - Windows helper script
- `eval_ablation.sh` - Git Bash helper script
- `ABLATION_GUIDE.md` - This file

## Next Steps

1. **Train a full model**: Run `train_pilot.py` with 50 epochs
2. **Run ablation**: Use `eval_ablation.bat` with the trained checkpoint
3. **Analyze results**: Look for heads with high specialization scores
4. **Visualize attention**: Use the stored attention weights to see what patterns specialized heads learn

## Troubleshooting

### "CUDA not available"

The script automatically falls back to CPU. To use GPU:
1. Ensure PyTorch with CUDA is installed: `.venv/Scripts/python.exe -c "import torch; print(torch.cuda.is_available())"`
2. Install CUDA-enabled PyTorch: See `GPU_SETUP.md`

### "Checkpoint not found"

Ensure you've trained a model first:
```bash
.venv/Scripts/python.exe train_pilot.py --config configs/pilot.yaml --output-dir runs/pilot --device cuda
```

### Low accuracy across all heads

The model needs more training. Increase `num_epochs` in `configs/pilot.yaml` and retrain.

### All heads show zero impact

Either:
1. Model hasn't learned the task (accuracy near 0%)
2. All heads contribute equally (redundant architecture)
3. Need a more challenging task (increase depth, add more operators)

---

**Status**: ✅ Implementation complete
**Date**: January 24, 2026
**Model**: Pilot arithmetic (128d, 4L, 4H)
**Test sets**: 100 flat + 100 paren expressions
