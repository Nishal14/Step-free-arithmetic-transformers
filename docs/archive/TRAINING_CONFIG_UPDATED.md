# Training Configuration Updated for New Run

## Changes Made

### 1. Increased Epochs: 40 → 50 ✓

**File:** `configs/scale_10m.yaml`

```yaml
training:
  num_epochs: 50  # Changed from 40
```

**Impact:**
- Training will run for 10 additional epochs
- Estimated time: ~9 seconds/epoch × 50 = ~7.5 minutes total
- More training may improve final validation metrics

### 2. New Random Seed: 42 → 123 ✓

**Files Updated:**
- `train_10m_scale.sh` - Changed `--seed 42` to `--seed 123`
- `train_10m_scale.bat` - Changed `--seed 42` to `--seed 123`

**Impact:**
- Different random initialization
- Different data shuffling order
- Independent training run from previous attempt
- Results can be compared for reproducibility

## New Training Command

The training scripts now use the updated configuration automatically:

```bash
# Git Bash
bash train_10m_scale.sh

# Windows CMD
train_10m_scale.bat
```

**Or manually:**
```bash
source .venv/Scripts/activate
python -m src.train \
    --config configs/scale_10m.yaml \
    --output-dir runs/scale_10m \
    --seed 123 \
    --device cuda \
    --fp16 \
    --gpu-only
```

## What to Expect

### Training Duration
- **50 epochs** at ~9 seconds/epoch
- **Estimated total time:** 7-8 minutes
- Evaluation every 2 epochs
- Checkpoints saved every 5 epochs

### Checkpoints
Saved to `runs/scale_10m/`:
- `checkpoint_epoch_5.pt`
- `checkpoint_epoch_10.pt`
- `checkpoint_epoch_15.pt`
- ...
- `checkpoint_epoch_50.pt`
- `checkpoint_best.pt` (best validation loss)
- `checkpoint_latest.pt`

### Metrics File
Saved to: `metrics/training_metrics_seed123.json`

Contains:
```json
{
  "seed": 123,
  "best_val_loss": X.XXXX,
  "best_val_perplexity": XX.XX,
  "best_val_epoch": YY,
  "metrics": [...]
}
```

### Final Summary
At the end of training:
```
==================================================
TRAINING SUMMARY
==================================================
Best validation loss:        X.XXXX
Best validation perplexity:  XX.XX
Achieved at epoch:           YY
==================================================
```

## Configuration Summary

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Model Size** | 10.65M params | 6 layers, 6 heads, d_model=384 |
| **Batch Size** | 8 | FP16 optimized for 4GB VRAM |
| **Epochs** | 50 | ← Increased from 40 |
| **Seed** | 123 | ← Changed from 42 |
| **Learning Rate** | 0.0001 | With 500-step warmup |
| **Device** | CUDA | GPU-only (RTX 3050) |
| **Precision** | FP16 | Mixed precision enabled |

## Comparison with Previous Run

If you completed a previous training with seed 42 (epochs 1-40):

| Run | Seed | Epochs | Best Val Loss | Best Val Perplexity | Epoch |
|-----|------|--------|---------------|---------------------|-------|
| Previous (partial) | 42 | 3 | 1.7083 | 5.52 | 2 |
| **New (full)** | **123** | **50** | **TBD** | **TBD** | **TBD** |

The new run will provide:
- Complete 50-epoch training
- Independent validation of model architecture
- Comparison point for reproducibility

## Files Modified

1. `configs/scale_10m.yaml` - num_epochs: 50
2. `train_10m_scale.sh` - seed: 123
3. `train_10m_scale.bat` - seed: 123

## Ready to Train

Start training with:
```bash
bash train_10m_scale.sh
```

Monitor with (separate terminal):
```bash
nvidia-smi -l 1
```

Expected VRAM usage: ~3-4 GB / 4.29 GB

---

**Status:** Configuration updated ✓
**Ready for:** Full 50-epoch training run with seed 123
**Estimated time:** 7-8 minutes
