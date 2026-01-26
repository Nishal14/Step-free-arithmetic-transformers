# 10M Scaling Experiment Setup

## Objective

Train a ~10M parameter transformer on the same arithmetic task to verify that parenthesis-sensitive mechanisms persist at larger scale.

**This is a GPU-ONLY experiment with HARD FAIL enforcement.**

## Model Configuration

**Config file:** `configs/scale_10m.yaml`

### Architecture
- **Parameters:** 10.65M (target: ~10M)
- **d_model:** 384
- **Layers:** 6
- **Heads:** 6
- **FFN dimension:** 1536
- **Max sequence length:** 32

### Training
- **Batch size:** 8
- **Epochs:** 40
- **Learning rate:** 1e-4
- **Warmup steps:** 500
- **FP16 mixed precision:** ENABLED

### Dataset
- **Training:** `data/pilot_train.jsonl` (2000 examples)
- **Validation:** `data/pilot_val.jsonl` (200 examples)
- Same dataset as 0.66M pilot model

## GPU Requirements

### Hard Constraints (NON-NEGOTIABLE)

❌ **No CPU training**
❌ **No silent device fallback**
❌ **No automatic downscaling**

If CUDA is not available, training will **ABORT with RuntimeError**.

### System Requirements

- **CUDA-capable GPU** (NVIDIA)
- **VRAM:** ~3-4 GB with batch_size=8, FP16
- **Recommended:** GTX 1060 6GB or better
- **CUDA toolkit** installed
- **PyTorch** with CUDA support

## Setup Steps

### Step 1: Verify Configuration

Run the verification script to check:
- Model parameter count
- GPU availability
- VRAM capacity

```bash
# Windows
.venv\Scripts\activate
python verify_10m_config.py

# Linux/Mac
source .venv/bin/activate
python verify_10m_config.py
```

**Expected output:**
```
Total parameters: 10,651,776 (10.65M)
Difference from 10M target: 6.5%
[OK] Configuration verified: ~10M parameters
[OK] Ready for GPU training
```

If you see `[ERROR] CUDA not available`, **DO NOT PROCEED**. Fix CUDA setup first.

### Step 2: Check GPU Status

```bash
nvidia-smi
```

Verify:
- GPU is detected
- VRAM available (should show total memory)
- No other processes using GPU

### Step 3: Run Training

**Windows:**
```bash
train_10m_scale.bat
```

**Linux/Mac:**
```bash
bash train_10m_scale.sh
```

**Manual command:**
```bash
python -m src.train \
    --config configs/scale_10m.yaml \
    --output-dir runs/scale_10m \
    --seed 42 \
    --device cuda \
    --fp16 \
    --gpu-only
```

### Critical Flags

- `--device cuda` - Use GPU (required)
- `--fp16` - Enable FP16 mixed precision (required for memory efficiency)
- `--gpu-only` - **HARD FAIL** if CUDA unavailable (required for this experiment)

## Training Behavior

### Startup Verification

Before training begins, the script will:

1. **Assert CUDA availability**
   - If `--gpu-only` flag is used AND CUDA not available → **RuntimeError**

2. **Print GPU information**
   ```
   GPU Name: NVIDIA GeForce RTX 3060
   GPU Memory: 12.00 GB
   FP16 Mixed Precision: ENABLED
   ```

3. **Create model and count parameters**
   ```
   Model parameters: 10.65M
   ```

If any check fails → **STOP IMMEDIATELY**

### Monitoring During Training

**Watch VRAM usage:**
```bash
watch -n 1 nvidia-smi
```

**Expected VRAM:** ~3-4 GB with batch_size=8, FP16

**If OOM (Out of Memory) occurs:**
1. Stop training (Ctrl+C)
2. Edit `configs/scale_10m.yaml`:
   ```yaml
   training:
     batch_size: 4  # Reduce from 8 to 4
   ```
3. Restart training (checkpoints will resume if saved)

### Checkpoint Saving

Checkpoints saved to: `runs/scale_10m/`

- `checkpoint_epoch_5.pt` - Every 5 epochs
- `checkpoint_epoch_10.pt`
- ...
- `checkpoint_best.pt` - Best validation loss
- `checkpoint_latest.pt` - Most recent (updated every epoch)

## Troubleshooting

### "CUDA not available" Error

**Cause:** PyTorch not installed with CUDA support or no GPU detected

**Fix:**
1. Check GPU: `nvidia-smi`
2. Reinstall PyTorch with CUDA:
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cu118
   ```
3. Verify: `python -c "import torch; print(torch.cuda.is_available())"`

### RuntimeError: CUDA requested but not available

**This is intentional behavior.** The `--gpu-only` flag enforces GPU-only training.

**Do NOT:**
- Remove `--gpu-only` flag
- Change `--device cuda` to `--device cpu`
- Modify train.py to bypass checks

**Do:**
- Fix CUDA installation
- Ensure GPU is detected
- Run on a GPU-capable machine

### Out of Memory (OOM)

```
RuntimeError: CUDA out of memory
```

**Solution 1:** Reduce batch size
```yaml
training:
  batch_size: 4  # or even 2
```

**Solution 2:** Check for memory leaks
```bash
nvidia-smi  # Should show only this training process
```

Kill any other GPU processes if needed.

### Training too slow

**Check:**
- FP16 enabled (`--fp16` flag)
- GPU utilization (`nvidia-smi` should show ~90-100%)
- Batch size not too small (< 4 may be inefficient)

## Success Criteria

Training is considered **successful** if:

1. ✓ Model trains to reasonable validation accuracy (~similar to 0.66M pilot)
2. ✓ Checkpoints are saved to `runs/scale_10m/`
3. ✓ No CPU usage occurred (GPU-only confirmed)
4. ✓ FP16 mixed precision was used
5. ✓ Training completes without OOM errors

## What Comes Next (DO NOT DO YET)

After training completes successfully:

1. Run baseline evaluation (flat vs parenthesized expressions)
2. Identify parenthesis-sensitive heads via ablation
3. Run minimal targeted ablation
4. Re-probe depth encoding with linear probes
5. Compare findings to 0.66M pilot model

**These steps will be provided separately. Do not proceed without instruction.**

## Files Created

- `configs/scale_10m.yaml` - Model and training configuration
- `verify_10m_config.py` - Verification script
- `train_10m_scale.sh` - Training launcher (Linux/Mac)
- `train_10m_scale.bat` - Training launcher (Windows)
- `SCALE_10M_SETUP.md` - This document

## Modified Files

- `src/train.py` - Added:
  - GPU-only enforcement (`--gpu-only` flag)
  - FP16 mixed precision support (`--fp16` flag)
  - GPU information printing
  - GradScaler for FP16
  - Hard fail on CUDA unavailability when `--gpu-only` is set

---

**REMINDER: This experiment REQUIRES GPU. Training will intentionally FAIL on CPU.**
