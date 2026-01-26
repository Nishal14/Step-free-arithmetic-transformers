# ✓ READY FOR GPU TRAINING

## Final Verification: ALL SYSTEMS GO

```
✓ CUDA available: True
✓ GPU: NVIDIA GeForce RTX 3050 Laptop GPU
✓ VRAM: 4.29 GB
✓ PyTorch: 2.5.1+cu121
✓ Model: 10.65M parameters
✓ Config: Verified
✓ FP16: Working
✓ GPU forward pass: Successful
```

## What Was Fixed

### 1. CUDA-Enabled PyTorch Installed ✓
```bash
uv pip install --reinstall --no-deps \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121
```

**Result:** PyTorch 2.5.1+cu121 with CUDA 12.1 support

### 2. Config Fixed ✓
**Changed:** `lr: 1e-4` → `lr: 0.0001`
**Reason:** YAML parser compatibility (float vs string)

### 3. GradScaler Deprecation Fixed ✓
**Changed:** `torch.cuda.amp.GradScaler()` → `torch.amp.GradScaler('cuda')`
**Reason:** Updated PyTorch API

### 4. Training Script Updated ✓
**Changed:** Uses direct venv activation (not `uv run`)
**Reason:** Prevents environment sync that reverts to CPU PyTorch

## Training Command

**Ready to execute:**

```bash
# Activate environment
source .venv/Scripts/activate

# Run training
python -m src.train \
    --config configs/scale_10m.yaml \
    --output-dir runs/scale_10m \
    --seed 42 \
    --device cuda \
    --fp16 \
    --gpu-only
```

**Or use the batch script:**
```bash
train_10m_scale.bat
```

## Expected Training Output

### Startup (First 10 seconds)
```
============================================================
GPU-ONLY MODE ENABLED
CPU training is disabled. Will abort if GPU unavailable.
============================================================

Configuration:
model:
  name: scale-10m-arithmetic
  ...

============================================================
GPU INFORMATION
============================================================
GPU Name: NVIDIA GeForce RTX 3050 Laptop GPU
GPU Memory: 4.29 GB
FP16 Mixed Precision: ENABLED
============================================================

Vocabulary size: 35
Model parameters: 10.65M
Train dataset size: 2000
Validation dataset size: 200
```

### During Training
```
Epoch 1/40
Train Loss: X.XXXX | Train Perplexity: XX.XX | Time: XX.XXs
Val Loss: X.XXXX | Val Perplexity: XX.XX
New best model!
Saved checkpoint to runs/scale_10m/checkpoint_epoch_1.pt
...
```

## Monitoring Commands

**In separate terminal:**
```bash
# Watch GPU every second
nvidia-smi -l 1
```

**Expected readings:**
- GPU Utilization: 90-100%
- Memory Usage: ~3-4 GB / 4.29 GB
- Temperature: < 85°C (safe operating range)

## Training Duration

**Estimated:** 1-3 hours (40 epochs on 2000 examples)

**Checkpoints saved:**
- Every 5 epochs: `checkpoint_epoch_5.pt`, `checkpoint_epoch_10.pt`, ...
- Best model: `checkpoint_best.pt`
- Latest: `checkpoint_latest.pt`

## If OOM (Out of Memory) Occurs

**Symptom:**
```
RuntimeError: CUDA out of memory
```

**Fix:**
1. Stop training (Ctrl+C)
2. Edit `configs/scale_10m.yaml`:
   ```yaml
   training:
     batch_size: 4  # Reduce from 8
   ```
3. Restart:
   ```bash
   train_10m_scale.bat
   ```

**Alternative if still OOM:**
```yaml
training:
  batch_size: 2  # Minimum for stable training
```

## Pre-Flight Checklist

- [x] CUDA enabled in PyTorch
- [x] GPU detected (RTX 3050)
- [x] Configuration verified (10.65M params)
- [x] VRAM sufficient (4.29 GB)
- [x] Dataset exists
- [x] FP16 working
- [x] Forward pass successful
- [x] Training script tested
- [x] Monitoring commands ready

## IMPORTANT: Environment Activation

**DO THIS:**
```bash
source .venv/Scripts/activate
python -m src.train ...
```

**DO NOT DO THIS:**
```bash
uv run python -m src.train ...  # Will revert to CPU PyTorch!
```

## What Happens Next

### During Training (40 epochs)
- Model trains on GPU with FP16
- Checkpoints saved every 5 epochs
- Validation every 2 epochs
- Best model tracked and saved

### After Training Completes
**DO NOT RUN YET - Wait for instructions:**
1. Baseline evaluation (flat vs paren)
2. Head identification via ablation
3. Targeted ablation experiments
4. Linear probing for depth encoding
5. Comparison with 0.66M pilot results

## Status

✅ **CUDA ENABLED**
✅ **SYSTEM VERIFIED**
✅ **READY TO TRAIN**

⏳ **AWAITING USER APPROVAL TO START TRAINING**

---

**To start training, run:**
```bash
train_10m_scale.bat
```

**Or manually:**
```bash
source .venv/Scripts/activate && python -m src.train --config configs/scale_10m.yaml --output-dir runs/scale_10m --seed 42 --device cuda --fp16 --gpu-only
```
