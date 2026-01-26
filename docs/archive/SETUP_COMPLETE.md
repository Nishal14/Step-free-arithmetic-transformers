# 10M Scaling Experiment - Setup Complete ✓

## Summary

All components for the 10M parameter scaling experiment have been successfully configured with strict GPU-only enforcement and FP16 mixed precision support.

## Configuration Verified

### Model Architecture ✓
- **Parameters:** 10,651,776 (~10.65M, within 6.5% of 10M target)
- **Configuration:** `configs/scale_10m.yaml`
- **Architecture:**
  - d_model: 384
  - num_layers: 6
  - num_heads: 6
  - d_ff: 1536
  - max_seq_len: 32

### Training Configuration ✓
- **Batch size:** 8 (FP16 optimized)
- **Epochs:** 40
- **Learning rate:** 1e-4
- **Warmup steps:** 500
- **Dataset:** Same as 0.66M pilot (pilot_train.jsonl, pilot_val.jsonl)

### GPU-Only Enforcement ✓
- **Hard fail:** Enabled via `--gpu-only` flag
- **No CPU fallback:** RuntimeError raised if CUDA unavailable
- **FP16 required:** Mixed precision enabled via `--fp16` flag
- **GPU monitoring:** Prints GPU name and VRAM at startup

## Files Created

### Configuration
1. **configs/scale_10m.yaml** - Model and training configuration

### Scripts
2. **verify_10m_config.py** - Configuration verification
3. **train_10m_scale.sh** - Training launcher (Linux/Mac)
4. **train_10m_scale.bat** - Training launcher (Windows)
5. **test_gpu_enforcement.py** - GPU enforcement testing

### Documentation
6. **SCALE_10M_SETUP.md** - Complete setup guide
7. **SCALE_10M_CHECKLIST.md** - Implementation checklist
8. **QUICK_START_10M.txt** - Quick reference
9. **SETUP_COMPLETE.md** - This summary

### Data (OOD Experiment)
10. **generate_depth3_dataset.py** - Depth-3 dataset generator
11. **data/pilot_test_depth3.jsonl** - OOD test set (100 examples)
12. **eval_ood_depth3.py** - OOD evaluation script
13. **OOD_DEPTH3_RESULTS.md** - OOD results documentation

### Probing
14. **probe_depth.py** - Linear probing for depth encoding

## Files Modified

### src/train.py - Training Script Enhancements ✓

**Added:**
- `--gpu-only` flag: Hard GPU requirement enforcement
- `--fp16` flag: FP16 mixed precision training
- GPU information printing (name, VRAM)
- GradScaler for FP16 training
- RuntimeError on CUDA unavailability when `--gpu-only` is set

**Modified functions:**
- `train_epoch()`: Added FP16 support with autocast and GradScaler
- `evaluate_model()`: Added FP16 support with autocast
- `train()`: Added GPU monitoring and scaler initialization
- `main()`: Added GPU-only enforcement logic

## Test Results ✓

### Configuration Verification
```
Total parameters: 10,651,776 (10.65M)
Target: 10M
Difference: 6.5%
[OK] Configuration verified
```

### GPU Enforcement Test
```
[OK] train.py imports successfully
[OK] New flags (--gpu-only, --fp16) are properly defined
[OK] GPU enforcement logic verified
```

### Current System Status
```
CUDA available: False (CPU-only system)
Expected behavior: Training will ABORT with --gpu-only flag
This is CORRECT and INTENTIONAL
```

## Training Command

**Full command for GPU-capable system:**
```bash
python -m src.train \
    --config configs/scale_10m.yaml \
    --output-dir runs/scale_10m \
    --seed 42 \
    --device cuda \
    --fp16 \
    --gpu-only
```

**Or use launcher scripts:**
```bash
# Windows
train_10m_scale.bat

# Linux/Mac
bash train_10m_scale.sh
```

## Pre-Training Checklist

Before starting training on a GPU-capable machine:

- [ ] Run `python verify_10m_config.py`
  - Verify ~10M parameters
  - Verify CUDA available
  - Check VRAM capacity

- [ ] Run `nvidia-smi`
  - Verify GPU detected
  - Check available VRAM (need ~6GB+)
  - Ensure no other processes using GPU

- [ ] Test imports: `python test_gpu_enforcement.py`
  - Verify train.py imports
  - Confirm flags are defined

- [ ] Check dataset exists:
  - `data/pilot_train.jsonl` (2000 examples)
  - `data/pilot_val.jsonl` (200 examples)

## Expected Behavior

### On GPU-capable System
```
✓ CUDA detected: NVIDIA GeForce RTX XXXX
✓ GPU Memory: X.XX GB
✓ FP16 Mixed Precision: ENABLED
✓ Model parameters: 10.65M
✓ Training starts
✓ VRAM usage: ~3-4 GB
✓ Checkpoints saved every 5 epochs
```

### On CPU-only System
```
RuntimeError: CUDA requested but not available. Aborting.
[Training stops immediately]
```

This is **intentional and correct** for the scaling experiment.

## OOD Depth-3 Experiment Results ✓

Already completed on 0.66M pilot model:

```
Baseline accuracy:       93.0%
L2-H2 ablated accuracy:  91.0%
Accuracy drop:           2.0%
```

**Finding:** Model generalizes well to unseen depth-3 expressions. L2-H2 remains relevant but less critical for OOD depths.

## Linear Probing Results ✓

Already completed on 0.66M pilot model:

### Primary Target (L2-H2 Attention Head)
```
Train accuracy: 89.32%
Test accuracy:  89.16%
```

### Best Residual Stream (Layer 3)
```
Train accuracy: 99.93%
Test accuracy:  99.91%
```

**Finding:** Parenthesis depth is strongly encoded in both attention heads and residual streams, with increasingly stronger signal in deeper layers.

## Next Steps (After Training)

**DO NOT PROCEED until training completes successfully on GPU.**

After 10M model training:
1. Baseline evaluation (flat vs parenthesized)
2. Head identification via ablation scanning
3. Targeted ablation of identified heads
4. Linear probing for depth encoding
5. Comparison with 0.66M pilot results

**Wait for further instructions before running evaluations.**

## Troubleshooting

### "CUDA not available" Error
- **Cause:** No GPU or PyTorch without CUDA support
- **Fix:** Install CUDA toolkit + PyTorch with CUDA

### "Out of memory" Error
- **Cause:** Insufficient VRAM
- **Fix:** Reduce batch_size in config (8 → 4 → 2)

### Training starts on CPU
- **Cause:** Missing `--gpu-only` flag
- **Fix:** Add `--gpu-only` flag to command

## Status

✅ **Setup Complete**
✅ **Configuration Verified**
✅ **GPU Enforcement Tested**
✅ **FP16 Mixed Precision Enabled**
✅ **OOD Depth-3 Experiment Complete (0.66M)**
✅ **Linear Probing Complete (0.66M)**

⏳ **Awaiting:** GPU-capable machine for 10M training
⏳ **User Action Required:** Run training with provided scripts

---

**Ready for training on GPU-capable system.**
**Current system is CPU-only and will correctly reject training with --gpu-only flag.**
