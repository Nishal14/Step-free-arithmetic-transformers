# GPU Setup Guide for RTX 3050

This guide documents the GPU training setup for math-compact on Windows with NVIDIA RTX 3050.

## ✅ Completed Setup

### Hardware
- **GPU**: NVIDIA GeForce RTX 3050 Laptop GPU
- **VRAM**: 4096 MB (4 GB)
- **Driver Version**: 566.07
- **CUDA Support**: 12.7

### Software
- **PyTorch**: 2.5.1+cu121 (CUDA 12.1)
- **torchvision**: 0.20.1+cu121
- **torchaudio**: 2.5.1+cu121
- **cuDNN**: 9.1.0

## Verification

All GPU checks passed:
```bash
✓ nvidia-smi shows RTX 3050
✓ torch.cuda.is_available() == True
✓ GPU name: NVIDIA GeForce RTX 3050 Laptop GPU
✓ Model runs on cuda:0
✓ Forward pass successful on GPU
✓ Memory efficient: ~11 MB for small model
```

## Running GPU Training

### ⚠️ Important: Don't use `uv run`

The `uv run` command re-syncs dependencies from `uv.lock`, which reverts to CPU-only PyTorch. Instead, use one of these methods:

### Method 1: Helper Scripts (Easiest)

**Windows CMD/PowerShell:**
```bash
train_gpu.bat
```

**Git Bash:**
```bash
bash train_gpu.sh
```

### Method 2: Direct Python Call

```bash
.venv/Scripts/python.exe -m src.train \
  --config configs/train.yaml \
  --output-dir runs/gpu_training \
  --device cuda \
  --seed 42
```

### Method 3: Activate Environment First

**Git Bash:**
```bash
source .venv/Scripts/activate
python -m src.train --config configs/train.yaml --output-dir runs/gpu_training --device cuda
```

**CMD:**
```cmd
.venv\Scripts\activate.bat
python -m src.train --config configs/train.yaml --output-dir runs/gpu_training --device cuda
```

**PowerShell:**
```powershell
.venv\Scripts\Activate.ps1
python -m src.train --config configs/train.yaml --output-dir runs/gpu_training --device cuda
```

## RTX 3050 Optimization Tips (4GB VRAM)

### Recommended Model Configuration

Edit `configs/train.yaml`:

```yaml
model:
  d_model: 256          # ✓ Good for 4GB VRAM
  num_layers: 6         # Can go up to 8
  num_heads: 8
  d_ff: 768             # Reduced from 1024 for memory efficiency
  max_seq_len: 512
  ffn_type: "gelu"      # Default, most memory-efficient

training:
  batch_size: 16        # Start here, reduce to 8 if OOM
  num_epochs: 20
```

### If You Get OOM (Out of Memory) Errors

**Option 1: Reduce Batch Size**
```yaml
training:
  batch_size: 8  # or even 4
```

**Option 2: Reduce Sequence Length**
```yaml
data:
  max_seq_len: 256  # Down from 512
```

**Option 3: Use Smaller Model**
```yaml
model:
  d_model: 128
  num_layers: 4
  d_ff: 512
```

**Option 4: Gradient Accumulation (future)**
```yaml
training:
  batch_size: 4
  gradient_accumulation_steps: 4  # Effective batch size = 16
```

## Monitoring GPU Usage

### Check GPU Status
```bash
nvidia-smi
```

### Watch GPU in Real-Time (Git Bash)
```bash
watch -n 1 nvidia-smi
```

### Check Memory During Training
```bash
# In another terminal
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv -l 1
```

## Troubleshooting

### Problem: `torch.cuda.is_available()` returns False

**Solution:** Check if you're using `uv run`:
```bash
# ❌ Wrong - reverts to CPU PyTorch
uv run python -c "import torch; print(torch.cuda.is_available())"

# ✓ Correct - uses CUDA PyTorch
.venv/Scripts/python.exe -c "import torch; print(torch.cuda.is_available())"
```

### Problem: Environment keeps reverting to CPU PyTorch

**Solution:** Always use `.venv/Scripts/python.exe` directly or activate the environment first. Do NOT use `uv run`.

### Problem: CUDA out of memory during training

**Solution:** Reduce batch size or model size (see optimization tips above).

### Problem: Training is slow even on GPU

**Check:**
1. GPU utilization with `nvidia-smi` - should be 50-100%
2. Batch size isn't too small (< 4)
3. Data isn't bottlenecking (use `num_workers > 0` on Linux, keep 0 on Windows)

## Performance Benchmarks

### Small Model (0.67M params)
- Config: d_model=256, num_layers=2, d_ff=1024
- Batch size: 32
- GPU memory: ~200 MB
- Speed: ~2000 tokens/sec

### Medium Model (5-10M params) - Recommended
- Config: d_model=256, num_layers=6, d_ff=768
- Batch size: 16
- GPU memory: ~1-2 GB
- Speed: ~1000 tokens/sec

### Large Model (20-30M params) - May need tuning
- Config: d_model=512, num_layers=8, d_ff=1536
- Batch size: 8
- GPU memory: ~3-3.5 GB
- Speed: ~500 tokens/sec

## Next Steps

1. **Test with small dataset:**
   ```bash
   bash train_gpu.sh
   ```

2. **Generate training data with steps:**
   ```bash
   .venv/Scripts/python.exe -m src.data.generate \
     --task add \
     --instances 20000 \
     --with_steps \
     --output_dir data
   ```

3. **Monitor training:**
   - Watch GPU utilization: `nvidia-smi`
   - Check training logs in `runs/gpu_training/`
   - View metrics: `metrics/training_metrics_seed42.json`

4. **Run interpretability analysis:**
   ```bash
   .venv/Scripts/python.exe example_interpretability.py
   ```

## Additional Resources

- [PyTorch CUDA Installation](https://pytorch.org/get-started/locally/)
- [NVIDIA Driver Downloads](https://www.nvidia.com/Download/index.aspx)
- [RTX 3050 Specs](https://www.nvidia.com/en-us/geforce/graphics-cards/30-series/rtx-3050-3050ti-laptop/)

---

**Status**: ✅ GPU training fully operational!
**Last Updated**: January 23, 2026
**GPU**: NVIDIA GeForce RTX 3050 Laptop GPU (4GB VRAM)
