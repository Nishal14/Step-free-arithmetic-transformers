# Quick Start: GPU Training

## ⚡ Fastest Way to Train on GPU

```bash
# Windows CMD/PowerShell
train_gpu.bat

# Git Bash
bash train_gpu.sh
```

## 🔍 Verify GPU Works

```bash
.venv/Scripts/python.exe -c "import torch; print(torch.cuda.is_available())"
# Should print: True
```

## 📊 Monitor GPU

```bash
nvidia-smi
```

## ⚠️ Critical: Don't Use `uv run`

```bash
# ❌ WRONG - reverts to CPU
uv run python -m src.train

# ✓ CORRECT - uses GPU
.venv/Scripts/python.exe -m src.train
```

Or use the helper scripts above!

## 🎯 RTX 3050 Recommended Settings

**Edit `configs/train.yaml`:**

```yaml
model:
  d_model: 256
  num_layers: 6
  d_ff: 768  # Reduced for 4GB VRAM

training:
  batch_size: 16  # Reduce to 8 if OOM
```

## 💾 Memory Issues?

Reduce batch size:
```yaml
training:
  batch_size: 8  # or 4
```

## 📖 Full Documentation

See `GPU_SETUP.md` for complete guide.

---

**Status**: ✅ GPU Ready!
