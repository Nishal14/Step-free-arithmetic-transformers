# Why `uv run` Doesn't Work with CUDA PyTorch

## The Problem

```bash
# This FAILS - shows CUDA unavailable
uv run verify_10m_config.py

# This WORKS - shows CUDA available
source .venv/Scripts/activate
python verify_10m_config.py
```

## Root Cause

**`uv run` synchronizes the environment with `pyproject.toml` dependencies before running.**

### What Happens:

1. You manually install CUDA PyTorch:
   ```bash
   uv pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
   # Result: torch==2.5.1+cu121 (CUDA-enabled)
   ```

2. You run `uv run script.py`:
   ```bash
   uv run verify_10m_config.py
   ```

3. **uv syncs environment:**
   - Reads `pyproject.toml`: `"torch>=2.0.0"` (generic, no CUDA)
   - Resolves from default PyPI (not CUDA index)
   - Reinstalls `torch==2.10.0` (CPU-only)
   - **Your CUDA PyTorch is replaced with CPU version!**

4. Script runs with CPU PyTorch:
   ```python
   torch.cuda.is_available()  # False
   ```

## Why This Happens

PyTorch CUDA builds require a **custom package index**:
- CUDA index: `https://download.pytorch.org/whl/cu121`
- Default PyPI: `https://pypi.org/simple`

Standard dependency managers (pip, uv, poetry) default to PyPI, which only has CPU PyTorch.

## Solutions

### Solution 1: Use Direct venv Activation (RECOMMENDED)

**This is what we're using for training:**

```bash
# Activate environment directly
source .venv/Scripts/activate  # Linux/Mac
.venv\Scripts\activate.bat     # Windows

# Run scripts normally
python verify_10m_config.py
python -m src.train --config configs/scale_10m.yaml ...
```

**Why this works:**
- Bypasses uv's sync mechanism
- Uses manually installed CUDA PyTorch
- No environment modification

### Solution 2: Lock CUDA PyTorch in uv.lock (Advanced)

**Not recommended** - uv's lockfile management is complex and may not preserve custom index builds.

### Solution 3: Use uv pip directly (for testing only)

```bash
# Don't use: uv run python script.py
# Instead use:
uv pip list | grep torch  # Check current installation
source .venv/Scripts/activate
python script.py
```

## Best Practice for This Project

### For GPU Training (ALWAYS use direct activation):

```bash
# ✓ CORRECT
source .venv/Scripts/activate
python -m src.train --config configs/scale_10m.yaml --device cuda --fp16 --gpu-only

# ✗ WRONG - will revert to CPU PyTorch
uv run python -m src.train --config configs/scale_10m.yaml --device cuda --fp16 --gpu-only
```

### For Non-GPU Scripts (either works):

```bash
# Both OK for CPU-only scripts
uv run python some_analysis.py
# OR
source .venv/Scripts/activate && python some_analysis.py
```

## Verify Current Installation

```bash
# Method 1: Check via activated venv
source .venv/Scripts/activate
python -c "import torch; print(f'Version: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

# Expected output:
# Version: 2.5.1+cu121
# CUDA: True
```

```bash
# Method 2: Check installed packages
uv pip list | grep torch

# Expected output:
# torch                  2.5.1+cu121
# torchaudio             2.5.1+cu121
# torchvision            0.20.1+cu121
```

**If you see `torch 2.10.0` (no +cu121), CUDA PyTorch was replaced!**

## Reinstall CUDA PyTorch if Needed

```bash
uv pip install --reinstall --no-deps \
    torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121
```

## Training Scripts Updated

All training scripts now use **direct venv activation**:

- `train_10m_scale.bat` - Uses `.venv\Scripts\activate.bat`
- `train_10m_scale.sh` - Uses `source .venv/bin/activate`

**Do NOT modify these to use `uv run`!**

## Summary

| Command | CUDA Status | Use Case |
|---------|-------------|----------|
| `uv run script.py` | ❌ Reverts to CPU | Don't use for GPU |
| `source .venv/Scripts/activate && python script.py` | ✓ Keeps CUDA | **Use for GPU training** |
| `uv pip install torch ... --index-url` | ✓ Installs CUDA | Initial setup |

---

**TL;DR:** Always use direct venv activation (`source .venv/Scripts/activate`) for GPU training. Never use `uv run` with GPU workloads.
