# Reproducibility Guide

This document provides complete instructions for reproducing all experimental results.

## Random Seeds

All experiments use fixed random seeds for reproducibility:

- **Pilot model training**: seed 42
- **10M model training**: seed 123
- **Dataset generation**: seed 42

## Hardware Requirements

### Minimum Requirements
- GPU: NVIDIA GPU with 4GB VRAM (e.g., RTX 3050)
- CUDA: 12.1+ (bundled with PyTorch)
- RAM: 8GB system memory
- Storage: ~2GB for datasets and checkpoints

### Tested Configuration
- GPU: NVIDIA GeForce RTX 3050 (4GB VRAM)
- CUDA: 12.7 (driver), cu121 (PyTorch runtime)
- OS: Windows 11 / Git Bash
- Python: 3.10

## Environment Setup

### 1. Install uv Package Manager

```bash
# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Create Virtual Environment

```bash
cd step-free-arithmetic-transformers
uv venv --python 3.10
```

### 3. Install Dependencies

```bash
# Install all dependencies
uv sync

# CRITICAL: Install CUDA-enabled PyTorch
uv pip install --reinstall --no-deps torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121
```

### 4. Activate Environment

```bash
# Git Bash (Windows)
source .venv/Scripts/activate

# Windows CMD
.venv\Scripts\activate.bat
```

Always use direct activation. Do NOT use uv run for training/evaluation.

### 5. Verify CUDA

```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

Expected output: CUDA: True

## Dataset Generation

### Pilot Dataset

```bash
python src/data/generate_pilot_dataset.py \
    --num-train 1000 \
    --num-val 100 \
    --num-test 100 \
    --max-depth 2 \
    --seed 42 \
    --output-dir data
```

## Model Training

### Pilot Model

```bash
scripts/train_pilot.bat
```

Expected duration: ~2 minutes

### 10M Model

```bash
scripts/train_10m_scale.bat
```

Expected duration: ~7.5 minutes

## Experiments

### 1. Ablation

```bash
python eval/eval_pilot_ablation.py \
    --checkpoint runs/scale_10m/checkpoint_best.pt \
    --device cuda
```

### 2. Linear Probing

```bash
python probe/probe_depth.py \
    --checkpoint runs/pilot/checkpoint_best.pt \
    --layer 2 --head 2 --device cuda
```

### 3. OOD Generalization

```bash
python eval/eval_ood_depth3.py \
    --checkpoint runs/pilot/checkpoint_best.pt \
    --device cuda
```

## Common Issues

### CUDA Not Available

```bash
uv pip install --reinstall --no-deps torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121
```

### Out of Memory

Reduce batch size in config from 8 to 4.

### Module Not Found

Activate virtual environment:
```bash
source .venv/Scripts/activate
```

## Expected Results

**10M Model Ablation:**
- Baseline: Flat 96.0%, Paren 92.0%
- L0-H2: Flat 96.0%, Paren 89.0%
- L0-H3: Flat 96.0%, Paren 90.0%

**Pilot Model Probing:**
- L2-H2 output: 89.16% test accuracy
- L3 residual: 99.91% test accuracy

**OOD Generalization:**
- Baseline: 93.0%
- L2-H2 ablated: 91.0%

## Timing

Total reproduction time: ~20 minutes on RTX 3050
