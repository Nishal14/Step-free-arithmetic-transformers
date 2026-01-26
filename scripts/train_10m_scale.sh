#!/bin/bash
#
# 10M Scaling Experiment Training Script
# GPU-ONLY with FP16 mixed precision
#
# This script will ABORT if CUDA is not available.
# Do NOT run on CPU.
#

set -e

echo "=========================================="
echo "10M Scaling Experiment"
echo "=========================================="
echo ""

# Activate virtual environment directly (NOT uv run - prevents CPU PyTorch sync)
source .venv/Scripts/activate

# Check CUDA availability
if ! python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'" 2>/dev/null; then
    echo "ERROR: CUDA not available. Aborting."
    echo "This experiment requires GPU."
    echo ""
    echo "Run: uv pip install --reinstall --no-deps torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
    exit 1
fi

echo "CUDA detected"
echo ""

# Print GPU info
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}'); print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')"
echo ""

# Verify config
echo "Verifying configuration..."
python verify_10m_config.py
echo ""

# Training command
echo "Starting training..."
echo ""

python -m src.train \
    --config configs/scale_10m.yaml \
    --output-dir runs/scale_10m \
    --seed 123 \
    --device cuda \
    --fp16 \
    --gpu-only

echo ""
echo "=========================================="
echo "Training complete!"
echo "=========================================="
