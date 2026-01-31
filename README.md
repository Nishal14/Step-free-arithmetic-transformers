# Step-Free Arithmetic Transformers

**Mechanistic interpretability study of how transformers learn arithmetic reasoning under final-answer-only supervision.**

## Overview

This repository contains the code and experimental setup for investigating how compact transformer models (0.66M–10M parameters) learn to process arithmetic expressions with parentheses when trained exclusively on final answers, without intermediate computational steps.

We study whether attention heads specialize for structural processing (parenthesis matching), investigate how depth information is encoded in internal representations, and test whether learned circuits generalize to out-of-distribution nesting depths.

## Experiments

This repository includes implementations of three core experiments:

1. **Attention Head Ablation** – Targeted ablations to test causal contributions of individual attention heads, measuring differential accuracy on flat vs. parenthesized expressions.

2. **Linear Probing** – Binary linear probes to test whether parenthesis depth is decodable from attention head outputs and residual stream activations.

3. **Out-of-Distribution Generalization** – Evaluation on expressions with nesting depth exceeding training distribution (depth=3 vs. training on depth≤2), with ablations to test whether specialized heads remain necessary.

## Requirements

- Python 3.10+
- CUDA-capable GPU (4GB+ VRAM)
- [uv](https://github.com/astral-sh/uv) package manager

## Installation

```bash
# Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh  # macOS/Linux
# OR: powershell -c "irm https://astral.sh/uv/install.ps1 | iex"  # Windows

# Clone repository
cd step-free-arithmetic-transformers

# Create environment and install dependencies
uv venv --python 3.10
uv sync

# Install CUDA-enabled PyTorch
uv pip install --reinstall --no-deps torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

# Activate environment
source .venv/Scripts/activate  # Git Bash
# OR: .venv\Scripts\activate.bat  # Windows CMD
```

**Note:** For GPU training, always activate the virtual environment directly. Avoid using `uv run`, as it may reinstall CPU-only PyTorch.

## Quick Start

### Generate Dataset

```bash
python src/data/generate_pilot_dataset.py \
    --num-train 1000 --num-val 100 --num-test 100 \
    --max-depth 2 --seed 42 --output-dir data
```

### Train Pilot Model (0.66M parameters)

```bash
python -m src.train \
    --config configs/pilot.yaml \
    --output-dir runs/pilot \
    --seed 42 --device cuda
```

Training time: ~2 minutes on RTX 3050

### Train 10M Model

```bash
python -m src.train \
    --config configs/scale_10m.yaml \
    --output-dir runs/scale_10m \
    --seed 123 --device cuda --fp16 --gpu-only
```

Training time: ~7.5 minutes on RTX 3050

### Run Experiments

```bash
# Attention head ablation
python eval/eval_pilot_ablation.py \
    --checkpoint runs/scale_10m/checkpoint_best.pt \
    --device cuda

# Linear probing
python probe/probe_depth.py \
    --checkpoint runs/pilot/checkpoint_best.pt \
    --layer 2 --head 2 --device cuda

# OOD generalization
python eval/eval_ood_depth3.py \
    --checkpoint runs/pilot/checkpoint_best.pt \
    --device cuda
```

## Repository Structure

```
step-free-arithmetic-transformers/
├── configs/           # Model and training configurations
├── src/               # Model architecture and training code
│   └── data/          # Dataset generation scripts
├── eval/              # Ablation and evaluation scripts
├── probe/             # Linear probing experiments
├── scripts/           # Training automation scripts
└── docs/              # Detailed documentation
```

## Documentation

- **[EXPERIMENTS.md](docs/EXPERIMENTS.md)** – Detailed methodology for ablation, probing, and OOD testing
- **[RESULTS.md](docs/RESULTS.md)** – Complete numerical results and performance metrics
- **[ARCHITECTURE.md](docs/ARCHITECTURE.md)** – Model architecture details (RoPE, Pre-LN, weight tying)
- **[REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md)** – Step-by-step reproduction instructions

## Summary of Findings

We find evidence that certain attention heads specialize for structural processing, with ablations causing larger drops on parenthesized expressions than on flat ones. Linear probing reveals that parenthesis depth information is decodable from attention head outputs and is strongly encoded in residual stream activations. The model generalizes to out-of-distribution nesting depths not seen during training, and ablating specialized heads degrades performance on these OOD examples, suggesting a causal role. Scaling from 0.66M to 10M parameters, we observe that specialization shifts to earlier layers while maintaining similar functional patterns.

For detailed numerical results and performance metrics, see **[docs/RESULTS.md](docs/RESULTS.md)**.

## Hardware

**Tested configuration:** NVIDIA GeForce RTX 3050 (4GB VRAM)
**Minimum requirements:** CUDA-capable GPU with 4GB+ VRAM

## License

MIT License - See LICENSE file for details.

## Citation

```bibtex
@software{step_free_arithmetic_2026,
  title={Step-Free Arithmetic Transformers: Mechanistic Analysis of Final-Answer-Only Learning},
  author={Nishal Thomas},
  year={2026},
  url={https://github.com/Nishal14/Step-free-arithmetic-transformers}
}
```

## Contact

For questions or issues, please open an issue on GitHub.
