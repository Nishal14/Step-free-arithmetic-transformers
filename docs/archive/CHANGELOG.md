# Changelog

All notable changes to the math-compact project will be documented in this file.

## [0.1.0] - 2025-01-23

### Initial Release

#### Added
- **Synthetic Dataset Generator** (`src/data/generate.py`)
  - Support for 4 mathematical tasks:
    - Addition with carry propagation
    - Multiplication with long multiplication
    - Base conversion (decimal to binary/hexadecimal)
    - Polynomial expansion (FOIL method)
  - Stepwise reasoning trace generation
  - Configurable dataset size and complexity
  - Train/dev/test split generation (80/10/10)
  - JSONL output format with metadata

- **Compact Transformer Model** (`src/model.py`)
  - Configurable architecture (5-50M parameters)
  - Rotary Position Embeddings (RoPE) for better position encoding
  - Pre-LayerNorm architecture for training stability
  - Multi-head attention with causal masking
  - Weight tying between input/output embeddings
  - Autoregressive text generation support

- **Training Pipeline** (`src/train.py`)
  - Character-level tokenization for math expressions
  - AdamW optimizer with linear warmup and decay
  - Gradient clipping for stability
  - Configurable training hyperparameters via YAML
  - Periodic checkpointing (latest, best, epoch-wise)
  - Weights & Biases integration for experiment tracking
  - Validation evaluation during training

- **Evaluation Script** (`src/eval.py`)
  - Perplexity computation
  - Exact match accuracy measurement
  - Sample prediction visualization
  - JSON output for metrics

- **Configuration System**
  - YAML-based configuration (`configs/train.yaml`)
  - OmegaConf integration for flexible config management
  - Separate sections for model, data, training, and logging

- **Development Tools**
  - Installation test script (`test_installation.py`)
  - Quickstart scripts for Windows and Unix (`quickstart.bat`, `quickstart.sh`)
  - Comprehensive README with usage examples
  - .gitignore for Python/ML projects

- **Dependency Management**
  - UV package manager integration
  - Locked dependencies for reproducibility (`uv.lock`)
  - Python 3.10+ support

#### Core Features
- **Reproducibility**: Deterministic training with seed control
- **Scalability**: Support for CPU and CUDA devices
- **Modularity**: Clean separation of data, model, training, and evaluation
- **Extensibility**: Easy to add new tasks and model variants

#### Dependencies
- torch>=2.0.0
- transformers>=4.35.0
- datasets>=2.10.0
- accelerate>=0.20.0
- einops>=0.6.0
- omegaconf>=2.2.0
- wandb>=0.15.0
- numpy>=1.24.0
- tqdm>=4.65.0
- pyyaml>=6.0

#### Known Limitations
- Character-level tokenization (no BPE/WordPiece yet)
- No distributed training support
- Limited to autoregressive generation
- CPU-only training can be slow for large models

#### Testing
- All core components tested and verified
- Installation validation script included
- Example workflow demonstrated in quickstart

---

## Future Roadmap

### Planned Features
- [ ] BPE/WordPiece tokenization for efficiency
- [ ] Distributed training support (DDP, FSDP)
- [ ] Knowledge distillation from larger models
- [ ] Auxiliary loss functions for better reasoning
- [ ] More mathematical tasks (calculus, linear algebra)
- [ ] Interactive evaluation mode
- [ ] Model interpretability tools
- [ ] Curriculum learning support
- [ ] Mixed precision training (FP16/BF16)
- [ ] Model quantization for deployment

### Improvements
- [ ] Batch generation for faster evaluation
- [ ] Data augmentation strategies
- [ ] Hyperparameter tuning utilities
- [ ] Experiment tracking dashboard
- [ ] Unit tests for all modules
- [ ] Continuous integration setup
- [ ] Docker container for easy deployment
- [ ] Benchmark suite for performance comparison

---

## Version History

- **v0.1.0** (2025-01-23) - Initial release with core functionality
