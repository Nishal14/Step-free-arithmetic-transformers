"""
Verify that the scale_10m config produces ~10M parameters.
"""

import torch
import yaml
from pathlib import Path
from src.model import create_model


def main():
    # Load config
    config_path = Path("configs/scale_10m.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print("="*60)
    print("10M Scaling Config Verification")
    print("="*60)

    # Extract model config
    model_config = config["model"]
    print("\nModel Configuration:")
    for key, value in model_config.items():
        print(f"  {key}: {value}")

    # Create model
    print("\nCreating model...")
    model = create_model(model_config)

    # Count parameters
    total_params = model.count_parameters()
    print(f"\nTotal parameters: {total_params:,}")
    print(f"                  {total_params / 1e6:.2f}M")

    # Check if close to 10M
    target = 10e6
    diff = abs(total_params - target) / target * 100
    print(f"Target: 10M")
    print(f"Difference: {diff:.1f}%")

    if diff < 10:
        print("\n[OK] Configuration verified: ~10M parameters")
    else:
        print("\n[WARNING] Parameter count significantly differs from 10M")

    # Training config
    print("\n" + "="*60)
    print("Training Configuration:")
    print("="*60)
    train_config = config["training"]
    for key, value in train_config.items():
        if isinstance(value, dict):
            print(f"\n{key}:")
            for k, v in value.items():
                print(f"  {k}: {v}")
        else:
            print(f"{key}: {value}")

    # GPU requirements
    print("\n" + "="*60)
    print("GPU Requirements:")
    print("="*60)
    print("Required: CUDA-capable GPU")
    print("Expected VRAM: ~3-4 GB with batch_size=8, FP16")
    print("Recommended: GTX 1060 6GB or better")

    if torch.cuda.is_available():
        print(f"\nCUDA detected: {torch.cuda.get_device_name(0)}")
        print(f"VRAM available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print("[OK] Ready for GPU training")
    else:
        print("\n[ERROR] CUDA not available")
        print("Training will FAIL with --gpu-only flag")

    print("\n" + "="*60)


if __name__ == "__main__":
    main()
