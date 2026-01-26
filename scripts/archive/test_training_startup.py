"""
Test training startup without actually training.
Verifies GPU detection, model creation, and configuration parsing.
"""

import yaml
import torch
from pathlib import Path
from src.model import create_model

def test_startup():
    """Test that training startup works with GPU."""

    print("="*60)
    print("Training Startup Test")
    print("="*60)
    print()

    # Check CUDA
    print("1. Checking CUDA availability...")
    if not torch.cuda.is_available():
        print("[ERROR] CUDA not available!")
        return False

    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print("   [OK]")
    print()

    # Load config
    print("2. Loading configuration...")
    config_path = Path("configs/scale_10m.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print(f"   Model: {config['model']['name']}")
    print(f"   Batch size: {config['training']['batch_size']}")
    print(f"   Learning rate: {config['training']['optimizer']['lr']}")
    print(f"   Learning rate type: {type(config['training']['optimizer']['lr'])}")
    print("   [OK]")
    print()

    # Create model
    print("3. Creating model...")
    model_config = config["model"]
    model_config["vocab_size"] = 35
    model = create_model(model_config)
    print(f"   Parameters: {model.count_parameters() / 1e6:.2f}M")
    print("   [OK]")
    print()

    # Move to GPU
    print("4. Moving model to GPU...")
    device = torch.device("cuda")
    model = model.to(device)
    print("   [OK]")
    print()

    # Test FP16
    print("5. Testing FP16 autocast...")
    try:
        scaler = torch.amp.GradScaler('cuda')
        print("   [OK]")
    except Exception as e:
        print(f"   [ERROR] {e}")
        return False
    print()

    # Test forward pass
    print("6. Testing forward pass on GPU...")
    try:
        dummy_input = torch.randint(0, 35, (2, 16), device=device)
        with torch.cuda.amp.autocast():
            logits, _, _ = model(dummy_input)
        print(f"   Output shape: {logits.shape}")
        print("   [OK]")
    except Exception as e:
        print(f"   [ERROR] {e}")
        return False
    print()

    print("="*60)
    print("All checks passed! Ready for training.")
    print("="*60)
    return True


if __name__ == "__main__":
    success = test_startup()
    if not success:
        print("\n[FAILED] Fix errors before training")
        exit(1)
    else:
        print("\n[SUCCESS] System ready for GPU training")
