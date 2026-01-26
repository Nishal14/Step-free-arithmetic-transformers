#!/usr/bin/env python3
"""
Test script to verify installation and basic functionality.
Run: uv run -- python test_installation.py
"""

import sys
from pathlib import Path

def test_imports():
    """Test that all required packages can be imported."""
    print("Testing imports...")
    try:
        import torch
        import transformers
        import datasets
        import accelerate
        import einops
        import omegaconf
        import numpy
        import tqdm
        import yaml
        print("[OK] All required packages imported successfully")
        print(f"  - PyTorch version: {torch.__version__}")
        print(f"  - Transformers version: {transformers.__version__}")
        return True
    except ImportError as e:
        print(f"[FAIL] Import error: {e}")
        return False


def test_model_creation():
    """Test that model can be created."""
    print("\nTesting model creation...")
    try:
        from src.model import create_model

        config = {
            "vocab_size": 100,
            "d_model": 64,
            "num_layers": 2,
            "num_heads": 4,
            "d_ff": 256,
            "max_seq_len": 128
        }

        model = create_model(config)
        num_params = model.count_parameters()
        print(f"[OK] Model created successfully")
        print(f"  - Parameters: {num_params:,} ({num_params/1e6:.2f}M)")
        return True
    except Exception as e:
        print(f"[FAIL] Model creation error: {e}")
        return False


def test_tokenizer():
    """Test tokenizer functionality."""
    print("\nTesting tokenizer...")
    try:
        from src.eval import SimpleTokenizer

        tokenizer = SimpleTokenizer()
        text = "123 + 456 = 579"
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)

        print(f"[OK] Tokenizer working")
        print(f"  - Vocabulary size: {tokenizer.vocab_size}")
        print(f"  - Test encoding: '{text}' -> {len(tokens)} tokens")
        print(f"  - Test decoding: {tokens[:5]}... -> '{decoded[:10]}...'")
        return True
    except Exception as e:
        print(f"[FAIL] Tokenizer error: {e}")
        return False


def test_data_generator():
    """Test data generation functionality."""
    print("\nTesting data generator...")
    try:
        from src.data.generate import AdditionGenerator

        generator = AdditionGenerator(max_length=10, seed=42)
        sample = generator.generate(with_steps=True)

        print(f"[OK] Data generator working")
        print(f"  - Sample input: {sample['input']}")
        print(f"  - Sample output: {sample['gold']}")
        print(f"  - Number of steps: {len(sample.get('steps', []))}")
        return True
    except Exception as e:
        print(f"[FAIL] Data generator error: {e}")
        return False


def test_config_loading():
    """Test configuration loading."""
    print("\nTesting configuration loading...")
    try:
        import yaml
        from pathlib import Path

        config_path = Path("configs/train.yaml")
        if not config_path.exists():
            print(f"[FAIL] Config file not found: {config_path}")
            return False

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        print(f"[OK] Configuration loaded successfully")
        print(f"  - Model layers: {config['model']['num_layers']}")
        print(f"  - Model dimension: {config['model']['d_model']}")
        print(f"  - Batch size: {config['training']['batch_size']}")
        return True
    except Exception as e:
        print(f"[FAIL] Config loading error: {e}")
        return False


def test_cuda_availability():
    """Test CUDA availability."""
    print("\nTesting CUDA availability...")
    try:
        import torch

        cuda_available = torch.cuda.is_available()
        if cuda_available:
            print(f"[OK] CUDA is available")
            print(f"  - Device count: {torch.cuda.device_count()}")
            print(f"  - Device name: {torch.cuda.get_device_name(0)}")
        else:
            print(f"[WARN] CUDA is not available (CPU only)")
            print(f"  - You can still train on CPU, but it will be slower")
        return True
    except Exception as e:
        print(f"[FAIL] CUDA test error: {e}")
        return False


def main():
    """Run all tests."""
    print("="*60)
    print("Math-Compact Installation Test")
    print("="*60)
    print()

    tests = [
        ("Package Imports", test_imports),
        ("Model Creation", test_model_creation),
        ("Tokenizer", test_tokenizer),
        ("Data Generator", test_data_generator),
        ("Config Loading", test_config_loading),
        ("CUDA", test_cuda_availability),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"[FAIL] Unexpected error in {name}: {e}")
            results.append((name, False))

    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "[OK] PASS" if result else "[FAIL] FAIL"
        print(f"{status}: {name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\nSUCCESS: All tests passed! Installation is working correctly.")
        print("\nYou can now:")
        print("  1. Generate datasets: uv run -- python -m src.data.generate --help")
        print("  2. Train models: uv run -- python -m src.train --help")
        print("  3. Evaluate models: uv run -- python -m src.eval --help")
        print("\nOr run the quickstart script:")
        print("  - Windows: quickstart.bat")
        print("  - Linux/Mac: bash quickstart.sh")
    else:
        print("\n[WARN] Some tests failed. Please check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
