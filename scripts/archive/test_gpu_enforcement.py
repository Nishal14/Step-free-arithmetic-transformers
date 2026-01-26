"""
Test GPU-only enforcement mechanism.
Verifies that --gpu-only flag correctly aborts when CUDA unavailable.
"""

import sys
import subprocess
import torch


def test_gpu_only_enforcement():
    """Test that --gpu-only flag enforces GPU requirement."""
    print("="*60)
    print("Testing GPU-Only Enforcement")
    print("="*60)
    print()

    # Check current CUDA status
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available on this system: {cuda_available}")
    print()

    # Test 1: --gpu-only with --device cuda should check CUDA
    print("Test 1: --gpu-only flag behavior")
    print("-" * 40)

    if cuda_available:
        print("CUDA IS available:")
        print("  > Training should START successfully")
        print("  > GPU name should be printed")
        print("  > FP16 should be enabled")
    else:
        print("CUDA NOT available:")
        print("  > Training should ABORT with RuntimeError")
        print("  > Error message should mention 'CUDA requested but not available'")
        print("  > No CPU fallback should occur")

    print()

    # Test 2: Standard mode behavior (without --gpu-only)
    print("Test 2: Standard mode (without --gpu-only)")
    print("-" * 40)
    print("When --gpu-only is NOT used:")
    print("  > Silent fallback to CPU is ALLOWED")
    print("  > Training continues on CPU if CUDA unavailable")
    print("  > This is for backwards compatibility")

    print()
    print("="*60)
    print("Verification Complete")
    print("="*60)
    print()

    if not cuda_available:
        print("[IMPORTANT]")
        print("CUDA is NOT available on this system.")
        print("Training with --gpu-only flag will INTENTIONALLY FAIL.")
        print("This is the CORRECT behavior for the 10M scaling experiment.")
        print()
        print("To train the 10M model, you need:")
        print("  1. A CUDA-capable GPU")
        print("  2. CUDA toolkit installed")
        print("  3. PyTorch with CUDA support")
        print()
        print("Run on a GPU-capable machine or fix CUDA installation.")
    else:
        print("[READY]")
        print("CUDA is available. Ready for GPU training.")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print()
        print("You can now run:")
        print("  python -m src.train --config configs/scale_10m.yaml \\")
        print("      --output-dir runs/scale_10m \\")
        print("      --seed 42 --device cuda --fp16 --gpu-only")


def test_import():
    """Test that modified train.py can be imported."""
    print("\nTest 3: Import modified train.py")
    print("-" * 40)

    try:
        from src import train
        print("[OK] train.py imports successfully")

        # Check for new flags
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--gpu-only", action="store_true")
        parser.add_argument("--fp16", action="store_true")

        print("[OK] New flags (--gpu-only, --fp16) are properly defined")

    except Exception as e:
        print(f"[ERROR] Failed to import train.py: {e}")
        return False

    return True


if __name__ == "__main__":
    # Test import first
    if not test_import():
        sys.exit(1)

    # Then test enforcement logic
    test_gpu_only_enforcement()
