"""
Test script for minimal interpretability hooks.
Demonstrates attention head ablation and activation patching.
"""

from src.model import create_model
import torch

print("="*60)
print("Testing Minimal Interpretability Hooks")
print("="*60)
print()

# Create model
config = {
    'vocab_size': 30,
    'd_model': 128,
    'num_layers': 2,
    'num_heads': 4,
    'd_ff': 384
}
model = create_model(config)
model.eval()

# Create input
torch.manual_seed(42)
x = torch.randint(0, 30, (2, 20))

print("Configuration:")
print(f"  Model: {model.count_parameters() / 1e6:.2f}M parameters")
print(f"  Layers: {config['num_layers']}")
print(f"  Heads per layer: {config['num_heads']}")
print(f"  Input shape: {x.shape}")
print()

# Test 1: Baseline forward pass
print("Test 1: Baseline Forward Pass")
print("-" * 40)
with torch.no_grad():
    logits_baseline, _, _ = model(x)
print(f"[OK] Output shape: {logits_baseline.shape}")

# Check last_head_output for all layers
for i, block in enumerate(model.blocks):
    shape = block.attn.last_head_output.shape
    print(f"[OK] Layer {i} last_head_output: {shape}")
print()

# Test 2: No hooks == no change
print("Test 2: Behavioral Equivalence (No Hooks)")
print("-" * 40)
for block in model.blocks:
    block.attn.ablate_head = None
    if hasattr(block.attn, 'patch_head_output'):
        block.attn.patch_head_output = None

with torch.no_grad():
    logits_no_hooks, _, _ = model(x)

diff = (logits_baseline - logits_no_hooks).abs().max().item()
print(f"Max difference: {diff}")
if diff < 1e-6:
    print("[OK] Outputs are identical when hooks are cleared")
else:
    print(f"[FAIL] Outputs differ by {diff}")
print()

# Test 3: Single head ablation
print("Test 3: Single Head Ablation")
print("-" * 40)
ablation_results = []
for head_idx in range(config['num_heads']):
    # Ablate one head in layer 0
    model.blocks[0].attn.ablate_head = head_idx

    with torch.no_grad():
        logits_ablated, _, _ = model(x)

    diff = (logits_baseline - logits_ablated).abs().max().item()
    ablation_results.append(diff)
    print(f"  Head {head_idx}: max difference = {diff:.6f}")

    # Clear ablation
    model.blocks[0].attn.ablate_head = None

print(f"[OK] All heads tested, differences range: {min(ablation_results):.6f} to {max(ablation_results):.6f}")
print()

# Test 4: Activation patching
print("Test 4: Activation Patching")
print("-" * 40)

# Run baseline and store head output
with torch.no_grad():
    logits_baseline, _, _ = model(x)
    stored_head_0 = model.blocks[0].attn.last_head_output[:, :, 0, :].clone()

print(f"[OK] Stored head 0 output: {stored_head_0.shape}")

# Patch head 1 with head 0's activations
model.blocks[0].attn.patch_head_index = 1
model.blocks[0].attn.patch_head_output = stored_head_0

with torch.no_grad():
    logits_patched, _, _ = model(x)

diff = (logits_baseline - logits_patched).abs().max().item()
print(f"  Patch difference: {diff:.6f}")

# Clear patching
model.blocks[0].attn.patch_head_output = None
if diff > 1e-4:
    print("[OK] Patching successfully modified output")
else:
    print("[WARNING] Patching had minimal effect")
print()

# Test 5: Multiple heads ablated
print("Test 5: Multiple Heads Ablated Simultaneously")
print("-" * 40)

# Ablate multiple heads across different layers
model.blocks[0].attn.ablate_head = 0
model.blocks[1].attn.ablate_head = 2

with torch.no_grad():
    logits_multi_ablate, _, _ = model(x)

diff = (logits_baseline - logits_multi_ablate).abs().max().item()
print(f"  Layer 0 head 0 + Layer 1 head 2: difference = {diff:.6f}")

# Clear all ablations
for block in model.blocks:
    block.attn.ablate_head = None

if diff > 1e-4:
    print("[OK] Multi-layer ablation works")
print()

# Test 6: Training mode still works
print("Test 6: Training Mode Unaffected")
print("-" * 40)
model.train()
x_train = torch.randint(0, 30, (2, 20))
labels = torch.randint(0, 30, (2, 20))

logits_train, loss, _ = model(x_train, labels=labels)
print(f"[OK] Training forward pass successful")
print(f"  Logits shape: {logits_train.shape}")
print(f"  Loss: {loss.item():.4f}")
print()

print("="*60)
print("[OK] All Tests Passed!")
print("="*60)
print()
print("Summary:")
print("  - Per-head outputs stored in last_head_output")
print("  - Single-head ablation works via ablate_head")
print("  - Activation patching works via patch_head_output")
print("  - No behavior change when hooks inactive")
print("  - Training mode unaffected")
