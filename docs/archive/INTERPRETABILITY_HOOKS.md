# Interpretability Hooks

Minimal hooks for mechanistic interpretability in `math-compact`.

## Overview

Three surgical additions to `MultiHeadAttention` enable:
1. **Capturing** per-head outputs
2. **Ablating** individual attention heads
3. **Patching** head activations

**Zero impact on training** when hooks are not used.

## Usage

### 1. Capture Per-Head Outputs

Every forward pass automatically stores per-head outputs:

```python
from src.model import create_model
import torch

model = create_model({'vocab_size': 30})
model.eval()

x = torch.randint(0, 30, (2, 20))
logits, _, _ = model(x)

# Access per-head outputs from any layer
head_outputs = model.blocks[0].attn.last_head_output
# Shape: (batch, seq_len, num_heads, head_dim)
```

### 2. Ablate Single Attention Head

Zero out a specific head's contribution:

```python
# Ablate head 0 in layer 0
model.blocks[0].attn.ablate_head = 0

logits_ablated, _, _ = model(x)

# Clear ablation
model.blocks[0].attn.ablate_head = None
```

### 3. Activation Patching

Replace one head's activations with stored values:

```python
# Run baseline and capture head 0 output
with torch.no_grad():
    logits_baseline, _, _ = model(x)
    stored_head_0 = model.blocks[0].attn.last_head_output[:, :, 0, :].clone()

# Patch head 1 with head 0's activations
model.blocks[0].attn.patch_head_index = 1
model.blocks[0].attn.patch_head_output = stored_head_0

logits_patched, _, _ = model(x)

# Clear patching
model.blocks[0].attn.patch_head_output = None
```

## Examples

### Find Most Important Head

```python
model.eval()
x = torch.randint(0, vocab_size, (1, seq_len))

# Baseline
with torch.no_grad():
    logits_base, _, _ = model(x)

# Test each head
for layer_idx, block in enumerate(model.blocks):
    for head_idx in range(model.num_heads):
        block.attn.ablate_head = head_idx

        with torch.no_grad():
            logits_ablated, _, _ = model(x)

        impact = (logits_base - logits_ablated).abs().mean().item()
        print(f"Layer {layer_idx}, Head {head_idx}: impact = {impact:.4f}")

        block.attn.ablate_head = None
```

### Activation Patching Across Prompts

```python
# Clean run
model.eval()
clean_input = torch.tensor([[1, 2, 3, 4, 5]])
with torch.no_grad():
    _, _, _ = model(clean_input)
    clean_activation = model.blocks[0].attn.last_head_output[:, :, 0, :].clone()

# Corrupted run with patched activation
corrupt_input = torch.tensor([[1, 2, 99, 4, 5]])  # Token 99 is corrupted
model.blocks[0].attn.patch_head_index = 0
model.blocks[0].attn.patch_head_output = clean_activation

with torch.no_grad():
    logits_patched, _, _ = model(corrupt_input)

# Does patching restore clean behavior?
```

### Multi-Head Ablation

```python
# Ablate multiple heads simultaneously
model.blocks[0].attn.ablate_head = 0
model.blocks[1].attn.ablate_head = 2
model.blocks[2].attn.ablate_head = 1

with torch.no_grad():
    logits, _, _ = model(x)

# Clear all
for block in model.blocks:
    block.attn.ablate_head = None
```

## Implementation Details

### What Was Changed

**File**: `src/model.py`
**Location**: `MultiHeadAttention.forward()`, after line 180

**Added 3 lines**:
1. Store: `self.last_head_output = out.detach()`
2. Ablation check: `if hasattr(self, "ablate_head") and self.ablate_head is not None: ...`
3. Patching check: `if hasattr(self, "patch_head_output") and self.patch_head_output is not None: ...`

### Tensor Shapes

- **`last_head_output`**: `(batch, seq_len, num_heads, head_dim)`
- **`ablate_head`**: integer (0 to num_heads-1) or `None`
- **`patch_head_output`**: `(batch, seq_len, head_dim)`
- **`patch_head_index`**: integer (0 to num_heads-1)

### Safety

✓ **No behavior change** when hooks inactive
✓ **No gradient issues** (uses `.detach()`)
✓ **Training unaffected** (hooks only inspect/modify activations)
✓ **No performance overhead** when not using hooks

## Testing

Run the test suite:

```bash
.venv/Scripts/python.exe test_interpretability_hooks.py
```

Expected output:
- Test 1: Baseline forward pass ✓
- Test 2: No hooks == no change ✓
- Test 3: Single head ablation ✓
- Test 4: Activation patching ✓
- Test 5: Multiple heads ablated ✓
- Test 6: Training mode unaffected ✓

## Limitations

1. **`last_head_output` is overwritten** each forward pass (store it explicitly if needed)
2. **Patching requires matching dimensions** (batch, seq_len, head_dim)
3. **No automatic cleanup** - manually set hooks to `None`
4. **Eval mode recommended** for interpretability (avoids dropout interference)

## Next Steps

Combine with existing interpretability tools:
- `return_attention_weights=True` - Get attention patterns
- `forward_analysis()` - Convenience wrapper
- `register_hook()` - Capture activations at any layer

See `example_interpretability.py` for complete examples.

---

**Status**: ✓ Implemented and tested
**Files modified**: `src/model.py` only
**Lines added**: 12 (including comments)
