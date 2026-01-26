# Model Architecture

This document describes the transformer architecture used in this study.

## Overview

We use a compact decoder-only transformer architecture optimized for mathematical reasoning tasks. The model follows modern design choices while remaining interpretable and efficient.

## Core Components

### 1. Token Embeddings

```python
self.token_embeddings = nn.Embedding(vocab_size, d_model)
```

- Simple embedding layer mapping tokens to d_model dimensions
- Vocabulary size: 35 tokens
  - Digits: 0-9
  - Operators: +, -, *, /
  - Parentheses: (, )
  - Special: `<BOS>`, `<EOS>`, `<PAD>`
  - Equals sign: =

### 2. Rotary Position Embeddings (RoPE)

Instead of absolute position embeddings, we use Rotary Position Embeddings (RoFormer, Su et al. 2021).

**Advantages:**
- Better extrapolation to unseen sequence lengths
- Relative position encoding via rotation
- No explicit position embedding parameters

**Implementation:**
```python
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=512):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
    def forward(self, seq_len):
        t = torch.arange(seq_len, device=self.inv_freq.device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb
```

Applied to queries and keys before attention:
```python
q = apply_rotary_pos_emb(q, cos, sin)
k = apply_rotary_pos_emb(k, cos, sin)
```

### 3. Transformer Blocks

Each block consists of:
1. Pre-LayerNorm
2. Multi-head self-attention
3. Residual connection
4. Pre-LayerNorm
5. Feed-forward network
6. Residual connection

**Pre-LayerNorm architecture** (instead of Post-LN):
```python
# Pre-LN (our choice)
x = x + attn(norm(x))
x = x + ffn(norm(x))

# vs Post-LN (original transformer)
x = norm(x + attn(x))
x = norm(x + ffn(x))
```

Pre-LN provides better training stability for smaller models.

### 4. Multi-Head Self-Attention

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
```

**Key features:**
- Causal masking for autoregressive generation
- Scaled dot-product attention
- Dropout for regularization
- Support for head-level ablation (for interpretability)

**Ablation support:**
```python
if hasattr(self, 'ablate_head') and self.ablate_head is not None:
    attn_output[:, :, self.ablate_head, :] = 0
```

This allows zeroing specific heads without retraining.

### 5. Feed-Forward Network

```python
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
```

- Two-layer MLP with GELU activation
- Expansion ratio: d_ff = 4 × d_model (standard)
- Dropout after activation

### 6. Output Layer

```python
self.output = nn.Linear(d_model, vocab_size, bias=False)
```

**Weight tying:** Output layer shares weights with input embeddings:
```python
if tie_weights:
    self.output.weight = self.token_embeddings.weight
```

Benefits:
- Reduces parameter count (~10%)
- Improves generalization (empirically validated)
- Common in language models

## Model Configurations

### Pilot Model (0.66M parameters)

```yaml
model:
  vocab_size: 35
  d_model: 128
  num_layers: 4
  num_heads: 4
  d_ff: 512       # 4 × 128
  max_seq_len: 32
  dropout: 0.1
  use_rope: true
  tie_weights: true
```

**Parameter breakdown:**
- Embeddings: 35 × 128 = 4,480
- 4 layers × (attention + FFN) ≈ 655,000
- Total: **0.66M parameters**

### 10M Scaling Model

```yaml
model:
  vocab_size: 35
  d_model: 384
  num_layers: 6
  num_heads: 6
  d_ff: 1536      # 4 × 384
  max_seq_len: 32
  dropout: 0.1
  use_rope: true
  tie_weights: true
```

**Parameter breakdown:**
- Embeddings: 35 × 384 = 13,440
- 6 layers × (attention + FFN) ≈ 10,635,000
- Total: **10.65M parameters**

## Design Choices

### Why RoPE?
- Generalizes better to longer sequences than absolute embeddings
- No learned position parameters
- Proven effective in modern LLMs (LLaMA, GPT-NeoX)

### Why Pre-LayerNorm?
- Better gradient flow in smaller models
- Reduces training instability
- Slight performance improvement over Post-LN in our experiments

### Why Weight Tying?
- Standard practice for small vocabularies
- Reduces parameters with minimal performance loss
- Improves sample efficiency

### Why GELU?
- Smooth activation (vs ReLU)
- Better gradient properties
- Standard in transformers since BERT

## Interpretability Features

### 1. Attention Head Outputs Stored
```python
self.last_head_output = attn_output  # [batch, seq, num_heads, head_dim]
```

Allows extraction of individual head outputs for probing.

### 2. Ablation Support
```python
self.ablate_head = None  # Set to head index to ablate
```

Dynamic ablation without model modification.

### 3. Clean Residual Stream
Pre-LayerNorm makes residual stream easier to interpret:
```
Layer 0 output = embeddings + attn(norm(embeddings))
Layer 1 output = Layer 0 output + attn(norm(Layer 0 output))
...
```

Each layer's contribution is additive.

## Training Details

### Optimization
- **Optimizer**: AdamW
- **Learning rate**: 1e-4 (pilot), 1e-4 (10M)
- **Weight decay**: 0.01
- **Warmup steps**: 500
- **Scheduler**: Linear warmup + cosine decay

### Regularization
- Dropout: 0.1 in attention and FFN
- Weight decay: 0.01
- Gradient clipping: max norm 1.0

### Mixed Precision (10M model)
- FP16 training with automatic mixed precision
- Gradient scaling to prevent underflow
- 2× memory reduction, ~1.5× speedup

```python
scaler = torch.amp.GradScaler('cuda')
with torch.amp.autocast('cuda'):
    loss = model(input_ids, labels)
scaler.scale(loss).backward()
```

## Computational Efficiency

| Model | Params | VRAM (FP16) | Speed | Time/Epoch |
|-------|--------|-------------|-------|------------|
| Pilot | 0.66M | ~1.5 GB | ~30 it/s | ~3 sec |
| 10M | 10.65M | ~3.5 GB | ~27 it/s | ~9 sec |

Hardware: NVIDIA RTX 3050 (4GB VRAM)

## Comparison to Standard Transformers

| Feature | Our Model | GPT-2 Small | LLaMA 7B |
|---------|-----------|-------------|----------|
| Position encoding | RoPE | Learned | RoPE |
| LayerNorm position | Pre-LN | Post-LN | Pre-LN |
| Activation | GELU | GELU | SwiGLU |
| Weight tying | Yes | Yes | No |
| Bias in linear | No | Yes | No |

Our architecture follows modern practices (RoPE, Pre-LN, no bias) while remaining small and interpretable.

## References

- **RoPE**: Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding", 2021
- **Pre-LayerNorm**: Xiong et al., "On Layer Normalization in the Transformer Architecture", 2020
- **Weight Tying**: Press & Wolf, "Using the Output Embedding to Improve Language Models", 2017
- **Attention**: Vaswani et al., "Attention Is All You Need", 2017
