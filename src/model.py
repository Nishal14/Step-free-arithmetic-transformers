"""
Compact Transformer implementation for mathematical reasoning.
A small, efficient transformer model (5-50M parameters).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Optional, Tuple, List


class RotaryPositionalEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) for improved position encoding."""

    def __init__(self, dim: int, max_seq_len: int = 2048, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Precompute frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Cache for rotary embeddings
        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None

    def _update_cache(self, seq_len: int, device: torch.device):
        """Update cached cos/sin values if needed."""
        if seq_len > self._seq_len_cached:
            self._seq_len_cached = seq_len
            t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self._cos_cached = emb.cos()[None, :, None, :]
            self._sin_cached = emb.sin()[None, :, None, :]

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary embeddings to queries and keys.

        Args:
            q: Query tensor of shape (batch, seq_len, num_heads, head_dim)
            k: Key tensor of shape (batch, seq_len, num_heads, head_dim)

        Returns:
            Tuple of (q_rotated, k_rotated) with same shapes as inputs
        """
        # Verify input shapes
        assert q.dim() == 4, f"Expected 4D query tensor, got shape {q.shape}"
        assert k.dim() == 4, f"Expected 4D key tensor, got shape {k.shape}"
        assert q.shape == k.shape, f"Query and key shapes must match: q={q.shape}, k={k.shape}"

        batch_size, seq_len, num_heads, head_dim = q.shape
        assert head_dim == self.dim, f"Head dim {head_dim} doesn't match RoPE dim {self.dim}"

        self._update_cache(seq_len, q.device)

        # Cached cos/sin have shape (1, seq_len, 1, head_dim) for broadcasting
        cos = self._cos_cached[:, :seq_len, :, :]
        sin = self._sin_cached[:, :seq_len, :, :]

        # Rotate queries and keys
        q_rot = (q * cos) + (self._rotate_half(q) * sin)
        k_rot = (k * cos) + (self._rotate_half(k) * sin)

        return q_rot, k_rot

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        """Rotate half the hidden dims of the input."""
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with optional RoPE."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        use_rope: bool = True,
        max_seq_len: int = 2048
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5

        # Q, K, V projections
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.use_rope = use_rope

        if use_rope:
            self.rope = RotaryPositionalEmbedding(self.head_dim, max_seq_len)

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        is_causal: bool = True,
        return_attention_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            padding_mask: Optional boolean mask of shape (batch, seq_len) where True means
                         the token should be attended to, False means it should be masked out (PAD)
            is_causal: Whether to apply causal masking
            return_attention_weights: If True, return attention weights along with output

        Returns:
            Tuple of (output, attention_weights) where attention_weights is None unless requested
        """
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        qkv = self.qkv_proj(x)
        q, k, v = rearrange(
            qkv,
            'b s (three h d) -> three b s h d',
            three=3,
            h=self.num_heads
        )

        # Apply RoPE if enabled
        if self.use_rope:
            q, k = self.rope(q, k)

        # Scaled dot-product attention
        q = q * self.scale
        attn_scores = torch.einsum('bshd,bthd->bhst', q, k)

        # Build combined attention mask
        # Start with all positions allowed
        combined_mask = None

        # Apply causal mask (prevents attending to future positions)
        if is_causal:
            # Shape: (seq_len, seq_len) where True means "mask out" (upper triangle)
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
                diagonal=1
            )
            combined_mask = causal_mask

        # Apply padding mask (prevents attending to PAD tokens)
        if padding_mask is not None:
            # padding_mask shape: (batch, seq_len) where True = attend, False = mask out
            # Expand to (batch, 1, 1, seq_len) for broadcasting across heads and query positions
            # We need to invert it: True -> False (don't mask), False -> True (mask)
            pad_mask = ~padding_mask.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, seq_len)

            if combined_mask is None:
                combined_mask = pad_mask
            else:
                # Broadcast causal mask to (batch, 1, seq_len, seq_len) and combine with padding
                combined_mask = combined_mask.unsqueeze(0) | pad_mask  # Both True means mask

        # Apply the combined mask to attention scores
        if combined_mask is not None:
            attn_scores = attn_scores.masked_fill(combined_mask, float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights_dropout = self.dropout(attn_weights)

        # Aggregate values
        out = torch.einsum('bhst,bthd->bshd', attn_weights_dropout, v)

        # Store per-head outputs for interpretability
        self.last_head_output = out.detach()

        # Apply single-head ablation if requested
        if hasattr(self, "ablate_head") and self.ablate_head is not None:
            out[:, :, self.ablate_head, :] = 0

        # Apply activation patching if requested
        if hasattr(self, "patch_head_output") and self.patch_head_output is not None:
            out[:, :, self.patch_head_index, :] = self.patch_head_output

        out = rearrange(out, 'b s h d -> b s (h d)')

        output = self.out_proj(out)

        # Return attention weights if requested (before dropout for interpretability)
        if return_attention_weights:
            return output, attn_weights
        else:
            return output, None


class FeedForward(nn.Module):
    """Position-wise feed-forward network with GELU activation."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class GatedFeedForward(nn.Module):
    """
    Gated feed-forward network (GEGLU or SwiGLU variant).
    Provides better mechanistic interpretability by separating gating from transformation.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1, activation: str = "gelu"):
        super().__init__()
        self.activation = activation

        # Gated FFN projects to 2x d_ff (one for gate, one for value)
        self.fc1_gate = nn.Linear(d_model, d_ff)
        self.fc1_value = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Gate and value branches
        gate = self.fc1_gate(x)
        value = self.fc1_value(x)

        # Apply activation to gate
        if self.activation == "gelu":
            gate = F.gelu(gate)
        elif self.activation == "silu":  # SwiGLU uses SiLU (Swish)
            gate = F.silu(gate)
        else:
            raise ValueError(f"Unknown activation: {self.activation}")

        # Element-wise gating
        x = gate * value
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class TransformerBlock(nn.Module):
    """Single transformer block with pre-LayerNorm."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        use_rope: bool = True,
        max_seq_len: int = 2048,
        ffn_type: str = "gelu"
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads, dropout, use_rope, max_seq_len)
        self.ln2 = nn.LayerNorm(d_model)

        # Select FFN type
        if ffn_type == "gelu":
            self.ff = FeedForward(d_model, d_ff, dropout)
        elif ffn_type == "geglu":
            self.ff = GatedFeedForward(d_model, d_ff, dropout, activation="gelu")
        elif ffn_type == "swiglu":
            self.ff = GatedFeedForward(d_model, d_ff, dropout, activation="silu")
        else:
            raise ValueError(f"Unknown ffn_type: {ffn_type}. Choose from: gelu, geglu, swiglu")

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        return_attention_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            padding_mask: Optional boolean mask of shape (batch, seq_len)
            return_attention_weights: If True, return attention weights

        Returns:
            Tuple of (output, attention_weights)
        """
        # Pre-LN: attention block
        attn_out, attn_weights = self.attn(self.ln1(x), padding_mask, return_attention_weights=return_attention_weights)
        x = x + self.dropout(attn_out)

        # Pre-LN: feed-forward block
        x = x + self.dropout(self.ff(self.ln2(x)))

        return x, attn_weights


class CompactTransformer(nn.Module):
    """
    Compact Transformer for mathematical reasoning tasks.
    Target parameter range: 5-50M parameters.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        d_ff: int = 1024,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        use_rope: bool = True,
        tie_weights: bool = True,
        ffn_type: str = "gelu"
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len

        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # Positional embeddings (only if not using RoPE)
        if not use_rope:
            self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        else:
            self.pos_embedding = None

        self.dropout = nn.Dropout(dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout, use_rope, max_seq_len, ffn_type)
            for _ in range(num_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)

        # Output projection
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Tie input and output embeddings
        if tie_weights:
            self.lm_head.weight = self.token_embedding.weight

        # Hook registry for capturing activations
        self._activation_hooks = {}

        self._init_weights()

    def _init_weights(self):
        """Initialize weights using scaled initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                torch.nn.init.ones_(module.weight)
                torch.nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_attention_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[torch.Tensor]]]:
        """
        Args:
            input_ids: Token IDs of shape (batch, seq_len)
            attention_mask: Optional boolean mask of shape (batch, seq_len) where True means
                           the token should be attended to, False means it's a PAD token
            labels: Optional labels for computing loss
            return_attention_weights: If True, return attention weights from all layers

        Returns:
            logits: Output logits of shape (batch, seq_len, vocab_size)
            loss: Optional loss if labels are provided
            attention_weights: Optional list of attention weight tensors (one per layer)
        """
        batch_size, seq_len = input_ids.shape

        # Token embeddings
        x = self.token_embedding(input_ids)

        # Add positional embeddings if not using RoPE
        if self.pos_embedding is not None:
            positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
            x = x + self.pos_embedding(positions)

        x = self.dropout(x)

        # Collect attention weights from all layers if requested
        all_attention_weights = [] if return_attention_weights else None

        # Apply transformer blocks (pass attention_mask as padding_mask)
        for block in self.blocks:
            x, attn_weights = block(x, padding_mask=attention_mask, return_attention_weights=return_attention_weights)
            if return_attention_weights and attn_weights is not None:
                all_attention_weights.append(attn_weights)

        x = self.ln_f(x)

        # Project to vocabulary
        logits = self.lm_head(x)

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                labels.view(-1),
                ignore_index=-100
            )

        return logits, loss, all_attention_weights

    def count_parameters(self) -> int:
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def generate_next_token(
        self,
        input_ids: torch.Tensor,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_logits: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Generate a single next token (for step-by-step analysis).

        Args:
            input_ids: Input token IDs of shape (batch, seq_len)
            temperature: Sampling temperature
            top_k: Top-k sampling
            attention_mask: Optional attention mask
            return_logits: If True, return logits along with sampled token

        Returns:
            Tuple of (next_token, logits) where logits is None unless requested
        """
        # Get logits for next token
        logits, _, _ = self.forward(input_ids, attention_mask=attention_mask)
        next_token_logits = logits[:, -1, :] / temperature

        # Apply top-k filtering
        if top_k is not None:
            indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
            next_token_logits[indices_to_remove] = float('-inf')

        # Sample next token
        probs = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        if return_logits:
            return next_token, next_token_logits
        else:
            return next_token, None

    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Generate text autoregressively.

        Args:
            input_ids: Input token IDs of shape (batch, seq_len)
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling
            eos_token_id: End-of-sequence token ID
            attention_mask: Optional attention mask (will be extended during generation)

        Returns:
            Generated token IDs of shape (batch, max_length)
        """
        self.eval()
        batch_size = input_ids.shape[0]

        with torch.no_grad():
            for _ in range(max_length - input_ids.shape[1]):
                # Generate next token
                next_token, _ = self.generate_next_token(
                    input_ids,
                    temperature=temperature,
                    top_k=top_k,
                    attention_mask=attention_mask
                )

                # Append to input
                input_ids = torch.cat([input_ids, next_token], dim=1)

                # Extend attention mask if provided
                if attention_mask is not None:
                    attention_mask = torch.cat([
                        attention_mask,
                        torch.ones((batch_size, 1), dtype=torch.bool, device=attention_mask.device)
                    ], dim=1)

                # Check for EOS
                if eos_token_id is not None and (next_token == eos_token_id).all():
                    break

        return input_ids

    def register_hook(self, name: str, module: nn.Module, hook_fn):
        """
        Register a forward hook on a module for capturing activations.

        Args:
            name: Identifier for this hook
            module: The module to attach the hook to
            hook_fn: Hook function with signature: hook_fn(module, input, output) -> None

        Example:
            def save_activation(module, input, output):
                activations[name] = output.detach()

            model.register_hook("layer_0_attn", model.blocks[0].attn, save_activation)
        """
        handle = module.register_forward_hook(hook_fn)
        self._activation_hooks[name] = handle

    def clear_hooks(self):
        """Remove all registered activation hooks."""
        for handle in self._activation_hooks.values():
            handle.remove()
        self._activation_hooks.clear()

    @torch.no_grad()
    def forward_analysis(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_attention_weights: bool = True
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """
        Convenience method for analysis: runs forward pass in eval mode with no gradients.

        Args:
            input_ids: Token IDs of shape (batch, seq_len)
            attention_mask: Optional attention mask
            return_attention_weights: Whether to return attention weights (default True for analysis)

        Returns:
            Tuple of (logits, attention_weights)
        """
        was_training = self.training
        self.eval()

        logits, _, attention_weights = self.forward(
            input_ids,
            attention_mask=attention_mask,
            return_attention_weights=return_attention_weights
        )

        if was_training:
            self.train()

        return logits, attention_weights


def create_model(config: dict) -> CompactTransformer:
    """Factory function to create model from config."""
    return CompactTransformer(
        vocab_size=config.get("vocab_size", 10000),
        d_model=config.get("d_model", 256),
        num_layers=config.get("num_layers", 6),
        num_heads=config.get("num_heads", 8),
        d_ff=config.get("d_ff", 1024),
        max_seq_len=config.get("max_seq_len", 512),
        dropout=config.get("dropout", 0.1),
        use_rope=config.get("use_rope", True),
        tie_weights=config.get("tie_weights", True),
        ffn_type=config.get("ffn_type", "gelu")
    )


if __name__ == "__main__":
    # Test model creation
    config = {
        "vocab_size": 10000,
        "d_model": 256,
        "num_layers": 6,
        "num_heads": 8,
        "d_ff": 1024,
        "max_seq_len": 512
    }

    model = create_model(config)
    print(f"Model parameters: {model.count_parameters() / 1e6:.2f}M")

    # Test forward pass
    batch_size = 2
    seq_len = 32
    input_ids = torch.randint(0, config["vocab_size"], (batch_size, seq_len))

    logits, _ = model(input_ids)
    print(f"Output shape: {logits.shape}")
