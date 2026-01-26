"""
Utility functions for interpretability analysis.
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np


class SimpleTokenizer:
    """Simple character-level tokenizer for mathematical expressions."""

    def __init__(self):
        self.chars = list("0123456789+-*/()=ABCDEFabcdef ,")
        self.special_tokens = ["<PAD>", "<BOS>", "<EOS>", "<UNK>"]
        self.vocab = self.special_tokens + self.chars
        self.char_to_idx = {ch: idx for idx, ch in enumerate(self.vocab)}
        self.idx_to_char = {idx: ch for idx, ch in enumerate(self.vocab)}
        self.pad_token_id = self.char_to_idx["<PAD>"]
        self.bos_token_id = self.char_to_idx["<BOS>"]
        self.eos_token_id = self.char_to_idx["<EOS>"]
        self.unk_token_id = self.char_to_idx["<UNK>"]

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        tokens = []
        if add_special_tokens:
            tokens.append(self.bos_token_id)
        for ch in text:
            tokens.append(self.char_to_idx.get(ch, self.unk_token_id))
        if add_special_tokens:
            tokens.append(self.eos_token_id)
        return tokens

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        text = []
        for idx in token_ids:
            if skip_special_tokens and idx in [self.pad_token_id, self.bos_token_id, self.eos_token_id]:
                continue
            text.append(self.idx_to_char.get(idx, "<UNK>"))
        return "".join(text)


def load_model_and_tokenizer(checkpoint_path: Path, device: str = "cpu"):
    """Load model and tokenizer from checkpoint."""
    from src.model import create_model

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_config = checkpoint.get("model_config", {})

    # Initialize tokenizer
    tokenizer = SimpleTokenizer()
    model_config["vocab_size"] = tokenizer.vocab_size

    # Create and load model
    model = create_model(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model, tokenizer, checkpoint


def prepare_example(text: str, tokenizer: SimpleTokenizer, device: str = "cpu") -> torch.Tensor:
    """Tokenize and prepare a single example."""
    tokens = tokenizer.encode(text, add_special_tokens=True)
    return torch.tensor([tokens], dtype=torch.long, device=device)


class ActivationCache:
    """Cache for storing intermediate activations during forward pass."""

    def __init__(self):
        self.activations = {}
        self.attention_maps = {}
        self.hooks = []

    def clear(self):
        """Clear all cached activations."""
        self.activations.clear()
        self.attention_maps.clear()

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


def register_activation_hooks(model: nn.Module, cache: ActivationCache):
    """Register hooks to capture activations at each layer."""

    def make_hook(name: str):
        def hook(module, input, output):
            if isinstance(output, tuple):
                cache.activations[name] = output[0].detach()
            else:
                cache.activations[name] = output.detach()
        return hook

    # Hook transformer blocks
    for i, block in enumerate(model.blocks):
        hook = block.register_forward_hook(make_hook(f"block_{i}"))
        cache.hooks.append(hook)

        # Hook attention within each block
        attn_hook = block.attn.register_forward_hook(make_hook(f"attn_{i}"))
        cache.hooks.append(attn_hook)

        # Hook feedforward within each block
        ff_hook = block.ff.register_forward_hook(make_hook(f"ff_{i}"))
        cache.hooks.append(ff_hook)

    return cache


def register_attention_hooks(model: nn.Module, cache: ActivationCache):
    """Register hooks to capture attention patterns."""

    def make_attention_hook(layer_idx: int):
        def hook(module, input, output):
            # Capture attention weights before dropout
            x = input[0]
            batch_size, seq_len, _ = x.shape

            # Recompute attention scores
            qkv = module.qkv_proj(x)
            from einops import rearrange
            q, k, v = rearrange(
                qkv,
                'b s (three h d) -> three b s h d',
                three=3,
                h=module.num_heads
            )

            # Apply RoPE if used
            if module.use_rope:
                q, k = module.rope(q, k)

            # Compute attention scores
            q = q * module.scale
            attn_scores = torch.einsum('bshd,bthd->bhst', q, k)

            # Apply causal mask
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
                diagonal=1
            )
            attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))

            # Softmax to get attention weights
            attn_weights = torch.softmax(attn_scores, dim=-1)

            cache.attention_maps[f"layer_{layer_idx}"] = attn_weights.detach()

        return hook

    # Register attention hooks for each layer
    for i, block in enumerate(model.blocks):
        hook = block.attn.register_forward_hook(make_attention_hook(i))
        cache.hooks.append(hook)

    return cache


def extract_activations(
    model: nn.Module,
    input_ids: torch.Tensor,
    layer_names: Optional[List[str]] = None
) -> Dict[str, torch.Tensor]:
    """Extract activations from specified layers."""
    cache = ActivationCache()
    register_activation_hooks(model, cache)

    with torch.no_grad():
        model(input_ids)

    activations = cache.activations.copy()
    cache.remove_hooks()

    if layer_names:
        activations = {k: v for k, v in activations.items() if k in layer_names}

    return activations


def get_attention_patterns(
    model: nn.Module,
    input_ids: torch.Tensor
) -> Dict[str, torch.Tensor]:
    """Extract attention patterns from all layers."""
    cache = ActivationCache()
    register_attention_hooks(model, cache)

    with torch.no_grad():
        model(input_ids)

    attention_maps = cache.attention_maps.copy()
    cache.remove_hooks()

    return attention_maps


def compute_head_importance(
    model: nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor
) -> Dict[str, torch.Tensor]:
    """
    Compute importance score for each attention head based on gradient.
    Higher score = more important for the task.
    """
    model.train()  # Need gradients

    importance_scores = {}

    # Forward pass
    logits, loss = model(input_ids, labels=labels)

    # Backward pass
    loss.backward()

    # Compute importance for each head
    for i, block in enumerate(model.blocks):
        # Get gradients from attention output projection
        if block.attn.out_proj.weight.grad is not None:
            grad = block.attn.out_proj.weight.grad.abs().mean()
            importance_scores[f"layer_{i}"] = grad.item()

    model.zero_grad()
    model.eval()

    return importance_scores


def get_token_strings(input_ids: torch.Tensor, tokenizer: SimpleTokenizer) -> List[str]:
    """Convert token IDs to readable strings."""
    tokens = input_ids[0].cpu().tolist()
    return [tokenizer.idx_to_char.get(tok, "<UNK>") for tok in tokens]


def compute_kl_divergence(p: torch.Tensor, q: torch.Tensor) -> float:
    """Compute KL divergence between two probability distributions."""
    p = p + 1e-10  # Avoid log(0)
    q = q + 1e-10
    return (p * (p.log() - q.log())).sum().item()


def compute_js_divergence(p: torch.Tensor, q: torch.Tensor) -> float:
    """Compute Jensen-Shannon divergence between two probability distributions."""
    m = 0.5 * (p + q)
    return 0.5 * compute_kl_divergence(p, m) + 0.5 * compute_kl_divergence(q, m)


def find_top_k_neurons(
    activations: torch.Tensor,
    k: int = 10,
    reduction: str = "mean"
) -> Tuple[List[int], List[float]]:
    """
    Find the top-k most active neurons in a layer.

    Args:
        activations: Tensor of shape (batch, seq_len, hidden_dim)
        k: Number of top neurons to return
        reduction: How to aggregate across batch/sequence ("mean", "max", "sum")

    Returns:
        Tuple of (neuron_indices, activation_values)
    """
    if reduction == "mean":
        neuron_activations = activations.abs().mean(dim=(0, 1))
    elif reduction == "max":
        neuron_activations = activations.abs().flatten(0, 1).max(dim=0)[0]
    elif reduction == "sum":
        neuron_activations = activations.abs().sum(dim=(0, 1))
    else:
        raise ValueError(f"Unknown reduction: {reduction}")

    top_k_values, top_k_indices = torch.topk(neuron_activations, k)

    return top_k_indices.cpu().tolist(), top_k_values.cpu().tolist()


def normalize_attention(attn_weights: torch.Tensor) -> torch.Tensor:
    """Normalize attention weights to [0, 1] for visualization."""
    min_val = attn_weights.min()
    max_val = attn_weights.max()
    if max_val - min_val < 1e-8:
        return torch.zeros_like(attn_weights)
    return (attn_weights - min_val) / (max_val - min_val)
