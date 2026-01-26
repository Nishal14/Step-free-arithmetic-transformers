"""
Intervention and Ablation Experiments.

Systematically ablate (remove/zero) specific heads, layers, or neurons
to measure their causal impact on model performance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from .utils import prepare_example, get_token_strings


class InterventionAnalyzer:
    """
    Perform systematic interventions to measure component importance.

    Types of interventions:
    - Zero ablation: Set activations to zero
    - Mean ablation: Replace with mean activation
    - Random ablation: Replace with random values
    - Head pruning: Remove attention heads
    """

    def __init__(self, model, tokenizer, device: str = "cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()

    def ablate_attention_head(
        self,
        input_ids: torch.Tensor,
        layer_idx: int,
        head_idx: int,
        ablation_type: str = "zero"
    ) -> torch.Tensor:
        """
        Ablate a specific attention head and return model output.

        Args:
            input_ids: Input token IDs
            layer_idx: Which layer
            head_idx: Which head to ablate
            ablation_type: "zero", "mean", or "random"

        Returns:
            Output logits with ablated head
        """
        hooks = []

        def make_ablation_hook(head_idx: int, ablation_type: str):
            def hook(module, input, output):
                # Output is after out_proj, shape: (batch, seq_len, d_model)
                # We need to ablate before out_proj

                # Get the attention module
                x = input[0]
                batch_size, seq_len, _ = x.shape

                # Recompute attention to intervene
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

                # Compute attention
                q = q * module.scale
                attn_scores = torch.einsum('bshd,bthd->bhst', q, k)

                # Apply causal mask
                causal_mask = torch.triu(
                    torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
                    diagonal=1
                )
                attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))

                attn_weights = F.softmax(attn_scores, dim=-1)
                attn_weights = module.dropout(attn_weights)

                # ABLATE THE SPECIFIC HEAD
                if ablation_type == "zero":
                    attn_weights[:, head_idx, :, :] = 0.0
                elif ablation_type == "mean":
                    # Replace with uniform attention
                    uniform_attn = torch.ones_like(attn_weights[:, head_idx, :, :]) / seq_len
                    attn_weights[:, head_idx, :, :] = uniform_attn
                elif ablation_type == "random":
                    random_attn = torch.rand_like(attn_weights[:, head_idx, :, :])
                    random_attn = F.softmax(random_attn, dim=-1)
                    attn_weights[:, head_idx, :, :] = random_attn

                # Aggregate values
                out = torch.einsum('bhst,bthd->bshd', attn_weights, v)
                out = rearrange(out, 'b s h d -> b s (h d)')

                # Apply output projection
                return module.out_proj(out)

            return hook

        # Register hook
        attn_module = self.model.blocks[layer_idx].attn
        hook = attn_module.register_forward_hook(make_ablation_hook(head_idx, ablation_type))
        hooks.append(hook)

        # Forward pass
        with torch.no_grad():
            logits, _ = self.model(input_ids)

        # Clean up
        for h in hooks:
            h.remove()

        return logits

    def ablate_layer(
        self,
        input_ids: torch.Tensor,
        layer_idx: int,
        component: str = "full",
        ablation_type: str = "zero"
    ) -> torch.Tensor:
        """
        Ablate an entire layer or component.

        Args:
            input_ids: Input token IDs
            layer_idx: Which layer to ablate
            component: "full", "attn", or "ff"
            ablation_type: "zero", "mean", or "identity"

        Returns:
            Output logits with ablated layer
        """
        hooks = []

        def make_layer_ablation_hook(ablation_type: str):
            def hook(module, input, output):
                if ablation_type == "zero":
                    if isinstance(output, tuple):
                        return (torch.zeros_like(output[0]),) + output[1:]
                    return torch.zeros_like(output)
                elif ablation_type == "identity":
                    # Pass through input unchanged
                    return input[0]
                elif ablation_type == "mean":
                    # Replace with mean across batch and sequence
                    if isinstance(output, tuple):
                        out = output[0]
                        mean_val = out.mean()
                        return (torch.full_like(out, mean_val),) + output[1:]
                    mean_val = output.mean()
                    return torch.full_like(output, mean_val)
                return output

            return hook

        # Select component to ablate
        if component == "full":
            target = self.model.blocks[layer_idx]
        elif component == "attn":
            target = self.model.blocks[layer_idx].attn
        elif component == "ff":
            target = self.model.blocks[layer_idx].ff
        else:
            raise ValueError(f"Unknown component: {component}")

        hook = target.register_forward_hook(make_layer_ablation_hook(ablation_type))
        hooks.append(hook)

        # Forward pass
        with torch.no_grad():
            logits, _ = self.model(input_ids)

        # Clean up
        for h in hooks:
            h.remove()

        return logits

    def measure_head_importance(
        self,
        examples: List[str],
        target_position: int = -1,
        ablation_type: str = "zero"
    ) -> Dict[str, float]:
        """
        Measure importance of each attention head by ablating it.

        Args:
            examples: List of input expressions
            target_position: Position to measure loss (-1 for all positions)
            ablation_type: Type of ablation

        Returns:
            Dict mapping "layer_X_head_Y" to importance score (delta in loss)
        """
        importance_scores = {}

        # Get baseline loss
        baseline_losses = []
        for text in examples:
            input_ids = prepare_example(text, self.tokenizer, self.device)

            # Create labels (shifted input_ids)
            labels = input_ids[:, 1:].clone()
            input_ids_for_loss = input_ids[:, :-1]

            with torch.no_grad():
                logits, loss = self.model(input_ids_for_loss, labels=labels)
                baseline_losses.append(loss.item())

        baseline_loss = np.mean(baseline_losses)

        # Test each head
        num_layers = len(self.model.blocks)

        for layer_idx in range(num_layers):
            num_heads = self.model.blocks[layer_idx].attn.num_heads

            for head_idx in range(num_heads):
                head_key = f"layer_{layer_idx}_head_{head_idx}"

                ablated_losses = []

                for text in examples:
                    input_ids = prepare_example(text, self.tokenizer, self.device)

                    labels = input_ids[:, 1:].clone()
                    input_ids_for_loss = input_ids[:, :-1]

                    # Ablate head
                    ablated_logits = self.ablate_attention_head(
                        input_ids_for_loss,
                        layer_idx,
                        head_idx,
                        ablation_type
                    )

                    # Compute loss
                    loss = F.cross_entropy(
                        ablated_logits.view(-1, ablated_logits.shape[-1]),
                        labels.view(-1),
                        ignore_index=-100
                    )
                    ablated_losses.append(loss.item())

                # Importance = increase in loss
                ablated_loss = np.mean(ablated_losses)
                importance = ablated_loss - baseline_loss

                importance_scores[head_key] = importance

                print(f"{head_key}: importance = {importance:.4f}")

        return importance_scores

    def measure_layer_importance(
        self,
        examples: List[str],
        component: str = "full",
        ablation_type: str = "zero"
    ) -> Dict[str, float]:
        """
        Measure importance of each layer by ablating it.

        Args:
            examples: List of input expressions
            component: "full", "attn", or "ff"
            ablation_type: Type of ablation

        Returns:
            Dict mapping layer names to importance scores
        """
        importance_scores = {}

        # Baseline
        baseline_losses = []
        for text in examples:
            input_ids = prepare_example(text, self.tokenizer, self.device)
            labels = input_ids[:, 1:].clone()
            input_ids_for_loss = input_ids[:, :-1]

            with torch.no_grad():
                _, loss = self.model(input_ids_for_loss, labels=labels)
                baseline_losses.append(loss.item())

        baseline_loss = np.mean(baseline_losses)

        # Test each layer
        num_layers = len(self.model.blocks)

        for layer_idx in range(num_layers):
            layer_key = f"layer_{layer_idx}_{component}"

            ablated_losses = []

            for text in examples:
                input_ids = prepare_example(text, self.tokenizer, self.device)
                labels = input_ids[:, 1:].clone()
                input_ids_for_loss = input_ids[:, :-1]

                # Ablate layer
                ablated_logits = self.ablate_layer(
                    input_ids_for_loss,
                    layer_idx,
                    component,
                    ablation_type
                )

                # Compute loss
                loss = F.cross_entropy(
                    ablated_logits.view(-1, ablated_logits.shape[-1]),
                    labels.view(-1),
                    ignore_index=-100
                )
                ablated_losses.append(loss.item())

            ablated_loss = np.mean(ablated_losses)
            importance = ablated_loss - baseline_loss

            importance_scores[layer_key] = importance

            print(f"{layer_key}: importance = {importance:.4f}")

        return importance_scores

    def visualize_head_importance(
        self,
        importance_scores: Dict[str, float],
        save_path: Optional[Path] = None
    ):
        """
        Visualize attention head importance as a heatmap.

        Args:
            importance_scores: Dict from measure_head_importance
            save_path: Optional save path
        """
        # Parse layer and head indices
        data = []
        for key, score in importance_scores.items():
            parts = key.split("_")
            layer_idx = int(parts[1])
            head_idx = int(parts[3])
            data.append((layer_idx, head_idx, score))

        # Create matrix
        num_layers = max(d[0] for d in data) + 1
        num_heads = max(d[1] for d in data) + 1

        matrix = np.zeros((num_layers, num_heads))
        for layer_idx, head_idx, score in data:
            matrix[layer_idx, head_idx] = score

        # Plot
        fig, ax = plt.subplots(figsize=(12, 8))

        im = sns.heatmap(
            matrix,
            cmap="YlOrRd",
            annot=True,
            fmt=".3f",
            cbar_kws={'label': 'Importance (Δ Loss)'},
            ax=ax
        )

        ax.set_xlabel('Head Index')
        ax.set_ylabel('Layer Index')
        ax.set_title('Attention Head Importance (Ablation Impact)')

        plt.tight_layout()

        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved head importance visualization to {save_path}")

        return fig

    def visualize_layer_importance(
        self,
        importance_scores: Dict[str, float],
        save_path: Optional[Path] = None
    ):
        """
        Visualize layer importance as a bar chart.

        Args:
            importance_scores: Dict from measure_layer_importance
            save_path: Optional save path
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        layers = sorted(importance_scores.keys())
        scores = [importance_scores[layer] for layer in layers]

        colors = ['red' if s > 0 else 'blue' for s in scores]

        ax.bar(range(len(layers)), scores, color=colors, alpha=0.7)
        ax.set_xticks(range(len(layers)))
        ax.set_xticklabels(layers, rotation=45, ha='right')
        ax.set_ylabel('Importance (Δ Loss)')
        ax.set_xlabel('Layer')
        ax.set_title('Layer Importance via Ablation')
        ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved layer importance visualization to {save_path}")

        return fig

    def cumulative_ablation_experiment(
        self,
        examples: List[str],
        sorted_heads: List[Tuple[str, float]],
        max_ablations: int = 10
    ) -> Dict[int, float]:
        """
        Cumulatively ablate heads in order of importance.

        Tests if removing multiple heads compounds their effects.

        Args:
            examples: List of input expressions
            sorted_heads: List of (head_key, importance) sorted by importance
            max_ablations: Maximum number of heads to ablate

        Returns:
            Dict mapping num_ablated to average loss
        """
        # Get baseline loss
        baseline_losses = []
        for text in examples:
            input_ids = prepare_example(text, self.tokenizer, self.device)
            labels = input_ids[:, 1:].clone()
            input_ids_for_loss = input_ids[:, :-1]

            with torch.no_grad():
                _, loss = self.model(input_ids_for_loss, labels=labels)
                baseline_losses.append(loss.item())

        results = {0: np.mean(baseline_losses)}

        # Cumulatively ablate heads
        heads_to_ablate = []

        for i in range(min(max_ablations, len(sorted_heads))):
            head_key = sorted_heads[i][0]
            heads_to_ablate.append(head_key)

            # Measure loss with all heads ablated
            ablated_losses = []

            for text in examples:
                input_ids = prepare_example(text, self.tokenizer, self.device)
                labels = input_ids[:, 1:].clone()
                input_ids_for_loss = input_ids[:, :-1]

                # Ablate multiple heads
                logits = self._ablate_multiple_heads(input_ids_for_loss, heads_to_ablate)

                loss = F.cross_entropy(
                    logits.view(-1, logits.shape[-1]),
                    labels.view(-1),
                    ignore_index=-100
                )
                ablated_losses.append(loss.item())

            results[i + 1] = np.mean(ablated_losses)

            print(f"Ablated {i+1} heads: loss = {results[i+1]:.4f}")

        return results

    def _ablate_multiple_heads(
        self,
        input_ids: torch.Tensor,
        head_keys: List[str]
    ) -> torch.Tensor:
        """Helper to ablate multiple heads at once."""
        # Parse head keys
        heads_to_ablate = {}
        for key in head_keys:
            parts = key.split("_")
            layer_idx = int(parts[1])
            head_idx = int(parts[3])

            if layer_idx not in heads_to_ablate:
                heads_to_ablate[layer_idx] = []
            heads_to_ablate[layer_idx].append(head_idx)

        # Register hooks for all heads
        hooks = []

        for layer_idx, head_indices in heads_to_ablate.items():
            def make_multi_head_ablation_hook(head_indices: List[int]):
                def hook(module, input, output):
                    # Similar to single head ablation but for multiple heads
                    x = input[0]
                    batch_size, seq_len, _ = x.shape

                    qkv = module.qkv_proj(x)
                    from einops import rearrange

                    q, k, v = rearrange(qkv, 'b s (three h d) -> three b s h d', three=3, h=module.num_heads)

                    if module.use_rope:
                        q, k = module.rope(q, k)

                    q = q * module.scale
                    attn_scores = torch.einsum('bshd,bthd->bhst', q, k)

                    causal_mask = torch.triu(
                        torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
                        diagonal=1
                    )
                    attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))

                    attn_weights = F.softmax(attn_scores, dim=-1)

                    # Ablate all specified heads
                    for head_idx in head_indices:
                        attn_weights[:, head_idx, :, :] = 0.0

                    out = torch.einsum('bhst,bthd->bshd', attn_weights, v)
                    out = rearrange(out, 'b s h d -> b s (h d)')

                    return module.out_proj(out)

                return hook

            attn_module = self.model.blocks[layer_idx].attn
            hook = attn_module.register_forward_hook(make_multi_head_ablation_hook(head_indices))
            hooks.append(hook)

        # Forward pass
        with torch.no_grad():
            logits, _ = self.model(input_ids)

        # Clean up
        for hook in hooks:
            hook.remove()

        return logits
