"""
Attention Pattern Analysis and Visualization.

Analyzes which tokens attend to which during computation, helping identify
specialized attention heads for specific operations (e.g., carry detection,
operand tracking).
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from .utils import (
    get_attention_patterns,
    get_token_strings,
    normalize_attention
)


class AttentionAnalyzer:
    """
    Analyze and visualize attention patterns in transformer models.
    """

    def __init__(self, model, tokenizer, device: str = "cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()

    def get_attention_for_example(
        self,
        text: str
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], List[str]]:
        """
        Get attention patterns for a single example.

        Returns:
            input_ids: Token IDs
            attention_maps: Dict mapping layer names to attention weights
            token_strings: Human-readable token strings
        """
        from .utils import prepare_example

        input_ids = prepare_example(text, self.tokenizer, self.device)
        attention_maps = get_attention_patterns(self.model, input_ids)
        token_strings = get_token_strings(input_ids, self.tokenizer)

        return input_ids, attention_maps, token_strings

    def plot_attention_heads(
        self,
        text: str,
        layer_idx: int,
        save_path: Optional[Path] = None,
        figsize: Tuple[int, int] = (16, 12)
    ):
        """
        Plot attention patterns for all heads in a specific layer.

        Args:
            text: Input text to analyze
            layer_idx: Which transformer layer to visualize
            save_path: Optional path to save the figure
            figsize: Figure size
        """
        input_ids, attention_maps, tokens = self.get_attention_for_example(text)

        layer_key = f"layer_{layer_idx}"
        if layer_key not in attention_maps:
            raise ValueError(f"Layer {layer_idx} not found. Available: {list(attention_maps.keys())}")

        # Get attention weights: (batch=1, num_heads, seq_len, seq_len)
        attn_weights = attention_maps[layer_key][0].cpu().numpy()
        num_heads = attn_weights.shape[0]

        # Create subplot grid
        grid_size = int(np.ceil(np.sqrt(num_heads)))
        fig, axes = plt.subplots(grid_size, grid_size, figsize=figsize)
        axes = axes.flatten() if num_heads > 1 else [axes]

        for head_idx in range(num_heads):
            ax = axes[head_idx]

            # Plot heatmap
            sns.heatmap(
                attn_weights[head_idx],
                xticklabels=tokens,
                yticklabels=tokens,
                cmap="viridis",
                cbar=True,
                square=True,
                ax=ax,
                vmin=0,
                vmax=1
            )

            ax.set_title(f"Head {head_idx}", fontsize=10)
            ax.set_xlabel("Key", fontsize=8)
            ax.set_ylabel("Query", fontsize=8)
            ax.tick_params(axis='both', labelsize=7)

        # Hide empty subplots
        for idx in range(num_heads, len(axes)):
            axes[idx].axis('off')

        plt.suptitle(f"Attention Patterns - Layer {layer_idx}\nInput: {text}", fontsize=14)
        plt.tight_layout()

        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved attention plot to {save_path}")

        return fig

    def plot_attention_single_head(
        self,
        text: str,
        layer_idx: int,
        head_idx: int,
        save_path: Optional[Path] = None,
        figsize: Tuple[int, int] = (10, 8)
    ):
        """Plot attention pattern for a single head with enhanced detail."""
        input_ids, attention_maps, tokens = self.get_attention_for_example(text)

        layer_key = f"layer_{layer_idx}"
        attn_weights = attention_maps[layer_key][0, head_idx].cpu().numpy()

        fig, ax = plt.subplots(figsize=figsize)

        # Plot heatmap
        im = ax.imshow(attn_weights, cmap="viridis", aspect="auto", vmin=0, vmax=1)

        # Set ticks
        ax.set_xticks(np.arange(len(tokens)))
        ax.set_yticks(np.arange(len(tokens)))
        ax.set_xticklabels(tokens, fontsize=10)
        ax.set_yticklabels(tokens, fontsize=10)

        # Rotate x labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Attention Weight", rotation=270, labelpad=20)

        # Add text annotations for high attention values
        for i in range(len(tokens)):
            for j in range(len(tokens)):
                if attn_weights[i, j] > 0.1:  # Only show significant attention
                    text_color = "white" if attn_weights[i, j] > 0.5 else "black"
                    ax.text(j, i, f"{attn_weights[i, j]:.2f}",
                           ha="center", va="center", color=text_color, fontsize=8)

        ax.set_title(f"Attention Pattern - Layer {layer_idx}, Head {head_idx}\nInput: {text}")
        ax.set_xlabel("Key Tokens")
        ax.set_ylabel("Query Tokens")

        plt.tight_layout()

        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved single head attention plot to {save_path}")

        return fig

    def analyze_attention_statistics(
        self,
        text: str
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute statistics about attention patterns.

        Returns statistics like:
        - Average attention to previous token
        - Attention entropy (how focused vs. diffuse)
        - Maximum attention weight
        """
        input_ids, attention_maps, tokens = self.get_attention_for_example(text)

        stats = {}

        for layer_key, attn_weights in attention_maps.items():
            # attn_weights: (batch=1, num_heads, seq_len, seq_len)
            attn = attn_weights[0].cpu().numpy()
            num_heads, seq_len, _ = attn.shape

            layer_stats = {}

            for head_idx in range(num_heads):
                head_attn = attn[head_idx]

                # Compute entropy for each query position
                entropies = []
                for i in range(seq_len):
                    query_attn = head_attn[i]
                    # Avoid log(0)
                    query_attn = query_attn + 1e-10
                    entropy = -(query_attn * np.log(query_attn)).sum()
                    entropies.append(entropy)

                # Statistics
                layer_stats[f"head_{head_idx}"] = {
                    "mean_entropy": float(np.mean(entropies)),
                    "max_attention": float(head_attn.max()),
                    "mean_attention_to_prev": float(np.mean([head_attn[i, i-1] for i in range(1, seq_len)])),
                    "diagonal_attention": float(np.mean([head_attn[i, i] for i in range(seq_len)])),
                }

            stats[layer_key] = layer_stats

        return stats

    def find_specialized_heads(
        self,
        examples: List[str],
        operation_positions: List[List[int]],
        threshold: float = 0.3
    ) -> Dict[str, List[Tuple[int, int]]]:
        """
        Find attention heads that consistently attend to specific positions.

        Args:
            examples: List of text examples
            operation_positions: For each example, list of critical token positions
                                (e.g., positions where carry occurs)
            threshold: Minimum attention weight to consider

        Returns:
            Dict mapping "layer_X_head_Y" to list of (query_pos, key_pos) patterns
        """
        specialized_heads = {}

        for example, positions in zip(examples, operation_positions):
            input_ids, attention_maps, tokens = self.get_attention_for_example(example)

            for layer_key, attn_weights in attention_maps.items():
                attn = attn_weights[0].cpu().numpy()
                num_heads = attn.shape[0]

                for head_idx in range(num_heads):
                    head_key = f"{layer_key}_head_{head_idx}"
                    head_attn = attn[head_idx]

                    # Check if this head attends to operation positions
                    attending_patterns = []
                    for query_pos in range(len(tokens)):
                        for key_pos in positions:
                            if key_pos < len(tokens) and head_attn[query_pos, key_pos] > threshold:
                                attending_patterns.append((query_pos, key_pos))

                    if attending_patterns:
                        if head_key not in specialized_heads:
                            specialized_heads[head_key] = []
                        specialized_heads[head_key].extend(attending_patterns)

        return specialized_heads

    def compare_attention_across_layers(
        self,
        text: str,
        query_pos: int,
        save_path: Optional[Path] = None
    ):
        """
        Visualize how attention from a specific query position evolves across layers.

        Args:
            text: Input text
            query_pos: Query position to track
            save_path: Optional save path
        """
        input_ids, attention_maps, tokens = self.get_attention_for_example(text)

        num_layers = len(attention_maps)
        fig, axes = plt.subplots(1, num_layers, figsize=(4 * num_layers, 4))

        if num_layers == 1:
            axes = [axes]

        for layer_idx, (layer_key, attn_weights) in enumerate(attention_maps.items()):
            ax = axes[layer_idx]

            # Average attention across all heads
            avg_attn = attn_weights[0, :, query_pos, :].mean(dim=0).cpu().numpy()

            ax.bar(range(len(tokens)), avg_attn)
            ax.set_xticks(range(len(tokens)))
            ax.set_xticklabels(tokens, rotation=45, ha='right')
            ax.set_title(f"Layer {layer_idx}")
            ax.set_ylabel("Avg Attention Weight")
            ax.set_ylim(0, 1)

        plt.suptitle(f"Attention Evolution for Query Token '{tokens[query_pos]}' (pos {query_pos})")
        plt.tight_layout()

        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def export_attention_data(
        self,
        text: str,
        output_path: Path
    ):
        """Export attention patterns to numpy file for further analysis."""
        import json

        input_ids, attention_maps, tokens = self.get_attention_for_example(text)

        data = {
            "text": text,
            "tokens": tokens,
            "attention_maps": {
                k: v[0].cpu().numpy() for k, v in attention_maps.items()
            }
        }

        # Save as npz
        np.savez(output_path, **data)
        print(f"Exported attention data to {output_path}")

        # Also save metadata as JSON
        metadata_path = output_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump({
                "text": text,
                "tokens": tokens,
                "num_layers": len(attention_maps),
                "num_heads": list(attention_maps.values())[0].shape[1],
                "seq_len": len(tokens)
            }, f, indent=2)
