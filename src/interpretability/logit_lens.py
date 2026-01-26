"""
Logit Lens and Tuned Lens Analysis.

Decode intermediate layer representations to see what the model "thinks"
at each layer. This reveals when and how the model commits to predictions.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from .utils import ActivationCache, register_activation_hooks, get_token_strings


class LogitLens:
    """
    Implement Logit Lens to decode intermediate representations.

    The logit lens projects activations from intermediate layers through
    the language model head to see what tokens the model is "thinking about"
    at each layer.
    """

    def __init__(self, model, tokenizer, device: str = "cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()

    def get_layer_predictions(
        self,
        text: str,
        apply_ln: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Get predictions from each layer's activations.

        Args:
            text: Input text
            apply_ln: Whether to apply final layer norm before projection

        Returns:
            Dict mapping layer names to predicted logits
        """
        from .utils import prepare_example

        input_ids = prepare_example(text, self.tokenizer, self.device)

        # Get activations from all layers
        cache = ActivationCache()
        register_activation_hooks(self.model, cache)

        with torch.no_grad():
            final_logits, _ = self.model(input_ids)

        activations = cache.activations.copy()
        cache.remove_hooks()

        # Project each layer's activations through the LM head
        layer_predictions = {}

        for layer_name, activation in activations.items():
            if layer_name.startswith("block_"):
                # Apply layer norm if requested
                if apply_ln:
                    activation = self.model.ln_f(activation)

                # Project through LM head
                logits = self.model.lm_head(activation)
                layer_predictions[layer_name] = logits

        # Add final output
        layer_predictions["final"] = final_logits

        return layer_predictions

    def get_top_k_predictions(
        self,
        text: str,
        position: int,
        k: int = 5
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Get top-k predicted tokens at each layer for a specific position.

        Args:
            text: Input text
            position: Token position to analyze
            k: Number of top predictions to return

        Returns:
            Dict mapping layer names to list of (token, probability) tuples
        """
        layer_predictions = self.get_layer_predictions(text)

        results = {}

        for layer_name, logits in layer_predictions.items():
            # Get logits at the position
            position_logits = logits[0, position, :]

            # Get probabilities
            probs = F.softmax(position_logits, dim=-1)

            # Get top-k
            top_k_probs, top_k_indices = torch.topk(probs, k)

            # Convert to tokens
            top_k_tokens = [
                (self.tokenizer.idx_to_char.get(idx.item(), "<UNK>"),
                 prob.item())
                for idx, prob in zip(top_k_indices, top_k_probs)
            ]

            results[layer_name] = top_k_tokens

        return results

    def visualize_prediction_evolution(
        self,
        text: str,
        position: int,
        save_path: Optional[Path] = None,
        figsize: Tuple[int, int] = (12, 6)
    ):
        """
        Visualize how the top prediction evolves across layers.

        Args:
            text: Input text
            position: Token position to track
            save_path: Optional path to save figure
            figsize: Figure size
        """
        layer_predictions = self.get_layer_predictions(text)

        # Track top prediction at each layer
        layers = []
        top_tokens = []
        top_probs = []

        for layer_name, logits in sorted(layer_predictions.items()):
            position_logits = logits[0, position, :]
            probs = F.softmax(position_logits, dim=-1)

            top_prob, top_idx = torch.max(probs, dim=-1)
            top_token = self.tokenizer.idx_to_char.get(top_idx.item(), "<UNK>")

            layers.append(layer_name)
            top_tokens.append(top_token)
            top_probs.append(top_prob.item())

        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Plot 1: Top token at each layer
        colors = ['green' if i == len(layers) - 1 else 'blue' for i in range(len(layers))]
        ax1.bar(range(len(layers)), top_probs, color=colors, alpha=0.7)
        ax1.set_xticks(range(len(layers)))
        ax1.set_xticklabels(layers, rotation=45, ha='right')
        ax1.set_ylabel('Probability')
        ax1.set_title(f'Top Token Probability Across Layers (Position {position})')
        ax1.set_ylim(0, 1)

        # Add token labels on bars
        for i, (token, prob) in enumerate(zip(top_tokens, top_probs)):
            ax1.text(i, prob + 0.02, token, ha='center', va='bottom', fontsize=10, fontweight='bold')

        # Plot 2: Token transitions
        unique_tokens = list(set(top_tokens))
        token_to_idx = {t: i for i, t in enumerate(unique_tokens)}
        token_indices = [token_to_idx[t] for t in top_tokens]

        ax2.plot(range(len(layers)), token_indices, marker='o', linewidth=2, markersize=8)
        ax2.set_xticks(range(len(layers)))
        ax2.set_xticklabels(layers, rotation=45, ha='right')
        ax2.set_yticks(range(len(unique_tokens)))
        ax2.set_yticklabels(unique_tokens)
        ax2.set_title('Top Token Evolution Across Layers')
        ax2.set_xlabel('Layer')
        ax2.grid(True, alpha=0.3)

        plt.suptitle(f'Logit Lens Analysis: "{text}" at position {position}', fontsize=14)
        plt.tight_layout()

        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved logit lens visualization to {save_path}")

        return fig

    def visualize_layer_predictions_heatmap(
        self,
        text: str,
        top_k: int = 10,
        save_path: Optional[Path] = None,
        figsize: Tuple[int, int] = (14, 10)
    ):
        """
        Create a heatmap showing top-k predictions at each position and layer.

        Args:
            text: Input text
            top_k: Number of top predictions to show
            save_path: Optional path to save figure
            figsize: Figure size
        """
        from .utils import prepare_example

        input_ids = prepare_example(text, self.tokenizer, self.device)
        tokens = get_token_strings(input_ids, self.tokenizer)

        layer_predictions = self.get_layer_predictions(text)

        num_layers = len(layer_predictions)
        seq_len = len(tokens)

        # Create grid of subplots
        fig, axes = plt.subplots(num_layers, 1, figsize=figsize)
        if num_layers == 1:
            axes = [axes]

        for idx, (layer_name, logits) in enumerate(sorted(layer_predictions.items())):
            ax = axes[idx]

            # Get top-k predictions for each position
            probs = F.softmax(logits[0], dim=-1)
            top_k_probs, top_k_indices = torch.topk(probs, top_k, dim=-1)

            # Create matrix: (seq_len, top_k)
            prob_matrix = top_k_probs.detach().cpu().numpy()

            # Plot heatmap
            im = ax.imshow(prob_matrix.T, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1)

            # Set labels
            ax.set_yticks(range(top_k))
            ax.set_xticks(range(seq_len))
            ax.set_xticklabels(tokens, rotation=45, ha='right', fontsize=8)
            ax.set_ylabel('Top-k Rank', fontsize=9)
            ax.set_title(f'{layer_name}', fontsize=10)

            # Add token labels for top predictions
            for pos in range(seq_len):
                for rank in range(min(3, top_k)):  # Show top-3 token labels
                    token_idx = top_k_indices[pos, rank].item()
                    token = self.tokenizer.idx_to_char.get(token_idx, "?")
                    prob = prob_matrix[pos, rank]
                    if prob > 0.1:  # Only show if significant
                        ax.text(pos, rank, token, ha='center', va='center',
                               fontsize=7, color='white' if prob > 0.5 else 'black')

        plt.suptitle(f'Layer Predictions for: "{text}"', fontsize=14)
        plt.tight_layout()

        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved prediction heatmap to {save_path}")

        return fig

    def compute_prediction_entropy(
        self,
        text: str
    ) -> Dict[str, np.ndarray]:
        """
        Compute entropy of predictions at each layer and position.

        Higher entropy = more uncertain predictions.

        Returns:
            Dict mapping layer names to entropy arrays (seq_len,)
        """
        layer_predictions = self.get_layer_predictions(text)

        entropies = {}

        for layer_name, logits in layer_predictions.items():
            # Compute probabilities
            probs = F.softmax(logits[0], dim=-1)

            # Compute entropy for each position
            layer_entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)

            entropies[layer_name] = layer_entropy.detach().cpu().numpy()

        return entropies

    def analyze_convergence(
        self,
        text: str,
        target_token_pos: int,
        target_token: str
    ) -> Dict[str, float]:
        """
        Analyze when the model converges to the correct prediction.

        Args:
            text: Input text
            target_token_pos: Position of target token
            target_token: Expected token at that position

        Returns:
            Dict mapping layer names to probability of correct token
        """
        layer_predictions = self.get_layer_predictions(text)

        target_token_id = self.tokenizer.char_to_idx.get(target_token, self.tokenizer.unk_token_id)

        convergence = {}

        for layer_name, logits in layer_predictions.items():
            position_logits = logits[0, target_token_pos, :]
            probs = F.softmax(position_logits, dim=-1)

            # Get probability of target token
            target_prob = probs[target_token_id].item()
            convergence[layer_name] = target_prob

        return convergence

    def compare_examples(
        self,
        examples: List[Tuple[str, int, str]],
        save_path: Optional[Path] = None
    ):
        """
        Compare convergence patterns across multiple examples.

        Args:
            examples: List of (text, target_pos, target_token) tuples
            save_path: Optional path to save figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        for text, pos, target in examples:
            convergence = self.analyze_convergence(text, pos, target)

            layers = list(convergence.keys())
            probs = list(convergence.values())

            ax.plot(range(len(layers)), probs, marker='o', label=f'"{text}" → {target}')

        ax.set_xticks(range(len(layers)))
        ax.set_xticklabels(layers, rotation=45, ha='right')
        ax.set_ylabel('Target Token Probability')
        ax.set_xlabel('Layer')
        ax.set_title('Prediction Convergence Across Examples')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)

        plt.tight_layout()

        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig
