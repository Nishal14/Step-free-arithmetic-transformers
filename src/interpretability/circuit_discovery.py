"""
Circuit Discovery Tools.

Identify specific attention heads and neurons responsible for particular
operations like carry detection, digit addition, operator recognition, etc.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict
from .utils import (
    get_attention_patterns,
    extract_activations,
    find_top_k_neurons,
    get_token_strings
)


class CircuitDiscovery:
    """
    Discover computational circuits in the transformer.

    Identifies which components (heads, neurons) are responsible for
    specific algorithmic operations.
    """

    def __init__(self, model, tokenizer, device: str = "cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()

    def find_carry_detection_heads(
        self,
        addition_examples: List[Tuple[str, List[int]]],
        threshold: float = 0.3
    ) -> Dict[str, float]:
        """
        Find attention heads that detect carry positions in addition.

        Args:
            addition_examples: List of (expression, carry_positions)
                e.g., ("59 + 73", [0, 1]) means carry at positions 0 and 1
            threshold: Minimum attention weight to consider

        Returns:
            Dict mapping "layer_X_head_Y" to carry detection score
        """
        from .utils import prepare_example

        head_scores = defaultdict(float)
        head_counts = defaultdict(int)

        for text, carry_positions in addition_examples:
            input_ids = prepare_example(text, self.tokenizer, self.device)
            tokens = get_token_strings(input_ids, self.tokenizer)

            # Find digit positions in the input
            digit_positions = [i for i, token in enumerate(tokens) if token.isdigit()]

            if not digit_positions or not carry_positions:
                continue

            # Get attention patterns
            attention_maps = get_attention_patterns(self.model, input_ids)

            for layer_key, attn_weights in attention_maps.items():
                num_heads = attn_weights.shape[1]

                for head_idx in range(num_heads):
                    head_key = f"{layer_key}_head_{head_idx}"
                    head_attn = attn_weights[0, head_idx].cpu().numpy()

                    # Check if this head attends to carry positions
                    score = 0.0
                    count = 0

                    # For each digit that should have a carry
                    for digit_pos in digit_positions:
                        for carry_pos in carry_positions:
                            if carry_pos < len(tokens) and digit_pos < len(tokens):
                                # Check attention from result position to carry position
                                attn_weight = head_attn[digit_pos, carry_pos]
                                if attn_weight > threshold:
                                    score += attn_weight
                                    count += 1

                    if count > 0:
                        head_scores[head_key] += score / count
                        head_counts[head_key] += 1

        # Average scores across examples
        avg_scores = {
            head: score / head_counts[head]
            for head, score in head_scores.items()
            if head_counts[head] > 0
        }

        return dict(sorted(avg_scores.items(), key=lambda x: x[1], reverse=True))

    def find_operator_attention_heads(
        self,
        examples: List[str],
        operators: List[str] = ['+', '-', '*', '/']
    ) -> Dict[str, Dict[str, float]]:
        """
        Find heads that attend to mathematical operators.

        Args:
            examples: List of mathematical expressions
            operators: List of operators to track

        Returns:
            Dict mapping operator to dict of head scores
        """
        from .utils import prepare_example

        operator_head_scores = {op: defaultdict(float) for op in operators}
        operator_head_counts = {op: defaultdict(int) for op in operators}

        for text in examples:
            input_ids = prepare_example(text, self.tokenizer, self.device)
            tokens = get_token_strings(input_ids, self.tokenizer)

            # Find operator positions
            operator_positions = {op: [] for op in operators}
            for i, token in enumerate(tokens):
                if token in operators:
                    operator_positions[token].append(i)

            # Get attention patterns
            attention_maps = get_attention_patterns(self.model, input_ids)

            for layer_key, attn_weights in attention_maps.items():
                num_heads = attn_weights.shape[1]

                for head_idx in range(num_heads):
                    head_key = f"{layer_key}_head_{head_idx}"
                    head_attn = attn_weights[0, head_idx].cpu().numpy()

                    # For each operator
                    for op, positions in operator_positions.items():
                        if not positions:
                            continue

                        # Measure attention TO the operator from all positions
                        total_attn = 0.0
                        count = 0

                        for op_pos in positions:
                            # Average attention to this operator across all query positions
                            attn_to_op = head_attn[:, op_pos].mean()
                            total_attn += attn_to_op
                            count += 1

                        if count > 0:
                            operator_head_scores[op][head_key] += total_attn / count
                            operator_head_counts[op][head_key] += 1

        # Average and sort
        results = {}
        for op in operators:
            avg_scores = {
                head: score / operator_head_counts[op][head]
                for head, score in operator_head_scores[op].items()
                if operator_head_counts[op][head] > 0
            }
            results[op] = dict(sorted(avg_scores.items(), key=lambda x: x[1], reverse=True))

        return results

    def find_position_tracking_heads(
        self,
        examples: List[str]
    ) -> Dict[str, float]:
        """
        Find heads that track relative positions (e.g., for digit alignment).

        Looks for heads with strong diagonal or offset-diagonal attention.

        Returns:
            Dict mapping head to position-tracking score
        """
        from .utils import prepare_example

        head_scores = defaultdict(float)
        head_counts = defaultdict(int)

        for text in examples:
            input_ids = prepare_example(text, self.tokenizer, self.device)
            attention_maps = get_attention_patterns(self.model, input_ids)

            for layer_key, attn_weights in attention_maps.items():
                num_heads = attn_weights.shape[1]
                seq_len = attn_weights.shape[2]

                for head_idx in range(num_heads):
                    head_key = f"{layer_key}_head_{head_idx}"
                    head_attn = attn_weights[0, head_idx].cpu().numpy()

                    # Compute diagonal strength
                    diagonal_score = 0.0
                    for offset in [-1, 0, 1]:  # Check diagonal and adjacent
                        for i in range(max(0, -offset), min(seq_len, seq_len - offset)):
                            j = i + offset
                            if 0 <= j < seq_len:
                                diagonal_score += head_attn[i, j]

                    # Normalize by sequence length
                    diagonal_score /= (seq_len * 3)

                    head_scores[head_key] += diagonal_score
                    head_counts[head_key] += 1

        # Average scores
        avg_scores = {
            head: score / head_counts[head]
            for head, score in head_scores.items()
            if head_counts[head] > 0
        }

        return dict(sorted(avg_scores.items(), key=lambda x: x[1], reverse=True))

    def find_specialized_neurons(
        self,
        examples: List[str],
        labels: List[str],
        layer_name: str = "ff_0",
        top_k: int = 20
    ) -> Dict[str, List[Tuple[int, float]]]:
        """
        Find neurons specialized for different operations.

        Args:
            examples: List of expressions
            labels: List of operation labels for each example
            layer_name: Which feedforward layer to analyze
            top_k: Number of top neurons to return per label

        Returns:
            Dict mapping label to list of (neuron_idx, activation) tuples
        """
        from .utils import prepare_example

        neuron_activations = defaultdict(lambda: defaultdict(list))

        for text, label in zip(examples, labels):
            input_ids = prepare_example(text, self.tokenizer, self.device)

            # Extract activations
            activations = extract_activations(self.model, input_ids, [layer_name])

            if layer_name not in activations:
                continue

            # Get mean activation across sequence for each neuron
            layer_act = activations[layer_name][0]  # (seq_len, hidden_dim)
            mean_act = layer_act.mean(dim=0).cpu().numpy()  # (hidden_dim,)

            # Store activations for this label
            for neuron_idx, activation in enumerate(mean_act):
                neuron_activations[label][neuron_idx].append(activation)

        # Find top neurons for each label
        specialized_neurons = {}

        for label, neuron_dict in neuron_activations.items():
            # Compute mean activation for each neuron
            neuron_means = {
                neuron_idx: np.mean(activations)
                for neuron_idx, activations in neuron_dict.items()
            }

            # Sort and get top-k
            top_neurons = sorted(neuron_means.items(), key=lambda x: x[1], reverse=True)[:top_k]
            specialized_neurons[label] = top_neurons

        return specialized_neurons

    def visualize_head_specialization(
        self,
        head_scores_dict: Dict[str, Dict[str, float]],
        save_path: Optional[Path] = None,
        figsize: Tuple[int, int] = (12, 8)
    ):
        """
        Visualize which heads specialize in which operations.

        Args:
            head_scores_dict: Dict mapping operation to head scores
            save_path: Optional save path
            figsize: Figure size
        """
        # Collect all heads
        all_heads = set()
        for scores in head_scores_dict.values():
            all_heads.update(scores.keys())

        all_heads = sorted(all_heads)
        operations = list(head_scores_dict.keys())

        # Create matrix
        matrix = np.zeros((len(operations), len(all_heads)))

        for i, operation in enumerate(operations):
            for j, head in enumerate(all_heads):
                matrix[i, j] = head_scores_dict[operation].get(head, 0.0)

        # Normalize each row
        matrix = (matrix - matrix.min(axis=1, keepdims=True)) / (matrix.max(axis=1, keepdims=True) - matrix.min(axis=1, keepdims=True) + 1e-10)

        # Plot
        fig, ax = plt.subplots(figsize=figsize)

        im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')

        # Labels
        ax.set_xticks(range(len(all_heads)))
        ax.set_yticks(range(len(operations)))
        ax.set_xticklabels(all_heads, rotation=90, fontsize=8)
        ax.set_yticklabels(operations, fontsize=10)

        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Specialization Score (normalized)', rotation=270, labelpad=20)

        ax.set_title('Attention Head Specialization')
        ax.set_xlabel('Attention Head')
        ax.set_ylabel('Operation')

        plt.tight_layout()

        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved head specialization plot to {save_path}")

        return fig

    def analyze_circuit_for_example(
        self,
        text: str,
        target_operation: str
    ) -> Dict[str, any]:
        """
        Comprehensive circuit analysis for a single example.

        Args:
            text: Input expression
            target_operation: What operation to analyze (e.g., "addition", "carry")

        Returns:
            Dict with circuit information
        """
        from .utils import prepare_example

        input_ids = prepare_example(text, self.tokenizer, self.device)
        tokens = get_token_strings(input_ids, self.tokenizer)

        # Get attention patterns
        attention_maps = get_attention_patterns(self.model, input_ids)

        # Get activations
        activations = extract_activations(self.model, input_ids)

        # Analyze each component
        circuit = {
            "input": text,
            "tokens": tokens,
            "operation": target_operation,
            "attention_summary": {},
            "activation_summary": {},
            "important_heads": [],
            "important_neurons": {}
        }

        # Summarize attention patterns
        for layer_key, attn_weights in attention_maps.items():
            num_heads = attn_weights.shape[1]

            # Find heads with high max attention
            max_attns = []
            for head_idx in range(num_heads):
                head_attn = attn_weights[0, head_idx]
                max_attns.append(head_attn.max().item())

            circuit["attention_summary"][layer_key] = {
                "num_heads": num_heads,
                "max_attention_per_head": max_attns,
                "mean_attention": attn_weights.mean().item()
            }

            # Identify important heads (top 20% by max attention)
            threshold = np.percentile(max_attns, 80)
            for head_idx, max_attn in enumerate(max_attns):
                if max_attn >= threshold:
                    circuit["important_heads"].append({
                        "layer": layer_key,
                        "head": head_idx,
                        "max_attention": max_attn
                    })

        # Analyze neurons
        for layer_name, activation in activations.items():
            if layer_name.startswith("ff_"):
                # Find top neurons
                top_neurons, top_values = find_top_k_neurons(activation, k=10)

                circuit["important_neurons"][layer_name] = [
                    {"neuron": idx, "activation": val}
                    for idx, val in zip(top_neurons, top_values)
                ]

        return circuit

    def export_circuit_graph(
        self,
        circuit: Dict,
        output_path: Path
    ):
        """
        Export circuit as a graph representation.

        Args:
            circuit: Circuit dict from analyze_circuit_for_example
            output_path: Where to save the graph
        """
        import json

        # Convert to JSON-serializable format
        circuit_json = {
            "input": circuit["input"],
            "tokens": circuit["tokens"],
            "operation": circuit["operation"],
            "nodes": [],
            "edges": []
        }

        # Add nodes for important heads
        for head in circuit["important_heads"]:
            circuit_json["nodes"].append({
                "id": f"{head['layer']}_head_{head['head']}",
                "type": "attention_head",
                "layer": head["layer"],
                "head": head["head"],
                "importance": head["max_attention"]
            })

        # Add nodes for important neurons
        for layer_name, neurons in circuit["important_neurons"].items():
            for neuron_info in neurons:
                circuit_json["nodes"].append({
                    "id": f"{layer_name}_neuron_{neuron_info['neuron']}",
                    "type": "neuron",
                    "layer": layer_name,
                    "neuron": neuron_info["neuron"],
                    "importance": neuron_info["activation"]
                })

        # Save
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(circuit_json, f, indent=2)

        print(f"Exported circuit graph to {output_path}")
