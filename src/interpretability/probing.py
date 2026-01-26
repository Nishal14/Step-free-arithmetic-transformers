"""
Probing Classifiers.

Test what information is encoded at each layer using linear probes.
Trains classifiers to predict algorithmic properties from hidden states.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable
import matplotlib.pyplot as plt
import seaborn as sns
from .utils import extract_activations, prepare_example


class ProbingClassifier:
    """
    Train and evaluate probing classifiers on transformer representations.

    Probes test what information is linearly accessible in the hidden states.
    """

    def __init__(self, model, tokenizer, device: str = "cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()
        self.trained_probes = {}

    def collect_probe_data(
        self,
        examples: List[str],
        labels: List[int],
        layer_name: str,
        position_fn: Optional[Callable[[str], int]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Collect hidden states and labels for probe training.

        Args:
            examples: List of input texts
            labels: List of integer labels (what we're probing for)
            layer_name: Which layer to extract activations from
            position_fn: Function to extract relevant position from each example
                        (e.g., lambda text: text.index('+') for operator position)
                        If None, uses mean pooling across all positions

        Returns:
            X: Feature matrix (num_examples, hidden_dim)
            y: Labels (num_examples,)
        """
        X = []
        y = []

        for text, label in zip(examples, labels):
            input_ids = prepare_example(text, self.tokenizer, self.device)

            # Extract activations
            activations = extract_activations(self.model, input_ids, [layer_name])

            if layer_name not in activations:
                continue

            # Get hidden states: (batch=1, seq_len, hidden_dim)
            hidden_states = activations[layer_name][0].cpu().numpy()

            # Extract representation at specific position or use mean pooling
            if position_fn is not None:
                try:
                    pos = position_fn(text)
                    if 0 <= pos < hidden_states.shape[0]:
                        representation = hidden_states[pos]
                    else:
                        continue
                except:
                    continue
            else:
                # Mean pooling
                representation = hidden_states.mean(axis=0)

            X.append(representation)
            y.append(label)

        return np.array(X), np.array(y)

    def train_probe(
        self,
        probe_name: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        max_iter: int = 1000
    ) -> Dict[str, float]:
        """
        Train a linear probing classifier.

        Args:
            probe_name: Name for this probe
            X_train: Training features
            y_train: Training labels
            X_val: Optional validation features
            y_val: Optional validation labels
            max_iter: Maximum iterations for training

        Returns:
            Dict with training metrics
        """
        # Train logistic regression probe
        probe = LogisticRegression(max_iter=max_iter, random_state=42)
        probe.fit(X_train, y_train)

        # Evaluate
        train_acc = accuracy_score(y_train, probe.predict(X_train))

        results = {
            "train_accuracy": train_acc,
            "num_train": len(y_train),
        }

        if X_val is not None and y_val is not None:
            val_acc = accuracy_score(y_val, probe.predict(X_val))
            results["val_accuracy"] = val_acc
            results["num_val"] = len(y_val)

        # Store probe
        self.trained_probes[probe_name] = probe

        return results

    def probe_carry_detection(
        self,
        addition_examples: List[str],
        carry_labels: List[int],
        layer_name: str,
        carry_position_fn: Callable[[str], int]
    ) -> Dict[str, float]:
        """
        Train a probe to detect if a carry occurs.

        Args:
            addition_examples: List of addition expressions
            carry_labels: Binary labels (1 = carry occurs, 0 = no carry)
            layer_name: Which layer to probe
            carry_position_fn: Function to get the position to check for carry

        Returns:
            Probe performance metrics
        """
        # Split into train/val
        split_idx = int(0.8 * len(addition_examples))

        X_all, y_all = self.collect_probe_data(
            addition_examples,
            carry_labels,
            layer_name,
            carry_position_fn
        )

        X_train, y_train = X_all[:split_idx], y_all[:split_idx]
        X_val, y_val = X_all[split_idx:], y_all[split_idx:]

        probe_name = f"carry_detection_{layer_name}"
        results = self.train_probe(probe_name, X_train, y_train, X_val, y_val)

        print(f"Carry Detection Probe ({layer_name}):")
        print(f"  Train Accuracy: {results['train_accuracy']:.3f}")
        if 'val_accuracy' in results:
            print(f"  Val Accuracy: {results['val_accuracy']:.3f}")

        return results

    def probe_digit_value(
        self,
        examples: List[str],
        digit_values: List[int],
        layer_name: str,
        digit_position_fn: Callable[[str], int]
    ) -> Dict[str, float]:
        """
        Train a probe to predict digit values (0-9).

        Args:
            examples: List of expressions
            digit_values: List of digit values at specific positions
            layer_name: Which layer to probe
            digit_position_fn: Function to get position of digit to probe

        Returns:
            Probe performance metrics
        """
        split_idx = int(0.8 * len(examples))

        X_all, y_all = self.collect_probe_data(
            examples,
            digit_values,
            layer_name,
            digit_position_fn
        )

        X_train, y_train = X_all[:split_idx], y_all[:split_idx]
        X_val, y_val = X_all[split_idx:], y_all[split_idx:]

        probe_name = f"digit_value_{layer_name}"
        results = self.train_probe(probe_name, X_train, y_train, X_val, y_val)

        print(f"Digit Value Probe ({layer_name}):")
        print(f"  Train Accuracy: {results['train_accuracy']:.3f}")
        if 'val_accuracy' in results:
            print(f"  Val Accuracy: {results['val_accuracy']:.3f}")

        return results

    def probe_operation_type(
        self,
        examples: List[str],
        operation_labels: List[int],
        layer_name: str
    ) -> Dict[str, float]:
        """
        Train a probe to classify operation type (addition, multiplication, etc.).

        Args:
            examples: List of mathematical expressions
            operation_labels: Integer labels for operation types
                             (e.g., 0=add, 1=mul, 2=base_convert, 3=poly)
            layer_name: Which layer to probe

        Returns:
            Probe performance metrics
        """
        split_idx = int(0.8 * len(examples))

        # Use mean pooling since operation is global property
        X_all, y_all = self.collect_probe_data(
            examples,
            operation_labels,
            layer_name,
            position_fn=None  # Mean pooling
        )

        X_train, y_train = X_all[:split_idx], y_all[:split_idx]
        X_val, y_val = X_all[split_idx:], y_all[split_idx:]

        probe_name = f"operation_type_{layer_name}"
        results = self.train_probe(probe_name, X_train, y_train, X_val, y_val)

        print(f"Operation Type Probe ({layer_name}):")
        print(f"  Train Accuracy: {results['train_accuracy']:.3f}")
        if 'val_accuracy' in results:
            print(f"  Val Accuracy: {results['val_accuracy']:.3f}")

        return results

    def probe_across_layers(
        self,
        examples: List[str],
        labels: List[int],
        probe_type: str,
        position_fn: Optional[Callable[[str], int]] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Train probes across all layers to see where information emerges.

        Args:
            examples: List of input texts
            labels: Labels for the probe
            probe_type: Name/description of what we're probing
            position_fn: Optional position extraction function

        Returns:
            Dict mapping layer names to probe results
        """
        results = {}

        # Get all layer names
        sample_input = prepare_example(examples[0], self.tokenizer, self.device)
        all_activations = extract_activations(self.model, sample_input)
        layer_names = [k for k in all_activations.keys() if k.startswith("block_")]

        print(f"\nProbing for: {probe_type}")
        print("=" * 50)

        for layer_name in layer_names:
            split_idx = int(0.8 * len(examples))

            X_all, y_all = self.collect_probe_data(
                examples,
                labels,
                layer_name,
                position_fn
            )

            if len(X_all) == 0:
                continue

            X_train, y_train = X_all[:split_idx], y_all[:split_idx]
            X_val, y_val = X_all[split_idx:], y_all[split_idx:]

            probe_name = f"{probe_type}_{layer_name}"
            layer_results = self.train_probe(probe_name, X_train, y_train, X_val, y_val)

            results[layer_name] = layer_results

            print(f"{layer_name}: Train={layer_results['train_accuracy']:.3f}, "
                  f"Val={layer_results.get('val_accuracy', 0.0):.3f}")

        return results

    def visualize_probe_results_across_layers(
        self,
        results: Dict[str, Dict[str, float]],
        probe_type: str,
        save_path: Optional[Path] = None
    ):
        """
        Visualize how probe accuracy evolves across layers.

        Args:
            results: Results from probe_across_layers
            probe_type: Description of probe
            save_path: Optional save path
        """
        layers = sorted(results.keys())
        train_accs = [results[layer]['train_accuracy'] for layer in layers]
        val_accs = [results[layer].get('val_accuracy', 0.0) for layer in layers]

        fig, ax = plt.subplots(figsize=(10, 6))

        x = range(len(layers))
        ax.plot(x, train_accs, marker='o', label='Train Accuracy', linewidth=2)
        ax.plot(x, val_accs, marker='s', label='Val Accuracy', linewidth=2)

        ax.set_xticks(x)
        ax.set_xticklabels(layers, rotation=45, ha='right')
        ax.set_xlabel('Layer')
        ax.set_ylabel('Accuracy')
        ax.set_title(f'Probing Accuracy Across Layers: {probe_type}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)

        # Add chance level line
        num_classes = len(set([results[layer].get('num_train', 0) for layer in layers]))
        if num_classes > 0:
            chance = 1.0 / num_classes
            ax.axhline(y=chance, color='r', linestyle='--', label=f'Chance ({chance:.2f})', alpha=0.5)

        plt.tight_layout()

        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved probe results to {save_path}")

        return fig

    def compare_multiple_probes(
        self,
        probe_results_dict: Dict[str, Dict[str, Dict[str, float]]],
        save_path: Optional[Path] = None
    ):
        """
        Compare multiple probing experiments side-by-side.

        Args:
            probe_results_dict: Dict mapping probe_name to results dict
            save_path: Optional save path
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        for probe_name, results in probe_results_dict.items():
            layers = sorted(results.keys())
            val_accs = [results[layer].get('val_accuracy', 0.0) for layer in layers]

            x = range(len(layers))
            ax.plot(x, val_accs, marker='o', label=probe_name, linewidth=2)

        ax.set_xticks(range(len(layers)))
        ax.set_xticklabels(layers, rotation=45, ha='right')
        ax.set_xlabel('Layer')
        ax.set_ylabel('Validation Accuracy')
        ax.set_title('Comparison of Probing Tasks Across Layers')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)

        plt.tight_layout()

        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def get_probe_weights(
        self,
        probe_name: str
    ) -> Optional[np.ndarray]:
        """
        Get the learned weights of a trained probe.

        Returns:
            Weight matrix of shape (num_classes, hidden_dim)
        """
        if probe_name not in self.trained_probes:
            print(f"Probe '{probe_name}' not found.")
            return None

        probe = self.trained_probes[probe_name]
        return probe.coef_

    def analyze_probe_weights(
        self,
        probe_name: str,
        class_names: Optional[List[str]] = None,
        top_k: int = 20
    ) -> Dict[str, List[Tuple[int, float]]]:
        """
        Analyze which features (dimensions) are most important for each class.

        Args:
            probe_name: Name of trained probe
            class_names: Optional names for classes
            top_k: Number of top features to return per class

        Returns:
            Dict mapping class names to top features
        """
        weights = self.get_probe_weights(probe_name)

        if weights is None:
            return {}

        num_classes = weights.shape[0]

        if class_names is None:
            class_names = [f"Class_{i}" for i in range(num_classes)]

        results = {}

        for class_idx, class_name in enumerate(class_names):
            class_weights = weights[class_idx]

            # Get top positive and negative weights
            top_positive_idx = np.argsort(class_weights)[-top_k:][::-1]
            top_negative_idx = np.argsort(class_weights)[:top_k]

            results[class_name] = {
                "positive": [(int(idx), float(class_weights[idx])) for idx in top_positive_idx],
                "negative": [(int(idx), float(class_weights[idx])) for idx in top_negative_idx]
            }

        return results
