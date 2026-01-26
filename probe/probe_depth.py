"""
Linear probing to test whether parenthesis depth is encoded in internal activations.

Probes:
- Primary: Layer 2, Head 2 attention head output
- Secondary: Residual stream output of each layer

Classification: Binary (depth > 0 vs depth == 0)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import torch
import numpy as np
from typing import List, Dict, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from src.model import create_model


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


def load_checkpoint(checkpoint_path: Path, device: str = "cpu"):
    """Load model checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_config = checkpoint.get("model_config", {})
    model = create_model(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model, checkpoint


def load_dataset(data_path: Path) -> List[Dict]:
    """Load JSONL dataset."""
    data = []
    with open(data_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def extract_features_and_labels(
    model,
    dataset: List[Dict],
    tokenizer: SimpleTokenizer,
    device: str,
    layer_idx: int = 2,
    head_idx: int = 2,
    extract_residual: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract features from model activations and align with depth labels.

    Args:
        model: Trained transformer model
        dataset: List of examples with 'expr', 'result', and 'depth' fields
        tokenizer: Tokenizer instance
        device: Device to run on
        layer_idx: Layer index to extract from (0-indexed)
        head_idx: Head index to extract from (0-indexed)
        extract_residual: If True, extract residual stream instead of attention head

    Returns:
        X: Feature matrix (num_samples, feature_dim)
        y: Binary labels (num_samples,) where 1 = depth > 0, 0 = depth == 0
    """
    model.eval()
    all_features = []
    all_labels = []

    # Storage for residual stream if needed
    residual_activations = {}

    def capture_residual(layer_idx):
        """Hook to capture residual stream output."""
        def hook(module, input, output):
            residual_activations['output'] = output[0].detach()
        return hook

    with torch.no_grad():
        for item in dataset:
            expr = item["expr"]
            depth = item["depth"]

            # Skip flat expressions (all depth == 0)
            if max(depth) == 0:
                continue

            # Tokenize expression only (not the " = result" part)
            # This is because depth labels correspond to the expression characters
            tokens = tokenizer.encode(expr, add_special_tokens=True)
            input_ids = torch.tensor([tokens], dtype=torch.long, device=device)

            # Register hook if extracting residual stream
            if extract_residual:
                hook_handle = model.blocks[layer_idx].register_forward_hook(
                    capture_residual(layer_idx)
                )

            # Forward pass
            logits, _, _ = model(input_ids)

            # Extract features
            if extract_residual:
                # Residual stream output: (batch=1, seq_len, d_model)
                residual_output = residual_activations['output']
                features = residual_output[0].cpu().numpy()
                hook_handle.remove()
            else:
                # Extract attention head output from specified layer and head
                # Shape: (batch=1, seq_len, num_heads, head_dim)
                head_output = model.blocks[layer_idx].attn.last_head_output

                # Select specific head: (batch=1, seq_len, head_dim)
                features = head_output[0, :, head_idx, :].cpu().numpy()

            # Align with depth labels
            # tokens = [<BOS>, char0, char1, ..., charN, <EOS>]
            # depth = [depth0, depth1, ..., depthN]
            # We need to align token positions with depth positions
            # Skip <BOS> (position 0) and <EOS> (last position)

            # Token positions 1 to len(expr) correspond to expression characters
            for pos_idx in range(1, min(len(expr) + 1, features.shape[0] - 1)):
                char_idx = pos_idx - 1  # Map token position to character position
                if char_idx < len(depth):
                    feature_vec = features[pos_idx]
                    depth_label = 1 if depth[char_idx] > 0 else 0

                    all_features.append(feature_vec)
                    all_labels.append(depth_label)

    X = np.array(all_features)
    y = np.array(all_labels)

    return X, y


def probe_attention_head(
    model,
    train_data: List[Dict],
    test_data: List[Dict],
    tokenizer: SimpleTokenizer,
    device: str,
    layer_idx: int = 2,
    head_idx: int = 2,
    extract_residual: bool = False
) -> Dict[str, float]:
    """
    Probe a specific attention head or residual stream for depth encoding.

    Returns:
        Dictionary with train_acc and test_acc
    """
    if extract_residual:
        print(f"Extracting features from Layer {layer_idx} residual stream...")
    else:
        print(f"Extracting features from Layer {layer_idx}, Head {head_idx}...")

    # Extract training features
    X_train, y_train = extract_features_and_labels(
        model, train_data, tokenizer, device, layer_idx, head_idx, extract_residual
    )

    # Extract test features
    X_test, y_test = extract_features_and_labels(
        model, test_data, tokenizer, device, layer_idx, head_idx, extract_residual
    )

    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    print(f"Train class distribution: {np.bincount(y_train)}")
    print(f"Test class distribution: {np.bincount(y_test)}")

    # Train logistic regression probe
    print("Training probe...")
    probe = LogisticRegression(max_iter=1000, random_state=42)
    probe.fit(X_train, y_train)

    # Evaluate
    train_pred = probe.predict(X_train)
    test_pred = probe.predict(X_test)

    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)

    return {
        "train_acc": train_acc,
        "test_acc": test_acc,
        "train_samples": len(X_train),
        "test_samples": len(X_test)
    }


def probe_all_heads(
    model,
    train_data: List[Dict],
    test_data: List[Dict],
    tokenizer: SimpleTokenizer,
    device: str
) -> Dict[str, Dict[str, float]]:
    """
    Probe all attention heads across all layers.

    Returns:
        Dictionary mapping (layer, head) to results
    """
    num_layers = model.num_layers
    num_heads = model.blocks[0].attn.num_heads

    results = {}

    for layer_idx in range(num_layers):
        for head_idx in range(num_heads):
            print(f"\n--- Probing Layer {layer_idx}, Head {head_idx} ---")

            try:
                result = probe_attention_head(
                    model, train_data, test_data, tokenizer, device,
                    layer_idx, head_idx
                )
                results[f"L{layer_idx}-H{head_idx}"] = result

                print(f"Train accuracy: {result['train_acc']:.4f}")
                print(f"Test accuracy:  {result['test_acc']:.4f}")

            except Exception as e:
                print(f"Error probing L{layer_idx}-H{head_idx}: {e}")
                continue

    return results


def main():
    """Main probing procedure."""
    # Configuration
    checkpoint_path = Path("runs/pilot/checkpoint_best.pt")
    train_data_path = Path("data/pilot_train.jsonl")
    test_data_path = Path("data/pilot_test_paren.jsonl")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("="*60)
    print("LINEAR PROBING: Parenthesis Depth Encoding")
    print("="*60)
    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Train data: {train_data_path}")
    print(f"Test data: {test_data_path}")
    print()

    # Load model
    print("Loading model...")
    model, checkpoint = load_checkpoint(checkpoint_path, device)
    print(f"Model parameters: {model.count_parameters() / 1e6:.2f}M")
    print(f"Model config: {checkpoint.get('model_config', {})}")
    print()

    # Load data
    print("Loading datasets...")
    train_data = load_dataset(train_data_path)
    test_data = load_dataset(test_data_path)
    print(f"Train examples: {len(train_data)}")
    print(f"Test examples: {len(test_data)}")
    print()

    # Initialize tokenizer
    tokenizer = SimpleTokenizer()

    # Primary target: Layer 2, Head 2
    print("="*60)
    print("PRIMARY PROBE: Layer 2, Head 2")
    print("="*60)

    result_l2h2 = probe_attention_head(
        model, train_data, test_data, tokenizer, device,
        layer_idx=2, head_idx=2
    )

    print()
    print("RESULTS:")
    print(f"L2-H2 probe:")
    print(f"  Train accuracy: {result_l2h2['train_acc']:.4f}")
    print(f"  Test accuracy:  {result_l2h2['test_acc']:.4f}")
    print()

    # Calculate chance level
    chance_level = 0.5  # Binary classification baseline
    print(f"Chance level: {chance_level:.4f}")
    print(f"Above chance: {result_l2h2['test_acc'] > chance_level}")
    print()

    # Probe residual streams for comparison
    print("="*60)
    print("SECONDARY COMPARISON: Residual Stream Outputs")
    print("="*60)

    residual_results = {}
    num_layers = model.num_layers

    for layer_idx in range(num_layers):
        print(f"\n--- Probing Layer {layer_idx} Residual Stream ---")
        try:
            result = probe_attention_head(
                model, train_data, test_data, tokenizer, device,
                layer_idx=layer_idx, head_idx=0, extract_residual=True
            )
            residual_results[f"L{layer_idx}-residual"] = result
            print(f"Train accuracy: {result['train_acc']:.4f}")
            print(f"Test accuracy:  {result['test_acc']:.4f}")
        except Exception as e:
            print(f"Error probing L{layer_idx} residual: {e}")

    # Find best residual layer
    if residual_results:
        best_residual = max(residual_results.items(), key=lambda x: x[1]['test_acc'])
        print()
        print("="*60)
        print("COMPARISON SUMMARY")
        print("="*60)
        print(f"\nL2-H2 (attention head):")
        print(f"  Train accuracy: {result_l2h2['train_acc']:.4f}")
        print(f"  Test accuracy:  {result_l2h2['test_acc']:.4f}")
        print(f"\nBest residual layer: {best_residual[0]}")
        print(f"  Train accuracy: {best_residual[1]['train_acc']:.4f}")
        print(f"  Test accuracy:  {best_residual[1]['test_acc']:.4f}")
        print()
        print(f"L2-H2 vs Best Residual: {result_l2h2['test_acc']:.4f} vs {best_residual[1]['test_acc']:.4f}")

    print()
    print("="*60)
    print("PROBING COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
