"""
OOD Depth-3 Generalization Experiment

Tests whether Layer 2, Head 2 remains selectively necessary
for expressions with parenthesis depth = 3 (unseen during training).

Training was on depth ≤ 2, this tests generalization to depth = 3.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader

from src.model import create_model
from src.pilot_dataset import PilotMathDataset
from src.train import SimpleTokenizer, collate_fn


@torch.no_grad()
def evaluate(model, dataloader, device):
    """
    Evaluate last-answer-token accuracy.
    Checks whether the model correctly predicts the FINAL answer token.
    """
    model.eval()
    correct = 0
    total = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids)
        logits = outputs[0]
        preds = torch.argmax(logits, dim=-1)

        batch_size = labels.size(0)
        for i in range(batch_size):
            label_seq = labels[i]
            pred_seq = preds[i]

            # Find last valid position
            valid_positions = (label_seq != -100).nonzero(as_tuple=True)[0]
            last_pos = valid_positions[-1].item()

            # Check if prediction matches label at last position
            if pred_seq[last_pos] == label_seq[last_pos]:
                correct += 1

            total += 1

    return correct / total if total > 0 else 0.0


def clear_ablation(model):
    """Remove ablation from all attention heads."""
    for block in model.blocks:
        if hasattr(block.attn, "ablate_head"):
            delattr(block.attn, "ablate_head")


def set_ablation(model, layer_idx, head_idx):
    """
    Ablate a specific attention head.

    Args:
        model: CompactTransformer model
        layer_idx: Layer index (0-based)
        head_idx: Head index (0-based)
    """
    model.blocks[layer_idx].attn.ablate_head = head_idx


def main():
    """Run OOD depth-3 generalization experiment."""

    # Configuration
    checkpoint_path = Path("runs/pilot/checkpoint_best.pt")
    test_data_path = Path("data/pilot_test_depth3.jsonl")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 32

    print("="*60)
    print("OOD Depth-3 Generalization Experiment")
    print("="*60)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Test data: {test_data_path}")
    print(f"Device: {device}")
    print()

    # Initialize tokenizer
    tokenizer = SimpleTokenizer()

    # Load checkpoint
    print("Loading checkpoint...")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device)

    # Get model config
    model_config = ckpt.get("model_config", {})
    if not model_config:
        raise ValueError("No model_config found in checkpoint")

    model_config["vocab_size"] = tokenizer.vocab_size

    # Create and load model
    model = create_model(model_config)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    print(f"Model loaded: {model.count_parameters() / 1e6:.2f}M parameters")
    print(f"Layers: {model_config.get('num_layers', 'N/A')}")
    print(f"Heads per layer: {model_config.get('num_heads', 'N/A')}")
    print()

    # Load depth-3 test dataset
    print("Loading depth-3 test dataset...")
    if not test_data_path.exists():
        raise FileNotFoundError(f"Test data not found: {test_data_path}")

    test_dataset = PilotMathDataset(
        test_data_path,
        tokenizer,
        max_seq_len=32
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, tokenizer.pad_token_id),
        num_workers=0
    )

    print(f"Test set size: {len(test_dataset)} examples")
    print()

    # (A) Baseline evaluation (no ablation)
    print("="*60)
    print("(A) BASELINE: No ablation")
    print("="*60)

    clear_ablation(model)
    baseline_acc = evaluate(model, test_loader, device)

    print(f"Accuracy: {baseline_acc:.4f} ({baseline_acc*100:.1f}%)")
    print()

    # (B) Layer 2, Head 2 ablation
    print("="*60)
    print("(B) TARGETED ABLATION: Layer 2, Head 2")
    print("="*60)

    clear_ablation(model)
    set_ablation(model, layer_idx=2, head_idx=2)
    ablated_acc = evaluate(model, test_loader, device)

    print(f"Accuracy: {ablated_acc:.4f} ({ablated_acc*100:.1f}%)")
    print()

    # Clear ablation
    clear_ablation(model)

    # Report results
    print()
    print("="*60)
    print("OOD Depth-3 Results")
    print("="*60)
    print(f"Baseline accuracy: {baseline_acc*100:.1f}%")
    print(f"L2-H2 ablated accuracy: {ablated_acc*100:.1f}%")
    print(f"Accuracy drop: {(baseline_acc - ablated_acc)*100:.1f}%")
    print("="*60)


if __name__ == "__main__":
    main()
