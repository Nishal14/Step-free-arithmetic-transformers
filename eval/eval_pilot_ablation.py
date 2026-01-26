"""
Ablation evaluation script for pilot dataset.
Tests whether ablating individual attention heads selectively hurts
parenthesized expressions more than flat ones.

Usage:
    python eval/eval_pilot_ablation.py --checkpoint runs/pilot_test/checkpoint_best.pt
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import torch
from torch.utils.data import DataLoader

from src.model import create_model
from src.pilot_dataset import PilotMathDataset
from src.train import SimpleTokenizer, collate_fn


@torch.no_grad()
@torch.no_grad()
@torch.no_grad()
@torch.no_grad()
def evaluate(model, dataloader, device):
    """
    Pilot evaluation:
    Checks whether the model correctly predicts the FINAL answer token
    (last non-padding label).
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

            valid_positions = (label_seq != -100).nonzero(as_tuple=True)[0]
            last_pos = valid_positions[-1].item()

            if pred_seq[last_pos] == label_seq[last_pos]:
                correct += 1

            total += 1

    return correct / total if total > 0 else 0.0



def clear_ablation(model):
    """Remove ablation from all attention heads."""
    for block in model.blocks:
        if hasattr(block.attn, "ablate_head"):
            delattr(block.attn, "ablate_head")


def set_ablation(model, layer_idx, head_id):
    """
    Ablate a specific attention head.

    Args:
        model: CompactTransformer model
        layer_idx: Layer index (0-based)
        head_id: Head index within layer (0-based)
    """
    model.blocks[layer_idx].attn.ablate_head = head_id


def main():
    parser = argparse.ArgumentParser(description="Ablation evaluation for pilot dataset")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint file")
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"], help="Device")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--ablate-layer", type=int, default=None)
    parser.add_argument("--ablate-head", type=int, default=None)

    args = parser.parse_args()

    # Check CUDA availability
    if args.device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA not available, using CPU")
        args.device = "cpu"

    device = torch.device(args.device)

    print("="*60)
    print("Pilot Ablation Evaluation")
    print("="*60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {device}")
    print(f"Batch size: {args.batch_size}")
    print()

    # Initialize tokenizer
    tokenizer = SimpleTokenizer()

    # Load checkpoint
    print("Loading checkpoint...")
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device)

    # Get model config from checkpoint
    model_config = ckpt.get("model_config", {})
    if not model_config:
        raise ValueError("No model_config found in checkpoint")

    # Vocab size should already be in the checkpoint config, but verify
    model_config["vocab_size"] = tokenizer.vocab_size

    # Create model
    model = create_model(model_config)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    print(f"Model loaded: {model.count_parameters() / 1e6:.2f}M parameters")
    print(f"Layers: {model_config.get('num_layers', 'N/A')}")
    print(f"Heads per layer: {model_config.get('num_heads', 'N/A')}")
    print()

    # Load test datasets
    print("Loading test datasets...")
    test_flat_dataset = PilotMathDataset(
        Path("data/pilot_test_flat.jsonl"),
        tokenizer,
        max_seq_len=32
    )

    test_paren_dataset = PilotMathDataset(
        Path("data/pilot_test_paren.jsonl"),
        tokenizer,
        max_seq_len=32
    )

    test_flat_loader = DataLoader(
        test_flat_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, tokenizer.pad_token_id),
        num_workers=0
    )

    test_paren_loader = DataLoader(
        test_paren_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, tokenizer.pad_token_id),
        num_workers=0
    )

    print(f"Test flat size: {len(test_flat_dataset)}")
    print(f"Test paren size: {len(test_paren_dataset)}")
    print()

    # Baseline evaluation (no ablation)
    print("="*60)
    print("BASELINE (No Ablation)")
    print("="*60)

    # Apply targeted ablation if specified, otherwise clear all ablations
    if args.ablate_layer is not None and args.ablate_head is not None:
        print(f"Ablating Layer {args.ablate_layer} Head {args.ablate_head}")
        set_ablation(model, args.ablate_layer, args.ablate_head)
    else:
        clear_ablation(model)
    flat_baseline = evaluate(model, test_flat_loader, device)
    paren_baseline = evaluate(model, test_paren_loader, device)

    print(f"Flat accuracy:   {flat_baseline:.4f} ({flat_baseline*100:.1f}%)")
    print(f"Paren accuracy:  {paren_baseline:.4f} ({paren_baseline*100:.1f}%)")
    print(f"Gap (paren-flat): {paren_baseline - flat_baseline:+.4f}")
    print()

    # Head ablation experiments
    print("="*60)
    print("HEAD ABLATION EXPERIMENTS")
    print("="*60)
    print(f"{'Layer':<8} {'Head':<6} {'Flat Acc':<12} {'Paren Acc':<12} {'Gap':<12} {'Impact':<12}")
    print("-"*60)

    num_layers = model_config.get("num_layers", 4)
    num_heads = model_config.get("num_heads", 4)

    results = []

    for layer_idx in range(num_layers):
        for head_idx in range(num_heads):
            # Ablate this head
            clear_ablation(model)
            set_ablation(model, layer_idx, head_idx)

            # Evaluate
            flat_acc = evaluate(model, test_flat_loader, device)
            paren_acc = evaluate(model, test_paren_loader, device)

            # Compute metrics
            gap = paren_acc - flat_acc
            impact = (paren_baseline - paren_acc) - (flat_baseline - flat_acc)

            results.append({
                "layer": layer_idx,
                "head": head_idx,
                "flat_acc": flat_acc,
                "paren_acc": paren_acc,
                "gap": gap,
                "impact": impact
            })

            print(f"L{layer_idx}     H{head_idx}    {flat_acc:.4f}       {paren_acc:.4f}       {gap:+.4f}       {impact:+.4f}")

    # Clear ablation
    clear_ablation(model)

    print()
    print("="*60)
    print("SUMMARY")
    print("="*60)

    # Find heads with highest specialization (most impact on paren expressions)
    results_sorted = sorted(results, key=lambda x: x["impact"], reverse=True)

    print("\nTop 5 heads by specialization (paren impact - flat impact):")
    print(f"{'Rank':<6} {'Layer':<8} {'Head':<6} {'Impact':<12} {'Paren Acc':<12}")
    print("-"*50)
    for i, r in enumerate(results_sorted[:5], 1):
        print(f"{i:<6} L{r['layer']:<7} H{r['head']:<5} {r['impact']:+.4f}       {r['paren_acc']:.4f}")

    print("\nBottom 5 heads by specialization:")
    print(f"{'Rank':<6} {'Layer':<8} {'Head':<6} {'Impact':<12} {'Paren Acc':<12}")
    print("-"*50)
    for i, r in enumerate(results_sorted[-5:], len(results_sorted)-4):
        print(f"{i:<6} L{r['layer']:<7} H{r['head']:<5} {r['impact']:+.4f}       {r['paren_acc']:.4f}")

    print()
    print("="*60)
    print("Evaluation Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
