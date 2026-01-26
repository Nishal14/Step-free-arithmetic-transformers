"""
Evaluation script for trained transformer models.
Computes accuracy, perplexity, and task-specific metrics.
"""

import argparse
import json
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from tqdm import tqdm
import numpy as np

from src.model import CompactTransformer, create_model


class SimpleTokenizer:
    """Simple character-level tokenizer for mathematical expressions."""

    def __init__(self):
        # Define vocabulary
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
        """Encode text to token IDs."""
        tokens = []
        if add_special_tokens:
            tokens.append(self.bos_token_id)

        for ch in text:
            tokens.append(self.char_to_idx.get(ch, self.unk_token_id))

        if add_special_tokens:
            tokens.append(self.eos_token_id)

        return tokens

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs to text."""
        text = []
        for idx in token_ids:
            if skip_special_tokens and idx in [self.pad_token_id, self.bos_token_id, self.eos_token_id]:
                continue
            text.append(self.idx_to_char.get(idx, "<UNK>"))
        return "".join(text)


def load_checkpoint(checkpoint_path: Path, device: str = "cpu") -> Tuple[CompactTransformer, Dict]:
    """Load model checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model_config = checkpoint.get("model_config", {})
    model = create_model(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model, checkpoint


def load_dataset(data_path: Path) -> List[Dict[str, Any]]:
    """Load JSONL dataset."""
    data = []
    with open(data_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def compute_perplexity(
    model: CompactTransformer,
    dataset: List[Dict[str, Any]],
    tokenizer: SimpleTokenizer,
    device: str,
    batch_size: int = 32,
    max_seq_len: int = 512
) -> float:
    """Compute perplexity on dataset."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for i in tqdm(range(0, len(dataset), batch_size), desc="Computing perplexity"):
            batch = dataset[i:i + batch_size]

            # Prepare batch
            input_ids_list = []
            labels_list = []

            for item in batch:
                # Concatenate input and output
                full_text = item["input"] + " = " + item["gold"]
                tokens = tokenizer.encode(full_text, add_special_tokens=True)

                # Truncate if needed
                if len(tokens) > max_seq_len:
                    tokens = tokens[:max_seq_len]

                input_ids_list.append(tokens[:-1])  # Input: all but last token
                labels_list.append(tokens[1:])      # Labels: all but first token

            # Pad sequences
            max_len = max(len(seq) for seq in input_ids_list)
            input_ids_padded = []
            labels_padded = []

            for input_ids, labels in zip(input_ids_list, labels_list):
                pad_len = max_len - len(input_ids)
                input_ids_padded.append(input_ids + [tokenizer.pad_token_id] * pad_len)
                labels_padded.append(labels + [-100] * pad_len)  # -100 is ignored in loss

            # Convert to tensors
            input_ids = torch.tensor(input_ids_padded, dtype=torch.long, device=device)
            labels = torch.tensor(labels_padded, dtype=torch.long, device=device)

            # Forward pass
            logits, loss = model(input_ids, labels=labels)

            # Accumulate loss
            num_tokens = (labels != -100).sum().item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

    perplexity = np.exp(total_loss / total_tokens)
    return perplexity


def evaluate_accuracy(
    model: CompactTransformer,
    dataset: List[Dict[str, Any]],
    tokenizer: SimpleTokenizer,
    device: str,
    max_gen_length: int = 100
) -> Dict[str, float]:
    """Evaluate exact match accuracy."""
    model.eval()
    correct = 0
    total = 0

    predictions = []
    targets = []

    with torch.no_grad():
        for item in tqdm(dataset, desc="Evaluating accuracy"):
            input_text = item["input"] + " = "
            target = item["gold"]

            # Encode input
            input_ids = tokenizer.encode(input_text, add_special_tokens=True)
            input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

            # Generate output
            output_ids = model.generate(
                input_tensor,
                max_length=len(input_ids) + max_gen_length,
                temperature=0.1,  # Low temperature for more deterministic output
                eos_token_id=tokenizer.eos_token_id
            )

            # Decode prediction
            predicted_text = tokenizer.decode(output_ids[0].tolist(), skip_special_tokens=True)

            # Extract answer (after "=")
            if " = " in predicted_text:
                predicted_answer = predicted_text.split(" = ")[-1].strip()
            else:
                predicted_answer = predicted_text[len(input_text):].strip()

            predictions.append(predicted_answer)
            targets.append(target)

            # Check if correct
            if predicted_answer == target:
                correct += 1
            total += 1

    accuracy = correct / total if total > 0 else 0.0

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "predictions": predictions[:10],  # Sample predictions
        "targets": targets[:10]
    }


def evaluate(
    checkpoint_path: Path,
    data_path: Path,
    device: str = "cpu",
    batch_size: int = 32,
    max_seq_len: int = 512,
    max_gen_length: int = 100,
    output_path: Optional[Path] = None
):
    """Run full evaluation."""
    print(f"Loading checkpoint from {checkpoint_path}")
    model, checkpoint = load_checkpoint(checkpoint_path, device)

    print(f"Model parameters: {model.count_parameters() / 1e6:.2f}M")

    print(f"Loading dataset from {data_path}")
    dataset = load_dataset(data_path)
    print(f"Dataset size: {len(dataset)}")

    # Initialize tokenizer
    tokenizer = SimpleTokenizer()

    # Compute metrics
    print("\nComputing perplexity...")
    perplexity = compute_perplexity(model, dataset, tokenizer, device, batch_size, max_seq_len)

    print("\nEvaluating accuracy...")
    accuracy_metrics = evaluate_accuracy(model, dataset, tokenizer, device, max_gen_length)

    # Compile results
    results = {
        "checkpoint": str(checkpoint_path),
        "data_path": str(data_path),
        "dataset_size": len(dataset),
        "model_parameters": model.count_parameters(),
        "perplexity": perplexity,
        **accuracy_metrics
    }

    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Perplexity: {perplexity:.4f}")
    print(f"Accuracy: {accuracy_metrics['accuracy']:.4f} ({accuracy_metrics['correct']}/{accuracy_metrics['total']})")
    print("\nSample predictions:")
    for i, (pred, target) in enumerate(zip(accuracy_metrics['predictions'], accuracy_metrics['targets'])):
        status = '[CORRECT]' if pred == target else '[WRONG]'
        print(f"  {i+1}. Predicted: {pred} | Target: {target} | {status}")

    # Save results
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained transformer model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--data", type=str, required=True, help="Path to evaluation data (JSONL)")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device to use")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation")
    parser.add_argument("--max_seq_len", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--max_gen_length", type=int, default=100, help="Maximum generation length")
    parser.add_argument("--output", type=str, default=None, help="Path to save results JSON")

    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    data_path = Path(args.data)
    output_path = Path(args.output) if args.output else None

    evaluate(
        checkpoint_path=checkpoint_path,
        data_path=data_path,
        device=args.device,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        max_gen_length=args.max_gen_length,
        output_path=output_path
    )


if __name__ == "__main__":
    main()
