"""
Training script for compact transformer on synthetic math datasets.
Supports checkpointing, logging to W&B, and periodic evaluation.
"""

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import yaml
from omegaconf import OmegaConf

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available, logging disabled")

from src.model import CompactTransformer, create_model
from src.pilot_dataset import PilotMathDataset


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


class MathDataset(Dataset):
    """Dataset for mathematical reasoning tasks."""

    def __init__(
        self,
        data_path: Path,
        tokenizer: SimpleTokenizer,
        max_seq_len: int = 512,
        use_steps: bool = False,
        step_masking_mode: str = "mask_steps"
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.use_steps = use_steps
        self.step_masking_mode = step_masking_mode
        self.truncation_count = 0  # Track how many samples are truncated

        # Validate step_masking_mode
        valid_modes = ["none", "mask_steps", "mask_answer"]
        if step_masking_mode not in valid_modes:
            raise ValueError(f"step_masking_mode must be one of {valid_modes}, got {step_masking_mode}")

        # Load data
        self.data = []
        with open(data_path, 'r') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]

        # Format input: "input = output"
        if self.use_steps and "steps" in item:
            # Include reasoning steps: "input = step1 ; step2 ; ... ; final_answer"
            # The final answer is the last element after the last " ; "
            steps_text = " ; ".join(item["steps"])
            full_text = item["input"] + " = " + steps_text
            answer_start_marker = " ; " + item["steps"][-1] if len(item["steps"]) > 0 else ""
        else:
            full_text = item["input"] + " = " + item["gold"]
            answer_start_marker = None

        # Tokenize
        tokens = self.tokenizer.encode(full_text, add_special_tokens=True)

        # Find where the final answer begins (for step masking)
        answer_start_idx = None
        if self.use_steps and answer_start_marker and self.step_masking_mode != "none":
            # Tokenize just the answer portion to find its start
            input_part = item["input"] + " = " + " ; ".join(item["steps"][:-1]) + " ; "
            input_tokens = self.tokenizer.encode(input_part, add_special_tokens=True)
            answer_start_idx = len(input_tokens) - 1  # -1 for BOS token handling

        # Truncate if too long
        if len(tokens) > self.max_seq_len:
            tokens = tokens[:self.max_seq_len]
            self.truncation_count += 1

        # Prepare input_ids and labels for next-token prediction
        input_ids = tokens[:-1]
        labels = tokens[1:]

        # Apply step masking if requested
        if self.use_steps and self.step_masking_mode != "none" and answer_start_idx is not None:
            labels = list(labels)  # Convert to list for mutation
            if self.step_masking_mode == "mask_steps":
                # Mask intermediate steps, keep only final answer
                for i in range(min(answer_start_idx, len(labels))):
                    labels[i] = -100
            elif self.step_masking_mode == "mask_answer":
                # Mask final answer, keep only intermediate steps
                for i in range(answer_start_idx, len(labels)):
                    labels[i] = -100

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long)
        }


def collate_fn(batch: List[Dict[str, torch.Tensor]], pad_token_id: int) -> Dict[str, torch.Tensor]:
    """Collate function to pad sequences in a batch."""
    # Find max length in batch
    max_len = max(item["input_ids"].shape[0] for item in batch)

    # Pad sequences
    input_ids = []
    labels = []
    attention_masks = []

    for item in batch:
        seq_len = item["input_ids"].shape[0]
        pad_len = max_len - seq_len

        # Pad input_ids
        padded_input = torch.cat([
            item["input_ids"],
            torch.full((pad_len,), pad_token_id, dtype=torch.long)
        ])
        input_ids.append(padded_input)

        # Pad labels (use -100 for padding, which is ignored in loss)
        padded_labels = torch.cat([
            item["labels"],
            torch.full((pad_len,), -100, dtype=torch.long)
        ])
        labels.append(padded_labels)

        # Create attention mask: True for real tokens, False for padding
        attention_mask = torch.cat([
            torch.ones(seq_len, dtype=torch.bool),
            torch.zeros(pad_len, dtype=torch.bool)
        ])
        attention_masks.append(attention_mask)

    return {
        "input_ids": torch.stack(input_ids),
        "labels": torch.stack(labels),
        "attention_mask": torch.stack(attention_masks)
    }


def get_linear_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int
):
    """Learning rate scheduler with linear warmup and decay."""
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0,
            float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_epoch(
    model: CompactTransformer,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.LambdaLR],
    device: str,
    epoch: int,
    log_interval: int = 100,
    use_wandb: bool = False,
    use_fp16: bool = False,
    scaler: Optional[torch.cuda.amp.GradScaler] = None
) -> Dict[str, float]:
    """Train for one epoch with optional FP16 mixed precision."""
    model.train()
    total_loss = 0.0
    total_tokens = 0
    start_time = time.time()

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for step, batch in enumerate(pbar):
        # Move batch to device
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        optimizer.zero_grad()

        # Forward pass with optional FP16
        if use_fp16 and scaler is not None:
            with torch.cuda.amp.autocast():
                logits, loss, _ = model(input_ids, attention_mask=attention_mask, labels=labels)

            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard FP32 training
            logits, loss, _ = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        # Update metrics
        num_tokens = (labels != -100).sum().item()
        total_loss += loss.item() * num_tokens
        total_tokens += num_tokens

        # Log
        if step % log_interval == 0:
            avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
            perplexity = np.exp(avg_loss)
            lr = optimizer.param_groups[0]['lr']

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'ppl': f'{perplexity:.2f}',
                'lr': f'{lr:.2e}'
            })

            if use_wandb and WANDB_AVAILABLE:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/perplexity": perplexity,
                    "train/lr": lr,
                    "train/step": epoch * len(dataloader) + step
                })

    epoch_time = time.time() - start_time
    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0

    return {
        "loss": avg_loss,
        "perplexity": np.exp(avg_loss),
        "time": epoch_time
    }


@torch.no_grad()
def evaluate_model(
    model: CompactTransformer,
    dataloader: DataLoader,
    device: str,
    use_fp16: bool = False
) -> Dict[str, float]:
    """Evaluate model on validation set with optional FP16."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for batch in tqdm(dataloader, desc="Evaluating"):
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        if use_fp16:
            with torch.cuda.amp.autocast():
                logits, loss, _ = model(input_ids, attention_mask=attention_mask, labels=labels)
        else:
            logits, loss, _ = model(input_ids, attention_mask=attention_mask, labels=labels)

        num_tokens = (labels != -100).sum().item()
        total_loss += loss.item() * num_tokens
        total_tokens += num_tokens

    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0

    return {
        "loss": avg_loss,
        "perplexity": np.exp(avg_loss)
    }


def save_checkpoint(
    model: CompactTransformer,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.LambdaLR],
    epoch: int,
    step: int,
    config: Dict,
    output_dir: Path,
    is_best: bool = False
):
    """Save model checkpoint."""
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "model_config": config.get("model", {}),
        "train_config": config
    }

    # Save regular checkpoint
    checkpoint_path = output_dir / f"checkpoint_epoch_{epoch}.pt"
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")

    # Save best checkpoint
    if is_best:
        best_path = output_dir / "checkpoint_best.pt"
        torch.save(checkpoint, best_path)
        print(f"Saved best checkpoint to {best_path}")

    # Save latest checkpoint
    latest_path = output_dir / "checkpoint_latest.pt"
    torch.save(checkpoint, latest_path)


def train(config_path: Path, output_dir: Path, seed: int, device: str, use_fp16: bool = False):
    """Main training function with optional FP16 mixed precision."""
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print("Configuration:")
    print(OmegaConf.to_yaml(OmegaConf.create(config)))

    # GPU monitoring
    if device == "cuda":
        print("\n" + "="*60)
        print("GPU INFORMATION")
        print("="*60)
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"FP16 Mixed Precision: {'ENABLED' if use_fp16 else 'DISABLED'}")
        print("="*60 + "\n")

    # Set seed
    set_seed(seed)

    # Initialize W&B
    use_wandb = config.get("logging", {}).get("use_wandb", False) and WANDB_AVAILABLE
    if use_wandb:
        wandb.init(
            project=config.get("logging", {}).get("wandb_project", "math-compact"),
            config=config,
            name=f"{config.get('model', {}).get('name', 'transformer')}_seed{seed}"
        )

    # Initialize tokenizer
    tokenizer = SimpleTokenizer()
    print(f"Vocabulary size: {tokenizer.vocab_size}")

    # Update model config with vocab size
    model_config = config.get("model", {})
    model_config["vocab_size"] = tokenizer.vocab_size

    # Create model
    model = create_model(model_config)
    model = model.to(device)
    print(f"Model parameters: {model.count_parameters() / 1e6:.2f}M")

    # Initialize gradient scaler for FP16
    scaler = torch.amp.GradScaler('cuda') if use_fp16 else None

    # Load datasets
    data_config = config.get("data", {})
    train_path = Path(data_config["train_path"])
    val_path = Path(data_config["val_path"])

    # Detect if using pilot dataset (simpler format)
    is_pilot = "pilot" in str(train_path).lower()

    if is_pilot:
        print("Detected pilot dataset format")
        train_dataset = PilotMathDataset(
            train_path,
            tokenizer,
            max_seq_len=data_config.get("max_seq_len", 512)
        )
        val_dataset = PilotMathDataset(
            val_path,
            tokenizer,
            max_seq_len=data_config.get("max_seq_len", 512)
        )
    else:
        train_dataset = MathDataset(
            train_path,
            tokenizer,
            max_seq_len=data_config.get("max_seq_len", 512),
            use_steps=data_config.get("use_steps", False),
            step_masking_mode=data_config.get("step_masking_mode", "mask_steps")
        )
        val_dataset = MathDataset(
            val_path,
            tokenizer,
            max_seq_len=data_config.get("max_seq_len", 512),
            use_steps=data_config.get("use_steps", False),
            step_masking_mode=data_config.get("step_masking_mode", "mask_steps")
        )

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")

    # Report truncation statistics after first epoch of iteration
    # (truncation_count increments during __getitem__ calls)
    # We'll report after dataloaders are created by doing a quick count
    temp_trunc_train = train_dataset.truncation_count
    temp_trunc_val = val_dataset.truncation_count
    if temp_trunc_train > 0 or temp_trunc_val > 0:
        print(f"Warning: Truncation detected - Train: {temp_trunc_train}, Val: {temp_trunc_val}")

    # Create dataloaders
    train_config = config.get("training", {})
    batch_size = train_config.get("batch_size", 32)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, tokenizer.pad_token_id),
        num_workers=0  # Windows compatibility
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, tokenizer.pad_token_id),
        num_workers=0
    )

    # Setup optimizer
    optimizer_config = train_config.get("optimizer", {})
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=optimizer_config.get("lr", 1e-4),
        weight_decay=optimizer_config.get("weight_decay", 0.01),
        betas=(0.9, 0.95)
    )

    # Setup scheduler
    num_epochs = train_config.get("num_epochs", 10)
    num_training_steps = len(train_loader) * num_epochs
    num_warmup_steps = optimizer_config.get("warmup_steps", num_training_steps // 10)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    # Training loop
    best_val_loss = float('inf')
    best_val_ppl = float('inf')
    best_val_epoch = None
    log_interval = config.get("logging", {}).get("log_interval", 100)
    eval_interval = train_config.get("eval_interval", 1)
    save_interval = train_config.get("save_interval", 1)

    metrics_history = []

    for epoch in range(1, num_epochs + 1):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch}/{num_epochs}")
        print(f"{'='*50}")

        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler,
            device, epoch, log_interval, use_wandb, use_fp16, scaler
        )

        print(f"\nTrain Loss: {train_metrics['loss']:.4f} | "
              f"Train Perplexity: {train_metrics['perplexity']:.2f} | "
              f"Time: {train_metrics['time']:.2f}s")

        # Evaluate
        if epoch % eval_interval == 0:
            val_metrics = evaluate_model(model, val_loader, device, use_fp16)
            print(f"Val Loss: {val_metrics['loss']:.4f} | "
                  f"Val Perplexity: {val_metrics['perplexity']:.2f}")

            if use_wandb:
                wandb.log({
                    "val/loss": val_metrics['loss'],
                    "val/perplexity": val_metrics['perplexity'],
                    "epoch": epoch
                })

            # Save metrics
            metrics_history.append({
                "epoch": epoch,
                "train_loss": train_metrics['loss'],
                "train_perplexity": train_metrics['perplexity'],
                "val_loss": val_metrics['loss'],
                "val_perplexity": val_metrics['perplexity']
            })

            # Check if best model
            is_best = val_metrics['loss'] < best_val_loss
            if is_best:
                best_val_loss = val_metrics['loss']
                best_val_ppl = val_metrics['perplexity']
                best_val_epoch = epoch
                print("New best model!")

            # Save checkpoint
            if epoch % save_interval == 0:
                save_checkpoint(
                    model, optimizer, scheduler,
                    epoch, epoch * len(train_loader),
                    config, output_dir, is_best
                )

    # Save final metrics
    metrics_dir = Path("metrics")
    metrics_dir.mkdir(exist_ok=True)
    metrics_file = metrics_dir / f"training_metrics_seed{seed}.json"

    with open(metrics_file, 'w') as f:
        json.dump({
            "config": config,
            "seed": seed,
            "metrics": metrics_history,
            "best_val_loss": best_val_loss,
            "best_val_perplexity": best_val_ppl,
            "best_val_epoch": best_val_epoch
        }, f, indent=2)

    print(f"\nMetrics saved to {metrics_file}")

    if use_wandb:
        wandb.finish()

    # Print final summary
    print("\n" + "="*50)
    print("TRAINING SUMMARY")
    print("="*50)
    if best_val_epoch is not None:
        print(f"Best validation loss:        {best_val_loss:.4f}")
        print(f"Best validation perplexity:  {best_val_ppl:.2f}")
        print(f"Achieved at epoch:           {best_val_epoch}")
    else:
        print("No validation metrics available")
    print("="*50)

    print("\nTraining complete!")


def main():
    parser = argparse.ArgumentParser(description="Train compact transformer")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for checkpoints")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device to use")
    parser.add_argument("--fp16", action="store_true", help="Use FP16 mixed precision training (GPU only)")
    parser.add_argument("--gpu-only", action="store_true", help="Enforce GPU-only training (hard fail if CUDA unavailable)")

    args = parser.parse_args()

    # GPU-only enforcement for scaling experiments
    if args.gpu_only:
        if args.device != "cuda":
            raise RuntimeError(
                "GPU-only mode requires --device cuda. "
                "Aborting to prevent accidental CPU training."
            )
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA requested but not available. "
                "GPU-only mode cannot fall back to CPU. "
                "Aborting."
            )
        print("\n" + "="*60)
        print("GPU-ONLY MODE ENABLED")
        print("CPU training is disabled. Will abort if GPU unavailable.")
        print("="*60 + "\n")

    # Standard CUDA availability check (with optional fallback)
    elif args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = "cpu"

    # FP16 requires GPU
    if args.fp16 and args.device != "cuda":
        raise RuntimeError("FP16 mixed precision requires CUDA device")

    config_path = Path(args.config)
    output_dir = Path(args.output_dir)

    train(config_path, output_dir, args.seed, args.device, args.fp16)


if __name__ == "__main__":
    main()
