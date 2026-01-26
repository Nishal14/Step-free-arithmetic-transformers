"""
Training script for pilot arithmetic dataset.
Simplified version of src/train.py for quick experiments.
"""

import argparse
from pathlib import Path

# Import training utilities
from src.train import (
    set_seed,
    SimpleTokenizer,
    collate_fn,
    train_epoch,
    evaluate_model,
    save_checkpoint,
    get_linear_schedule_with_warmup
)
from src.model import create_model
from src.pilot_dataset import create_pilot_dataloaders

import yaml
from omegaconf import OmegaConf
import torch


def train_pilot(config_path: Path, output_dir: Path, seed: int, device: str):
    """Train on pilot dataset."""

    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print("="*60)
    print("Pilot Dataset Training")
    print("="*60)
    print()
    print("Configuration:")
    print(OmegaConf.to_yaml(OmegaConf.create(config)))

    # Set seed
    set_seed(seed)

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

    # Load datasets
    data_config = config.get("data", {})
    train_loader, val_loader, train_dataset, val_dataset = create_pilot_dataloaders(
        train_path=data_config["train_path"],
        val_path=data_config["val_path"],
        tokenizer=tokenizer,
        batch_size=config.get("training", {}).get("batch_size", 32),
        max_seq_len=data_config.get("max_seq_len", 32),
        collate_fn=lambda batch: collate_fn(batch, tokenizer.pad_token_id)
    )

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")

    # Report truncation
    if train_dataset.truncation_count > 0 or val_dataset.truncation_count > 0:
        print(f"Warning: Truncation detected - Train: {train_dataset.truncation_count}, Val: {val_dataset.truncation_count}")

    # Setup optimizer
    train_config = config.get("training", {})
    optimizer_config = train_config.get("optimizer", {})
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=optimizer_config.get("lr", 3e-4),
        weight_decay=optimizer_config.get("weight_decay", 0.01),
        betas=(0.9, 0.95)
    )

    # Setup scheduler
    num_epochs = train_config.get("num_epochs", 50)
    num_training_steps = len(train_loader) * num_epochs
    num_warmup_steps = optimizer_config.get("warmup_steps", 100)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    # Training loop
    best_val_loss = float('inf')
    log_interval = config.get("logging", {}).get("log_interval", 25)
    eval_interval = train_config.get("eval_interval", 5)
    save_interval = train_config.get("save_interval", 10)

    print()
    print("="*60)
    print("Starting Training")
    print("="*60)

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        print("-" * 40)

        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler,
            device, epoch, log_interval, use_wandb=False
        )

        print(f"Train Loss: {train_metrics['loss']:.4f} | "
              f"Train Perplexity: {train_metrics['perplexity']:.2f} | "
              f"Time: {train_metrics['time']:.2f}s")

        # Evaluate
        if epoch % eval_interval == 0:
            val_metrics = evaluate_model(model, val_loader, device)
            print(f"Val Loss: {val_metrics['loss']:.4f} | "
                  f"Val Perplexity: {val_metrics['perplexity']:.2f}")

            # Check if best model
            is_best = val_metrics['loss'] < best_val_loss
            if is_best:
                best_val_loss = val_metrics['loss']
                print("New best model!")

            # Save checkpoint
            if epoch % save_interval == 0 or is_best:
                save_checkpoint(
                    model, optimizer, scheduler,
                    epoch, epoch * len(train_loader),
                    config, output_dir, is_best
                )

    print()
    print("="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Best validation loss: {best_val_loss:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train on pilot arithmetic dataset")
    parser.add_argument("--config", type=str, default="configs/pilot.yaml", help="Path to config YAML")
    parser.add_argument("--output-dir", type=str, default="runs/pilot", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"], help="Device")

    args = parser.parse_args()

    # Check CUDA availability
    if args.device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA not available, using CPU")
        args.device = "cpu"

    config_path = Path(args.config)
    output_dir = Path(args.output_dir)

    train_pilot(config_path, output_dir, args.seed, args.device)


if __name__ == "__main__":
    main()
