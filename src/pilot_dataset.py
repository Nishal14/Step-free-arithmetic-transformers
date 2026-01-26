"""
Dataset loader for pilot arithmetic expressions.
Handles the pilot format: {"expr": "...", "result": ..., "depth": [...]}
"""

import json
import torch
from pathlib import Path
from typing import Dict
from torch.utils.data import Dataset


class PilotMathDataset(Dataset):
    """Dataset for pilot arithmetic expressions."""

    def __init__(
        self,
        data_path: Path,
        tokenizer,
        max_seq_len: int = 32
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.truncation_count = 0

        # Load data
        self.data = []
        with open(data_path, 'r') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]

        # Format: "expr = result"
        # Example: "(1+2)*3 = 9"
        full_text = item["expr"] + " = " + str(item["result"])

        # Tokenize
        tokens = self.tokenizer.encode(full_text, add_special_tokens=True)

        # Truncate if too long
        if len(tokens) > self.max_seq_len:
            tokens = tokens[:self.max_seq_len]
            self.truncation_count += 1

        # Prepare input_ids and labels for next-token prediction
        input_ids = tokens[:-1]
        labels = tokens[1:]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long)
        }


def create_pilot_dataloaders(
    train_path: str,
    val_path: str,
    tokenizer,
    batch_size: int = 32,
    max_seq_len: int = 32,
    collate_fn=None
):
    """
    Create dataloaders for pilot dataset.

    Args:
        train_path: Path to training data
        val_path: Path to validation data
        tokenizer: Tokenizer instance
        batch_size: Batch size
        max_seq_len: Maximum sequence length
        collate_fn: Collate function

    Returns:
        train_loader, val_loader
    """
    from torch.utils.data import DataLoader

    train_dataset = PilotMathDataset(
        Path(train_path),
        tokenizer,
        max_seq_len=max_seq_len
    )

    val_dataset = PilotMathDataset(
        Path(val_path),
        tokenizer,
        max_seq_len=max_seq_len
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )

    return train_loader, val_loader, train_dataset, val_dataset
