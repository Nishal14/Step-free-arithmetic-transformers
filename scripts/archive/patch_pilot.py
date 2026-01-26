"""
Activation patching experiment for pilot model.
Tests causal sufficiency of Layer 2, Head 2 for parenthesis handling.

Usage:
    .venv/Scripts/python.exe patch_pilot.py --checkpoint runs/pilot_test/checkpoint_best.pt
"""

import argparse
import torch
from pathlib import Path
import json

from src.model import create_model
from src.train import SimpleTokenizer


def clear_patching(model):
    """Remove patching from all attention modules."""
    for block in model.blocks:
        if hasattr(block.attn, "patch_head_index"):
            block.attn.patch_head_index = None
        if hasattr(block.attn, "patch_head_output"):
            block.attn.patch_head_output = None


def predict_final_token(model, input_ids, device):
    """
    Get model's prediction for the final answer token.

    Returns:
        predicted_token: int
        logits: tensor for analysis
    """
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids.to(device))
        logits = outputs[0]  # (batch, seq_len, vocab_size)

        # Get prediction for last position
        last_logits = logits[0, -1, :]  # (vocab_size,)
        predicted_token = torch.argmax(last_logits).item()

        return predicted_token, logits


def find_examples(test_path, tokenizer, model, device):
    """
    Find one correct and one incorrect example from test set.
    Prioritize examples with similar sequence lengths for clean patching.

    Returns:
        good_example: dict with 'expr', 'result', 'input_ids', 'true_token'
        bad_example: dict with 'expr', 'result', 'input_ids', 'true_token'
    """
    examples = []
    with open(test_path, 'r') as f:
        for line in f:
            examples.append(json.loads(line))

    # First pass: collect all examples with predictions
    good_examples = []
    bad_examples = []

    for example in examples:
        # Format and tokenize
        full_text = example["expr"] + " = " + str(example["result"])
        tokens = tokenizer.encode(full_text, add_special_tokens=True)

        # Get input_ids (all but last) and true final token
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long).unsqueeze(0)  # (1, seq_len)
        true_final_token = tokens[-1]

        # Get prediction
        pred_token, _ = predict_final_token(model, input_ids, device)

        # Check if correct
        is_correct = (pred_token == true_final_token)

        example_data = {
            "expr": example["expr"],
            "result": example["result"],
            "input_ids": input_ids,
            "true_token": true_final_token,
            "depth": example.get("depth", []),
            "seq_len": input_ids.size(1)
        }

        if is_correct:
            good_examples.append(example_data)
        else:
            bad_examples.append(example_data)

    # Second pass: find pair with matching sequence lengths
    good_example = None
    bad_example = None

    for good in good_examples:
        for bad in bad_examples:
            if good["seq_len"] == bad["seq_len"]:
                good_example = good
                bad_example = bad
                break
        if good_example:
            break

    # Fallback: just pick first of each if no exact match
    if not good_example and good_examples:
        good_example = good_examples[0]
    if not bad_example and bad_examples:
        bad_example = bad_examples[0]

    return good_example, bad_example


def main():
    parser = argparse.ArgumentParser(description="Activation patching for pilot model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--test-path", type=str, default="data/pilot_test_paren.jsonl")

    args = parser.parse_args()

    # Check CUDA
    if args.device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA not available, using CPU")
        args.device = "cpu"

    device = torch.device(args.device)

    print("="*60)
    print("Activation Patching Experiment")
    print("="*60)
    print(f"Target: Layer 2, Head 2")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {device}")
    print()

    # Load model
    tokenizer = SimpleTokenizer()
    checkpoint_path = Path(args.checkpoint)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device)
    model_config = ckpt.get("model_config", {})
    if not model_config:
        raise ValueError("No model_config in checkpoint")

    model = create_model(model_config)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    print(f"Model loaded: {model.count_parameters() / 1e6:.2f}M parameters")
    print()

    # Find examples
    print("Finding examples...")
    good_example, bad_example = find_examples(args.test_path, tokenizer, model, device)

    if not good_example:
        print("ERROR: Could not find a correctly predicted example")
        return
    if not bad_example:
        print("ERROR: Could not find an incorrectly predicted example")
        return

    print(f"Good example: {good_example['expr']} = {good_example['result']} (seq_len: {good_example['seq_len']})")
    print(f"Bad example:  {bad_example['expr']} = {bad_example['result']} (seq_len: {bad_example['seq_len']})")
    if good_example['seq_len'] != bad_example['seq_len']:
        print(f"WARNING: Sequence lengths differ. Will truncate activation to shorter length.")
    print()

    # ========================================================================
    # EXPERIMENT 1: Main patching test
    # ========================================================================
    print("="*60)
    print("EXPERIMENT 1: Patch L2-H2 from correct to incorrect example")
    print("="*60)

    # Step 1: Run correct example and save L2-H2 activation
    clear_patching(model)

    with torch.no_grad():
        outputs = model(good_example["input_ids"].to(device))
        logits_good = outputs[0]

        # Extract L2-H2 activation: last_head_output has shape (batch, seq_len, num_heads, head_dim)
        good_state = model.blocks[2].attn.last_head_output[:, :, 2, :].clone()
        # good_state shape: (1, seq_len, head_dim)

    pred_good, _ = predict_final_token(model, good_example["input_ids"], device)
    print(f"Good example prediction: {pred_good} (true: {good_example['true_token']}) [OK]")

    # Step 2: Run bad example WITHOUT patch
    clear_patching(model)
    pred_bad_nopatch, _ = predict_final_token(model, bad_example["input_ids"], device)
    print(f"Bad example (no patch): {pred_bad_nopatch} (true: {bad_example['true_token']}) [WRONG]")

    # Step 3: Run bad example WITH patch
    attn = model.blocks[2].attn
    attn.patch_head_index = 2

    # Handle sequence length mismatch by truncating to shorter length
    target_seq_len = bad_example["input_ids"].size(1)
    if good_state.size(1) != target_seq_len:
        patch_state = good_state[:, :target_seq_len, :]
    else:
        patch_state = good_state

    attn.patch_head_output = patch_state

    pred_bad_patched, _ = predict_final_token(model, bad_example["input_ids"], device)

    # Clear patch immediately
    attn.patch_head_index = None
    attn.patch_head_output = None

    print(f"Bad example (patched):  {pred_bad_patched} (true: {bad_example['true_token']}) ", end="")
    if pred_bad_patched == bad_example["true_token"]:
        print("[FIXED!]")
    elif pred_bad_patched != pred_bad_nopatch:
        print("[Changed but still wrong]")
    else:
        print("[No change]")

    print()

    # ========================================================================
    # CONTROL 1: Patch wrong head (L2-H1 instead of L2-H2)
    # ========================================================================
    print("="*60)
    print("CONTROL 1: Patch different head (L2-H1 instead of L2-H2)")
    print("="*60)

    # Extract L2-H1 from good example
    with torch.no_grad():
        outputs = model(good_example["input_ids"].to(device))
        control_state = model.blocks[2].attn.last_head_output[:, :, 1, :].clone()

    # Patch L2-H1 instead
    attn = model.blocks[2].attn
    attn.patch_head_index = 1  # Different head

    # Handle sequence length mismatch
    target_seq_len = bad_example["input_ids"].size(1)
    if control_state.size(1) != target_seq_len:
        patch_state = control_state[:, :target_seq_len, :]
    else:
        patch_state = control_state

    attn.patch_head_output = patch_state

    pred_control1, _ = predict_final_token(model, bad_example["input_ids"], device)

    attn.patch_head_index = None
    attn.patch_head_output = None

    print(f"Bad example (L2-H1 patch): {pred_control1} (true: {bad_example['true_token']}) ", end="")
    if pred_control1 == bad_example["true_token"]:
        print("[Fixed - unexpected!]")
    else:
        print("[No fix - expected]")

    print()

    # ========================================================================
    # CONTROL 2: Patch from another incorrect example
    # ========================================================================
    print("="*60)
    print("CONTROL 2: Patch from wrong->wrong example")
    print("="*60)

    # Find another incorrect example with matching sequence length
    other_bad = None
    target_seq_len = bad_example["seq_len"]

    with open(args.test_path, 'r') as f:
        for line in f:
            example = json.loads(line)
            if example["expr"] == bad_example["expr"]:
                continue  # Skip the same one

            full_text = example["expr"] + " = " + str(example["result"])
            tokens = tokenizer.encode(full_text, add_special_tokens=True)
            input_ids = torch.tensor(tokens[:-1], dtype=torch.long).unsqueeze(0)
            true_token = tokens[-1]

            # Check sequence length matches
            if input_ids.size(1) != target_seq_len:
                continue

            pred, _ = predict_final_token(model, input_ids, device)
            if pred != true_token:
                other_bad = {
                    "expr": example["expr"],
                    "result": example["result"],
                    "input_ids": input_ids,
                    "true_token": true_token
                }
                break

    if other_bad:
        # Extract L2-H2 from wrong example
        with torch.no_grad():
            outputs = model(other_bad["input_ids"].to(device))
            wrong_state = model.blocks[2].attn.last_head_output[:, :, 2, :].clone()

        # Patch bad example with wrong state
        attn = model.blocks[2].attn
        attn.patch_head_index = 2

        # Handle sequence length mismatch
        target_seq_len = bad_example["input_ids"].size(1)
        if wrong_state.size(1) != target_seq_len:
            patch_state = wrong_state[:, :target_seq_len, :]
        else:
            patch_state = wrong_state

        attn.patch_head_output = patch_state

        pred_control2, _ = predict_final_token(model, bad_example["input_ids"], device)

        attn.patch_head_index = None
        attn.patch_head_output = None

        print(f"Other wrong example: {other_bad['expr']} = {other_bad['result']}")
        print(f"Bad example (wrong->wrong patch): {pred_control2} (true: {bad_example['true_token']}) ", end="")
        if pred_control2 == bad_example["true_token"]:
            print("[Fixed - suspicious!]")
        else:
            print("[No fix - expected]")
    else:
        print("Could not find another incorrect example for control")

    print()

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Target: Layer 2, Head 2")
    print(f"Good example: {good_example['expr']} = {good_example['result']}")
    print(f"Bad example:  {bad_example['expr']} = {bad_example['result']}")
    print()
    print("Results:")
    print(f"  Unpatched:        {pred_bad_nopatch} (wrong)")
    print(f"  Patched (L2-H2):  {pred_bad_patched} ", end="")
    if pred_bad_patched == bad_example["true_token"]:
        print("(CORRECT)")
    else:
        print("(wrong)")
    print(f"  Control (L2-H1):  {pred_control1} ", end="")
    if pred_control1 == bad_example["true_token"]:
        print("(CORRECT)")
    else:
        print("(wrong)")
    print()

    if pred_bad_patched == bad_example["true_token"] and pred_control1 != bad_example["true_token"]:
        print("[SUCCESS] L2-H2 patching fixes the prediction")
        print("[SUCCESS] Specificity: L2-H1 does not fix it")
    elif pred_bad_patched == bad_example["true_token"]:
        print("[PARTIAL] L2-H2 fixes it, but control also worked")
    else:
        print("[NO EFFECT] Patching did not fix the prediction")
        print("  (Model may need more training, or head is not causally sufficient)")

    print()
    print("="*60)


if __name__ == "__main__":
    main()
