"""
Pilot dataset generator for arithmetic expressions with parentheses.
Generates small-scale datasets for initial mechanistic interpretability experiments.
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Any


def compute_depth(expr: str) -> List[int]:
    """
    Compute parenthesis depth for each character.

    Rules:
    - For '(': increment depth, then record
    - For ')': decrement depth, then record
    - For others: record current depth
    """
    depth_list = []
    current_depth = 0

    for char in expr:
        if char == '(':
            current_depth += 1
            depth_list.append(current_depth)
        elif char == ')':
            current_depth -= 1
            depth_list.append(current_depth)
        else:
            depth_list.append(current_depth)

    return depth_list


def evaluate_expr(expr: str) -> tuple[bool, int]:
    """
    Safely evaluate expression and return (valid, result).

    Returns:
        (False, 0) if invalid
        (True, result) if valid
    """
    try:
        result = eval(expr)

        # Check if result is integer
        if not isinstance(result, int):
            return False, 0

        # Check if result is within reasonable bounds
        if abs(result) > 2**63 - 1:
            return False, 0

        return True, result
    except:
        return False, 0


def generate_number() -> str:
    """Generate a 1-2 digit number (no leading zeros)."""
    return str(random.randint(1, 99))


def generate_flat_expression(num_count: int) -> str:
    """Generate expression without parentheses."""
    expr_parts = [generate_number()]

    for _ in range(num_count - 1):
        op = random.choice(['+', '-', '*'])
        num = generate_number()
        expr_parts.append(op)
        expr_parts.append(num)

    return ''.join(expr_parts)


def generate_depth1_expression(num_count: int) -> str:
    """Generate expression with 1 level of parentheses."""
    # Strategy: create a flat expression and wrap part of it

    if num_count == 2:
        # Simple: (a+b) or (a*b) etc
        num1 = generate_number()
        op = random.choice(['+', '-', '*'])
        num2 = generate_number()
        return f"({num1}{op}{num2})"

    elif num_count == 3:
        # Options: (a+b)+c, a+(b+c), (a+b)*c, etc
        choice = random.randint(0, 2)

        if choice == 0:
            # (a op b) op c
            num1, num2, num3 = generate_number(), generate_number(), generate_number()
            op1 = random.choice(['+', '-', '*'])
            op2 = random.choice(['+', '-', '*'])
            return f"({num1}{op1}{num2}){op2}{num3}"
        elif choice == 1:
            # a op (b op c)
            num1, num2, num3 = generate_number(), generate_number(), generate_number()
            op1 = random.choice(['+', '-', '*'])
            op2 = random.choice(['+', '-', '*'])
            return f"{num1}{op1}({num2}{op2}{num3})"
        else:
            # (a op b op c)
            num1, num2, num3 = generate_number(), generate_number(), generate_number()
            op1 = random.choice(['+', '-', '*'])
            op2 = random.choice(['+', '-', '*'])
            return f"({num1}{op1}{num2}{op2}{num3})"

    else:  # num_count == 4
        # (a op b) op (c op d), a op (b op c) op d, etc
        choice = random.randint(0, 2)
        num1, num2, num3, num4 = [generate_number() for _ in range(4)]
        ops = [random.choice(['+', '-', '*']) for _ in range(3)]

        if choice == 0:
            # (a op b) op (c op d)
            return f"({num1}{ops[0]}{num2}){ops[1]}({num3}{ops[2]}{num4})"
        elif choice == 1:
            # (a op b op c) op d
            return f"({num1}{ops[0]}{num2}{ops[1]}{num3}){ops[2]}{num4}"
        else:
            # a op (b op c op d)
            return f"{num1}{ops[0]}({num2}{ops[1]}{num3}{ops[2]}{num4})"


def generate_depth2_expression(num_count: int) -> str:
    """Generate expression with 2 levels of parentheses."""
    # Strategy: nest parentheses

    if num_count == 2:
        # ((a+b))
        num1 = generate_number()
        op = random.choice(['+', '-', '*'])
        num2 = generate_number()
        return f"(({num1}{op}{num2}))"

    elif num_count == 3:
        # Options: ((a+b)+c), (a+(b+c)), ((a+b))*c, etc
        choice = random.randint(0, 3)
        num1, num2, num3 = generate_number(), generate_number(), generate_number()
        ops = [random.choice(['+', '-', '*']) for _ in range(2)]

        if choice == 0:
            # ((a op b) op c)
            return f"(({num1}{ops[0]}{num2}){ops[1]}{num3})"
        elif choice == 1:
            # (a op (b op c))
            return f"({num1}{ops[0]}({num2}{ops[1]}{num3}))"
        elif choice == 2:
            # ((a op b)) op c
            return f"(({num1}{ops[0]}{num2})){ops[1]}{num3}"
        else:
            # a op ((b op c))
            return f"{num1}{ops[0]}(({num2}{ops[1]}{num3}))"

    else:  # num_count == 4
        choice = random.randint(0, 2)
        nums = [generate_number() for _ in range(4)]
        ops = [random.choice(['+', '-', '*']) for _ in range(3)]

        if choice == 0:
            # ((a op b) op (c op d))
            return f"(({nums[0]}{ops[0]}{nums[1]}){ops[1]}({nums[2]}{ops[2]}{nums[3]}))"
        elif choice == 1:
            # (a op (b op c)) op d
            return f"({nums[0]}{ops[0]}({nums[1]}{ops[1]}{nums[2]})){ops[2]}{nums[3]}"
        else:
            # a op ((b op c) op d)
            return f"{nums[0]}{ops[0]}(({nums[1]}{ops[1]}{nums[2]}){ops[2]}{nums[3]})"


def generate_expression(allow_parens: bool = True, max_depth: int = 2) -> str:
    """Generate a random expression."""
    num_count = random.randint(2, 4)

    if not allow_parens:
        return generate_flat_expression(num_count)

    # Choose depth
    depth = random.randint(0, max_depth)

    if depth == 0:
        return generate_flat_expression(num_count)
    elif depth == 1:
        return generate_depth1_expression(num_count)
    else:  # depth == 2
        return generate_depth2_expression(num_count)


def generate_dataset_split(
    size: int,
    allow_parens: bool = True,
    max_depth: int = 2,
    max_attempts: int = 1000
) -> tuple[List[Dict[str, Any]], int]:
    """
    Generate a dataset split.

    Returns:
        (examples, rejected_count)
    """
    examples = []
    rejected = 0
    attempts = 0

    while len(examples) < size and attempts < size * max_attempts:
        attempts += 1

        # Generate expression
        expr = generate_expression(allow_parens, max_depth)

        # Evaluate
        valid, result = evaluate_expr(expr)

        if not valid:
            rejected += 1
            continue

        # Compute depth
        depth = compute_depth(expr)

        # Create example
        example = {
            "expr": expr,
            "result": result,
            "depth": depth
        }

        examples.append(example)

    if len(examples) < size:
        print(f"Warning: Could only generate {len(examples)}/{size} examples after {attempts} attempts")

    return examples, rejected


def save_jsonl(examples: List[Dict[str, Any]], filepath: Path):
    """Save examples to JSONL file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'w') as f:
        for example in examples:
            f.write(json.dumps(example) + '\n')

    print(f"Saved {len(examples)} examples to {filepath}")


def main():
    """Generate pilot datasets."""
    print("="*60)
    print("Pilot Dataset Generation")
    print("="*60)
    print()

    random.seed(42)

    # Generate training set (mix of flat and parenthesized)
    print("Generating training set (2000 examples)...")
    train_examples, train_rejected = generate_dataset_split(
        size=2000,
        allow_parens=True,
        max_depth=2
    )
    print(f"  Rejected: {train_rejected}")

    # Generate validation set (same distribution)
    print("Generating validation set (200 examples)...")
    val_examples, val_rejected = generate_dataset_split(
        size=200,
        allow_parens=True,
        max_depth=2
    )
    print(f"  Rejected: {val_rejected}")

    # Generate test set - flat only
    print("Generating test set - flat (100 examples)...")
    test_flat_examples, test_flat_rejected = generate_dataset_split(
        size=100,
        allow_parens=False,
        max_depth=0
    )
    print(f"  Rejected: {test_flat_rejected}")

    # Generate test set - parenthesized only
    print("Generating test set - parenthesized (100 examples)...")
    test_paren_examples = []
    test_paren_rejected = 0
    attempts = 0
    max_attempts = 100000

    while len(test_paren_examples) < 100 and attempts < max_attempts:
        attempts += 1

        # Generate expression with parentheses
        expr = generate_expression(allow_parens=True, max_depth=2)

        # Skip if no parentheses
        if '(' not in expr:
            continue

        # Evaluate
        valid, result = evaluate_expr(expr)

        if not valid:
            test_paren_rejected += 1
            continue

        # Compute depth
        depth = compute_depth(expr)

        # Create example
        example = {
            "expr": expr,
            "result": result,
            "depth": depth
        }

        test_paren_examples.append(example)

    print(f"  Rejected: {test_paren_rejected}")

    # Save all splits
    print()
    print("Saving datasets...")
    save_jsonl(train_examples, Path("data/pilot_train.jsonl"))
    save_jsonl(val_examples, Path("data/pilot_val.jsonl"))
    save_jsonl(test_flat_examples, Path("data/pilot_test_flat.jsonl"))
    save_jsonl(test_paren_examples, Path("data/pilot_test_paren.jsonl"))

    # Print statistics
    print()
    print("="*60)
    print("Dataset Statistics")
    print("="*60)

    def print_stats(name: str, examples: List[Dict[str, Any]]):
        has_parens = sum(1 for ex in examples if '(' in ex['expr'])
        max_depth = max(max(ex['depth']) if ex['depth'] else 0 for ex in examples)
        min_len = min(len(ex['expr']) for ex in examples)
        max_len = max(len(ex['expr']) for ex in examples)
        avg_len = sum(len(ex['expr']) for ex in examples) / len(examples)

        print(f"\n{name}:")
        print(f"  Total examples: {len(examples)}")
        print(f"  With parentheses: {has_parens} ({100*has_parens/len(examples):.1f}%)")
        print(f"  Max depth: {max_depth}")
        print(f"  Expression length: {min_len}-{max_len} (avg: {avg_len:.1f})")

    print_stats("Training", train_examples)
    print_stats("Validation", val_examples)
    print_stats("Test (Flat)", test_flat_examples)
    print_stats("Test (Paren)", test_paren_examples)

    # Show examples
    print()
    print("="*60)
    print("Sample Examples")
    print("="*60)

    for i, ex in enumerate(train_examples[:5]):
        print(f"\nExample {i+1}:")
        print(f"  Expression: {ex['expr']}")
        print(f"  Result: {ex['result']}")
        print(f"  Depth: {ex['depth']}")

    print()
    print("="*60)
    print("Generation Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
