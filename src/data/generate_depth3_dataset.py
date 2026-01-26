"""
Generate OOD test dataset with parenthesis depth = exactly 3.
Used to test generalization to unseen depth levels.
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Any


def compute_depth(expr: str) -> List[int]:
    """Compute parenthesis depth for each character."""
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
    """Safely evaluate expression and return (valid, result)."""
    try:
        result = eval(expr)
        if not isinstance(result, int):
            return False, 0
        if abs(result) > 2**63 - 1:
            return False, 0
        return True, result
    except:
        return False, 0


def generate_number() -> str:
    """Generate a 1-2 digit number (no leading zeros)."""
    return str(random.randint(1, 99))


def generate_depth3_expression(num_count: int) -> str:
    """
    Generate expression with exactly 3 levels of parentheses.
    Uses triple nesting to achieve depth = 3.
    """
    if num_count == 2:
        # (((a+b)))
        num1 = generate_number()
        op = random.choice(['+', '-', '*'])
        num2 = generate_number()
        return f"((({num1}{op}{num2})))"

    elif num_count == 3:
        # Options with depth 3:
        # (((a+b)+c)), ((a+(b+c))), (((a+b)))+c, etc
        choice = random.randint(0, 5)
        num1, num2, num3 = generate_number(), generate_number(), generate_number()
        ops = [random.choice(['+', '-', '*']) for _ in range(2)]

        if choice == 0:
            # (((a op b) op c))
            return f"((({num1}{ops[0]}{num2}){ops[1]}{num3}))"
        elif choice == 1:
            # ((a op (b op c)))
            return f"(({num1}{ops[0]}({num2}{ops[1]}{num3})))"
        elif choice == 2:
            # (((a op b))) op c
            return f"((({num1}{ops[0]}{num2})){ops[1]}{num3}"
        elif choice == 3:
            # a op (((b op c)))
            return f"{num1}{ops[0]}((({num2}{ops[1]}{num3})))"
        elif choice == 4:
            # ((a op b) op (c))
            # Actually this gives depth 2, let me fix
            # (((a op b))) op (c)
            return f"((({num1}{ops[0]}{num2})){ops[1]}({num3})"
        else:
            # (a) op (((b op c)))
            return f"({num1}){ops[0]}((({num2}{ops[1]}{num3})))"

    else:  # num_count == 4
        choice = random.randint(0, 3)
        nums = [generate_number() for _ in range(4)]
        ops = [random.choice(['+', '-', '*']) for _ in range(3)]

        if choice == 0:
            # (((a op b) op c) op d)
            return f"((({nums[0]}{ops[0]}{nums[1]}){ops[1]}{nums[2]}){ops[2]}{nums[3]})"
        elif choice == 1:
            # ((a op (b op c)) op d)
            return f"(({nums[0]}{ops[0]}({nums[1]}{ops[1]}{nums[2]})){ops[2]}{nums[3]})"
        elif choice == 2:
            # (((a op b)) op (c op d))
            return f"((({nums[0]}{ops[0]}{nums[1]})){ops[1]}({nums[2]}{ops[2]}{nums[3]}))"
        else:
            # (a op ((b op c) op d))
            return f"({nums[0]}{ops[0]}(({nums[1]}{ops[1]}{nums[2]}){ops[2]}{nums[3]}))"


def generate_depth3_dataset(size: int = 100) -> List[Dict[str, Any]]:
    """
    Generate dataset with exactly depth=3 expressions.

    Returns:
        List of examples with 'expr', 'result', and 'depth' fields
    """
    examples = []
    rejected = 0
    attempts = 0
    max_attempts = 100000

    while len(examples) < size and attempts < max_attempts:
        attempts += 1

        # Generate expression
        num_count = random.randint(2, 4)
        expr = generate_depth3_expression(num_count)

        # Evaluate
        valid, result = evaluate_expr(expr)

        if not valid:
            rejected += 1
            continue

        # Compute depth
        depth = compute_depth(expr)

        # Verify that max depth is exactly 3
        if max(depth) != 3:
            rejected += 1
            continue

        # Create example
        example = {
            "expr": expr,
            "result": result,
            "depth": depth
        }

        examples.append(example)

    if len(examples) < size:
        print(f"Warning: Could only generate {len(examples)}/{size} examples after {attempts} attempts")

    print(f"Generated {len(examples)} examples, rejected {rejected}")
    return examples


def save_jsonl(examples: List[Dict[str, Any]], filepath: Path):
    """Save examples to JSONL file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'w') as f:
        for example in examples:
            f.write(json.dumps(example) + '\n')

    print(f"Saved {len(examples)} examples to {filepath}")


def main():
    """Generate depth-3 OOD test dataset."""
    print("="*60)
    print("OOD Depth-3 Dataset Generation")
    print("="*60)
    print()

    random.seed(42)

    # Generate exactly 100 depth-3 examples
    print("Generating depth-3 test set (100 examples)...")
    examples = generate_depth3_dataset(size=100)

    # Save dataset
    output_path = Path("data/pilot_test_depth3.jsonl")
    save_jsonl(examples, output_path)

    # Print statistics
    print()
    print("="*60)
    print("Dataset Statistics")
    print("="*60)

    max_depth = max(max(ex['depth']) for ex in examples)
    min_len = min(len(ex['expr']) for ex in examples)
    max_len = max(len(ex['expr']) for ex in examples)
    avg_len = sum(len(ex['expr']) for ex in examples) / len(examples)

    print(f"Total examples: {len(examples)}")
    print(f"Max depth: {max_depth}")
    print(f"Expression length: {min_len}-{max_len} (avg: {avg_len:.1f})")

    # Show sample examples
    print()
    print("="*60)
    print("Sample Examples")
    print("="*60)

    for i, ex in enumerate(examples[:5]):
        print(f"\nExample {i+1}:")
        print(f"  Expression: {ex['expr']}")
        print(f"  Result: {ex['result']}")
        print(f"  Max depth: {max(ex['depth'])}")
        print(f"  Depth: {ex['depth']}")

    print()
    print("="*60)
    print("Generation Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
