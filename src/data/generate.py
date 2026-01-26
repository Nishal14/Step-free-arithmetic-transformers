#!/usr/bin/env python3
"""
Synthetic dataset generator for mathematical reasoning tasks.
Generates train/dev/test splits with optional stepwise traces.
"""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


class TaskGenerator:
    """Base class for synthetic task generators."""

    def __init__(self, max_length: int, seed: int):
        self.max_length = max_length
        self.seed = seed

    def generate(self, with_steps: bool = False) -> Dict[str, Any]:
        """Generate a single instance. Override in subclasses."""
        raise NotImplementedError

    def get_metadata(self) -> Dict[str, Any]:
        """Return task-specific metadata."""
        raise NotImplementedError


class AdditionGenerator(TaskGenerator):
    """Generate addition problems with stepwise traces."""

    def generate(self, with_steps: bool = False) -> Dict[str, Any]:
        # Generate two random integers
        num_digits = random.randint(2, self.max_length)
        a = random.randint(10**(num_digits-1), 10**num_digits - 1)
        b = random.randint(10**(num_digits-1), 10**num_digits - 1)

        result = a + b
        input_str = f"{a} + {b}"

        record = {
            "input": input_str,
            "gold": str(result),
            "task": "add",
            "seed": self.seed,
            "ground_truth_rule": "elementwise_add"
        }

        if with_steps:
            steps = self._generate_steps(a, b, result)
            record["steps"] = steps

        return record

    def _generate_steps(self, a: int, b: int, result: int) -> List[str]:
        """Generate step-by-step addition trace."""
        steps = []
        str_a = str(a)
        str_b = str(b)
        str_result = str(result)

        # Pad to same length
        max_len = max(len(str_a), len(str_b))
        str_a = str_a.zfill(max_len)
        str_b = str_b.zfill(max_len)

        steps.append(f"Align numbers: {str_a} + {str_b}")

        carry = 0
        partial_results = []
        for i in range(max_len - 1, -1, -1):
            digit_a = int(str_a[i])
            digit_b = int(str_b[i])
            sum_digits = digit_a + digit_b + carry
            digit_result = sum_digits % 10
            carry = sum_digits // 10
            partial_results.insert(0, str(digit_result))
            steps.append(f"Position {max_len - i}: {digit_a} + {digit_b} + carry({carry if i < max_len - 1 else 0}) = {sum_digits} -> digit={digit_result}, new_carry={carry}")

        if carry > 0:
            partial_results.insert(0, str(carry))
            steps.append(f"Final carry: {carry}")

        steps.append(f"Result: {result}")
        return steps

    def get_metadata(self) -> Dict[str, Any]:
        return {"task": "add", "seed": self.seed, "ground_truth_rule": "elementwise_add"}


class MultiplicationGenerator(TaskGenerator):
    """Generate multiplication problems with stepwise traces."""

    def generate(self, with_steps: bool = False) -> Dict[str, Any]:
        # Generate two random integers (smaller for multiplication)
        num_digits = random.randint(1, min(self.max_length // 2, 4))
        a = random.randint(10**(num_digits-1) if num_digits > 1 else 1, 10**num_digits - 1)
        b = random.randint(10**(num_digits-1) if num_digits > 1 else 1, 10**num_digits - 1)

        result = a * b
        input_str = f"{a} * {b}"

        record = {
            "input": input_str,
            "gold": str(result),
            "task": "mul",
            "seed": self.seed,
            "ground_truth_rule": "long_multiplication"
        }

        if with_steps:
            steps = self._generate_steps(a, b, result)
            record["steps"] = steps

        return record

    def _generate_steps(self, a: int, b: int, result: int) -> List[str]:
        """Generate step-by-step multiplication trace."""
        steps = []
        str_a = str(a)
        str_b = str(b)

        steps.append(f"Multiply {a} * {b}")

        # Long multiplication method
        partial_products = []
        for i, digit_b in enumerate(reversed(str_b)):
            digit_b_int = int(digit_b)
            partial = a * digit_b_int
            shifted_partial = partial * (10 ** i)
            partial_products.append(shifted_partial)
            steps.append(f"Partial product: {a} * {digit_b_int} * 10^{i} = {shifted_partial}")

        steps.append(f"Sum partial products: {' + '.join(map(str, partial_products))} = {result}")
        steps.append(f"Result: {result}")
        return steps

    def get_metadata(self) -> Dict[str, Any]:
        return {"task": "mul", "seed": self.seed, "ground_truth_rule": "long_multiplication"}


class BaseConvertGenerator(TaskGenerator):
    """Generate base conversion problems (decimal to binary/hex)."""

    def generate(self, with_steps: bool = False) -> Dict[str, Any]:
        # Generate random decimal number
        num = random.randint(1, 10**self.max_length - 1)

        # Randomly choose target base
        target_base = random.choice([2, 16])
        base_name = "binary" if target_base == 2 else "hex"

        if target_base == 2:
            result = bin(num)[2:]  # Remove '0b' prefix
        else:
            result = hex(num)[2:].upper()  # Remove '0x' prefix

        input_str = f"{num} to {base_name}"

        record = {
            "input": input_str,
            "gold": result,
            "task": "base_convert",
            "seed": self.seed,
            "ground_truth_rule": f"decimal_to_{base_name}"
        }

        if with_steps:
            steps = self._generate_steps(num, target_base, result)
            record["steps"] = steps

        return record

    def _generate_steps(self, num: int, base: int, result: str) -> List[str]:
        """Generate step-by-step base conversion trace."""
        steps = []
        base_name = "binary" if base == 2 else "hexadecimal"
        steps.append(f"Convert {num} (decimal) to {base_name}")

        # Division method
        n = num
        remainders = []
        while n > 0:
            remainder = n % base
            n = n // base
            remainders.append(remainder)
            if base == 16 and remainder >= 10:
                rem_str = chr(ord('A') + remainder - 10)
            else:
                rem_str = str(remainder)
            steps.append(f"Divide {n * base + remainder} by {base}: quotient={n}, remainder={rem_str}")

        steps.append(f"Read remainders in reverse: {result}")
        return steps

    def get_metadata(self) -> Dict[str, Any]:
        return {"task": "base_convert", "seed": self.seed, "ground_truth_rule": "decimal_to_base"}


class PolynomialExpandGenerator(TaskGenerator):
    """Generate polynomial expansion problems (a+b)(c+d)."""

    def generate(self, with_steps: bool = False) -> Dict[str, Any]:
        # Generate simple binomial expansion: (a+b)(c+d)
        a = random.randint(1, min(self.max_length, 20))
        b = random.randint(1, min(self.max_length, 20))
        c = random.randint(1, min(self.max_length, 20))
        d = random.randint(1, min(self.max_length, 20))

        input_str = f"({a}+{b})({c}+{d})"

        # Expand: ac + ad + bc + bd
        result_val = (a + b) * (c + d)
        result = f"{a*c}+{a*d}+{b*c}+{b*d}={result_val}"

        record = {
            "input": input_str,
            "gold": result,
            "task": "poly_expand",
            "seed": self.seed,
            "ground_truth_rule": "foil_method"
        }

        if with_steps:
            steps = self._generate_steps(a, b, c, d, result_val)
            record["steps"] = steps

        return record

    def _generate_steps(self, a: int, b: int, c: int, d: int, result: int) -> List[str]:
        """Generate step-by-step polynomial expansion trace."""
        steps = []
        steps.append(f"Expand ({a}+{b})({c}+{d}) using FOIL")
        steps.append(f"First: {a} * {c} = {a*c}")
        steps.append(f"Outer: {a} * {d} = {a*d}")
        steps.append(f"Inner: {b} * {c} = {b*c}")
        steps.append(f"Last: {b} * {d} = {b*d}")
        steps.append(f"Sum: {a*c} + {a*d} + {b*c} + {b*d} = {result}")
        return steps

    def get_metadata(self) -> Dict[str, Any]:
        return {"task": "poly_expand", "seed": self.seed, "ground_truth_rule": "foil_method"}


def get_generator(task: str, max_length: int, seed: int) -> TaskGenerator:
    """Factory function to get task generator."""
    generators = {
        "add": AdditionGenerator,
        "mul": MultiplicationGenerator,
        "base_convert": BaseConvertGenerator,
        "poly_expand": PolynomialExpandGenerator,
    }

    if task not in generators:
        raise ValueError(f"Unknown task: {task}. Choose from {list(generators.keys())}")

    return generators[task](max_length, seed)


def generate_dataset(
    task: str,
    instances: int,
    max_length: int,
    with_steps: bool,
    seed: int,
    output_dir: Path
):
    """Generate complete dataset with train/dev/test splits."""
    set_seed(seed)

    # Create output directory
    task_dir = output_dir / task
    task_dir.mkdir(parents=True, exist_ok=True)

    # Split ratios
    train_ratio = 0.8
    dev_ratio = 0.1
    test_ratio = 0.1

    train_size = int(instances * train_ratio)
    dev_size = int(instances * dev_ratio)
    test_size = instances - train_size - dev_size

    splits = {
        "train": train_size,
        "dev": dev_size,
        "test": test_size
    }

    generator = get_generator(task, max_length, seed)

    for split_name, split_size in splits.items():
        output_file = task_dir / f"{split_name}.jsonl"

        with open(output_file, 'w') as f:
            for i in range(split_size):
                # Use different seed for each instance
                instance_seed = seed + hash(f"{split_name}_{i}") % 1000000
                set_seed(instance_seed)

                record = generator.generate(with_steps=with_steps)
                record["split"] = split_name
                record["instance_id"] = i

                f.write(json.dumps(record) + '\n')

        print(f"Generated {split_size} {split_name} instances -> {output_file}")

    print(f"\nDataset generation complete for task '{task}'")
    print(f"Total instances: {instances}")
    print(f"Output directory: {task_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic mathematical reasoning datasets"
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["add", "mul", "base_convert", "poly_expand"],
        help="Task type to generate"
    )
    parser.add_argument(
        "--instances",
        type=int,
        default=20000,
        help="Total number of instances to generate"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=20,
        help="Maximum token/digit length"
    )
    parser.add_argument(
        "--with_steps",
        action="store_true",
        help="Generate stepwise traces"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data",
        help="Output directory for generated datasets"
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    generate_dataset(
        task=args.task,
        instances=args.instances,
        max_length=args.max_length,
        with_steps=args.with_steps,
        seed=args.seed,
        output_dir=output_dir
    )


if __name__ == "__main__":
    main()
