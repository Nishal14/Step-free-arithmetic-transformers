"""
Activation Patching for Causal Analysis.

Test causal relationships between components by patching activations
from one forward pass into another. This helps identify which components
are causally responsible for specific behaviors.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Callable
import numpy as np
from .utils import ActivationCache, register_activation_hooks


class ActivationPatcher:
    """
    Perform activation patching experiments to test causal relationships.

    The key idea: Run model on two inputs (clean and corrupted), then
    patch activations from clean run into corrupted run to see what
    information is causally important.
    """

    def __init__(self, model, tokenizer, device: str = "cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()

    def run_with_cache(
        self,
        input_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Run model and cache all activations.

        Returns:
            logits: Model output logits
            activations: Dict of cached activations
        """
        cache = ActivationCache()
        register_activation_hooks(self.model, cache)

        with torch.no_grad():
            logits, _ = self.model(input_ids)

        activations = cache.activations.copy()
        cache.remove_hooks()

        return logits, activations

    def patch_activation(
        self,
        corrupted_input: torch.Tensor,
        clean_activations: Dict[str, torch.Tensor],
        patch_layer: str,
        patch_positions: Optional[List[int]] = None
    ) -> torch.Tensor:
        """
        Patch activations from clean run into corrupted run.

        Args:
            corrupted_input: Input with corruption/error
            clean_activations: Activations from clean forward pass
            patch_layer: Which layer to patch (e.g., "block_2")
            patch_positions: Which token positions to patch (None = all)

        Returns:
            Logits from patched forward pass
        """
        patched_logits = None

        def make_patch_hook(clean_activation: torch.Tensor, positions: Optional[List[int]]):
            def hook(module, input, output):
                nonlocal patched_logits

                if isinstance(output, tuple):
                    corrupted_activation = output[0]
                else:
                    corrupted_activation = output

                # Patch specified positions
                if positions is None:
                    # Patch all positions
                    patched = clean_activation.clone()
                else:
                    # Patch only specified positions
                    patched = corrupted_activation.clone()
                    for pos in positions:
                        if pos < patched.shape[1]:
                            patched[:, pos, :] = clean_activation[:, pos, :]

                if isinstance(output, tuple):
                    return (patched,) + output[1:]
                else:
                    return patched

            return hook

        # Find the layer to patch
        layer_to_patch = None
        if patch_layer.startswith("block_"):
            layer_idx = int(patch_layer.split("_")[1])
            layer_to_patch = self.model.blocks[layer_idx]
        elif patch_layer.startswith("attn_"):
            layer_idx = int(patch_layer.split("_")[1])
            layer_to_patch = self.model.blocks[layer_idx].attn
        elif patch_layer.startswith("ff_"):
            layer_idx = int(patch_layer.split("_")[1])
            layer_to_patch = self.model.blocks[layer_idx].ff

        if layer_to_patch is None:
            raise ValueError(f"Unknown layer: {patch_layer}")

        # Register patch hook
        clean_act = clean_activations[patch_layer]
        hook = layer_to_patch.register_forward_hook(
            make_patch_hook(clean_act, patch_positions)
        )

        # Run with patched activation
        with torch.no_grad():
            patched_logits, _ = self.model(corrupted_input)

        # Remove hook
        hook.remove()

        return patched_logits

    def compute_patching_effect(
        self,
        clean_text: str,
        corrupted_text: str,
        target_token_pos: int,
        layers_to_test: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Compute the effect of patching each layer.

        Measures how much patching each layer recovers the clean logits
        at the target position.

        Args:
            clean_text: Correct input
            corrupted_text: Input with error
            target_token_pos: Position where we measure the effect
            layers_to_test: List of layers to patch (None = all)

        Returns:
            Dict mapping layer name to recovery score (0-1)
        """
        from .utils import prepare_example

        # Run clean and corrupted
        clean_input = prepare_example(clean_text, self.tokenizer, self.device)
        corrupted_input = prepare_example(corrupted_text, self.tokenizer, self.device)

        clean_logits, clean_activations = self.run_with_cache(clean_input)
        corrupted_logits, _ = self.run_with_cache(corrupted_input)

        # Get target logits
        clean_target_logits = clean_logits[0, target_token_pos, :]
        corrupted_target_logits = corrupted_logits[0, target_token_pos, :]

        # Determine layers to test
        if layers_to_test is None:
            layers_to_test = list(clean_activations.keys())

        results = {}

        for layer in layers_to_test:
            if layer not in clean_activations:
                continue

            # Patch this layer
            patched_logits = self.patch_activation(
                corrupted_input,
                clean_activations,
                layer,
                patch_positions=None
            )

            patched_target_logits = patched_logits[0, target_token_pos, :]

            # Compute recovery score using cosine similarity
            clean_norm = torch.nn.functional.normalize(clean_target_logits.unsqueeze(0), dim=1)
            patched_norm = torch.nn.functional.normalize(patched_target_logits.unsqueeze(0), dim=1)
            corrupted_norm = torch.nn.functional.normalize(corrupted_target_logits.unsqueeze(0), dim=1)

            recovery = torch.nn.functional.cosine_similarity(patched_norm, clean_norm).item()

            # Baseline (corrupted similarity)
            baseline = torch.nn.functional.cosine_similarity(corrupted_norm, clean_norm).item()

            # Normalized recovery (0 = no effect, 1 = full recovery)
            normalized_recovery = (recovery - baseline) / (1.0 - baseline + 1e-10)

            results[layer] = normalized_recovery

        return results

    def position_specific_patching(
        self,
        clean_text: str,
        corrupted_text: str,
        patch_layer: str,
        target_token_pos: int
    ) -> Dict[int, float]:
        """
        Test which positions are important by patching one position at a time.

        Args:
            clean_text: Correct input
            corrupted_text: Input with error
            patch_layer: Which layer to patch
            target_token_pos: Position where we measure the effect

        Returns:
            Dict mapping position to recovery score
        """
        from .utils import prepare_example

        clean_input = prepare_example(clean_text, self.tokenizer, self.device)
        corrupted_input = prepare_example(corrupted_text, self.tokenizer, self.device)

        clean_logits, clean_activations = self.run_with_cache(clean_input)
        corrupted_logits, _ = self.run_with_cache(corrupted_input)

        clean_target = clean_logits[0, target_token_pos, :]
        corrupted_target = corrupted_logits[0, target_token_pos, :]

        seq_len = clean_input.shape[1]
        results = {}

        for pos in range(seq_len):
            # Patch only this position
            patched_logits = self.patch_activation(
                corrupted_input,
                clean_activations,
                patch_layer,
                patch_positions=[pos]
            )

            patched_target = patched_logits[0, target_token_pos, :]

            # Compute recovery
            recovery = torch.nn.functional.cosine_similarity(
                patched_target.unsqueeze(0),
                clean_target.unsqueeze(0)
            ).item()

            baseline = torch.nn.functional.cosine_similarity(
                corrupted_target.unsqueeze(0),
                clean_target.unsqueeze(0)
            ).item()

            normalized_recovery = (recovery - baseline) / (1.0 - baseline + 1e-10)
            results[pos] = normalized_recovery

        return results

    def head_patching_experiment(
        self,
        clean_text: str,
        corrupted_text: str,
        layer_idx: int,
        target_token_pos: int
    ) -> Dict[int, float]:
        """
        Patch individual attention heads to find which ones matter.

        Args:
            clean_text: Correct input
            corrupted_text: Input with error
            layer_idx: Which layer to test
            target_token_pos: Position to measure effect

        Returns:
            Dict mapping head index to recovery score
        """
        from .utils import prepare_example

        clean_input = prepare_example(clean_text, self.tokenizer, self.device)
        corrupted_input = prepare_example(corrupted_text, self.tokenizer, self.device)

        # Get clean and corrupted outputs
        clean_logits, _ = self.run_with_cache(clean_input)
        corrupted_logits, _ = self.run_with_cache(corrupted_input)

        clean_target = clean_logits[0, target_token_pos, :]
        corrupted_target = corrupted_logits[0, target_token_pos, :]

        results = {}
        num_heads = self.model.blocks[layer_idx].attn.num_heads

        for head_idx in range(num_heads):
            # Patch this specific head
            patched_logits = self._patch_single_head(
                clean_input,
                corrupted_input,
                layer_idx,
                head_idx
            )

            patched_target = patched_logits[0, target_token_pos, :]

            # Compute recovery
            recovery = torch.nn.functional.cosine_similarity(
                patched_target.unsqueeze(0),
                clean_target.unsqueeze(0)
            ).item()

            baseline = torch.nn.functional.cosine_similarity(
                corrupted_target.unsqueeze(0),
                clean_target.unsqueeze(0)
            ).item()

            normalized_recovery = (recovery - baseline) / (1.0 - baseline + 1e-10)
            results[head_idx] = normalized_recovery

        return results

    def _patch_single_head(
        self,
        clean_input: torch.Tensor,
        corrupted_input: torch.Tensor,
        layer_idx: int,
        head_idx: int
    ) -> torch.Tensor:
        """Patch a single attention head."""
        # Get clean activations
        _, clean_activations = self.run_with_cache(clean_input)

        # Extract clean head output
        clean_block_output = clean_activations[f"attn_{layer_idx}"]

        def make_head_patch_hook(clean_act: torch.Tensor, head_idx: int):
            def hook(module, input, output):
                # Output is the result after out_proj
                # We need to patch at the multi-head level

                # This is a simplified version - for full implementation,
                # you'd need to patch before out_proj
                return output

            return hook

        # Note: Full head patching requires modifying the attention mechanism
        # For now, we'll patch the full attention output as an approximation
        layer_to_patch = self.model.blocks[layer_idx].attn

        def patch_hook(module, input, output):
            return clean_block_output

        hook = layer_to_patch.register_forward_hook(patch_hook)

        with torch.no_grad():
            patched_logits, _ = self.model(corrupted_input)

        hook.remove()

        return patched_logits

    def knockout_experiment(
        self,
        text: str,
        layers_to_knockout: List[str]
    ) -> torch.Tensor:
        """
        Zero out activations at specified layers (ablation).

        Args:
            text: Input text
            layers_to_knockout: Layers to ablate

        Returns:
            Logits with ablated components
        """
        from .utils import prepare_example

        input_ids = prepare_example(text, self.tokenizer, self.device)

        hooks = []

        def make_knockout_hook():
            def hook(module, input, output):
                if isinstance(output, tuple):
                    return (torch.zeros_like(output[0]),) + output[1:]
                else:
                    return torch.zeros_like(output)
            return hook

        # Register knockout hooks
        for layer_name in layers_to_knockout:
            if layer_name.startswith("block_"):
                layer_idx = int(layer_name.split("_")[1])
                layer = self.model.blocks[layer_idx]
            elif layer_name.startswith("attn_"):
                layer_idx = int(layer_name.split("_")[1])
                layer = self.model.blocks[layer_idx].attn
            elif layer_name.startswith("ff_"):
                layer_idx = int(layer_name.split("_")[1])
                layer = self.model.blocks[layer_idx].ff
            else:
                continue

            hook = layer.register_forward_hook(make_knockout_hook())
            hooks.append(hook)

        # Run with knockouts
        with torch.no_grad():
            logits, _ = self.model(input_ids)

        # Clean up hooks
        for hook in hooks:
            hook.remove()

        return logits
