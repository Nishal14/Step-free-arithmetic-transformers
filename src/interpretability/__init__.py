"""
Mechanistic Interpretability Tools for Compact Transformers.

This module provides tools for analyzing how compact transformers learn
mathematical reasoning, including attention analysis, activation patching,
circuit discovery, and probing classifiers.
"""

from .attention_patterns import AttentionAnalyzer
from .activation_patching import ActivationPatcher
from .logit_lens import LogitLens
from .circuit_discovery import CircuitDiscovery
from .probing import ProbingClassifier
from .interventions import InterventionAnalyzer
from .utils import (
    load_model_and_tokenizer,
    prepare_example,
    extract_activations,
    get_attention_patterns
)

__all__ = [
    "AttentionAnalyzer",
    "ActivationPatcher",
    "LogitLens",
    "CircuitDiscovery",
    "ProbingClassifier",
    "InterventionAnalyzer",
    "load_model_and_tokenizer",
    "prepare_example",
    "extract_activations",
    "get_attention_patterns"
]
