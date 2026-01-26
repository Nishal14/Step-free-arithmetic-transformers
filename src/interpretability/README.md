# Interpretability Tools for Math-Compact

Comprehensive mechanistic interpretability toolkit for analyzing how compact transformers learn mathematical reasoning.

## Overview

This module provides six major categories of interpretability tools:

1. **Attention Pattern Analysis** - Visualize which tokens attend to which
2. **Activation Patching** - Test causal relationships between components
3. **Logit Lens** - Decode intermediate layer representations
4. **Circuit Discovery** - Identify specialized heads and neurons
5. **Probing Classifiers** - Test what information is encoded at each layer
6. **Intervention Experiments** - Ablate components to measure impact

## Quick Start

### Basic Analysis

```bash
# Run comprehensive analysis on a trained model
python -m src.analyze \
    --checkpoint runs/seed42/checkpoint_best.pt \
    --all \
    --examples "12 + 34" "56 * 78" "100 + 200"
```

### Specific Analyses

```bash
# Only attention analysis
python -m src.analyze --checkpoint <path> --attention --examples "59 + 73"

# Only logit lens
python -m src.analyze --checkpoint <path> --logit-lens --examples "12 + 34"

# Circuit discovery + probing
python -m src.analyze --checkpoint <path> --circuits --probing
```

## Tools in Detail

### 1. Attention Pattern Analysis

**Purpose:** Understand which tokens the model attends to during computation.

**Key Features:**
- Visualize attention heatmaps for all heads
- Track attention evolution across layers
- Find specialized attention heads (carry detection, operator tracking)
- Compute attention statistics (entropy, max attention, diagonal attention)

**Example Usage:**

```python
from src.interpretability import AttentionAnalyzer, load_model_and_tokenizer

model, tokenizer, _ = load_model_and_tokenizer("runs/model.pt")
analyzer = AttentionAnalyzer(model, tokenizer)

# Visualize all heads in layer 2
analyzer.plot_attention_heads("59 + 73", layer_idx=2, save_path="attn.png")

# Analyze attention statistics
stats = analyzer.analyze_attention_statistics("12 + 34")

# Find heads that attend to carry positions
carry_examples = [("59 + 73", [0, 1]), ("88 + 99", [0, 1])]
specialized = analyzer.find_specialized_heads(carry_examples, [ex[1] for ex in carry_examples])
```

**Key Findings to Look For:**
- Heads that consistently attend to operator tokens
- Heads with diagonal attention (position tracking)
- Heads that attend to carry positions in addition
- Heads that attend to specific digit positions

### 2. Activation Patching

**Purpose:** Establish causal relationships by patching activations from correct runs into incorrect runs.

**Key Features:**
- Patch activations at specific layers
- Test position-specific patching
- Measure recovery scores
- Individual head patching

**Example Usage:**

```python
from src.interpretability import ActivationPatcher

patcher = ActivationPatcher(model, tokenizer)

# Test which layers are important for correct computation
effects = patcher.compute_patching_effect(
    clean_text="12 + 34",
    corrupted_text="12 + 94",  # Wrong digit
    target_token_pos=-2
)

# Find which positions matter
position_effects = patcher.position_specific_patching(
    clean_text="12 + 34",
    corrupted_text="12 + 94",
    patch_layer="block_2",
    target_token_pos=-2
)

# Test individual heads
head_effects = patcher.head_patching_experiment(
    clean_text="12 + 34",
    corrupted_text="12 + 94",
    layer_idx=2,
    target_token_pos=-2
)
```

**Key Findings to Look For:**
- Which layers are causally important (high recovery scores)
- Which positions contain critical information
- Which heads are necessary for correct computation

### 3. Logit Lens

**Purpose:** See what the model "thinks" at each layer by projecting intermediate activations through the language head.

**Key Features:**
- Track prediction evolution across layers
- Visualize when model commits to answer
- Compute prediction entropy
- Analyze convergence patterns

**Example Usage:**

```python
from src.interpretability import LogitLens

lens = LogitLens(model, tokenizer)

# See prediction evolution for a specific position
lens.visualize_prediction_evolution(
    "12 + 34",
    position=-1,  # Last position
    save_path="evolution.png"
)

# Get top-k predictions at each layer
predictions = lens.get_top_k_predictions("12 + 34", position=-1, k=5)

# Analyze when model converges to correct answer
convergence = lens.analyze_convergence("12 + 34", target_token_pos=-1, target_token="6")
```

**Key Findings to Look For:**
- Does the model compute the answer gradually or decide early?
- Which layer first predicts the correct answer?
- High entropy = uncertain, low entropy = confident
- Do predictions change in later layers?

### 4. Circuit Discovery

**Purpose:** Identify specific computational circuits - which heads and neurons implement specific operations.

**Key Features:**
- Find carry detection heads
- Find operator-specialized heads
- Find position-tracking heads
- Identify specialized neurons for different operations

**Example Usage:**

```python
from src.interpretability import CircuitDiscovery

discovery = CircuitDiscovery(model, tokenizer)

# Find heads that detect carries
carry_examples = [
    ("59 + 73", [0, 1]),
    ("88 + 99", [0, 1])
]
carry_heads = discovery.find_carry_detection_heads(carry_examples)

# Find heads that attend to operators
examples = ["12 + 34", "56 * 78", "10 / 2"]
operator_heads = discovery.find_operator_attention_heads(examples, ['+', '*', '/'])

# Find neurons specialized for different operations
neurons = discovery.find_specialized_neurons(
    examples=["12 + 34", "56 * 78"],
    labels=["add", "mul"],
    layer_name="ff_2"
)

# Complete circuit analysis for one example
circuit = discovery.analyze_circuit_for_example("59 + 73", "carry_detection")
```

**Key Findings to Look For:**
- Specific heads that specialize in carry detection
- Heads that always attend to the operator
- Neurons that activate strongly for specific operations
- Hierarchical circuits (early layers detect features, later layers combine them)

### 5. Probing Classifiers

**Purpose:** Test what information is linearly accessible in the hidden states.

**Key Features:**
- Train linear probes for various properties
- Test across all layers
- Measure when information emerges
- Analyze probe weights

**Example Usage:**

```python
from src.interpretability import ProbingClassifier

prober = ProbingClassifier(model, tokenizer)

# Probe for carry detection
carry_examples = ["59 + 73", "88 + 99", "12 + 34"]
carry_labels = [1, 1, 0]  # 1 = has carry, 0 = no carry

results = prober.probe_carry_detection(
    carry_examples,
    carry_labels,
    layer_name="block_2",
    carry_position_fn=lambda text: text.index('+')
)

# Probe across all layers
all_layer_results = prober.probe_across_layers(
    examples=["12 + 34", "56 * 78"],
    labels=[0, 1],  # 0=add, 1=mul
    probe_type="operation_type"
)

# Visualize results
prober.visualize_probe_results_across_layers(all_layer_results, "operation_type")
```

**Key Findings to Look For:**
- When does carry information become linearly accessible?
- Which layers encode digit values?
- When does the model "know" what operation it's doing?
- Accuracy jumping from chance to high = information emerging at that layer

### 6. Intervention Experiments

**Purpose:** Measure component importance through systematic ablation.

**Key Features:**
- Ablate individual attention heads
- Ablate entire layers (attention or feedforward)
- Cumulative ablation experiments
- Multiple ablation types (zero, mean, identity)

**Example Usage:**

```python
from src.interpretability import InterventionAnalyzer

intervener = InterventionAnalyzer(model, tokenizer)

# Measure importance of each layer
layer_importance = intervener.measure_layer_importance(
    examples=["12 + 34", "56 * 78"],
    component="attn",
    ablation_type="zero"
)

# Measure importance of each head (expensive!)
head_importance = intervener.measure_head_importance(
    examples=["12 + 34"],
    ablation_type="zero"
)

# Visualize
intervener.visualize_layer_importance(layer_importance, save_path="layer_imp.png")
intervener.visualize_head_importance(head_importance, save_path="head_imp.png")

# Cumulative ablation
sorted_heads = sorted(head_importance.items(), key=lambda x: x[1], reverse=True)
cumulative_results = intervener.cumulative_ablation_experiment(
    examples=["12 + 34"],
    sorted_heads=sorted_heads,
    max_ablations=5
)
```

**Key Findings to Look For:**
- Which heads are critical? (large loss increase when ablated)
- Which layers matter most?
- Are effects additive? (cumulative ablation)
- Negative importance = head was hurting performance!

## Typical Analysis Workflow

### Phase 1: Exploratory Analysis

1. **Start with attention patterns** to get intuition
2. **Use logit lens** to see how predictions evolve
3. **Run circuit discovery** to identify specialized components

### Phase 2: Hypothesis Testing

4. **Use probing** to test specific hypotheses about what's encoded
5. **Use activation patching** to establish causal relationships

### Phase 3: Validation

6. **Use interventions** to confirm which components are necessary
7. **Compare** models trained with different settings

## Configuration

Edit `configs/interpretability.yaml` to customize:

```yaml
attention:
  enabled: true
  layers_to_analyze: [0, 2, 4]
  find_specialized_heads: true

circuits:
  carry_detection:
    examples_with_carry:
      - text: "59 + 73"
        carry_positions: [0, 1]
    threshold: 0.3

probing:
  probes:
    - name: "carry_detection"
      type: "binary"
```

## Output Structure

Analysis results are saved to `reports/interpretability/`:

```
reports/interpretability/
├── attention/
│   ├── example_0_layer_0_all_heads.png
│   ├── example_0_layer_0_head_0.png
│   └── example_0_stats.json
├── patching/
│   └── patching_results.json
├── logit_lens/
│   ├── example_0_evolution.png
│   ├── example_0_predictions.png
│   └── example_0_entropy.json
├── circuits/
│   ├── operator_heads.json
│   ├── position_heads.json
│   ├── specialized_neurons.json
│   └── head_specialization.png
├── probing/
│   ├── has_addition_results.json
│   └── has_addition_results.png
└── interventions/
    ├── layer_importance.json
    └── layer_importance.png
```

## Tips for Mathematical Reasoning Analysis

### For Addition
- Look for **carry detection heads** that attend to positions where carries occur
- Check if **digit sum neurons** activate when processing digit pairs
- Use probing to test if carry information is encoded

### For Multiplication
- Look for **partial product heads** that track intermediate results
- Check for **position tracking** to align digits correctly
- Test if model uses long multiplication algorithm

### For Base Conversion
- Look for **power-of-base neurons** that represent 2^n or 16^n
- Check if heads attend to digit positions systematically
- Test if division-by-base is encoded in activations

## Common Pitfalls

1. **Don't over-interpret single examples** - always test on multiple examples
2. **High attention ≠ causal importance** - use patching to establish causality
3. **Probing accuracy can be misleading** - high accuracy might be from spurious features
4. **Layer importance is context-dependent** - results vary by task and training

## Advanced Usage

### Custom Probes

```python
# Define custom position function
def get_carry_position(text: str) -> int:
    # Find position where carry should occur
    return text.index('+') + 2

# Use in probing
results = prober.probe_carry_detection(
    examples,
    labels,
    "block_2",
    carry_position_fn=get_carry_position
)
```

### Custom Circuit Discovery

```python
# Define custom attention pattern
def check_custom_pattern(attn_weights, positions):
    # Your custom logic
    return score

# Integrate into circuit discovery
circuit = discovery.analyze_circuit_for_example(example, "custom_operation")
```

## Performance Notes

- **Attention analysis**: Fast, run on all examples
- **Logit lens**: Fast, run on all examples
- **Activation patching**: Moderate, subset of examples
- **Circuit discovery**: Fast for attention, slower for neurons
- **Probing**: Fast with small datasets, needs 100+ examples
- **Interventions**: Very slow (O(layers × heads)), use sparingly

## References

- **Logit Lens**: nostalgebraist (2020)
- **Activation Patching**: Meng et al. (2022) "Locating and Editing Factual Associations"
- **Circuit Discovery**: Elhage et al. (2021) "A Mathematical Framework for Transformer Circuits"
- **Probing**: Belinkov (2021) "Probing Classifiers: Promises, Shortcomings, and Advances"

## Troubleshooting

**"Hook registration failed"**
- Make sure model is in eval mode
- Check that layer names match model architecture

**"OOM error during head ablation"**
- Reduce number of examples
- Use smaller batch size
- Test fewer heads at once

**"Probing accuracy at chance"**
- Check if labels are correct
- Try different layers
- Increase number of training examples

**"All attention heads look similar"**
- Try different tasks (multiplication vs addition)
- Look at later layers
- Use carry-heavy examples

## Support

For issues or questions:
1. Check the main project README
2. Review example outputs in `reports/`
3. Open an issue on GitHub
