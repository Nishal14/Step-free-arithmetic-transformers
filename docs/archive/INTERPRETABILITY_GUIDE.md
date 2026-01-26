# Interpretability Tools - Quick Start Guide

## What Was Added

Your math-compact project now includes a comprehensive mechanistic interpretability toolkit with 6 major analysis tools:

```
src/interpretability/
├── __init__.py                  # Module exports
├── README.md                    # Detailed documentation
├── utils.py                     # Shared utilities
├── attention_patterns.py        # Attention visualization
├── activation_patching.py       # Causal interventions
├── logit_lens.py               # Intermediate predictions
├── circuit_discovery.py         # Specialized components
├── probing.py                  # Linear probes
└── interventions.py            # Ablation experiments
```

## Installation

Update dependencies:

```bash
# Sync new dependencies (matplotlib, seaborn, scikit-learn)
uv sync
```

## Quick Start

### 1. Train a Model First

```bash
# Generate data
uv run -- python -m src.data.generate \
    --task add \
    --instances 20000 \
    --max_length 20 \
    --with_steps \
    --seed 42

# Train model
uv run -- python -m src.train \
    --config configs/train.yaml \
    --output-dir runs/seed42 \
    --seed 42 \
    --device cpu
```

### 2. Run Comprehensive Analysis

```bash
# Analyze everything
uv run -- python -m src.analyze \
    --checkpoint runs/seed42/checkpoint_best.pt \
    --all \
    --examples "59 + 73" "12 + 34" "88 + 99"

# Results saved to: reports/interpretability/
```

### 3. Run Specific Analyses

```bash
# Only attention patterns
uv run -- python -m src.analyze \
    --checkpoint runs/seed42/checkpoint_best.pt \
    --attention \
    --examples "59 + 73"

# Attention + logit lens
uv run -- python -m src.analyze \
    --checkpoint runs/seed42/checkpoint_best.pt \
    --attention --logit-lens \
    --examples "59 + 73"

# Circuit discovery + probing
uv run -- python -m src.analyze \
    --checkpoint runs/seed42/checkpoint_best.pt \
    --circuits --probing \
    --examples "59 + 73" "12 * 34"
```

### 4. Try the Example Script

```bash
# Run interactive example
uv run -- python example_interpretability.py
```

## The Six Tools

### 1. Attention Pattern Analysis
**What it does:** Shows which tokens attend to which during computation

**Use cases:**
- Find heads that attend to operator tokens (+, *, etc.)
- Identify carry-detection heads in addition
- Discover position-tracking patterns

**Example:**
```python
from src.interpretability import AttentionAnalyzer, load_model_and_tokenizer

model, tokenizer, _ = load_model_and_tokenizer("runs/model.pt")
analyzer = AttentionAnalyzer(model, tokenizer)

# Visualize all heads in layer 2
analyzer.plot_attention_heads("59 + 73", layer_idx=2, save_path="attention.png")
```

### 2. Activation Patching
**What it does:** Tests causal relationships by patching activations

**Use cases:**
- Determine which layers are causally important
- Find which positions contain critical information
- Test if specific heads are necessary

**Example:**
```python
from src.interpretability import ActivationPatcher

patcher = ActivationPatcher(model, tokenizer)

# Test which layer matters for correct computation
effects = patcher.compute_patching_effect(
    clean_text="12 + 34",
    corrupted_text="12 + 94",  # Wrong digit
    target_token_pos=-2
)
# High recovery score = layer is causally important
```

### 3. Logit Lens
**What it does:** Decodes what model "thinks" at each layer

**Use cases:**
- See when model commits to answer
- Track prediction evolution
- Measure uncertainty (entropy)

**Example:**
```python
from src.interpretability import LogitLens

lens = LogitLens(model, tokenizer)

# Visualize prediction evolution
lens.visualize_prediction_evolution("12 + 34", position=-1, save_path="evolution.png")

# When does model predict correct answer?
convergence = lens.analyze_convergence("12 + 34", target_token_pos=-1, target_token="6")
```

### 4. Circuit Discovery
**What it does:** Identifies specialized heads and neurons

**Use cases:**
- Find carry-detection heads
- Find operator-specialized heads
- Identify digit-recognition neurons

**Example:**
```python
from src.interpretability import CircuitDiscovery

discovery = CircuitDiscovery(model, tokenizer)

# Find heads that detect carries
carry_examples = [("59 + 73", [0, 1]), ("88 + 99", [0, 1])]
carry_heads = discovery.find_carry_detection_heads(carry_examples)

# Find heads that attend to operators
examples = ["12 + 34", "56 * 78"]
operator_heads = discovery.find_operator_attention_heads(examples, ['+', '*'])
```

### 5. Probing Classifiers
**What it does:** Tests what information is linearly accessible

**Use cases:**
- Test if carry information is encoded
- Find when operation type becomes known
- Test if digit values are represented

**Example:**
```python
from src.interpretability import ProbingClassifier

prober = ProbingClassifier(model, tokenizer)

# Probe across all layers: when does model know the operation?
examples = ["12 + 34", "56 * 78"]
labels = [0, 1]  # 0=add, 1=mul

results = prober.probe_across_layers(examples, labels, "operation_type")
# High accuracy = information is encoded at that layer
```

### 6. Intervention Experiments
**What it does:** Measures importance through ablation

**Use cases:**
- Quantify head importance
- Measure layer contribution
- Test if components are necessary

**Example:**
```python
from src.interpretability import InterventionAnalyzer

intervener = InterventionAnalyzer(model, tokenizer)

# Measure layer importance
layer_importance = intervener.measure_layer_importance(
    examples=["12 + 34", "56 * 78"],
    component="attn"
)
# Large delta = layer is important
```

## Typical Research Workflow

### Week 1: Exploratory Analysis
```bash
# Day 1-2: Attention patterns
python -m src.analyze --checkpoint <path> --attention --examples <many examples>

# Day 3-4: Logit lens
python -m src.analyze --checkpoint <path> --logit-lens --examples <many examples>

# Day 5: Initial circuit discovery
python -m src.analyze --checkpoint <path> --circuits --examples <targeted examples>
```

### Week 2: Hypothesis Formation
Based on Week 1, form hypotheses:
- "Layer 2, Head 3 detects carries"
- "Neurons 45-50 in FF layer 1 encode digit sums"
- "Model commits to answer at layer 4"

### Week 3: Hypothesis Testing
```bash
# Test with probing
python -m src.analyze --checkpoint <path> --probing

# Test causality with patching
python -m src.analyze --checkpoint <path> --patching

# Verify importance with interventions
python -m src.analyze --checkpoint <path> --interventions
```

### Week 4: Documentation
- Write up findings
- Create visualizations for paper
- Compare models trained differently

## Configuration

Edit `configs/interpretability.yaml` to customize:

```yaml
# Enable/disable specific analyses
attention:
  enabled: true
  layers_to_analyze: [0, 2, 4]

circuits:
  carry_detection:
    examples_with_carry:
      - text: "59 + 73"
        carry_positions: [0, 1]

probing:
  probes:
    - name: "carry_detection"
      type: "binary"
```

## Output Files

After running analysis, check:

```
reports/interpretability/
├── attention/
│   ├── example_0_layer_0_all_heads.png    # All heads visualization
│   ├── example_0_layer_0_head_0.png       # Single head detail
│   └── example_0_stats.json               # Attention statistics
│
├── logit_lens/
│   ├── example_0_evolution.png            # Prediction evolution
│   ├── example_0_predictions.png          # Layer predictions heatmap
│   └── example_0_entropy.json             # Entropy per layer
│
├── circuits/
│   ├── operator_heads.json                # Operator-specialized heads
│   ├── position_heads.json                # Position-tracking heads
│   ├── specialized_neurons.json           # Operation-specific neurons
│   └── head_specialization.png            # Specialization heatmap
│
├── probing/
│   └── <probe_name>_results.png           # Accuracy across layers
│
└── interventions/
    └── layer_importance.png               # Ablation impact
```

## Performance Tips

**Fast (run on many examples):**
- Attention analysis
- Logit lens
- Circuit discovery (attention-based)

**Moderate (use subset of examples):**
- Activation patching
- Circuit discovery (neuron-based)
- Probing classifiers

**Slow (use few examples):**
- Head importance measurement (O(layers × heads))
- Cumulative ablation experiments

## Common Research Questions

### For Addition Tasks

**Q: Does the model learn carry propagation?**
```bash
# Use circuit discovery + probing
python -m src.analyze --checkpoint <path> --circuits --probing \
    --examples "59 + 73" "88 + 99" "999 + 1"
```

**Q: Which layers compute the answer?**
```bash
# Use logit lens + interventions
python -m src.analyze --checkpoint <path> --logit-lens --interventions \
    --examples "12 + 34"
```

### For Multiplication Tasks

**Q: Does it use long multiplication?**
```bash
# Use attention patterns + circuit discovery
python -m src.analyze --checkpoint <path> --attention --circuits \
    --examples "12 * 34" "56 * 78"
```

### Comparing Models

**Q: What's different between models trained with/without steps?**
```bash
# Analyze both models
python -m src.analyze --checkpoint runs/with_steps/checkpoint_best.pt --all
python -m src.analyze --checkpoint runs/without_steps/checkpoint_best.pt --all

# Compare output files manually or write custom comparison script
```

## Troubleshooting

**"Module not found: matplotlib"**
```bash
uv sync  # Install new dependencies
```

**"Checkpoint not found"**
```bash
# Train a model first
python -m src.train --config configs/train.yaml --output-dir runs/test --seed 42
```

**"OOM during head ablation"**
```bash
# Use fewer examples or disable head ablation in config
# Head ablation is expensive: O(num_layers × num_heads × num_examples)
```

**"All attention heads look the same"**
- Try examples with carries: "59 + 73", "88 + 99"
- Look at later layers (layer 3-5)
- Try different tasks (multiplication vs addition)

## Next Steps

1. **Read the detailed docs:** `src/interpretability/README.md`
2. **Run the example:** `python example_interpretability.py`
3. **Analyze your model:** `python -m src.analyze --checkpoint <path> --all`
4. **Form hypotheses** based on initial visualizations
5. **Test hypotheses** with probing and patching
6. **Document findings** for your research

## Resources

- **Logit Lens:** [nostalgebraist blog](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens)
- **Activation Patching:** [ROME paper](https://arxiv.org/abs/2202.05262)
- **Circuits:** [Anthropic's Circuits Thread](https://transformer-circuits.pub/)
- **Probing:** [Probing Classifiers Survey](https://arxiv.org/abs/2102.12452)

## Support

- Detailed documentation: `src/interpretability/README.md`
- Example script: `example_interpretability.py`
- Configuration: `configs/interpretability.yaml`
- Main README: `README.md`

Happy analyzing! 🔍
