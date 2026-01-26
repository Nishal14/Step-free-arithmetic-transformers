"""
Example script demonstrating interpretability tools usage.

This script shows how to use each interpretability tool to analyze
a trained model's behavior on mathematical reasoning tasks.
"""

from pathlib import Path
from src.interpretability import (
    AttentionAnalyzer,
    ActivationPatcher,
    LogitLens,
    CircuitDiscovery,
    ProbingClassifier,
    InterventionAnalyzer,
    load_model_and_tokenizer
)


def main():
    # Load trained model
    checkpoint_path = Path("runs/seed42/checkpoint_best.pt")

    if not checkpoint_path.exists():
        print(f"Checkpoint not found at {checkpoint_path}")
        print("Please train a model first using:")
        print("  uv run -- python -m src.train --config configs/train.yaml --output-dir runs/seed42 --seed 42")
        return

    print("Loading model...")
    model, tokenizer, checkpoint = load_model_and_tokenizer(checkpoint_path, device="cpu")
    print(f"Model loaded: {model.count_parameters() / 1e6:.2f}M parameters\n")

    # Example inputs
    addition_example = "59 + 73"  # Has carry
    simple_example = "12 + 34"   # No carry

    # ========================================
    # 1. ATTENTION PATTERN ANALYSIS
    # ========================================
    print("="*60)
    print("1. ATTENTION PATTERN ANALYSIS")
    print("="*60)

    analyzer = AttentionAnalyzer(model, tokenizer)

    # Get attention patterns
    input_ids, attention_maps, tokens = analyzer.get_attention_for_example(addition_example)
    print(f"\nTokens: {tokens}")
    print(f"Number of layers: {len(attention_maps)}")

    # Compute attention statistics
    stats = analyzer.analyze_attention_statistics(addition_example)
    print(f"\nAttention statistics for '{addition_example}':")
    for layer, layer_stats in list(stats.items())[:2]:  # Show first 2 layers
        print(f"\n{layer}:")
        for head, head_stats in list(layer_stats.items())[:2]:  # Show first 2 heads
            print(f"  {head}:")
            print(f"    Mean entropy: {head_stats['mean_entropy']:.3f}")
            print(f"    Max attention: {head_stats['max_attention']:.3f}")

    # ========================================
    # 2. LOGIT LENS
    # ========================================
    print("\n" + "="*60)
    print("2. LOGIT LENS - PREDICTION EVOLUTION")
    print("="*60)

    lens = LogitLens(model, tokenizer)

    # Get top predictions at each layer for the last position
    predictions = lens.get_top_k_predictions(simple_example, position=-2, k=3)

    print(f"\nTop-3 predictions for '{simple_example}' at position -2:")
    for layer, top_preds in list(predictions.items())[:3]:  # Show first 3 layers
        print(f"\n{layer}:")
        for token, prob in top_preds:
            print(f"  '{token}': {prob:.3f}")

    # Analyze convergence to correct answer
    convergence = lens.analyze_convergence(simple_example, target_token_pos=-2, target_token="4")
    print(f"\nConvergence to '4' at position -2:")
    for layer, prob in list(convergence.items())[:3]:
        print(f"  {layer}: {prob:.3f}")

    # ========================================
    # 3. CIRCUIT DISCOVERY
    # ========================================
    print("\n" + "="*60)
    print("3. CIRCUIT DISCOVERY")
    print("="*60)

    discovery = CircuitDiscovery(model, tokenizer)

    # Find operator attention heads
    examples = [simple_example, addition_example, "12 * 34"]
    operator_heads = discovery.find_operator_attention_heads(examples, operators=['+', '*'])

    print("\nOperator-specialized heads:")
    for op, heads in operator_heads.items():
        print(f"\n{op}:")
        for head, score in list(heads.items())[:3]:  # Show top 3
            print(f"  {head}: {score:.3f}")

    # Find position tracking heads
    position_heads = discovery.find_position_tracking_heads(examples)
    print("\nTop 5 position-tracking heads:")
    for head, score in list(position_heads.items())[:5]:
        print(f"  {head}: {score:.3f}")

    # ========================================
    # 4. ACTIVATION PATCHING
    # ========================================
    print("\n" + "="*60)
    print("4. ACTIVATION PATCHING")
    print("="*60)

    patcher = ActivationPatcher(model, tokenizer)

    # Test patching: correct vs incorrect example
    clean_text = "12 + 34"
    corrupted_text = "12 + 94"  # Wrong digit

    print(f"\nPatching experiment:")
    print(f"  Clean: {clean_text}")
    print(f"  Corrupted: {corrupted_text}")

    effects = patcher.compute_patching_effect(
        clean_text,
        corrupted_text,
        target_token_pos=-2
    )

    print("\nLayer patching effects (recovery scores):")
    for layer, effect in sorted(effects.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {layer}: {effect:.3f}")

    # ========================================
    # 5. PROBING CLASSIFIERS
    # ========================================
    print("\n" + "="*60)
    print("5. PROBING CLASSIFIERS")
    print("="*60)

    prober = ProbingClassifier(model, tokenizer)

    # Simple probe: does input contain addition?
    probe_examples = [
        "12 + 34", "56 + 78", "99 + 11",  # Addition
        "12 * 34", "56 * 78", "99 * 11"   # Multiplication
    ]
    probe_labels = [1, 1, 1, 0, 0, 0]  # 1 = has '+', 0 = no '+'

    print("\nTraining probe: 'has addition operator'")
    results = prober.probe_across_layers(
        probe_examples,
        probe_labels,
        "has_addition"
    )

    print("\nProbing results (validation accuracy):")
    for layer, layer_results in list(results.items())[:3]:
        val_acc = layer_results.get('val_accuracy', 0.0)
        print(f"  {layer}: {val_acc:.3f}")

    # ========================================
    # 6. INTERVENTION EXPERIMENTS
    # ========================================
    print("\n" + "="*60)
    print("6. INTERVENTION EXPERIMENTS")
    print("="*60)

    intervener = InterventionAnalyzer(model, tokenizer)

    print("\nMeasuring layer importance (this may take a moment)...")
    test_examples = [simple_example, addition_example]

    layer_importance = intervener.measure_layer_importance(
        test_examples,
        component="attn",
        ablation_type="zero"
    )

    print("\nLayer importance (delta in loss when ablated):")
    for layer, importance in sorted(layer_importance.items(), key=lambda x: x[1], reverse=True):
        print(f"  {layer}: {importance:.4f}")

    # ========================================
    # SUMMARY
    # ========================================
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("""
This example demonstrated all six interpretability tools:

1. Attention Analysis - Found which tokens the model attends to
2. Logit Lens - Tracked prediction evolution across layers
3. Circuit Discovery - Identified specialized heads and neurons
4. Activation Patching - Tested causal relationships
5. Probing Classifiers - Measured what information is encoded
6. Interventions - Quantified component importance

For detailed analysis with visualizations, run:
    python -m src.analyze --checkpoint <path> --all --examples "59 + 73"

See src/interpretability/README.md for comprehensive documentation.
    """)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")  # Suppress matplotlib warnings

    main()
