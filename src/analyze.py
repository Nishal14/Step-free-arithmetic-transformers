"""
Main Interpretability Analysis Script.

Comprehensive analysis of trained models using all interpretability tools.
"""

import argparse
import json
from pathlib import Path
import torch
from typing import List, Dict, Optional
import matplotlib.pyplot as plt

from src.interpretability import (
    AttentionAnalyzer,
    ActivationPatcher,
    LogitLens,
    CircuitDiscovery,
    ProbingClassifier,
    InterventionAnalyzer,
    load_model_and_tokenizer
)


def analyze_attention_patterns(
    analyzer: AttentionAnalyzer,
    examples: List[str],
    output_dir: Path
):
    """Run attention pattern analysis."""
    print("\n" + "="*60)
    print("ATTENTION PATTERN ANALYSIS")
    print("="*60)

    attn_dir = output_dir / "attention"
    attn_dir.mkdir(parents=True, exist_ok=True)

    for idx, example in enumerate(examples):
        print(f"\nAnalyzing: {example}")

        # Plot all heads in layer 0
        fig = analyzer.plot_attention_heads(
            example,
            layer_idx=0,
            save_path=attn_dir / f"example_{idx}_layer_0_all_heads.png"
        )
        plt.close(fig)

        # Plot specific head in detail
        fig = analyzer.plot_attention_single_head(
            example,
            layer_idx=0,
            head_idx=0,
            save_path=attn_dir / f"example_{idx}_layer_0_head_0.png"
        )
        plt.close(fig)

        # Attention statistics
        stats = analyzer.analyze_attention_statistics(example)
        with open(attn_dir / f"example_{idx}_stats.json", 'w') as f:
            json.dump(stats, f, indent=2)

    print(f"\nAttention analysis saved to {attn_dir}")


def analyze_activation_patching(
    patcher: ActivationPatcher,
    clean_examples: List[str],
    corrupted_examples: List[str],
    output_dir: Path
):
    """Run activation patching experiments."""
    print("\n" + "="*60)
    print("ACTIVATION PATCHING ANALYSIS")
    print("="*60)

    patch_dir = output_dir / "patching"
    patch_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for idx, (clean, corrupted) in enumerate(zip(clean_examples, corrupted_examples)):
        print(f"\nPatching: Clean='{clean}' vs Corrupted='{corrupted}'")

        # Compute patching effects
        effects = patcher.compute_patching_effect(
            clean,
            corrupted,
            target_token_pos=-2  # Second to last token
        )

        print("Layer patching effects:")
        for layer, effect in sorted(effects.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {layer}: {effect:.3f}")

        results.append({
            "clean": clean,
            "corrupted": corrupted,
            "effects": effects
        })

    # Save results
    with open(patch_dir / "patching_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nPatching analysis saved to {patch_dir}")


def analyze_logit_lens(
    lens: LogitLens,
    examples: List[str],
    output_dir: Path
):
    """Run logit lens analysis."""
    print("\n" + "="*60)
    print("LOGIT LENS ANALYSIS")
    print("="*60)

    lens_dir = output_dir / "logit_lens"
    lens_dir.mkdir(parents=True, exist_ok=True)

    for idx, example in enumerate(examples):
        print(f"\nAnalyzing: {example}")

        # Prediction evolution
        fig = lens.visualize_prediction_evolution(
            example,
            position=-2,
            save_path=lens_dir / f"example_{idx}_evolution.png"
        )
        plt.close(fig)

        # Layer predictions heatmap
        fig = lens.visualize_layer_predictions_heatmap(
            example,
            top_k=5,
            save_path=lens_dir / f"example_{idx}_predictions.png"
        )
        plt.close(fig)

        # Entropy analysis
        entropies = lens.compute_prediction_entropy(example)
        with open(lens_dir / f"example_{idx}_entropy.json", 'w') as f:
            json.dump({k: v.tolist() for k, v in entropies.items()}, f, indent=2)

    print(f"\nLogit lens analysis saved to {lens_dir}")


def discover_circuits(
    discovery: CircuitDiscovery,
    examples: List[str],
    operation_labels: List[str],
    output_dir: Path
):
    """Run circuit discovery."""
    print("\n" + "="*60)
    print("CIRCUIT DISCOVERY")
    print("="*60)

    circuit_dir = output_dir / "circuits"
    circuit_dir.mkdir(parents=True, exist_ok=True)

    # Find operator attention heads
    operator_heads = discovery.find_operator_attention_heads(examples)

    print("\nOperator-specialized heads:")
    for op, heads in operator_heads.items():
        print(f"\n{op}:")
        for head, score in list(heads.items())[:3]:
            print(f"  {head}: {score:.3f}")

    # Save results
    with open(circuit_dir / "operator_heads.json", 'w') as f:
        json.dump(operator_heads, f, indent=2)

    # Find position tracking heads
    position_heads = discovery.find_position_tracking_heads(examples)

    print("\nPosition-tracking heads (top 5):")
    for head, score in list(position_heads.items())[:5]:
        print(f"  {head}: {score:.3f}")

    with open(circuit_dir / "position_heads.json", 'w') as f:
        json.dump(position_heads, f, indent=2)

    # Find specialized neurons
    specialized_neurons = discovery.find_specialized_neurons(
        examples,
        operation_labels,
        layer_name="ff_0",
        top_k=10
    )

    print("\nSpecialized neurons:")
    for label, neurons in specialized_neurons.items():
        print(f"\n{label}:")
        for neuron_idx, activation in neurons[:3]:
            print(f"  Neuron {neuron_idx}: {activation:.3f}")

    with open(circuit_dir / "specialized_neurons.json", 'w') as f:
        json.dump(specialized_neurons, f, indent=2)

    # Visualize
    fig = discovery.visualize_head_specialization(
        operator_heads,
        save_path=circuit_dir / "head_specialization.png"
    )
    plt.close(fig)

    print(f"\nCircuit discovery saved to {circuit_dir}")


def run_probing_experiments(
    prober: ProbingClassifier,
    examples: List[str],
    labels: List[int],
    probe_type: str,
    output_dir: Path
):
    """Run probing classifier experiments."""
    print("\n" + "="*60)
    print(f"PROBING EXPERIMENTS: {probe_type}")
    print("="*60)

    probe_dir = output_dir / "probing"
    probe_dir.mkdir(parents=True, exist_ok=True)

    # Probe across layers
    results = prober.probe_across_layers(
        examples,
        labels,
        probe_type,
        position_fn=None  # Use mean pooling
    )

    # Visualize
    fig = prober.visualize_probe_results_across_layers(
        results,
        probe_type,
        save_path=probe_dir / f"{probe_type}_results.png"
    )
    plt.close(fig)

    # Save results
    with open(probe_dir / f"{probe_type}_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nProbing results saved to {probe_dir}")

    return results


def run_intervention_experiments(
    intervener: InterventionAnalyzer,
    examples: List[str],
    output_dir: Path
):
    """Run intervention and ablation experiments."""
    print("\n" + "="*60)
    print("INTERVENTION EXPERIMENTS")
    print("="*60)

    intervention_dir = output_dir / "interventions"
    intervention_dir.mkdir(parents=True, exist_ok=True)

    # Measure layer importance
    print("\nMeasuring layer importance...")
    layer_importance = intervener.measure_layer_importance(
        examples[:5],  # Use subset for speed
        component="attn",
        ablation_type="zero"
    )

    # Visualize
    fig = intervener.visualize_layer_importance(
        layer_importance,
        save_path=intervention_dir / "layer_importance.png"
    )
    plt.close(fig)

    # Save results
    with open(intervention_dir / "layer_importance.json", 'w') as f:
        json.dump(layer_importance, f, indent=2)

    # Note: Head importance measurement is expensive, so we skip it by default
    # Uncomment below to run full head ablation:
    # print("\nMeasuring head importance (this may take a while)...")
    # head_importance = intervener.measure_head_importance(examples[:3])
    # fig = intervener.visualize_head_importance(head_importance, ...)

    print(f"\nIntervention results saved to {intervention_dir}")


def main():
    parser = argparse.ArgumentParser(description="Comprehensive interpretability analysis")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--output-dir", type=str, default="reports/interpretability", help="Output directory")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device")

    # Analysis selection
    parser.add_argument("--attention", action="store_true", help="Run attention analysis")
    parser.add_argument("--patching", action="store_true", help="Run activation patching")
    parser.add_argument("--logit-lens", action="store_true", help="Run logit lens analysis")
    parser.add_argument("--circuits", action="store_true", help="Run circuit discovery")
    parser.add_argument("--probing", action="store_true", help="Run probing experiments")
    parser.add_argument("--interventions", action="store_true", help="Run intervention experiments")
    parser.add_argument("--all", action="store_true", help="Run all analyses")

    # Example inputs
    parser.add_argument("--examples", type=str, nargs="+",
                       default=["12 + 34", "56 * 78", "100 + 200"],
                       help="Example inputs to analyze")

    args = parser.parse_args()

    # Check device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = "cpu"

    # Load model
    print(f"Loading model from {args.checkpoint}")
    model, tokenizer, checkpoint = load_model_and_tokenizer(Path(args.checkpoint), args.device)
    print(f"Model loaded: {model.count_parameters() / 1e6:.2f}M parameters")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Import matplotlib for plotting
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt

    # Initialize analyzers
    attention_analyzer = AttentionAnalyzer(model, tokenizer, args.device)
    patcher = ActivationPatcher(model, tokenizer, args.device)
    logit_lens = LogitLens(model, tokenizer, args.device)
    circuit_discovery = CircuitDiscovery(model, tokenizer, args.device)
    prober = ProbingClassifier(model, tokenizer, args.device)
    intervener = InterventionAnalyzer(model, tokenizer, args.device)

    # Run selected analyses
    run_all = args.all

    if run_all or args.attention:
        analyze_attention_patterns(attention_analyzer, args.examples, output_dir)

    if run_all or args.patching:
        # Create corrupted examples (simple corruption: change one digit)
        corrupted = [ex.replace('1', '9', 1) if '1' in ex else ex for ex in args.examples]
        analyze_activation_patching(patcher, args.examples, corrupted, output_dir)

    if run_all or args.logit_lens:
        analyze_logit_lens(logit_lens, args.examples, output_dir)

    if run_all or args.circuits:
        # Simple operation labels
        operation_labels = ["add" if '+' in ex else "mul" if '*' in ex else "other"
                          for ex in args.examples]
        discover_circuits(circuit_discovery, args.examples, operation_labels, output_dir)

    if run_all or args.probing:
        # Simple binary labels for probing
        labels = [1 if '+' in ex else 0 for ex in args.examples]
        run_probing_experiments(prober, args.examples, labels, "has_addition", output_dir)

    if run_all or args.interventions:
        run_intervention_experiments(intervener, args.examples, output_dir)

    print("\n" + "="*60)
    print(f"Analysis complete! Results saved to {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
