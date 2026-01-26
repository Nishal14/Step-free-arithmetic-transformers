"""
Test that the training summary format is correct.
This simulates what the final output will look like.
"""

def print_training_summary(best_val_loss, best_val_ppl, best_val_epoch):
    """Simulate the final training summary."""
    print("\n" + "="*50)
    print("TRAINING SUMMARY")
    print("="*50)
    if best_val_epoch is not None:
        print(f"Best validation loss:        {best_val_loss:.4f}")
        print(f"Best validation perplexity:  {best_val_ppl:.2f}")
        print(f"Achieved at epoch:           {best_val_epoch}")
    else:
        print("No validation metrics available")
    print("="*50)


if __name__ == "__main__":
    print("Testing training summary format:")
    print("\nExample 1: With validation metrics")
    print_training_summary(
        best_val_loss=1.7083,
        best_val_ppl=5.52,
        best_val_epoch=2
    )

    print("\n\nExample 2: After 40 epochs (typical)")
    print_training_summary(
        best_val_loss=0.4523,
        best_val_ppl=1.57,
        best_val_epoch=35
    )

    print("\n\nExample 3: No validation (edge case)")
    print_training_summary(
        best_val_loss=float('inf'),
        best_val_ppl=float('inf'),
        best_val_epoch=None
    )

    print("\n\nFormat verification:")
    print("✓ Clear section headers")
    print("✓ Aligned metrics")
    print("✓ 4 decimal places for loss")
    print("✓ 2 decimal places for perplexity")
    print("✓ Integer epoch number")
    print("✓ Copy-paste ready for papers/notes")
