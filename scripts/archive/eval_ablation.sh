#!/bin/bash
# Helper script to run ablation evaluation on GPU
# Usage: bash eval_ablation.sh [checkpoint_path]

CHECKPOINT=${1:-runs/pilot_test/checkpoint_best.pt}

echo "Running ablation evaluation..."
echo "Checkpoint: $CHECKPOINT"
echo ""

.venv/Scripts/python.exe eval_pilot_ablation.py --checkpoint "$CHECKPOINT" --device cuda --batch-size 32
