@echo off
REM Quickstart script for math-compact project (Windows)
REM This script demonstrates a complete workflow from dataset generation to training

echo ==========================================
echo Math-Compact Quickstart
echo ==========================================
echo.

REM Step 1: Generate dataset
echo Step 1: Generating addition dataset...
uv run -- python -m src.data.generate --task add --instances 1000 --max_length 10 --with_steps --seed 42 --output_dir data

if %errorlevel% neq 0 (
    echo Error generating dataset
    exit /b %errorlevel%
)

echo.
echo Dataset generated successfully!
echo Files created:
echo   - data/add/train.jsonl (800 examples^)
echo   - data/add/dev.jsonl (100 examples^)
echo   - data/add/test.jsonl (100 examples^)
echo.

REM Step 2: Train model (small version for quick testing)
echo Step 2: Training compact transformer...
echo Note: Training for 5 epochs on small dataset for demonstration
echo.

uv run -- python -m src.train --config configs/train.yaml --output-dir runs/quickstart --seed 42 --device cpu

if %errorlevel% neq 0 (
    echo Error during training
    exit /b %errorlevel%
)

echo.
echo Training complete!
echo Checkpoints saved to: runs/quickstart/
echo.

REM Step 3: Evaluate model
echo Step 3: Evaluating trained model...
echo.

uv run -- python -m src.eval --checkpoint runs/quickstart/checkpoint_best.pt --data data/add/test.jsonl --device cpu --output metrics/quickstart_eval.json

if %errorlevel% neq 0 (
    echo Error during evaluation
    exit /b %errorlevel%
)

echo.
echo ==========================================
echo Quickstart Complete!
echo ==========================================
echo.
echo Next steps:
echo   - Check training metrics: metrics/training_metrics_seed42.json
echo   - Check evaluation results: metrics/quickstart_eval.json
echo   - Generate larger datasets with more tasks
echo   - Adjust hyperparameters in configs/train.yaml
echo   - Train on GPU with --device cuda
echo.

pause
