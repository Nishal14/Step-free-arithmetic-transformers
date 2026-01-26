@echo off
REM GPU Training Helper Script for Windows
REM This script runs training using the CUDA-enabled Python environment

echo Starting GPU Training...
echo.

REM Run training with GPU
.venv\Scripts\python.exe -m src.train ^
  --config configs/train.yaml ^
  --output-dir runs/gpu_training ^
  --device cuda ^
  --seed 42

echo.
echo Training complete!
pause
