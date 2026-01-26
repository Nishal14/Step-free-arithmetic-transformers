@echo off
REM 10M Scaling Experiment Training Script (Windows)
REM GPU-ONLY with FP16 mixed precision
REM
REM This script will ABORT if CUDA is not available.
REM Do NOT run on CPU.

echo ==========================================
echo 10M Scaling Experiment
echo ==========================================
echo.

REM Activate virtual environment directly (NOT uv run - prevents CPU PyTorch sync)
call .venv\Scripts\activate.bat

REM Check CUDA availability
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'" 2>nul
if errorlevel 1 (
    echo ERROR: CUDA not available. Aborting.
    echo This experiment requires GPU.
    echo.
    echo Run: uv pip install --reinstall --no-deps torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    exit /b 1
)

echo CUDA detected
echo.

REM Print GPU info
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}'); print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')"
echo.

REM Verify config
echo Verifying configuration...
python verify_10m_config.py
if errorlevel 1 exit /b 1
echo.

REM Training command
echo Starting training...
echo.

python -m src.train ^
    --config configs/scale_10m.yaml ^
    --output-dir runs/scale_10m ^
    --seed 123 ^
    --device cuda ^
    --fp16 ^
    --gpu-only

if errorlevel 1 (
    echo.
    echo ERROR: Training failed!
    exit /b 1
)

echo.
echo ==========================================
echo Training complete!
echo ==========================================
