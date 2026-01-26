@echo off
REM Pilot Model Training Script (0.66M parameters)

echo ==========================================
echo Pilot Model Training (0.66M params)
echo ==========================================
echo.

call .venv\Scripts\activate.bat

python -m src.train ^
    --config configs/pilot.yaml ^
    --output-dir runs/pilot ^
    --seed 42 ^
    --device cuda

echo.
echo ==========================================
echo Training complete!
echo ==========================================
