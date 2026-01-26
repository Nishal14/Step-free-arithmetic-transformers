@echo off
REM Helper script to run ablation evaluation on GPU
REM Usage: eval_ablation.bat [checkpoint_path]

set CHECKPOINT=%1
if "%CHECKPOINT%"=="" set CHECKPOINT=runs/pilot_test/checkpoint_best.pt

echo Running ablation evaluation...
echo Checkpoint: %CHECKPOINT%
echo.

.venv\Scripts\python.exe eval_pilot_ablation.py --checkpoint %CHECKPOINT% --device cuda --batch-size 32
