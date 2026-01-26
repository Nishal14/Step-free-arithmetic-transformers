@echo off
REM Helper script to run activation patching experiment
REM Usage: patch_pilot.bat [checkpoint_path]

set CHECKPOINT=%1
if "%CHECKPOINT%"=="" set CHECKPOINT=runs/pilot_test/checkpoint_best.pt

echo Running activation patching experiment...
echo Checkpoint: %CHECKPOINT%
echo.

.venv\Scripts\python.exe patch_pilot.py --checkpoint %CHECKPOINT% --device cuda
