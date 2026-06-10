@echo off
setlocal

cd /d "%~dp0\.."
call conda activate light
if errorlevel 1 exit /b %errorlevel%

set "PYTHONPATH=%CD%;%PYTHONPATH%"

set "GPU_IDS=%~1"
if "%GPU_IDS%"=="" set "GPU_IDS=0"

python scripts/train_sdsd_indoor.py --gpus "%GPU_IDS%"
