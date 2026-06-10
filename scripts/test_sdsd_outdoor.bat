@echo off
setlocal

cd /d "%~dp0\.."
call conda activate light
if errorlevel 1 exit /b %errorlevel%

set "PYTHONPATH=%CD%;%PYTHONPATH%"

set "WEIGHTS=%~1"
if "%WEIGHTS%"=="" set "WEIGHTS=experiments/RetinexFormer_SDSD_outdoor/models/net_g_latest.pth"
if not "%~1"=="" shift

set "GPU_IDS=%~1"
if "%GPU_IDS%"=="" set "GPU_IDS=0"
if not "%~1"=="" shift

set "OUTPUT_DIR=%~1"
if "%OUTPUT_DIR%"=="" set "OUTPUT_DIR=results/SDSD_outdoor/enhanced"

python scripts/test_sdsd_outdoor.py --weights "%WEIGHTS%" --gpus "%GPU_IDS%" --output_dir "%OUTPUT_DIR%"
