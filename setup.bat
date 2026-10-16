@echo off
setlocal
cd /d %~dp0

set HF_HOME=%~dp0.hf
set HF_ENDPOINT=https://hf-mirror.com

if not exist .venv (
    echo [setup] creating venv...
    python -m venv .venv
)

call .venv\Scripts\activate.bat
python -m pip install -U pip
python -m pip install --index-url https://download.pytorch.org/whl/cu130 torch torchvision
pip install -r requirements.txt

echo.
echo [setup] done. HF_HOME = %HF_HOME%
echo [setup] Activate with:  .venv\Scripts\activate
echo [setup] then run:  run.bat --prompt "your prompt"
endlocal
