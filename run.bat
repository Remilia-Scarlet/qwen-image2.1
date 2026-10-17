@echo off
setlocal
cd /d %~dp0

set HF_HOME=%~dp0.hf
set HF_ENDPOINT=https://hf-mirror.com
set HF_HUB_DISABLE_XET=1
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
set HF_HUB_DOWNLOAD_TIMEOUT=60

if exist .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
)

python generate.py %*
endlocal
