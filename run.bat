@echo off
setlocal
cd /d %~dp0

set HF_HOME=%~dp0.hf
set HF_ENDPOINT=https://hf-mirror.com

if exist .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
)

python generate.py %*
endlocal
