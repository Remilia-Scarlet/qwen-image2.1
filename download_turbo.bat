@echo off
setlocal
cd /d %~dp0

set HF_HOME=%~dp0.hf
set HF_ENDPOINT=https://hf-mirror.com
set HF_HUB_DISABLE_XET=1
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
set HF_HUB_DOWNLOAD_TIMEOUT=60

echo Downloading official Qwen/Qwen-Image-2.1-Turbo (~30GB) into .hf cache ...
echo (resumable: just run this again if it stops)
.venv\Scripts\hf.exe download Qwen/Qwen-Image-2.1-Turbo

echo.
echo Done. Run with:  run.bat --prompt "your prompt"
pause
endlocal
