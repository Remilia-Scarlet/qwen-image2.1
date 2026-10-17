@echo off
setlocal
cd /d %~dp0

set HF_HOME=%~dp0.hf
set HF_ENDPOINT=https://hf-mirror.com
set HF_HUB_DISABLE_XET=1
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
set HF_HUB_DOWNLOAD_TIMEOUT=60

echo Downloading AtomicChat/Qwen-Image-2.1-Turbo-GGUF Q8_0 (7.59GB) into gguf\ ...
echo (resumable: just run this again if it stops)
.venv\Scripts\hf.exe download AtomicChat/Qwen-Image-2.1-Turbo-GGUF Qwen-Image-2.1-Turbo-Q8_0.gguf --local-dir gguf

echo.
echo Done. Use it with:  run.bat --gguf gguf\Qwen-Image-2.1-Turbo-Q8_0.gguf ...
pause
endlocal
