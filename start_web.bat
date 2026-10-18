@echo off
setlocal
pushd "%~dp0"
if errorlevel 1 exit /b 1

set "HF_HOME=%~dp0.hf"
set "HF_ENDPOINT=https://hf-mirror.com"
set "HF_HUB_DISABLE_XET=1"
set "HF_HUB_DISABLE_SYMLINKS_WARNING=1"
set "HF_HUB_DOWNLOAD_TIMEOUT=60"

if exist "%~dp0.venv\Scripts\activate.bat" (
    call "%~dp0.venv\Scripts\activate.bat"
)

python -B "%~dp0server.py" %*
set "WEB_EXIT_CODE=%ERRORLEVEL%"
popd
exit /b %WEB_EXIT_CODE%
