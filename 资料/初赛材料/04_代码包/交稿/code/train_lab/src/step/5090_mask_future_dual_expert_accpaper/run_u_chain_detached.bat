@echo off
cd /d "%~dp0"
echo %DATE% %TIME% launching U-series full chain >> u_chain_detached_launch.log
REM 5090 · conda activate cyy 后亦可直接双击；路径相对仓库根
set "REPO=%~dp0..\..\..\..\.."
set PY=python
where python >nul 2>&1
if errorlevel 1 (
  if exist "%REPO%\.venv\Scripts\python.exe" set "PY=%REPO%\.venv\Scripts\python.exe"
)
start "mask_future_5090_u_chain" /MIN "%PY%" -u chain_u_all.py
echo started > u_chain_detached.flag
echo launched U-chain with %PY%
