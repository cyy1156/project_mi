@echo off
cd /d "%~dp0"
echo %DATE% %TIME% launching mask-future-5090 full chain >> chain_detached_launch.log
REM 仓库相对路径（5090 机为 F:\Cyy\MI）；请先 conda activate cyy。禁止写死 D:\cyy\MI
set "REPO=%~dp0..\..\..\..\.."
set PY=python
where python >nul 2>&1
if errorlevel 1 (
  if exist "%REPO%\.venv\Scripts\python.exe" set "PY=%REPO%\.venv\Scripts\python.exe"
)
start "mask_future_5090_chain" /MIN "%PY%" -u chain_all.py
echo started > chain_detached.flag
echo launched with %PY%
