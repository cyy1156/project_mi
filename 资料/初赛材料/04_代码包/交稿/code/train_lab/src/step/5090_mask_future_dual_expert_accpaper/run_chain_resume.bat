@echo off
cd /d "%~dp0"
if "%~1"=="" (
  echo Usage: run_chain_resume.bat P1
  echo Steps: see chain_all.py / arms_registry.CHAIN_ORDER
  exit /b 1
)
echo %DATE% %TIME% resume mask-future-5090 from %~1 >> chain_detached_launch.log
REM 仓库相对路径（5090 机 F:\Cyy\MI）；请先 conda activate cyy。禁止写死 D:\cyy\MI
set "REPO=%~dp0..\..\..\..\.."
set PY=python
where python >nul 2>&1
if errorlevel 1 (
  if exist "%REPO%\.venv\Scripts\python.exe" set "PY=%REPO%\.venv\Scripts\python.exe"
)
start "mask_future_5090_resume" /MIN "%PY%" -u chain_all.py --from %~1
echo started > chain_detached.flag
