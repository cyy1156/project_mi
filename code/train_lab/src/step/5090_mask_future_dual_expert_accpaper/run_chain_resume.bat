@echo off
cd /d "%~dp0"
if "%~1"=="" (
  echo Usage: run_chain_resume.bat P1
  echo Steps: see chain_all.py / arms_registry.CHAIN_ORDER
  exit /b 1
)
echo %DATE% %TIME% resume mask-future-5090 from %~1 >> chain_detached_launch.log
set "REPO=%~dp0..\..\..\..\.."
set "PY=%REPO%\.venv\Scripts\python.exe"
if not exist "%PY%" set PY=python
start "mask_future_5090_resume" /MIN "%PY%" -u chain_all.py --from %~1
echo started > chain_detached.flag
