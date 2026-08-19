@echo off
cd /d "%~dp0"
if "%~1"=="" (
  echo Usage: run_chain_resume.bat P1
  echo Default gate steps: A0_ref A0 A1 P0 A2 P1 P2
  echo Full chain: python chain_all.py --full-chain --from P1
  exit /b 1
)
echo %DATE% %TIME% resume mask-future-5070 from %~1 >> chain_detached_launch.log
set "REPO=%~dp0..\..\..\..\.."
set "PY=%REPO%\.venv\Scripts\python.exe"
if not exist "%PY%" set PY=python
start "mask_future_5070_resume" /MIN "%PY%" -u chain_all.py --from %~1
echo started > chain_detached.flag
