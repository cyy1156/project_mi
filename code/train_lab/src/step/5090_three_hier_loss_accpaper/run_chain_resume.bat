@echo off
cd /d "%~dp0"
if "%~1"=="" (
  echo Usage: run_chain_resume.bat H1_three
  echo Steps: S0_three H1_three H2_three H3_three T0_task
  exit /b 1
)
echo %DATE% %TIME% resume scheme16-5090 from %~1 >> chain_detached_launch.log
set PY=D:\cyy\MI\.venv\Scripts\python.exe
if not exist "%PY%" set PY=python
start "scheme16_5090_resume" /MIN "%PY%" -u chain_all.py --from %~1
echo started > chain_detached.flag
