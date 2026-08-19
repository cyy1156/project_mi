@echo off
cd /d "%~dp0"
if "%~1"=="" (
  echo Usage: run_u_chain_resume.bat FROM_ARM
  echo Example: run_u_chain_resume.bat U3
  echo Example: run_u_chain_resume.bat U13
  exit /b 1
)
set "REPO=%~dp0..\..\..\..\.."
set PY=python
where python >nul 2>&1
if errorlevel 1 (
  if exist "%REPO%\.venv\Scripts\python.exe" set "PY=%REPO%\.venv\Scripts\python.exe"
)
"%PY%" -u chain_u_all.py --from %1
