@echo off
setlocal EnableExtensions
REM machine_id=5070_laptop
cd /d "%~dp0..\..\.."
if errorlevel 1 (
  echo ERROR: cannot cd to repo root
  pause
  exit /b 1
)
set "REPO=%CD%"
call "%~dp0_resolve_python.bat"
if not defined PY (
  echo ERROR: Python not found. conda activate cyy first.
  pause
  exit /b 1
)
echo [5070_laptop] Using: %PY%
"%PY%" "%~dp0check_deps.py"
set "ERR=%ERRORLEVEL%"
echo.
pause
exit /b %ERR%
