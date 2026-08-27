@echo off
setlocal EnableExtensions
REM machine_id=5070_laptop · do not use shared experiment_game\*.bat on this PC
cd /d "%~dp0..\..\.."
if errorlevel 1 (
  echo ERROR: cannot cd to repo root from %~dp0
  pause
  exit /b 1
)

set "REPO=%CD%"
call "%~dp0_resolve_python.bat"
if not defined PY (
  echo ERROR: Python not found for machine 5070_laptop.
  echo Prefer: conda activate cyy
  echo Then: python -m experiment_game.tools.open_induction
  echo Expected example: %%USERPROFILE%%\.conda\envs\cyy\python.exe
  pause
  exit /b 1
)

echo [5070_laptop] Using: %PY%
echo.
echo === Subject induction page ===
echo Prerequisite: open_operator.bat must already be running.
echo This script only opens http://127.0.0.1:8080/  (does NOT start Phase2).
echo.

"%PY%" -m experiment_game.tools.open_induction %*
set "ERR=%ERRORLEVEL%"
echo.
if not "%ERR%"=="0" (
  echo Exit code: %ERR%
  pause
)
exit /b %ERR%
