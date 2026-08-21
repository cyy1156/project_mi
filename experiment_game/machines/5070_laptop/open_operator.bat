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
  echo Then: python -m experiment_game.tools.open_operator
  echo Expected example: %%USERPROFILE%%\.conda\envs\cyy\python.exe
  pause
  exit /b 1
)

echo [5070_laptop] Using: %PY%
echo.
echo === Operator console ===
echo Browser: http://127.0.0.1:8080/operator.html#setup
echo Subject: http://127.0.0.1:8080/
echo Keep this window open. Close it to stop the server.
echo.

"%PY%" -m experiment_game.tools.open_operator %*
set "ERR=%ERRORLEVEL%"
echo.
if not "%ERR%"=="0" (
  echo Exit code: %ERR%
  pause
)
exit /b %ERR%
