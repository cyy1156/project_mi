@echo off
setlocal EnableExtensions
cd /d "%~dp0.."
if errorlevel 1 (
  echo ERROR: cannot cd to repo root from %~dp0
  pause
  exit /b 1
)

set "PY=%CD%\collect_data\LSL_connect_model\LSL_connect_model\.venv\Scripts\python.exe"
if not exist "%PY%" (
  echo ERROR: Python venv not found:
  echo   %PY%
  echo Create lsl_connect .venv and install experiment_game\requirements.txt
  pause
  exit /b 1
)

echo.
echo === MI induction page ===
echo Browser: http://127.0.0.1:8080/
echo Keep this window open. Close it to stop the server.
echo.

"%PY%" -m experiment_game.tools.open_induction %*
set "ERR=%ERRORLEVEL%"
echo.
if not "%ERR%"=="0" (
  echo Exit code: %ERR%
  pause
)
exit /b %ERR%