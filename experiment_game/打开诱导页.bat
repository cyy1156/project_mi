@echo off
setlocal EnableExtensions
cd /d "%~dp0.."
if errorlevel 1 (
  echo ERROR: cannot cd to repo root from %~dp0
  pause
  exit /b 1
)

set "PY=%CD%\.venv\Scripts\python.exe"
if not exist "%PY%" (
  echo ERROR: Python venv not found:
  echo   %PY%
  echo Create repo-root .venv and: pip install -r requirements.txt
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