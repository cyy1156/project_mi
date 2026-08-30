@echo off
setlocal EnableExtensions
REM machine_id=5070_laptop · LAN monitor (scheme B)
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
  pause
  exit /b 1
)

echo [5070_laptop] Using: %PY%
echo.
echo === Operator console ^(LAN / scheme B^) ===
echo Bind: 0.0.0.0
echo.
echo IMPORTANT for monitor PC:
echo   Copy the FULL URL printed below ^(must contain ?token=...^)
echo   Do NOT open bare operator.html without token.
echo Subject display stays on this PC: http://127.0.0.1:8080/
echo Keep this window open. Close it to stop the server.
echo First time: allow firewall inbound for TCP 8080 and 8765.
echo.

"%PY%" -m experiment_game.tools.open_operator --host 0.0.0.0 %*
set "ERR=%ERRORLEVEL%"
echo.
if not "%ERR%"=="0" (
  echo Exit code: %ERR%
  pause
)
exit /b %ERR%
