@echo off
setlocal EnableExtensions
pushd "%~dp0.."
if errorlevel 1 (
  echo ERROR: cannot cd to repo root from %~dp0
  pause
  exit /b 1
)

rem LAN mode: bind 0.0.0.0 for remote operator.html on same network
rem Firewall once as admin - see docs/remote_monitor_scheme_B.md

set "PY="
if exist "%CD%\.venv\Scripts\python.exe" set "PY=%CD%\.venv\Scripts\python.exe"
if not defined PY if exist "%USERPROFILE%\.conda\envs\cyy\python.exe" set "PY=%USERPROFILE%\.conda\envs\cyy\python.exe"
if not defined PY (
  echo ERROR: Python not found. Checked:
  echo   %CD%\.venv\Scripts\python.exe
  echo   %USERPROFILE%\.conda\envs\cyy\python.exe
  echo.
  echo Fix: conda activate cyy
  echo      cd /d %CD%
  echo      python -m experiment_game.tools.open_operator --host 0.0.0.0
  pause
  exit /b 1
)

echo.
echo === Operator console ^(LAN^) ===
echo Bind: 0.0.0.0
echo.
echo IMPORTANT for monitor PC:
echo   Do NOT open bare http://LAN-IP:8080/operator.html
echo   Prefer bookmarking the URL with ?token=... ^(token is fixed in config^)
echo   config: experiment_game\config\ws_control_token.txt
echo.
echo Subject screen on THIS PC: http://127.0.0.1:8080/
echo Repo root: %CD%
echo Python: %PY%
echo Keep this window open. Close it to stop the server.
echo First time: allow firewall for ports 8080 and 8765 ^(inbound^).
echo.

"%PY%" -m experiment_game.tools.open_operator --host 0.0.0.0 %*
set "ERR=%ERRORLEVEL%"
echo.
if not "%ERR%"=="0" (
  echo Exit code: %ERR%
  pause
)
popd
exit /b %ERR%
