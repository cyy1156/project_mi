@echo off
setlocal
set "HERE=%~dp0"
set "REPO=%HERE%..\..\.."
call "%HERE%_resolve_python.bat"
if not defined PY (
  echo ERROR: Python not found. Edit machine.json / _resolve_python.bat
  exit /b 1
)
cd /d "%REPO%"
"%PY%" -m experiment_game.tools.preflight %*
exit /b %ERRORLEVEL%
