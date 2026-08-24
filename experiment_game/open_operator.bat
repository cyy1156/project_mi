@echo off
setlocal EnableExtensions
pushd "%~dp0.."
if errorlevel 1 (
  echo ERROR: cannot cd to repo root from %~dp0
  pause
  exit /b 1
)

set "PY="
if exist "%CD%\.venv\Scripts\python.exe" set "PY=%CD%\.venv\Scripts\python.exe"
if not defined PY if exist "%USERPROFILE%\.conda\envs\cyy\python.exe" set "PY=%USERPROFILE%\.conda\envs\cyy\python.exe"
if not defined PY (
  echo ERROR: Python not found. Checked:
  echo   %CD%\.venv\Scripts\python.exe
  echo   %USERPROFILE%\.conda\envs\cyy\python.exe
  echo Create conda env: conda create -n cyy python=3.13
  pause
  exit /b 1
)

echo.
echo === Operator console (local) ===
echo Browser: http://127.0.0.1:8080/operator.html#setup
echo Subject: http://127.0.0.1:8080/
echo Repo root: %CD%
echo Python: %PY%
echo Keep this window open. Close it to stop the server.
echo.

"%PY%" -m experiment_game.tools.open_operator %*
set "ERR=%ERRORLEVEL%"
echo.
if not "%ERR%"=="0" (
  echo Exit code: %ERR%
  pause
)
popd
exit /b %ERR%
