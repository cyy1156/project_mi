@echo off
setlocal EnableExtensions
cd /d "%~dp0.."
if errorlevel 1 (
  echo ERROR: cannot cd to repo root from %~dp0
  pause
  exit /b 1
)

rem Python 解析顺序：仓库 .venv → conda cyy 环境（C:\Users\yy\.conda\envs\cyy）
set "PY=%CD%\.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=%USERPROFILE%\.conda\envs\cyy\python.exe"
if not exist "%PY%" (
  echo ERROR: Python not found. Checked:
  echo   %CD%\.venv\Scripts\python.exe
  echo   %USERPROFILE%\.conda\envs\cyy\python.exe
  echo Create one of them, or: conda create -n cyy python=3.13
  echo then pip install -r experiment_game\requirements.txt pylsl brainflow pyyaml
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