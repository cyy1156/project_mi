@echo off
cd /d "%~dp0"
echo %DATE% %TIME% launching scheme16-5090 full chain >> chain_detached_launch.log
REM 请先 conda activate cyy；否则回退仓库 .venv 或 PATH 上的 python
set "REPO=%~dp0..\..\..\..\.."
set "PY=%REPO%\.venv\Scripts\python.exe"
if not exist "%PY%" set PY=python
start "scheme16_5090_chain" /MIN "%PY%" -u chain_all.py
echo started > chain_detached.flag
echo launched with %PY%
