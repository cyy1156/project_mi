@echo off
cd /d "%~dp0"
echo %DATE% %TIME% launching mask-future-5070 gate chain (fold0) >> chain_detached_launch.log
set "REPO=%~dp0..\..\..\..\.."
set "PY=%REPO%\.venv\Scripts\python.exe"
if not exist "%PY%" set PY=python
start "mask_future_5070_chain" /MIN "%PY%" -u chain_all.py
echo started > chain_detached.flag
echo launched with %PY%
