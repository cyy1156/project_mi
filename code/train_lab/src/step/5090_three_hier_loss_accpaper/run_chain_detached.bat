@echo off
cd /d "%~dp0"
echo %DATE% %TIME% launching scheme16-5090 full chain >> chain_detached_launch.log
REM Prefer repo venv; fall back to py launcher
set PY=D:\cyy\MI\.venv\Scripts\python.exe
if not exist "%PY%" set PY=python
start "scheme16_5090_chain" /MIN "%PY%" -u chain_all.py
echo started > chain_detached.flag
echo launched with %PY%
