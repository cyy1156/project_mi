@echo off
REM Exp34 全量实验（5070 / conda cyy）
REM 日志: 资料\模型训练\34_...\logs\
setlocal
set PY=%USERPROFILE%\.conda\envs\cyy\python.exe
if not exist "%PY%" set PY=python
set SCRIPT=%~dp0run_exp34_full.py
echo Using %PY%
"%PY%" "%SCRIPT%" --resume %*
echo ExitCode=%ERRORLEVEL%
pause
