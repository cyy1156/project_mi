@echo off
REM Detached resume for scheme 14: B2 then S0
cd /d "%~dp0"
echo %DATE% %TIME% launching detached chain --from-arm B2 >> chain_detached_launch.log
start "scheme14_chain_B2S0" /MIN "D:\cyy\MI\.venv\Scripts\python.exe" -u chain_all.py --from-arm B2
echo started > chain_detached.flag
