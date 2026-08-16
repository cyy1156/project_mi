@echo off
cd /d "%~dp0"
echo %DATE% %TIME% resume scheme16 from S0_three >> chain_detached_launch.log
start "scheme16_chain" /MIN "D:\cyy\MI\.venv\Scripts\python.exe" -u chain_all.py --from S0_three
echo started > chain_detached.flag
