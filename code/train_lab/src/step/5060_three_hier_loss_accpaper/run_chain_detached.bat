@echo off
cd /d "%~dp0"
echo %DATE% %TIME% launching scheme16 full chain >> chain_detached_launch.log
start "scheme16_chain" /MIN "D:\cyy\MI\.venv\Scripts\python.exe" -u chain_all.py
echo started > chain_detached.flag
