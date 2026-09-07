@echo off
cd /d "%~dp0"
echo %DATE% %TIME% launching A1 >> a1_launch.log
start "scheme15_A1" /MIN "D:\cyy\MI\.venv\Scripts\python.exe" -u run_arm.py --arm A1 --num-workers 0
echo started > a1_running.flag
