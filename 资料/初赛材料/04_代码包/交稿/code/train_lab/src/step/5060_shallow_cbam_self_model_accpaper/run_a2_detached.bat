@echo off
cd /d "%~dp0"
echo %DATE% %TIME% launching A2 >> a2_launch.log
start "scheme15_A2" /MIN "D:\cyy\MI\.venv\Scripts\python.exe" -u run_arm.py --arm A2 --num-workers 0
echo started > a2_running.flag
