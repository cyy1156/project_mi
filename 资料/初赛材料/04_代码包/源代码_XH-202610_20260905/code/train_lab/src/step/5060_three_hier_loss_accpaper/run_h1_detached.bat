@echo off
cd /d "%~dp0"
echo %DATE% %TIME% launching H1 three-only >> h1_launch.log
start "scheme16_H1" /MIN "D:\cyy\MI\.venv\Scripts\python.exe" -u run_arm.py --arm H1 --num-workers 0
echo started > h1_running.flag
