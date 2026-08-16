@echo off
cd /d "%~dp0"
echo %DATE% %TIME% launching A2 5-fold >> a2f_launch.log
start "scheme15_A2f" /MIN "D:\cyy\MI\.venv\Scripts\python.exe" -u run_arm.py --arm A2 --max-folds 0 --num-workers 0
echo started > a2f_running.flag
