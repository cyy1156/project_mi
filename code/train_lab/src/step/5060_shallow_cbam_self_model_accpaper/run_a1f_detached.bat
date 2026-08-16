@echo off
cd /d "%~dp0"
echo %DATE% %TIME% launching A1 5-fold >> a1f_launch.log
start "scheme15_A1f" /MIN "D:\cyy\MI\.venv\Scripts\python.exe" -u run_arm.py --arm A1 --max-folds 0 --num-workers 0
echo started > a1f_running.flag
