@echo off
REM Sequential P1 then P2 full paper-config runs (8ch)
set PYTHON=D:\cyy\MI\.venv\Scripts\python.exe
set DIR=D:\cyy\MI\code\train_lab\src\step\5060_ciacnet_mi_accpaper
cd /d %DIR%
echo ==== START P1 %DATE% %TIME% ====
%PYTHON% -u run_p_track.py --arm P1
echo ==== START P2 %DATE% %TIME% ====
%PYTHON% -u run_p_track.py --arm P2
echo ==== ALL DONE %DATE% %TIME% ====
