@echo off
setlocal
call conda activate cyy
python "%~dp0run_exp35_full.py" %*
endlocal
