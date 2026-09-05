@echo off
setlocal
cd /d "%~dp0"
set ARM=%~1
if "%ARM%"=="" set ARM=L025
python chain_23_all.py --from %ARM% --max-folds 0
endlocal
