@echo off
REM 方案 21 断点续跑：run_21_chain_resume.bat A2_pt
setlocal
cd /d "%~dp0"
set FROM=%~1
if "%FROM%"=="" set FROM=F_mi_a
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_21_chain_guarded.ps1" -FromArm %FROM% -MaxFolds 0 -NoConsole
exit /b %ERRORLEVEL%
