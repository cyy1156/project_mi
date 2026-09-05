@echo off
REM machine: 5070_laptop (hostname cyy) — conda env cyy first; never require repo .venv
set "PY="

if /I "%CONDA_DEFAULT_ENV%"=="cyy" if exist "%CONDA_PREFIX%\python.exe" (
  set "PY=%CONDA_PREFIX%\python.exe"
  goto :eof
)

if exist "%USERPROFILE%\.conda\envs\cyy\python.exe" (
  set "PY=%USERPROFILE%\.conda\envs\cyy\python.exe"
  goto :eof
)

if exist "%USERPROFILE%\miniconda3\envs\cyy\python.exe" (
  set "PY=%USERPROFILE%\miniconda3\envs\cyy\python.exe"
  goto :eof
)

if exist "%USERPROFILE%\anaconda3\envs\cyy\python.exe" (
  set "PY=%USERPROFILE%\anaconda3\envs\cyy\python.exe"
  goto :eof
)

if exist "C:\Users\yy\.conda\envs\cyy\python.exe" (
  set "PY=C:\Users\yy\.conda\envs\cyy\python.exe"
  goto :eof
)

where python >nul 2>&1
if not errorlevel 1 (
  for /f "delims=" %%P in ('where python') do (
    set "PY=%%P"
    goto :eof
  )
)

if defined REPO if exist "%REPO%\.venv\Scripts\python.exe" (
  set "PY=%REPO%\.venv\Scripts\python.exe"
  goto :eof
)

set "PY="
