@echo off
REM machine template — conda env name from MACHINE_CONDA_ENV (default cyy); no hard-coded username
set "PY="
if not defined MACHINE_CONDA_ENV set "MACHINE_CONDA_ENV=cyy"

if /I "%CONDA_DEFAULT_ENV%"=="%MACHINE_CONDA_ENV%" if exist "%CONDA_PREFIX%\python.exe" (
  set "PY=%CONDA_PREFIX%\python.exe"
  goto :eof
)

if exist "%USERPROFILE%\.conda\envs\%MACHINE_CONDA_ENV%\python.exe" (
  set "PY=%USERPROFILE%\.conda\envs\%MACHINE_CONDA_ENV%\python.exe"
  goto :eof
)

if exist "%USERPROFILE%\miniconda3\envs\%MACHINE_CONDA_ENV%\python.exe" (
  set "PY=%USERPROFILE%\miniconda3\envs\%MACHINE_CONDA_ENV%\python.exe"
  goto :eof
)

if exist "%USERPROFILE%\anaconda3\envs\%MACHINE_CONDA_ENV%\python.exe" (
  set "PY=%USERPROFILE%\anaconda3\envs\%MACHINE_CONDA_ENV%\python.exe"
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
