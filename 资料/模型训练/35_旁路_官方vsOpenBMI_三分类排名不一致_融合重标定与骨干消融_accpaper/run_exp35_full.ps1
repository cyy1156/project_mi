# Exp35 全量（默认 P0+D；加 --with-h 才跑轨 H）
param(
  [Parameter(ValueFromRemainingArguments = $true)]
  [string[]]$ArgsRest
)
$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$py = Join-Path $env:USERPROFILE ".conda\envs\cyy\python.exe"
if (-not (Test-Path $py)) { $py = "python" }
& $py (Join-Path $here "run_exp35_full.py") @ArgsRest
exit $LASTEXITCODE
