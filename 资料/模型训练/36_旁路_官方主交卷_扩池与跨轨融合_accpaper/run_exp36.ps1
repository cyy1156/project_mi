# Exp36 Day0
param(
  [Parameter(ValueFromRemainingArguments = $true)]
  [string[]]$ArgsRest
)
$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$py = Join-Path $env:USERPROFILE ".conda\envs\cyy\python.exe"
if (-not (Test-Path $py)) { $py = "python" }
& $py (Join-Path $here "run_exp36.py") @ArgsRest
exit $LASTEXITCODE
