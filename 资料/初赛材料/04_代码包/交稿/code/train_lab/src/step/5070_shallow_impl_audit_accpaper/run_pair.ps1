# 方案 18 · 顺序跑 L0（库）→ S0（手写）· OpenBMI Acc_paper 五折旁路
param(
  [int]$MaxFolds = 0,
  [int]$NumWorkers = 2,
  [switch]$SmokeOnly
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$foldArg = if ($MaxFolds -gt 0) { @("--max-folds", "$MaxFolds") } else { @() }
$workerArg = @("--num-workers", "$NumWorkers")

Write-Host "=== compare A/B ===" -ForegroundColor Cyan
python compare_shallow_impl.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "=== L0 braindecode ===" -ForegroundColor Cyan
python baseline_shallow_lib.py @workerArg @foldArg
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "=== S0 self_model ===" -ForegroundColor Cyan
python baseline_shallow_self.py @workerArg @foldArg
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "DONE scheme18 pair" -ForegroundColor Green
