# Exp34 全量（PowerShell）
#   powershell -ExecutionPolicy Bypass -File .\run_exp34_full.ps1
#   powershell -File .\run_exp34_full.ps1 -DoBScratch -SkipTrackC

param(
  [string]$RunTag = "",
  [switch]$SkipTrackC,
  [switch]$DoBScratch,
  [switch]$Resume = $true,
  [ValidateSet("all","A","B","C")][string]$Only = "all"
)

$Py = Join-Path $env:USERPROFILE ".conda\envs\cyy\python.exe"
if (-not (Test-Path $Py)) { $Py = "python" }
$Script = Join-Path $PSScriptRoot "run_exp34_full.py"

$argsList = @($Script, "--only", $Only)
if ($RunTag) { $argsList += @("--run-tag", $RunTag) }
if ($Resume) { $argsList += "--resume" }
if ($SkipTrackC) { $argsList += "--skip-track-c" }
if ($DoBScratch) { $argsList += "--do-b-scratch" }

Write-Host "Exp34 FULL -> $Py $($argsList -join ' ')"
& $Py @argsList
exit $LASTEXITCODE
