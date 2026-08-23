# 方案 24 · W 腿：O3s_m 五折（5090 · 5090_mech_verify_accpaper 续）
#
#   powershell -File .\run_24_w_o3s_guarded.ps1
#   powershell -File .\run_24_w_o3s_guarded.ps1 -MaxFolds 1
param(
  [int]$MaxFolds = 0,
  [string]$ResumeDir = "",
  [switch]$NoConsole,
  [int]$TimeoutSec = 86400
)

$ErrorActionPreference = "Continue"
$WorkDir = $PSScriptRoot
$extra = "--max-folds $MaxFolds --num-workers 0"
if ($ResumeDir) { $extra += " --resume-dir `"$ResumeDir`"" }
$guardArgs = @(
  "-NoProfile", "-ExecutionPolicy", "Bypass",
  "-File", (Join-Path $WorkDir "run_with_mem_guard.ps1"),
  "-Arm", "O3s_m",
  "-ExtraArgs", $extra,
  "-TimeoutSec", "$TimeoutSec",
  "-MinSysFreeGB", "0.02"
)
if ($NoConsole) { $guardArgs += "-NoConsole" }
& powershell @guardArgs
exit $LASTEXITCODE
