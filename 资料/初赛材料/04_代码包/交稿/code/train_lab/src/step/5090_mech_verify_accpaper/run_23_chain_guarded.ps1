# 方案 23 · 机制验证链（5090）
# Tier1: O2s_m → O2s_f → O1s_m → O1s_f → O600 → L025 → L050 → A1_all
#
#   powershell -File .\run_23_chain_guarded.ps1 -MaxFolds 1
#   powershell -File .\run_23_chain_guarded.ps1 -MaxFolds 0 -NoConsole
#   powershell -File .\run_23_chain_guarded.ps1 -FromArm L025
param(
  [string]$FromArm = "O2s_m",
  [switch]$Tier2,
  [switch]$SkipCalibration,
  [switch]$NoConsole,
  [int]$MaxFolds = 0,
  [int]$TimeoutSecPerArm = 86400,
  [int]$CooldownBetweenSec = 180
)

$ErrorActionPreference = "Continue"
$WorkDir = $PSScriptRoot
$State = Join-Path $WorkDir "23_chain_guarded_state.json"
$ChainLog = Join-Path $WorkDir "23_chain_guarded.log"
$PidFile = Join-Path $WorkDir "23_chain_guarded.pid"

$Tier1 = @("O2s_m", "O2s_f", "O1s_m", "O1s_f", "O600", "L025", "L050", "A1_all")
$Tier2Arms = @("P1_local", "E1", "E2")
$Arms = if ($Tier2) { $Tier2Arms } else { $Tier1 }
if ($SkipCalibration -and -not $Tier2) {
  $Arms = @($Arms | Where-Object { $_ -ne "O2s_m" })
}

$start = [array]::IndexOf($Arms, $FromArm)
if ($start -lt 0) { throw "bad FromArm=$FromArm (queue: $($Arms -join ','))" }
$queue = $Arms[$start..($Arms.Count - 1)]

function Write-Chain([string]$msg) {
  $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $msg
  Write-Host $line
  Add-Content -Path $ChainLog -Value $line -Encoding utf8
}

function Save-State($obj) {
  try {
    $json = $obj | ConvertTo-Json -Depth 8
    [System.IO.File]::WriteAllText($State, $json, [System.Text.UTF8Encoding]::new($false))
  } catch {
    Write-Chain ("WARN Save-State failed: {0}" -f $_.Exception.Message)
  }
}

[System.IO.File]::WriteAllText($PidFile, "$PID", [System.Text.UTF8Encoding]::new($false))

$foldLabel = if ($MaxFolds -le 0) { "5fold" } else { "fold0..$($MaxFolds-1)" }
$mode = if ($Tier2) { "TIER2" } else { "TIER1" }
Write-Chain "START scheme23-5090 mode=$mode queue=$($queue -join ',') folds=$foldLabel batch=256/512"

$state = [ordered]@{
  package   = "5090_mech_verify_accpaper"
  mode      = $mode
  started   = (Get-Date -Format o)
  queue     = $queue
  done      = @()
  failed    = $null
  current   = $null
  max_folds = $MaxFolds
}
Save-State $state

$failCode = 0
foreach ($arm in $queue) {
  Write-Chain "RUN $arm"
  $state.current = $arm
  Save-State $state

  $extra = "--max-folds $MaxFolds --num-workers 0"
  $guardArgs = @(
    "-NoProfile", "-ExecutionPolicy", "Bypass",
    "-File", (Join-Path $WorkDir "run_with_mem_guard.ps1"),
    "-Arm", $arm,
    "-ExtraArgs", $extra,
    "-TimeoutSec", "$TimeoutSecPerArm",
    "-MinSysFreeGB", "0.02"
  )
  if ($NoConsole) { $guardArgs += "-NoConsole" }
  & powershell @guardArgs
  $code = $LASTEXITCODE

  if ($code -ne 0) {
    Write-Chain "FAIL $arm exit=$code"
    $state.failed = @{ step = $arm; code = $code }
    Save-State $state
    $failCode = $code
    break
  }

  Write-Chain "OK $arm"
  $state.done = @($state.done + $arm)
  $state.current = $null
  Save-State $state

  if ($arm -ne $queue[-1]) { Start-Sleep -Seconds $CooldownBetweenSec }
}

if ($failCode -eq 0) {
  $state.finished = (Get-Date -Format o)
  Save-State $state
  Write-Chain "ALL DONE scheme23-5090"
}
exit $failCode
