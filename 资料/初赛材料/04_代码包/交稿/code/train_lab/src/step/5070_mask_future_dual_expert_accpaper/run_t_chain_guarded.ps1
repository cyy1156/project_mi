# Scheme 17 · T 系列 v2 五折（5070 · Token + Phase Query Predictor）
# 顺序：T1 → T1_aux → T1_128
#
# 用法：
#   powershell -File .\run_t_chain_guarded.ps1 -MaxFolds 1          # fold0 冒烟
#   powershell -File .\run_t_chain_guarded.ps1 -MaxFolds 0 -NoConsole  # 正式五折
#   powershell -File .\run_t_chain_guarded.ps1 -FromArm T1_aux -SkipT1_128
param(
  [string]$FromArm = "T1",
  [switch]$NoConsole,
  [int]$MaxFolds = 1,
  [int]$TimeoutSecPerArm = 86400,
  [double]$MinFreeGB = 3.2,
  [int]$CooldownBetweenSec = 180,
  [switch]$SkipT1_128
)

$ErrorActionPreference = "Continue"
$WorkDir = $PSScriptRoot
$State = Join-Path $WorkDir "t_chain_guarded_state.json"
$ChainLog = Join-Path $WorkDir "t_chain_guarded.log"
$PidFile = Join-Path $WorkDir "t_chain_guarded.pid"

$Arms = @("T1", "T1_aux")
if (-not $SkipT1_128) { $Arms += "T1_128" }

$start = [array]::IndexOf($Arms, $FromArm)
if ($start -lt 0) { throw "bad FromArm=$FromArm (queue: $($Arms -join ','))" }
$queue = $Arms[$start..($Arms.Count - 1)]

function Write-Chain([string]$msg) {
  $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $msg
  Write-Host $line
  Add-Content -Path $ChainLog -Value $line -Encoding utf8
}

function Get-FreePhysGB {
  $os = Get-CimInstance Win32_OperatingSystem
  return [math]::Round($os.FreePhysicalMemory / 1MB, 2)
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
Write-Chain "START T-series queue=$($queue -join ',') folds=$foldLabel free=$(Get-FreePhysGB)G"

$state = [ordered]@{
  package   = "5070_mask_future_dual_expert_accpaper"
  mode      = "T_SERIES"
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
  $state.current = $arm
  Save-State $state

  $extra = "--max-folds $MaxFolds --num-workers 0"
  Write-Chain "RUN $arm ExtraArgs='$extra'"
  $t0 = Get-Date
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
  $dt = [math]::Round(((Get-Date) - $t0).TotalHours, 2)

  if ($code -ne 0) {
    Write-Chain "FAIL $arm exit=$code after ${dt}h"
    $state.failed = @{ step = $arm; code = $code; hours = $dt }
    $state.current = $null
    Save-State $state
    $failCode = $code
    break
  }

  Write-Chain "OK $arm in ${dt}h"
  $state.done = @($state.done + $arm)
  $state.current = $null
  Save-State $state

  if ($arm -ne $queue[-1]) {
    Write-Chain "cooldown ${CooldownBetweenSec}s"
    Start-Sleep -Seconds $CooldownBetweenSec
  }
}

if ($failCode -eq 0) {
  $state.finished = (Get-Date -Format o)
  Save-State $state
  Write-Chain "ALL DONE T-series"
}
exit $failCode
