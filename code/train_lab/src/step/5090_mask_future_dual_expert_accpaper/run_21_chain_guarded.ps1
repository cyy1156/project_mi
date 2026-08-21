# 方案 21 · LeJEPA 对齐链（5090）
# 顺序：F_mi_a → F_mi_080 → A2_pt → J1_tok
#
#   powershell -File .\run_21_chain_guarded.ps1 -MaxFolds 1
#   powershell -File .\run_21_chain_guarded.ps1 -MaxFolds 0 -NoConsole
#   powershell -File .\run_21_chain_guarded.ps1 -FromArm A2_pt
param(
  [string]$FromArm = "F_mi_a",
  [switch]$NoConsole,
  [int]$MaxFolds = 0,
  [int]$TimeoutSecPerArm = 86400,
  [double]$MinFreeGB = 3.2,
  [int]$CooldownBetweenSec = 180,
  [switch]$SkipFmi080,
  [switch]$SkipJ1,
  [switch]$WithA1_800
)

$ErrorActionPreference = "Continue"
$WorkDir = $PSScriptRoot
$State = Join-Path $WorkDir "21_chain_guarded_state.json"
$ChainLog = Join-Path $WorkDir "21_chain_guarded.log"
$PidFile = Join-Path $WorkDir "21_chain_guarded.pid"

$Arms = @("F_mi_a")
if (-not $SkipFmi080) { $Arms += "F_mi_080" }
$Arms += "A2_pt"
if (-not $SkipJ1) { $Arms += "J1_tok" }

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
Write-Chain "START scheme21 queue=$($queue -join ',') folds=$foldLabel free=$(Get-FreePhysGB)G"

$state = [ordered]@{
  package = "5090_mask_future_dual_expert_accpaper"
  mode    = "SCHEME_21"
  started = (Get-Date -Format o)
  queue   = $queue
  done    = @()
  failed  = $null
  current = $null
}
Save-State $state

$failCode = 0
foreach ($arm in $queue) {
  $free = Get-FreePhysGB
  Write-Chain "before $arm free_phys=${free}G"
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

  if ($WithA1_800 -and $arm -eq "F_mi_080") {
    Write-Chain "RUN A1_800 (optional control)"
    $guardArgs[5] = "A1_800"
    & powershell @guardArgs
    if ($LASTEXITCODE -ne 0) { $failCode = $LASTEXITCODE; break }
  }

  if ($arm -ne $queue[-1]) { Start-Sleep -Seconds $CooldownBetweenSec }
}

if ($failCode -eq 0) {
  $state.finished = (Get-Date -Format o)
  Save-State $state
  Write-Chain "ALL DONE scheme21"
}
exit $failCode
