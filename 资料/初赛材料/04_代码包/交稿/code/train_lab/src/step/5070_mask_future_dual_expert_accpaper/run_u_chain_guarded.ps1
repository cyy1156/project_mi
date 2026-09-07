# Scheme 17 · U 系列五折（相对 P2 · 必做臂）
# 顺序：U1 → U3 → U2（对齐实验方案；组合臂 U12/U13/U123 另开）
#
# 用法：
#   powershell -File .\run_u_chain_guarded.ps1
#   powershell -File .\run_u_chain_guarded.ps1 -FromArm U3
#   powershell -File .\run_u_chain_guarded.ps1 -NoConsole
param(
  [string]$FromArm = "U1",
  [switch]$NoConsole,
  [int]$MaxFolds = 0,                 # 0=五折
  [int]$TimeoutSecPerArm = 86400,     # 24h / 臂
  [double]$MinFreeGB = 3.2,
  [int]$CooldownBetweenSec = 180,
  [switch]$IncludeCombos              # 追加 U12/U13/U123
)

$ErrorActionPreference = "Continue"
$WorkDir = "D:\cyy\MI\code\train_lab\src\step\5070_mask_future_dual_expert_accpaper"
$State = Join-Path $WorkDir "u_chain_guarded_state.json"
$ChainLog = Join-Path $WorkDir "u_chain_guarded.log"
$PidFile = Join-Path $WorkDir "u_chain_guarded.pid"

$MustOrder = @("U1", "U3", "U2")
# 组合顺序对齐实验方案：U13 → U12 → U123
$ComboOrder = @("U13", "U12", "U123")
$Arms = if ($IncludeCombos) { $MustOrder + $ComboOrder } else { $MustOrder }

$start = [array]::IndexOf($Arms, $FromArm)
if ($start -lt 0) { throw "bad FromArm=$FromArm (not in queue: $($Arms -join ','))" }
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

$busy = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -EA SilentlyContinue |
  Where-Object {
    $_.CommandLine -and (
      $_.CommandLine -match '5070_mask_future_dual_expert_accpaper\\run_arm\.py' -or
      $_.CommandLine -match '5060_three_hier_loss_accpaper\\run_arm\.py'
    )
  })
if ($busy.Count -gt 0) {
  Write-Chain "REFUSE: training already running pid=$($busy[0].ProcessId)"
  exit 2
}

$os0 = Get-CimInstance Win32_OperatingSystem
$cLimit = [math]::Round($os0.TotalVirtualMemorySize / 1MB, 2)
$foldLabel = if ($MaxFolds -le 0) { "5fold" } else { "fold0..$($MaxFolds-1)" }
Write-Chain "START U-series queue=$($queue -join ',') folds=$foldLabel max_folds=$MaxFolds free=$(Get-FreePhysGB)G commit_limit=${cLimit}G"
if ($cLimit -lt 40) {
  Write-Chain "REFUSE: commit_limit=${cLimit}G < 40G"
  exit 3
}

$state = [ordered]@{
  package         = "5070_mask_future_dual_expert_accpaper"
  mode            = "U_SERIES"
  started         = (Get-Date -Format o)
  queue           = $queue
  done            = @()
  failed          = $null
  current         = $null
  commit_limit_gb = $cLimit
  hparams         = @{
    batch_train   = 128
    batch_eval    = 256
    sigreg_slices = 1024
    max_folds     = $MaxFolds
    num_workers   = 0
  }
}
Save-State $state

$failCode = 0
foreach ($arm in $queue) {
  $free = Get-FreePhysGB
  Write-Chain "before $arm free_phys=${free}G"
  $waited = 0
  while ($free -lt $MinFreeGB -and $waited -lt 900) {
    Write-Chain "low RAM ${free}G < ${MinFreeGB}G — sleep 30s (waited=${waited}s)"
    Start-Sleep -Seconds 30
    $waited += 30
    $free = Get-FreePhysGB
  }
  if ($free -lt $MinFreeGB) {
    Write-Chain ("FAIL {0}: free_phys={1}G still < {2}G" -f $arm, $free, $MinFreeGB)
    $state.failed = @{ step = $arm; reason = "low_ram"; free = $free }
    $state.current = $null
    Save-State $state
    $failCode = 3
    break
  }

  $state.current = $arm
  Save-State $state

  $extra = "--max-folds $MaxFolds --num-workers 0"
  Write-Chain "RUN $arm ExtraArgs='$extra' TimeoutSec=$TimeoutSecPerArm ShowConsole=$(-not $NoConsole)"
  $t0 = Get-Date
  $guardArgs = @(
    "-NoProfile", "-ExecutionPolicy", "Bypass",
    "-File", (Join-Path $WorkDir "run_with_mem_guard.ps1"),
    "-Arm", $arm,
    "-ExtraArgs", $extra,
    "-TimeoutSec", "$TimeoutSecPerArm",
    # 16GB 机训练谷底常 <0.1G；过严会误杀。进程内 mem_guard 已有 0.05G 底线。
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
    $failCode = if ($code -eq 0) { 1 } else { $code }
    break
  }

  Write-Chain "OK $arm in ${dt}h"
  $state.done = @($state.done + $arm)
  $state.current = $null
  Save-State $state

  if ($arm -ne $queue[-1]) {
    Write-Chain "cooldown ${CooldownBetweenSec}s before next arm"
    Start-Sleep -Seconds $CooldownBetweenSec
  }
}

if ($failCode -eq 0) {
  $state.finished = (Get-Date -Format o)
  Save-State $state
  Write-Chain "ALL DONE U-series"
}
exit $failCode
