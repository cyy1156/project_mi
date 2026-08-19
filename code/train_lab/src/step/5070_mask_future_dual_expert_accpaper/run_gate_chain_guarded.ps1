# Scheme 17 · 5060 全量主线管道（带双层内存看门狗）
# 默认 GATE：A0_ref → A0 → A1 → P0 → A2 → P1 → P2
# 完整消融：-FullChain（含 B/C）
#
# 用法：
#   powershell -File .\run_gate_chain_guarded.ps1
#   powershell -File .\run_gate_chain_guarded.ps1 -FromArm A1
#   powershell -File .\run_gate_chain_guarded.ps1 -FromArm A1 -MaxFolds 0   # 五折
#   powershell -File .\run_gate_chain_guarded.ps1 -FullChain
# 每臂默认弹出可见控制台（run_with_mem_guard.ps1）；静默加 -NoConsole
param(
  [string]$FromArm = "A0_ref",
  [switch]$FullChain,
  [switch]$NoConsole,
  [int]$MaxFolds = 1,            # 1=fold0；0=五折
  [int]$TimeoutSecPerArm = 86400, # 五折更久：默认 24h/臂
  [double]$MinFreeGB = 3.2,
  [int]$CooldownBetweenSec = 120
)

$ErrorActionPreference = "Continue"
$WorkDir = "D:\cyy\MI\code\train_lab\src\step\5070_mask_future_dual_expert_accpaper"
$LogDir = "D:\cyy\MI\code\train_lab\out\_ab_mem"
$State = Join-Path $WorkDir "gate_chain_guarded_state.json"
$ChainLog = Join-Path $WorkDir "gate_chain_guarded.log"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$GateOrder = @("A0_ref", "A0", "A1", "P0", "A2", "P1", "P2")
$FullOrder = @(
  "A0_ref", "A0", "A1", "P0", "A2", "P1",
  "B1", "B2", "B3", "B4", "B5a", "B5b", "B6", "B7", "B8", "B9", "B10",
  "P2", "C1", "C2a", "C2b", "C2c"
)
$Arms = if ($FullChain) { $FullOrder } else { $GateOrder }

$start = [array]::IndexOf($Arms, $FromArm)
if ($start -lt 0) { throw "bad FromArm=$FromArm (not in queue)" }
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

# Refuse competing trains (scheme16 / scheme17)
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
$mode = if ($FullChain) { "FULL_CHAIN" } else { "GATE" }
$foldLabel = if ($MaxFolds -le 0) { "5fold" } else { "fold0..$($MaxFolds-1)" }
Write-Chain "START scheme17 $mode queue=$($queue -join ',') folds=$foldLabel max_folds=$MaxFolds commit_limit=${cLimit}G free=$(Get-FreePhysGB)G timeout=${TimeoutSecPerArm}s batch=128/256 sigreg=1024"
if ($cLimit -lt 40) {
  Write-Chain "REFUSE: commit_limit=${cLimit}G < 40G (need D: pagefile ~48G)"
  exit 3
}

$state = [ordered]@{
  package         = "5070_mask_future_dual_expert_accpaper"
  mode            = $mode
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
  while ($free -lt $MinFreeGB -and $waited -lt 600) {
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
    "-MinSysFreeGB", "0.08"
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
  Write-Chain "ALL DONE scheme17 $mode"
}
exit $failCode
