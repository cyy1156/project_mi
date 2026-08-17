# Scheme 16 · 5060 最稳妥串行：Three fold0 × (S0→H1→H2→H3)
# 每臂：外部看门狗 + 进程内 mem_guard；workers=0；三分类 only；默认 max_epochs/patience
# 用法：
#   powershell -File .\run_fold0_chain_guarded.ps1
#   powershell -File .\run_fold0_chain_guarded.ps1 -FromArm H1
param(
  [ValidateSet("S0", "H1", "H2", "H3")]
  [string]$FromArm = "S0",
  [int]$TimeoutSecPerArm = 43200,  # 12h / arm
  [double]$MinFreeGB = 3.2,
  [int]$CooldownBetweenSec = 90
)

$ErrorActionPreference = "Continue"
$WorkDir = "D:\cyy\MI\code\train_lab\src\step\5060_three_hier_loss_accpaper"
$LogDir = "D:\cyy\MI\code\train_lab\out\_ab_mem"
$State = Join-Path $WorkDir "fold0_chain_state.json"
$ChainLog = Join-Path $WorkDir "fold0_chain_guarded.log"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$Arms = @("S0", "H1", "H2", "H3")
$start = $Arms.IndexOf($FromArm)
if ($start -lt 0) { throw "bad FromArm=$FromArm" }
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

# Refuse if another scheme16 already running
$existing = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -EA SilentlyContinue |
  Where-Object { $_.CommandLine -match '5060_three_hier_loss_accpaper\\run_arm\.py' })
if ($existing.Count -gt 0) {
  Write-Chain "REFUSE: scheme16 already running pid=$($existing[0].ProcessId)"
  exit 2
}

$os0 = Get-CimInstance Win32_OperatingSystem
$cLimit = [math]::Round($os0.TotalVirtualMemorySize / 1MB, 2)
Write-Chain "START fold0 chain queue=$($queue -join ',') commit_limit=${cLimit}G free=$(Get-FreePhysGB)G timeout=${TimeoutSecPerArm}s"
if ($cLimit -lt 40) {
  Write-Chain "REFUSE: commit_limit=${cLimit}G < 40G (need D: pagefile 49152/49152)"
  exit 3
}

$state = [ordered]@{
  started   = (Get-Date -Format o)
  queue     = $queue
  done      = @()
  failed    = $null
  current   = $null
  commit_limit_gb = $cLimit
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

  # Full training (no --max-epochs): SHARED max_epochs=300, patience=20
  $extra = "--three-only --max-folds 1 --num-workers 0"
  Write-Chain "RUN $arm ExtraArgs='$extra' TimeoutSec=$TimeoutSecPerArm"
  $t0 = Get-Date
  & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $WorkDir "run_with_mem_guard.ps1") `
    -Arm $arm `
    -ExtraArgs $extra `
    -TimeoutSec $TimeoutSecPerArm `
    -MinSysFreeGB 0.08
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
  Write-Chain "ALL DONE fold0 chain"
}
exit $failCode
