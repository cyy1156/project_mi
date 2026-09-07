# Scheme 17 · U 系列监督：熔断后冷却重启；已完成臂从 state 续跑
# 用法：powershell -File .\supervise_u_series.ps1
param(
  [double]$MinFreeGB = 3.5,
  [int]$CooldownSec = 180,
  [string]$FromArm = "U1",
  [int]$MaxFolds = 0,
  [switch]$IncludeCombos
)

$ErrorActionPreference = "Continue"
$WorkDir = "D:\cyy\MI\code\train_lab\src\step\5060_mask_future_dual_expert_accpaper"
$SupLog = Join-Path $WorkDir "supervise_u_series.log"
$ChainPid = Join-Path $WorkDir "u_chain_guarded.pid"
$State = Join-Path $WorkDir "u_chain_guarded_state.json"
$Status = Join-Path $WorkDir "supervise_u_status.txt"
$OutRoot = "D:\cyy\MI\code\train_lab\out\5060_mask_future_dual_expert_accpaper"

function Write-Sup([string]$msg) {
  $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $msg
  Write-Host $line
  Add-Content -Path $SupLog -Value $line -Encoding utf8
}

function Get-FreePhysGB {
  $os = Get-CimInstance Win32_OperatingSystem
  return [math]::Round($os.FreePhysicalMemory / 1MB, 2)
}

function Test-PidAlive([string]$path) {
  if (-not (Test-Path $path)) { return $false }
  $id = 0
  [void][int]::TryParse((Get-Content $path -Raw).Trim(), [ref]$id)
  if ($id -le 0) { return $false }
  return [bool](Get-Process -Id $id -EA SilentlyContinue)
}

function Get-BusyTrain {
  # NoConsole 启动时 CommandLine 往往只有 run_arm.py（无包路径）
  return @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -EA SilentlyContinue |
    Where-Object {
      $_.CommandLine -and (
        $_.CommandLine -match 'run_arm\.py' -or
        $_.CommandLine -match '5060_mask_future_dual_expert_accpaper' -or
        $_.CommandLine -match '5060_three_hier_loss_accpaper'
      )
    })
}

function Get-NextArm {
  $must = @("U1", "U3", "U2")
  $all = if ($IncludeCombos) { $must + @("U13", "U12", "U123") } else { $must }
  $done = @()
  if (Test-Path $State) {
    try {
      $j = Get-Content $State -Raw -Encoding utf8 | ConvertFrom-Json
      if ($j.finished) { return $null }
      if ($j.done) { $done = @($j.done) }
      if ($j.failed -and $j.failed.step) {
        $fail = [string]$j.failed.step
        $idx = [array]::IndexOf($all, $fail)
        if ($idx -ge 0) { return $all[$idx] }
      }
    } catch {}
  }
  foreach ($a in $all) {
    if ($done -notcontains $a) { return $a }
  }
  return $null
}

function Summarize-Progress {
  $lines = @()
  $lines += ("time={0}" -f (Get-Date -Format o))
  $lines += ("free_gb={0}" -f (Get-FreePhysGB))
  $busy = Get-BusyTrain
  $lines += ("busy_python={0}" -f $busy.Count)
  if ($busy.Count -gt 0) {
    $cmd = $busy[0].CommandLine
    if ($cmd.Length -gt 160) { $cmd = $cmd.Substring(0, 160) }
    $lines += ("busy_cmd={0}" -f $cmd)
  }
  $lines += ("chain_pid_alive={0}" -f (Test-PidAlive $ChainPid))
  if (Test-Path $State) {
    try {
      $j = Get-Content $State -Raw -Encoding utf8 | ConvertFrom-Json
      $lines += ("state_current={0}" -f $j.current)
      $lines += ("state_done={0}" -f (($j.done) -join ","))
      $lines += ("state_finished={0}" -f $j.finished)
      if ($j.failed) { $lines += ("state_failed={0}" -f ($j.failed | ConvertTo-Json -Compress)) }
    } catch {
      $lines += "state_parse_error"
    }
  } else {
    $lines += "state=missing"
  }
  # latest U* run dirs
  foreach ($arm in @("U1", "U3", "U2", "U12", "U13", "U123")) {
    $dirs = @(Get-ChildItem $OutRoot -Directory -EA SilentlyContinue |
      Where-Object { $_.Name -match "_${arm}$" } |
      Sort-Object LastWriteTime -Descending)
    if ($dirs.Count -eq 0) { continue }
    $d = $dirs[0].FullName
    $folds = 0
    foreach ($f in 0..4) {
      if (Test-Path (Join-Path $d "fold$f\metrics.json")) { $folds++ }
    }
    $sum = Test-Path (Join-Path $d "summary.json")
    $lines += ("run_{0}={1} folds_done={2}/5 summary={3}" -f $arm, $dirs[0].Name, $folds, $sum)
  }
  [System.IO.File]::WriteAllText($Status, ($lines -join "`n") + "`n", [System.Text.UTF8Encoding]::new($false))
  return ($lines -join " | ")
}

$prog = Summarize-Progress
Write-Sup "TICK $prog"

if (Test-Path $State) {
  try {
    $j = Get-Content $State -Raw -Encoding utf8 | ConvertFrom-Json
    if ($j.finished) {
      Write-Sup "DONE: U-series finished at $($j.finished)"
      exit 0
    }
  } catch {}
}

$chainAlive = Test-PidAlive $ChainPid
$busy = Get-BusyTrain
if ($chainAlive -or $busy.Count -gt 0) {
  Write-Sup "OK running (chainAlive=$chainAlive busy=$($busy.Count))"
  exit 0
}

$next = Get-NextArm
if ($null -eq $next) {
  Write-Sup "DONE: no next arm (all finished or empty)"
  exit 0
}

$free = Get-FreePhysGB
if ($free -lt $MinFreeGB) {
  Write-Sup "WAIT free=${free}G < ${MinFreeGB}G — cooldown ${CooldownSec}s then exit (caller rechecks)"
  Start-Sleep -Seconds $CooldownSec
  exit 0
}

Write-Sup "RESTART U-chain FromArm=$next MaxFolds=$MaxFolds free=${free}G"
$args = @(
  "-NoProfile", "-ExecutionPolicy", "Bypass",
  "-File", (Join-Path $WorkDir "run_u_chain_guarded.ps1"),
  "-FromArm", $next,
  "-MaxFolds", "$MaxFolds",
  "-NoConsole"
)
if ($IncludeCombos) { $args += "-IncludeCombos" }

$p = Start-Process -FilePath "powershell.exe" -ArgumentList $args -WorkingDirectory $WorkDir `
  -WindowStyle Hidden -PassThru `
  -RedirectStandardOutput (Join-Path $WorkDir "u_chain_guarded_stdout.log") `
  -RedirectStandardError (Join-Path $WorkDir "u_chain_guarded_stderr.log")
[System.IO.File]::WriteAllText($ChainPid, "$($p.Id)", [System.Text.UTF8Encoding]::new($false))
Write-Sup "launched chain pid=$($p.Id)"
exit 0
