# Scheme 17 · U 组合附报监督：熔断后冷却重启
# 队列：U13 → U12 → U123（全量，含 U123）
# 用法：powershell -File .\supervise_u_combo.ps1
param(
  [double]$MinFreeGB = 3.2,
  [int]$CooldownSec = 180,
  [string]$FromArm = "U13",
  [int]$MaxFolds = 0
)

$ErrorActionPreference = "Continue"
$WorkDir = "D:\cyy\MI\code\train_lab\src\step\5070_mask_future_dual_expert_accpaper"
$SupLog = Join-Path $WorkDir "supervise_u_combo.log"
$ChainPid = Join-Path $WorkDir "u_combo_chain_guarded.pid"
$State = Join-Path $WorkDir "u_combo_chain_guarded_state.json"
$Status = Join-Path $WorkDir "supervise_u_combo_status.txt"
$OutRoot = "D:\cyy\MI\code\train_lab\out\5070_mask_future_dual_expert_accpaper"
$All = @("U13", "U12", "U123")

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
  return @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -EA SilentlyContinue |
    Where-Object {
      $_.CommandLine -and (
        $_.CommandLine -match 'run_arm\.py' -or
        $_.CommandLine -match '5070_mask_future_dual_expert_accpaper' -or
        $_.CommandLine -match '5060_three_hier_loss_accpaper'
      )
    })
}

function Get-NextArm {
  $done = @()
  if (Test-Path $State) {
    try {
      $j = Get-Content $State -Raw -Encoding utf8 | ConvertFrom-Json
      if ($j.finished) { return $null }
      if ($j.done) { $done = @($j.done) }
      if ($j.failed -and $j.failed.step) {
        $fail = [string]$j.failed.step
        $idx = [array]::IndexOf($All, $fail)
        if ($idx -ge 0) { return $All[$idx] }
      }
    } catch {}
  }
  foreach ($a in $All) {
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
  foreach ($arm in $All) {
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
      Write-Sup "DONE: U-combo finished at $($j.finished)"
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
  $next = $FromArm
}

$free = Get-FreePhysGB
if ($free -lt $MinFreeGB) {
  Write-Sup "WAIT free=${free}G < ${MinFreeGB}G — cooldown ${CooldownSec}s then exit (caller rechecks)"
  Start-Sleep -Seconds $CooldownSec
  exit 0
}

Write-Sup "LAUNCH U-combo FromArm=$next MaxFolds=$MaxFolds free=${free}G"
$argList = @(
  "-NoProfile", "-ExecutionPolicy", "Bypass",
  "-File", (Join-Path $WorkDir "run_u_combo_chain_guarded.ps1"),
  "-FromArm", $next,
  "-MaxFolds", "$MaxFolds",
  "-NoConsole"
)

$p = Start-Process -FilePath "powershell.exe" -ArgumentList $argList -WorkingDirectory $WorkDir `
  -WindowStyle Hidden -PassThru `
  -RedirectStandardOutput (Join-Path $WorkDir "u_combo_chain_guarded_stdout.log") `
  -RedirectStandardError (Join-Path $WorkDir "u_combo_chain_guarded_stderr.log")
[System.IO.File]::WriteAllText($ChainPid, "$($p.Id)", [System.Text.UTF8Encoding]::new($false))
Write-Sup "launched combo chain pid=$($p.Id)"
exit 0
