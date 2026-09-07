# Scheme 17 · 5060 监督续跑：检查 pf1000 → 门控链 → 内存熔断后自动冷却重启
# 由 Cursor agent /loop 周期调用；也可手动：powershell -File .\supervise_gate17.ps1
param(
  [double]$MinFreeGB = 3.5,
  [int]$CooldownSec = 120,
  [string]$FromArm = "A1",
  [int]$MaxFolds = 0
)

$ErrorActionPreference = "Continue"
$WorkDir = "D:\cyy\MI\code\train_lab\src\step\5060_mask_future_dual_expert_accpaper"
$PfDir = "D:\cyy\MI\code\preprocess_lab\out\openbmi_2s_hop100_pf1000"
$LogDir = "D:\cyy\MI\code\train_lab\out\_ab_mem"
$SupLog = Join-Path $WorkDir "supervise_gate17.log"
$ChainPid = Join-Path $WorkDir "gate_chain_guarded.pid"
$RebuildPid = Join-Path $WorkDir "pf1000_rebuild.pid"
$GateState = Join-Path $WorkDir "gate_chain_guarded_state.json"
$ChainState = Join-Path $WorkDir "chain_state.json"

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

function Test-PfReady {
  $meta = Join-Path $PfDir "preprocess_meta.json"
  $xf = Join-Path $PfDir "openbmi_X_full.npy"
  if (-not ((Test-Path $meta) -and (Test-Path $xf))) { return $false }
  try {
    $j = Get-Content $meta -Raw -Encoding utf8 | ConvertFrom-Json
    if ($j.no_rest -eq $true) { return $false }
    $ver = 0
    if ($null -ne $j.protocol_version) { $ver = [int]$j.protocol_version }
    if ($ver -lt 3) { return $false }
    return $true
  } catch { return $false }
}

function Get-BusyTrain {
  return @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -EA SilentlyContinue |
    Where-Object {
      $_.CommandLine -and (
        $_.CommandLine -match '5060_mask_future_dual_expert_accpaper' -or
        $_.CommandLine -match 'openbmi_pf1000' -or
        $_.CommandLine -match '5060_three_hier_loss_accpaper'
      ) -and (
        $_.CommandLine -match 'run_arm\.py' -or
        $_.CommandLine -match 'chain_all\.py' -or
        $_.CommandLine -match 'openbmi_pf1000'
      )
    })
}

function Get-ResumeArm {
  # Prefer gate state; else chain_state failed step; else FromArm
  if (Test-Path $GateState) {
    try {
      $g = Get-Content $GateState -Raw -Encoding utf8 | ConvertFrom-Json
      if ($g.current) { return [string]$g.current }
      if ($g.failed -and $g.failed.step) { return [string]$g.failed.step }
      if ($g.done -and $g.done.Count -gt 0) {
        $order = @("A0_ref","A0","A1","P0","A2","P1","P2")
        $last = [string]$g.done[-1]
        $i = [array]::IndexOf($order, $last)
        if ($i -ge 0 -and $i -lt $order.Count - 1) { return $order[$i + 1] }
      }
      if ($g.finished) { return $null }
    } catch {}
  }
  if (Test-Path $ChainState) {
    try {
      $c = Get-Content $ChainState -Raw -Encoding utf8 | ConvertFrom-Json
      if ($c.failed -and $c.failed.step) { return [string]$c.failed.step }
      if ($c.done -and $c.done.Count -gt 0) {
        $last = [string]$c.done[-1].step
        $order = @("A0_ref","A0","A1","P0","A2","P1","P2")
        $i = [array]::IndexOf($order, $last)
        if ($i -ge 0 -and $i -lt $order.Count - 1) { return $order[$i + 1] }
      }
      if ($c.finished) { return $null }
    } catch {}
  }
  return $FromArm
}

function Get-LatestTrainSnippet {
  $arm = "?"
  $snip = "(no log)"
  $latest = Get-ChildItem $LogDir -Filter "guarded17_*_stdout.txt" -EA SilentlyContinue |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1
  if ($latest) {
    if ($latest.Name -match 'guarded17_([^_]+)_') { $arm = $Matches[1] }
    $ep = Select-String -Path $latest.FullName -Pattern 'fold0 ep \d+|early stop|HARD STOP' -EA SilentlyContinue |
      Select-Object -Last 1
    if ($ep) { $snip = $ep.Line.Trim() }
    else {
      $tail = Get-Content $latest.FullName -Tail 1 -EA SilentlyContinue
      if ($tail) { $snip = [string]$tail }
    }
  }
  $busy = Get-BusyTrain
  $live = @($busy | Where-Object { $_.CommandLine -match 'run_arm\.py --arm' })
  if ($live.Count -gt 0 -and $live[0].CommandLine -match '--arm (\S+)') {
    $arm = $Matches[1]
  }
  return @{ arm = $arm; snip = $snip; train_n = $live.Count; log = if ($latest) { $latest.Name } else { "" } }
}

function Write-StatusCard([string]$action) {
  $info = Get-LatestTrainSnippet
  $free = Get-FreePhysGB
  $card = @(
    "time=$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
    "action=$action"
    "arm=$($info.arm)"
    "train_procs=$($info.train_n)"
    "free_gb=$free"
    "latest=$($info.snip)"
    "log=$($info.log)"
    "chain_alive=$(Test-PidAlive $ChainPid)"
    "pf_ready=$(Test-PfReady)"
  ) -join "`n"
  $statusPath = Join-Path $WorkDir "supervise_status.txt"
  [System.IO.File]::WriteAllText($statusPath, $card + "`n", [System.Text.UTF8Encoding]::new($false))
  # 给 agent 一眼可抓的汇报行
  Write-Sup ("REPORT arm={0} free={1}G procs={2} | {3}" -f $info.arm, $free, $info.train_n, $info.snip)
}

$free = Get-FreePhysGB
Write-Sup "tick free=${free}G pf_ready=$(Test-PfReady) chain_alive=$(Test-PidAlive $ChainPid) rebuild_alive=$(Test-PidAlive $RebuildPid)"

# 1) rebuild still running (pid file OR live python batch)
$busy0 = Get-BusyTrain
$pfBusy = @($busy0 | Where-Object { $_.CommandLine -match 'openbmi_pf1000' })
if ((Test-PidAlive $RebuildPid) -or ($pfBusy.Count -gt 0)) {
  Write-Sup "WAIT pf1000 rebuild (python_busy=$($busy0.Count) pf=$($pfBusy.Count))"
  Write-StatusCard "WAIT_REBUILD"
  exit 10
}

# 2) pf not ready → auto resume rebuild (no --reset; skip finished shards)
if (-not (Test-PfReady)) {
  $free = Get-FreePhysGB
  if ($free -lt $MinFreeGB) {
    Write-Sup "BLOCK pf not ready + low RAM free=${free}G — wait"
    Write-StatusCard "BLOCK_PF"
    exit 11
  }
  $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
  $outLog = Join-Path $LogDir "pf1000_rebuild_${stamp}_stdout.txt"
  $py = "D:\cyy\MI\.venv\Scripts\python.exe"
  $pre = "D:\cyy\MI\code\preprocess_lab"
  Write-Sup "RESUME pf1000 batch (no --reset) log=$outLog"
  $p = Start-Process -FilePath $py -ArgumentList @("-u", "-m", "src.datasets.openbmi_pf1000.batch") `
    -WorkingDirectory $pre -RedirectStandardOutput $outLog -RedirectStandardError "$outLog.err" `
    -WindowStyle Hidden -PassThru
  Set-Content -LiteralPath $RebuildPid -Value $p.Id -Encoding ascii
  Write-Sup "LAUNCHED rebuild python pid=$($p.Id)"
  Write-StatusCard "LAUNCH_REBUILD"
  exit 10
}

# 3) chain already running
if (Test-PidAlive $ChainPid) {
  Write-Sup "OK chain already running"
  Write-StatusCard "RUNNING"
  exit 0
}
$busy = Get-BusyTrain
if ($busy.Count -gt 0) {
  Write-Sup "OK train busy pid=$($busy[0].ProcessId) — no relaunch"
  Write-StatusCard "RUNNING_BUSY"
  exit 0
}

# 4) low RAM cooldown
if ($free -lt $MinFreeGB) {
  Write-Sup "COOLDOWN free=${free}G < ${MinFreeGB}G sleep ${CooldownSec}s"
  Write-StatusCard "COOLDOWN"
  Start-Sleep -Seconds $CooldownSec
  $free = Get-FreePhysGB
  if ($free -lt $MinFreeGB) {
    Write-Sup "STILL low free=${free}G — skip launch"
    Write-StatusCard "STILL_LOW"
    exit 12
  }
}

$arm = Get-ResumeArm
if (-not $arm) {
  Write-Sup "ALL DONE (no resume arm)"
  Write-StatusCard "ALL_DONE"
  exit 0
}

Write-Sup "LAUNCH gate chain FromArm=$arm MaxFolds=$MaxFolds NoConsole"
$launchLog = Join-Path $WorkDir "gate_chain_detached_launch.log"
$args = @(
  "-NoProfile", "-ExecutionPolicy", "Bypass",
  "-File", (Join-Path $WorkDir "run_gate_chain_guarded.ps1"),
  "-FromArm", $arm,
  "-MaxFolds", "$MaxFolds",
  "-NoConsole"
)
$p = Start-Process -FilePath "powershell.exe" -ArgumentList $args -WorkingDirectory $WorkDir -WindowStyle Minimized -PassThru
Set-Content -LiteralPath $ChainPid -Value $p.Id -Encoding ascii
Add-Content -Path $launchLog -Value ("[{0}] supervise launch pid={1} FromArm={2}" -f (Get-Date -Format o), $p.Id, $arm) -Encoding utf8
Write-Sup "LAUNCHED pid=$($p.Id) FromArm=$arm"
Write-StatusCard "LAUNCHED_$arm"
exit 0
