# Scheme 16 external memory watchdog.
# Kills the whole python process tree if commit/WS/sys-free explode
# (backup when in-process mem_guard cannot schedule under thrashing).
param(
  [Parameter(Mandatory = $true)][string]$Arm,
  [string]$ExtraArgs = "--three-only --max-folds 1 --num-workers 0",
  [double]$MaxProcVirtGB = 0,   # 0 = auto from commit_limit
  [double]$MaxProcWsGB = 0,     # 0 = auto
  [double]$MinSysFreeGB = 0.20,
  [double]$MaxSysCommitGB = 0,  # 0 = auto
  [int]$TimeoutSec = 43200,     # 12h default for full fold0 (not smoke)
  [string]$WorkDir = "D:\cyy\MI\code\train_lab\src\step\5060_three_hier_loss_accpaper"
)

$ErrorActionPreference = "Continue"
$py = "D:\cyy\MI\.venv\Scripts\python.exe"
$logDir = "D:\cyy\MI\code\train_lab\out\_ab_mem"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$label = "guarded16_$Arm"
$out = Join-Path $logDir "${label}_${stamp}_stdout.txt"
$err = Join-Path $logDir "${label}_${stamp}_stderr.txt"
$mem = Join-Path $logDir "${label}_${stamp}_mem.csv"
$summary = Join-Path $logDir "${label}_${stamp}_summary.txt"

function Kill-Tree([int]$RootPid) {
  try {
    & taskkill.exe /PID $RootPid /T /F 2>$null | Out-Null
  } catch {}
  Get-CimInstance Win32_Process -Filter "Name='python.exe'" -EA SilentlyContinue |
    Where-Object { $_.CommandLine -match '5060_three_hier_loss_accpaper|run_arm\.py' } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force -EA SilentlyContinue }
}

# Refuse if another scheme16 already running
$existing = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -EA SilentlyContinue |
  Where-Object { $_.CommandLine -match '5060_three_hier_loss_accpaper\\run_arm\.py' })
if ($existing.Count -gt 0) {
  "REFUSE: scheme16 already running pid=$($existing[0].ProcessId)" | Tee-Object $summary
  exit 2
}

$os0 = Get-CimInstance Win32_OperatingSystem
$free0 = [math]::Round($os0.FreePhysicalMemory / 1MB, 2)
$cUsed0 = [math]::Round(($os0.TotalVirtualMemorySize - $os0.FreeVirtualMemory) / 1MB, 2)
$cLimit0 = [math]::Round($os0.TotalVirtualMemorySize / 1MB, 2)
# Registry max pagefile (MB) — allow growth up to this
$pfReg = @((Get-ItemProperty 'HKLM:\SYSTEM\CurrentControlSet\Control\Session Manager\Memory Management' -EA SilentlyContinue).PagingFiles)
$cfgMaxMb = 0
foreach ($line in $pfReg) {
  $parts = (@($line) -split '\s+')
  if ($parts.Count -ge 3) {
    $mx = 0; [void][int]::TryParse($parts[2], [ref]$mx)
    if ($mx -gt $cfgMaxMb) { $cfgMaxMb = $mx }
  }
}
$allocMb = 0
Get-CimInstance Win32_PageFileUsage -EA SilentlyContinue | ForEach-Object { $allocMb += [int]$_.AllocatedBaseSize }
$physGb = [math]::Round($os0.TotalVisibleMemorySize / 1MB, 2)
# Auto caps: absolute runaway + room for growable pagefile
if ($MaxSysCommitGB -le 0) {
  if ($cfgMaxMb -gt 0) {
    $MaxSysCommitGB = [math]::Round($physGb + ($cfgMaxMb / 1024.0) - 1.5, 2)
  } else {
    $MaxSysCommitGB = [math]::Max(10.0, $cLimit0 - 1.2)
  }
}
if ($MaxProcVirtGB -le 0) { $MaxProcVirtGB = 40.0 }
if ($MaxProcWsGB -le 0) { $MaxProcWsGB = 14.0 }
$canGrow = ($cfgMaxMb -gt 0) -and (($allocMb + 512) -lt $cfgMaxMb)
if ($free0 -lt 3.0) {
  "REFUSE: free_phys=${free0}G < 3.0G (commit_limit=${cLimit0}G)" | Tee-Object $summary
  exit 3
}
"INFO: commit_limit=${cLimit0}G alloc_pf=${allocMb}MB cfg_max_pf=${cfgMaxMb}MB can_grow=$canGrow caps proc<=${MaxProcVirtGB}G sys<=${MaxSysCommitGB}G" |
  Tee-Object -FilePath $summary -Append
if ($cLimit0 -lt 22.0 -and -not $canGrow) {
  "WARN: commit_limit low and pagefile cannot grow — likely early kill." |
    Tee-Object -FilePath $summary -Append
}

$argList = @("-u", "run_arm.py", "--arm", $Arm) + ($ExtraArgs -split '\s+' | Where-Object { $_ })
$launchMsg = "LAUNCH arm=$Arm args=$($argList -join ' ') free=${free0}G commitUsed=${cUsed0}G caps commit<=${MaxProcVirtGB}G ws<=${MaxProcWsGB}G sys<=${MaxSysCommitGB}G"
Add-Content -Path $summary -Value $launchMsg -Encoding utf8
if (-not (Test-Path $summary)) { Set-Content -Path $summary -Value $launchMsg -Encoding utf8 }

"sec,ws_gb,commit_gb,sys_free_gb,sys_commit_gb,action" | Set-Content $mem -Encoding utf8

$proc = Start-Process -FilePath $py -ArgumentList $argList `
  -WorkingDirectory $WorkDir `
  -RedirectStandardOutput $out -RedirectStandardError $err `
  -PassThru -NoNewWindow

$t0 = Get-Date
$peakWs = 0.0
$peakVirt = 0.0
$peakCommit = 0.0
$peakLimit = $cLimit0
$killed = $false
$reason = ""
$freeStrikes = 0
$freeStrikeNeed = 4
$freeStrikeNeedGrow = 30  # ~60s while pagefile can grow
$ratioStrikes = 0
$ratioStrikeNeed = 8

while (-not $proc.HasExited) {
  Start-Sleep -Seconds 2
  $elapsed = [int]((Get-Date) - $t0).TotalSeconds

  $pys = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -EA SilentlyContinue |
    Where-Object {
      $_.ProcessId -eq $proc.Id -or $_.ParentProcessId -eq $proc.Id -or
      ($_.CommandLine -and $_.CommandLine -match 'run_arm\.py|5060_three_hier')
    })
  if ($pys.Count -eq 0) { $pys = @($proc) }

  $ws = 0.0; $virt = 0.0
  foreach ($p in $pys) {
    try {
      $gp = Get-Process -Id $p.ProcessId -EA Stop
      $ws += $gp.WorkingSet64 / 1GB
      # PageFileUsage = commit charge (KB on Win32_Process)
      $pfKb = [double](Get-CimInstance Win32_Process -Filter "ProcessId=$($p.ProcessId)").PageFileUsage
      $virt += $pfKb / (1024.0 * 1024.0)
    } catch {}
  }
  $ws = [math]::Round($ws, 2)
  $virt = [math]::Round($virt, 2)
  if ($ws -gt $peakWs) { $peakWs = $ws }
  if ($virt -gt $peakVirt) { $peakVirt = $virt }

  $os = Get-CimInstance Win32_OperatingSystem
  $free = [math]::Round($os.FreePhysicalMemory / 1MB, 2)
  $cUsed = [math]::Round(($os.TotalVirtualMemorySize - $os.FreeVirtualMemory) / 1MB, 2)
  $cLimit = [math]::Round($os.TotalVirtualMemorySize / 1MB, 2)
  if ($cUsed -gt $peakCommit) { $peakCommit = $cUsed }
  if ($cLimit -gt $peakLimit) { $peakLimit = $cLimit }

  # refresh grow flag occasionally
  if (($elapsed % 10) -lt 2) {
    $allocMb = 0
    Get-CimInstance Win32_PageFileUsage -EA SilentlyContinue | ForEach-Object { $allocMb += [int]$_.AllocatedBaseSize }
    $canGrow = ($cfgMaxMb -gt 0) -and (($allocMb + 512) -lt $cfgMaxMb)
  }

  $action = "ok"
  $ratio = if ($cLimit -gt 0) { $cUsed / $cLimit } else { 1.0 }
  if ($virt -ge $MaxProcVirtGB) {
    $action = "KILL_COMMIT"; $killed = $true; $reason = "proc_commit=${virt}G>=${MaxProcVirtGB}G"
  } elseif ($ws -ge $MaxProcWsGB) {
    $action = "KILL_WS"; $killed = $true; $reason = "ws=${ws}G>=${MaxProcWsGB}G"
  } elseif ($cUsed -ge $MaxSysCommitGB) {
    $action = "KILL_SYS_COMMIT"; $killed = $true; $reason = "sys_commit=${cUsed}G>=${MaxSysCommitGB}G"
  } elseif ($ratio -ge 0.98) {
    $ratioStrikes++
    $action = "WARN_RATIO_$ratioStrikes"
    if ((-not $canGrow) -and ($ratioStrikes -ge 2)) {
      $action = "KILL_SYS_RATIO"; $killed = $true
      $reason = "sys_commit_ratio=$([math]::Round($ratio*100,1))% used=${cUsed}/${cLimit}G grow=0"
    }
  } elseif ($elapsed -ge $TimeoutSec) {
    $action = "KILL_TIMEOUT"; $killed = $true; $reason = "timeout=${elapsed}s"
  } elseif ($free -lt $MinSysFreeGB) {
    $freeStrikes++
    $needFree = if ($canGrow) { $freeStrikeNeedGrow } else { $freeStrikeNeed }
    $action = "WARN_SYS_FREE_$freeStrikes"
    if ($freeStrikes -ge $needFree) {
      $action = "KILL_SYS_FREE"; $killed = $true
      $reason = "sys_free=${free}G<${MinSysFreeGB}G x$freeStrikes grow=$canGrow"
    }
  } else {
    $freeStrikes = 0
    $ratioStrikes = 0
  }

  "$elapsed,$ws,$virt,$free,$cUsed,limit=$cLimit,grow=$canGrow,$action" | Add-Content $mem -Encoding utf8

  if ($killed) {
    $wdog = Join-Path $logDir "${label}_${stamp}_watchdog.txt"
    "$(Get-Date -Format o) WATCHDOG $action $reason" | Set-Content $wdog -Encoding utf8
    Kill-Tree -RootPid $proc.Id
    break
  }
}

Start-Sleep -Seconds 2
try { $proc.Refresh() } catch {}
$code = $proc.ExitCode
# Start-Process sometimes leaves ExitCode null even after clean exit; infer from logs
if (-not $killed) {
  $okHint = $false
  if (Test-Path $out) {
    $tail = Get-Content $out -Tail 30 -EA SilentlyContinue
    if ($tail -match 'THREE done|done\b|ALL DONE') { $okHint = $true }
  }
  if ($null -eq $code) {
    $code = if ($okHint) { 0 } else { 1 }
  } elseif (($code -lt 0) -and $okHint) {
    $code = 0
  }
} elseif ($null -eq $code) {
  $code = 99
}
$os1 = Get-CimInstance Win32_OperatingSystem
$result = @"
DONE arm=$Arm exit=$code killed=$killed reason=$reason
peakWS_GB=$peakWs peakVirt_GB=$peakVirt peakSysCommit_GB=$peakCommit peakCommitLimit_GB=$peakLimit
after_free_GB=$([math]::Round($os1.FreePhysicalMemory/1MB,2))
stdout=$out
stderr=$err
memcsv=$mem
"@
$result | Tee-Object $summary
if ($killed) { exit 99 } else { exit ([int]$code) }
