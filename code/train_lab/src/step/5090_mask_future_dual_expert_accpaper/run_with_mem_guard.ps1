# Scheme 17 · external memory watchdog (backup for in-process mem_guard).
# Kills the python process tree if commit/WS/sys-free explode.
#
# 默认弹出可见控制台窗口（Tee 同步写日志）；后台静默加 -NoConsole。
param(
  [Parameter(Mandatory = $true)][string]$Arm,
  [string]$ExtraArgs = "--max-folds 1 --num-workers 0",
  [double]$MaxProcVirtGB = 0,   # 0 = auto
  [double]$MaxProcWsGB = 0,     # 0 = auto
  [double]$MinSysFreeGB = 0.20,
  [double]$MaxSysCommitGB = 0,  # 0 = auto
  [int]$TimeoutSec = 43200,     # 12h / arm
  [string]$WorkDir = "D:\cyy\MI\code\train_lab\src\step\5090_mask_future_dual_expert_accpaper",
  [switch]$NoConsole
)

$ErrorActionPreference = "Continue"
$py = "D:\cyy\MI\.venv\Scripts\python.exe"
$logDir = "D:\cyy\MI\code\train_lab\out\_ab_mem"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$label = "guarded17_$Arm"
$out = Join-Path $logDir "${label}_${stamp}_stdout.txt"
$err = Join-Path $logDir "${label}_${stamp}_stderr.txt"
$mem = Join-Path $logDir "${label}_${stamp}_mem.csv"
$summary = Join-Path $logDir "${label}_${stamp}_summary.txt"
$exitFile = Join-Path $logDir "${label}_${stamp}_exitcode.txt"
$runner = Join-Path $logDir "${label}_${stamp}_runner.ps1"

function Kill-Tree([int]$RootPid) {
  try {
    & taskkill.exe /PID $RootPid /T /F 2>$null | Out-Null
  } catch {}
  Get-CimInstance Win32_Process -Filter "Name='python.exe'" -EA SilentlyContinue |
    Where-Object {
      $_.CommandLine -and (
        $_.CommandLine -match '5090_mask_future_dual_expert_accpaper' -or
        $_.CommandLine -match 'run_arm\.py'
      )
    } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force -EA SilentlyContinue }
}

$existing = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -EA SilentlyContinue |
  Where-Object {
    $_.CommandLine -and $_.CommandLine -match '5090_mask_future_dual_expert_accpaper\\run_arm\.py'
  })
if ($existing.Count -gt 0) {
  "REFUSE: scheme17 already running pid=$($existing[0].ProcessId)" | Tee-Object $summary
  exit 2
}

$os0 = Get-CimInstance Win32_OperatingSystem
$free0 = [math]::Round($os0.FreePhysicalMemory / 1MB, 2)
$cUsed0 = [math]::Round(($os0.TotalVirtualMemorySize - $os0.FreeVirtualMemory) / 1MB, 2)
$cLimit0 = [math]::Round($os0.TotalVirtualMemorySize / 1MB, 2)
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
"INFO: commit_limit=${cLimit0}G alloc_pf=${allocMb}MB cfg_max_pf=${cfgMaxMb}MB can_grow=$canGrow caps proc<=${MaxProcVirtGB}G sys<=${MaxSysCommitGB}G show_console=$(-not $NoConsole)" |
  Tee-Object -FilePath $summary -Append

$argList = @("-u", "run_arm.py", "--arm", $Arm) + ($ExtraArgs -split '\s+' | Where-Object { $_ })
$launchMsg = "LAUNCH arm=$Arm args=$($argList -join ' ') free=${free0}G commitUsed=${cUsed0}G caps commit<=${MaxProcVirtGB}G ws<=${MaxProcWsGB}G sys<=${MaxSysCommitGB}G"
Add-Content -Path $summary -Value $launchMsg -Encoding utf8

"sec,ws_gb,commit_gb,sys_free_gb,sys_commit_gb,action" | Set-Content $mem -Encoding utf8

$argLit = ($argList | ForEach-Object { "'" + ($_ -replace "'", "''") + "'" }) -join ", "
$runnerBody = @"
`$ErrorActionPreference = 'Continue'
try { `$Host.UI.RawUI.WindowTitle = 'scheme17 · $Arm · mem-guarded' } catch {}
Set-Location -LiteralPath '$WorkDir'
`$outLog = '$out'
`$errLog = '$err'
`$exitFile = '$exitFile'
Set-Content -LiteralPath `$outLog -Value '' -Encoding utf8
Set-Content -LiteralPath `$errLog -Value '' -Encoding utf8
`$pyArgs = @($argLit)
Write-Host ('[guarded17] start ' + (Get-Date -Format o) + ' arm=$Arm') -ForegroundColor Cyan
Write-Host ('[guarded17] ' + '$py' + ' ' + (`$pyArgs -join ' ')) -ForegroundColor DarkGray
# 2>&1 在 PS5 会把 stderr 变成 ErrorRecord（红字 NativeCommandError）；统一转成普通行再 Tee
& '$py' @pyArgs 2>&1 | ForEach-Object {
  if (`$_ -is [System.Management.Automation.ErrorRecord]) {
    `$line = `$_.ToString()
  } else {
    `$line = "`$_"
  }
  Write-Host `$line
  Add-Content -LiteralPath `$outLog -Value `$line -Encoding utf8
}
`$code = 0
if (`$null -ne `$LASTEXITCODE) { `$code = [int]`$LASTEXITCODE }
Set-Content -LiteralPath `$exitFile -Value ([string]`$code) -Encoding ascii
Write-Host ('[guarded17] exit=' + `$code + ' ' + (Get-Date -Format o)) -ForegroundColor Yellow
exit `$code
"@
Set-Content -LiteralPath $runner -Value $runnerBody -Encoding UTF8

if ($NoConsole) {
  $proc = Start-Process -FilePath $py -ArgumentList $argList `
    -WorkingDirectory $WorkDir `
    -RedirectStandardOutput $out -RedirectStandardError $err `
    -PassThru -WindowStyle Hidden
} else {
  $proc = Start-Process -FilePath "powershell.exe" `
    -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $runner) `
    -WorkingDirectory $WorkDir `
    -PassThru -WindowStyle Normal
}

$t0 = Get-Date
$peakWs = 0.0
$peakVirt = 0.0
$peakCommit = 0.0
$peakLimit = $cLimit0
$killed = $false
$reason = ""
$freeStrikes = 0
$freeStrikeNeed = 4
$freeStrikeNeedGrow = 30
$ratioStrikes = 0
$ratioStrikeNeed = 8

while (-not $proc.HasExited) {
  Start-Sleep -Seconds 2
  $elapsed = [int]((Get-Date) - $t0).TotalSeconds

  $pys = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -EA SilentlyContinue |
    Where-Object {
      $_.ProcessId -eq $proc.Id -or $_.ParentProcessId -eq $proc.Id -or
      ($_.CommandLine -and $_.CommandLine -match 'run_arm\.py|5090_mask_future')
    })
  if ($pys.Count -eq 0) { $pys = @($proc) }

  $ws = 0.0; $virt = 0.0
  foreach ($p in $pys) {
    try {
      $gp = Get-Process -Id $p.ProcessId -EA Stop
      $ws += $gp.WorkingSet64 / 1GB
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
if ((-not $killed) -and (Test-Path $exitFile)) {
  $ec = 0
  if ([int]::TryParse((Get-Content $exitFile -Raw).Trim(), [ref]$ec)) { $code = $ec }
}
if (-not $killed) {
  $okHint = $false
  if (Test-Path $out) {
    $tail = Get-Content $out -Tail 40 -EA SilentlyContinue
    if ($tail -match 'test_acc_paper|THREE done|mean|ALL DONE|done\b') { $okHint = $true }
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
