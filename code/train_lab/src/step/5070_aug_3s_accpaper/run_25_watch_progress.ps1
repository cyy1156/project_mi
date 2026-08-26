# Scheme 25 progress monitor (5090)
#
#   powershell -File .\run_25_watch_progress.ps1
#   powershell -File .\run_25_watch_progress.ps1 -PollSec 60
param(
  [int]$PollSec = 60
)

$ErrorActionPreference = "SilentlyContinue"
$Root = "F:\Cyy\MI"
$Pkg = Join-Path $Root "code\train_lab\src\step\5070_aug_3s_accpaper"
$G1Base = Join-Path $Root "code\train_lab\out\5090_aug_3s_accpaper\shallow_openbmi_3s_hop100_balbatch_accpaper\openbmi_3s_hop100"
$Results = Join-Path $Root "资料\模型训练\25_旁路_域增广训练_增量FT配套_openbmi_accpaper\results"
$ChainLog = Join-Path $Pkg "25_run_all_5090.log"

function Show-Status {
  Clear-Host
  $now = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
  Write-Host "=== Scheme25 5090 progress === $now  (Ctrl+C quit, refresh ${PollSec}s)" -ForegroundColor Cyan
  Write-Host ""

  Write-Host "[G1 train]" -ForegroundColor Yellow
  if (Test-Path $G1Base) {
    $runs = Get-ChildItem $G1Base -Directory -Filter "run_*" | Sort-Object Name -Descending
    if ($runs) {
      $latest = $runs[0]
      $done = 0
      foreach ($i in 0..4) {
        $fd = Join-Path $latest.FullName "three\fold$i\best_three.pt"
        if (Test-Path $fd) { $done++ }
      }
      $sumOk = Test-Path (Join-Path $latest.FullName "three\summary.json")
      Write-Host ("  latest={0}  folds={1}/5  summary={2}" -f $latest.Name, $done, $sumOk)
      $rlog = Join-Path $latest.FullName "run.log"
      if (Test-Path $rlog) {
        Write-Host "  run.log (last 3):"
        Get-Content $rlog -Tail 3 | ForEach-Object { Write-Host "    $_" -ForegroundColor DarkGray }
      }
    } else {
      Write-Host "  no G1 run yet"
    }
  } else {
    Write-Host "  out dir not created yet"
  }
  Write-Host ""

  Write-Host "[results]" -ForegroundColor Yellow
  foreach ($tag in @("S25-G1_openbmi_guard", "S25-A0_zeroshot", "S25-G1", "S25-G2", "S25-G3", "S25-A0")) {
    $dir = Join-Path $Results $tag
    if (Test-Path $dir) {
      $last = Get-ChildItem $dir -Directory | Sort-Object Name -Descending | Select-Object -First 1
      if ($last) { Write-Host ("  {0}: {1}" -f $tag, $last.Name) }
    }
  }
  Write-Host ""

  Write-Host "[chain log]" -ForegroundColor Yellow
  if (Test-Path $ChainLog) {
    Get-Content $ChainLog -Tail 5 | ForEach-Object { Write-Host "  $_" -ForegroundColor DarkGray }
  } else {
    Write-Host "  25_run_all_5090.log: not started"
  }
  Write-Host ""

  Write-Host "[processes]" -ForegroundColor Yellow
  $procs = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -match '5070_aug_3s_accpaper|baseline_shallow_aug|incremental_ft|eval_stieger|eval_openbmi_guard' }
  if ($procs) {
    foreach ($p in $procs) {
      $ws = [math]::Round($p.WorkingSet64 / 1GB, 2)
      $cmd = $p.CommandLine
      if ($cmd.Length -gt 110) { $cmd = $cmd.Substring(0, 110) + "..." }
      Write-Host ("  pid={0}  WS={1}GB  {2}" -f $p.ProcessId, $ws, $cmd)
    }
  } else {
    Write-Host "  no scheme25 python process"
  }
}

while ($true) {
  Show-Status
  Start-Sleep -Seconds $PollSec
}
