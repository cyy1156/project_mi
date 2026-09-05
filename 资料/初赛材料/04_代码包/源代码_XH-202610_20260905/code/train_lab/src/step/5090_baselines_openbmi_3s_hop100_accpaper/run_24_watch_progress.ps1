# Scheme 24 progress monitor (refresh every N seconds)
#
#   powershell -File .\run_24_watch_progress.ps1
#   powershell -File .\run_24_watch_progress.ps1 -PollSec 30
param(
  [int]$PollSec = 60
)

$ErrorActionPreference = "SilentlyContinue"
$Root = "F:\Cyy\MI"
$Pkg3 = Join-Path $Root "code\train_lab\src\step\5090_baselines_openbmi_3s_hop100_accpaper"
$OutW = Join-Path $Root "code\train_lab\out\5090_mech_verify_accpaper"
$S3Three = Join-Path $Root "code\train_lab\out\5090_alg_incr_3s_hop100_accpaper\shallow_openbmi_3s_hop100_balbatch_accpaper\openbmi_3s_hop100\run_20260823_095327\three"

function Show-Status {
  Clear-Host
  $now = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
  Write-Host "=== Scheme24 progress === $now  (Ctrl+C to quit, refresh ${PollSec}s)" -ForegroundColor Cyan
  Write-Host ""

  Write-Host "[S3/V/T/E]" -ForegroundColor Yellow
  $ef = Join-Path $S3Three "replay_e_fusion.json"
  if (Test-Path $ef) {
    $j = Get-Content $ef -Raw -Encoding UTF8 | ConvertFrom-Json
    Write-Host ("  E fusion test={0:N4}  delta={1:N2}pp" -f $j.test_acc_paper_fused, $j.delta_test_pp_vs_shallow)
  } else {
    Write-Host "  E fusion: pending"
  }
  $vr = Join-Path $S3Three "replay_v_weighted_vote.json"
  if (Test-Path $vr) {
    $v = Get-Content $vr -Raw -Encoding UTF8 | ConvertFrom-Json
    Write-Host ("  V vote   test={0:N4}  delta={1:N2}pp" -f $v.test_acc_paper_weighted, $v.delta_test_pp)
  }
  Write-Host ""

  Write-Host "[W O3s_m]" -ForegroundColor Yellow
  $o3 = Get-ChildItem $OutW -Directory -Filter "*O3s_m" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
  if ($o3) {
    $done = 0
    foreach ($i in 0..4) {
      $fd = Join-Path $o3.FullName "fold$i"
      if (Test-Path (Join-Path $fd "metrics.json")) { $done++ }
      elseif (Test-Path (Join-Path $fd "best.pt")) { $done++ }
    }
    $sumOk = Test-Path (Join-Path $o3.FullName "summary.json")
    Write-Host ("  run={0}" -f $o3.Name)
    Write-Host ("  folds={0}/5  summary={1}" -f $done, $sumOk)
    $log = Join-Path $o3.FullName "run.log"
    if (Test-Path $log) {
      Write-Host "  run.log (last 3 lines):"
      Get-Content $log -Tail 3 | ForEach-Object { Write-Host "    $_" -ForegroundColor DarkGray }
    }
  } else {
    Write-Host "  no O3s_m run dir yet"
  }
  Write-Host ""

  Write-Host "[W adaptive replay]" -ForegroundColor Yellow
  $replay = Get-ChildItem $OutW -Recurse -Filter "replay_w_adaptive.json" -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1
  if ($replay) {
    $w = Get-Content $replay.FullName -Raw -Encoding UTF8 | ConvertFrom-Json
    Write-Host ("  DONE  {0}" -f $replay.FullName)
    Write-Host ("  test_acc_adaptive={0:N4}  tau_conf={1:N2}" -f $w.test_acc_adaptive, $w.tau_conf_val)
  } else {
    Write-Host "  replay_w_adaptive.json: pending"
  }
  Write-Host ""

  Write-Host "[processes]" -ForegroundColor Yellow
  $procs = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -match '5090_mech_verify|5090_baselines|O3s_m|run_arm|replay_w' }
  if ($procs) {
    foreach ($p in $procs) {
      $ws = [math]::Round($p.WorkingSet64 / 1GB, 2)
      $cmd = $p.CommandLine
      if ($cmd.Length -gt 100) { $cmd = $cmd.Substring(0, 100) + "..." }
      Write-Host ("  pid={0}  WS={1}GB  {2}" -f $p.ProcessId, $ws, $cmd)
    }
  } else {
    Write-Host "  no related python process"
  }
  Write-Host ""

  Write-Host "[chain log 24_e_w_chain.log last 5 lines]" -ForegroundColor Yellow
  $chainLog = Join-Path $Pkg3 "24_e_w_chain.log"
  if (Test-Path $chainLog) {
    Get-Content $chainLog -Tail 5 | ForEach-Object { Write-Host "  $_" -ForegroundColor DarkGray }
  } else {
    Write-Host "  (none)"
  }
}

while ($true) {
  Show-Status
  Start-Sleep -Seconds $PollSec
}
