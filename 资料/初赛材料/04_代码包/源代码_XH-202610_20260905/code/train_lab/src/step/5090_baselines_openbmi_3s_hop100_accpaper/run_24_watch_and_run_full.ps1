# 等待 S3 shallow 完成 → 自动启动方案 24 全实验链
#
#   powershell -File .\run_24_watch_and_run_full.ps1
param(
  [int]$PollSec = 60,
  [int]$MaxWaitHours = 6
)

$ErrorActionPreference = "Stop"
$Pkg3 = "F:\Cyy\MI\code\train_lab\src\step\5090_baselines_openbmi_3s_hop100_accpaper"
$StartLog = Join-Path $Pkg3 "24_start_preprocess_s3.log"
$WatchLog = Join-Path $Pkg3 "24_watch_and_run_full.log"
$Out3 = "F:\Cyy\MI\code\train_lab\out\5090_alg_incr_3s_hop100_accpaper"

function Log([string]$msg) {
  $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $msg
  Write-Host $line
  Add-Content -Path $WatchLog -Value $line -Encoding utf8
}

Log "WATCH start poll=${PollSec}s max_wait=${MaxWaitHours}h"

$deadline = (Get-Date).AddHours($MaxWaitHours)
while ((Get-Date) -lt $deadline) {
  # 1) start script finished
  if (Test-Path $StartLog) {
    $lines = Get-Content $StartLog -ErrorAction SilentlyContinue
    $lastStart = ($lines | Select-String -Pattern "^\[.*\] START" | Select-Object -Last 1).LineNumber
    if (-not $lastStart) { $lastStart = 1 }
    $recent = $lines[($lastStart - 1)..($lines.Count - 1)] -join "`n"
    if ($recent -match "ALL DONE S3 shallow|FULL EXPERIMENT exit=0") {
      Log "detected pipeline done via start log"
      break
    }
    if ($recent -match "FAIL preprocess|FAIL S3|FULL EXPERIMENT exit=[1-9]") {
      Log "start script failed — check $StartLog"
      exit 1
    }
  }
  # 2) summary.json with 5 folds
  $summaries = Get-ChildItem $Out3 -Recurse -Filter "summary.json" -ErrorAction SilentlyContinue |
    Where-Object { $_.FullName -match "\\three\\summary\.json$" } |
    Sort-Object LastWriteTime -Descending
  if ($summaries) {
    try {
      $j = Get-Content $summaries[0].FullName -Raw | ConvertFrom-Json
      $nf = @($j.folds).Count
      if ($nf -ge 5) {
        Log "detected S3 summary with $nf folds: $($summaries[0].FullName)"
        break
      }
    } catch { }
  }
  $nOk = 0
  $man = "F:\Cyy\MI\code\preprocess_lab\out\openbmi_3s_hop100\manifest.json"
  if (Test-Path $man) {
    $nOk = (Select-String -Path $man -Pattern '"status": "ok"').Count
  }
  Log "waiting... preprocess_ok=$nOk/108"
  Start-Sleep -Seconds $PollSec
}

if ((Get-Date) -ge $deadline) {
  Log "TIMEOUT waiting for S3"
  exit 2
}

Log "LAUNCH run_24_full_experiment.ps1"
Push-Location $Pkg3
powershell -NoProfile -File .\run_24_full_experiment.ps1 -NoConsole
$code = $LASTEXITCODE
Pop-Location
Log "FULL EXPERIMENT exit=$code"
exit $code
