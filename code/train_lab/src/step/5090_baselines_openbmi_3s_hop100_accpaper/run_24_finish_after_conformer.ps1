# 等待 conformer summary → E fusion → W（方案 24 收尾）
#
#   powershell -File .\run_24_finish_after_conformer.ps1
param(
  [string]$S3RunDir = "F:\Cyy\MI\code\train_lab\out\5090_alg_incr_3s_hop100_accpaper\shallow_openbmi_3s_hop100_balbatch_accpaper\openbmi_3s_hop100\run_20260823_095327\three",
  [string]$EegRunDir = "F:\Cyy\MI\code\train_lab\out\5090_alg_incr_3s_hop100_accpaper\eegnet_openbmi_3s_hop100_balbatch_accpaper\openbmi_3s_hop100\run_20260823_131435\three",
  [string]$CfThreeDir = "F:\Cyy\MI\code\train_lab\out\5090_alg_incr_3s_hop100_accpaper\conformer_openbmi_3s_hop100_balbatch_accpaper\openbmi_3s_hop100\run_20260823_135213\three",
  [int]$PollSec = 120,
  [int]$MaxWaitHours = 8
)

$ErrorActionPreference = "Stop"
$Pkg3 = "F:\Cyy\MI\code\train_lab\src\step\5090_baselines_openbmi_3s_hop100_accpaper"
$Log = Join-Path $Pkg3 "24_finish_after_conformer.log"
$CfSummary = Join-Path $CfThreeDir "summary.json"

function Log([string]$msg) {
  $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $msg
  Write-Host $line
  Add-Content -Path $Log -Value $line -Encoding utf8
}

Log "WAIT conformer summary=$CfSummary"
$deadline = (Get-Date).AddHours($MaxWaitHours)
while ((Get-Date) -lt $deadline) {
  $done = 0
  foreach ($i in 0..4) {
    if (Test-Path (Join-Path $CfThreeDir "fold$i\best_three.pt")) { $done++ }
  }
  if (Test-Path $CfSummary) {
    Log "conformer DONE summary exists ($done/5 ckpts)"
    break
  }
  Log "conformer progress $done/5 folds"
  Start-Sleep -Seconds $PollSec
}

if (-not (Test-Path $CfSummary)) {
  Log "TIMEOUT waiting conformer"
  exit 1
}

Log "launch continue (fusion + W)"
& powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $Pkg3 "run_24_continue.ps1") `
  -S3RunDir $S3RunDir -EegRunDir $EegRunDir -CfThreeDir $CfThreeDir -SkipConformer -NoConsole
exit $LASTEXITCODE
