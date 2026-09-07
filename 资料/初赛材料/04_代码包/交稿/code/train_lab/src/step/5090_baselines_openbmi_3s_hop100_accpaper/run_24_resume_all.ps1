# 方案 24 · 续跑：preprocess(90/108) → S3 → 全实验链
#
#   powershell -File .\run_24_resume_all.ps1
param(
  [switch]$SkipPreprocess,
  [int]$MaxWaitHours = 8
)

$ErrorActionPreference = "Stop"
$Root = "F:\Cyy\MI"
$PreRoot = Join-Path $Root "code\preprocess_lab"
$Pkg = Join-Path $Root "code\train_lab\src\step\5090_baselines_openbmi_3s_hop100_accpaper"
$DataGlob = "F:/Cyy/MI/DATA/openbmi/openbmi/sess*_subj*_EEG_MI.mat"
$Out3s = Join-Path $PreRoot "out\openbmi_3s_hop100\openbmi_X.npy"
$Log = Join-Path $Pkg "24_resume_all.log"

function Log([string]$msg) {
  $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $msg
  Write-Host $line
  Add-Content -Path $Log -Value $line -Encoding utf8
}

Log "RESUME ALL pid=$PID SkipPreprocess=$SkipPreprocess"

if (-not $SkipPreprocess -and -not (Test-Path $Out3s)) {
  $nOk = 0
  $man = Join-Path $PreRoot "out\openbmi_3s_hop100\manifest.json"
  if (Test-Path $man) {
    $nOk = (Select-String -Path $man -Pattern '"status": "ok"').Count
  }
  Log "RUN preprocess resume ($nOk/108 shards done)"
  Push-Location $PreRoot
  python -m src.datasets.openbmi.batch_3s_hop100 --glob $DataGlob
  $preCode = $LASTEXITCODE
  Pop-Location
  if ($preCode -ne 0) {
    Log "FAIL preprocess exit=$preCode"
    exit $preCode
  }
  if (-not (Test-Path $Out3s)) {
    Log "FAIL missing $Out3s after preprocess"
    exit 2
  }
  Log "OK preprocess merged"
} elseif (Test-Path $Out3s) {
  Log "SKIP preprocess (openbmi_X.npy exists)"
} else {
  Log "SKIP preprocess (flag)"
}

Log "LAUNCH start_preprocess_and_s3 (SkipPreprocess) + full experiment chain"
Push-Location $Pkg
Add-Content -Path (Join-Path $Pkg "24_start_preprocess_s3.log") -Value ("[{0}] RESUME START pid=$PID" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss")) -Encoding utf8
powershell -NoProfile -File .\run_24_start_preprocess_and_s3.ps1 -SkipPreprocess -NoConsole
$code = $LASTEXITCODE
Pop-Location
Log "RESUME ALL exit=$code"
exit $code
