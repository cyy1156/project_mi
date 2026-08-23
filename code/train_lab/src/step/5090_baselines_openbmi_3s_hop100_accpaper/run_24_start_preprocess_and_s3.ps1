# 方案 24 · 5090 一键：3s preprocess → S3 shallow 五折校准
#
#   powershell -File .\run_24_start_preprocess_and_s3.ps1 -NoConsole
param(
  [switch]$NoConsole,
  [switch]$SkipPreprocess,
  [int]$MaxFolds = 0
)

$ErrorActionPreference = "Stop"
$Root = "F:\Cyy\MI"
$PreRoot = Join-Path $Root "code\preprocess_lab"
$Pkg = Join-Path $Root "code\train_lab\src\step\5090_baselines_openbmi_3s_hop100_accpaper"
$DataGlob = "F:/Cyy/MI/DATA/openbmi/openbmi/sess*_subj*_EEG_MI.mat"
$Out3s = Join-Path $PreRoot "out\openbmi_3s_hop100\openbmi_X.npy"
$Log = Join-Path $Pkg "24_start_preprocess_s3.log"
$PidFile = Join-Path $Pkg "24_start_preprocess_s3.pid"

function Log([string]$msg) {
  $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $msg
  Write-Host $line
  Add-Content -Path $Log -Value $line -Encoding utf8
}

Set-Content -Path $PidFile -Value $PID -Encoding ascii
Log "START pid=$PID SkipPreprocess=$SkipPreprocess MaxFolds=$MaxFolds"

if (-not $SkipPreprocess) {
  if (Test-Path $Out3s) {
    Log "SKIP preprocess (exists $Out3s)"
  } else {
    Log "RUN preprocess 3s hop100 glob=$DataGlob"
    Push-Location $PreRoot
    $preCmd = "python -m src.datasets.openbmi.batch_3s_hop100 --glob `"$DataGlob`""
    if ($NoConsole) {
      Invoke-Expression $preCmd
    } else {
      Invoke-Expression $preCmd
    }
    if ($LASTEXITCODE -ne 0) {
      Log "FAIL preprocess exit=$LASTEXITCODE"
      exit $LASTEXITCODE
    }
    Pop-Location
    if (-not (Test-Path $Out3s)) {
      Log "FAIL preprocess finished but missing $Out3s"
      exit 2
    }
    Log "OK preprocess"
  }
} else {
  Log "SKIP preprocess (flag)"
}

Log "RUN S3 shallow three-only max-folds=$MaxFolds"
Push-Location $Pkg
$foldArg = if ($MaxFolds -le 0) { 0 } else { $MaxFolds }
$trainCmd = "python baseline_shallow.py --three-only --max-folds $foldArg --num-workers 0"
Invoke-Expression $trainCmd
$code = $LASTEXITCODE
Pop-Location
if ($code -ne 0) {
  Log "FAIL S3 shallow exit=$code"
  exit $code
}
Log "ALL DONE S3 shallow — check out/5090_alg_incr_3s_hop100_accpaper/"

# 自动衔接方案 24 全实验（V/T/E/W）
Log "LAUNCH run_24_full_experiment.ps1"
$fullScript = Join-Path $Pkg "run_24_full_experiment.ps1"
if (Test-Path $fullScript) {
  & powershell -NoProfile -File $fullScript -NoConsole
  $fullCode = $LASTEXITCODE
  Log "FULL EXPERIMENT exit=$fullCode"
  exit $fullCode
}
exit 0
