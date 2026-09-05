# 方案 24 · 5090 续跑（conformer 断点 → E fusion → W）
#
#   powershell -File .\run_24_continue.ps1
#   powershell -File .\run_24_continue.ps1 -SkipConformer   # 仅 fusion + W
param(
  [string]$S3RunDir = "F:\Cyy\MI\code\train_lab\out\5090_alg_incr_3s_hop100_accpaper\shallow_openbmi_3s_hop100_balbatch_accpaper\openbmi_3s_hop100\run_20260823_095327\three",
  [string]$EegRunDir = "F:\Cyy\MI\code\train_lab\out\5090_alg_incr_3s_hop100_accpaper\eegnet_openbmi_3s_hop100_balbatch_accpaper\openbmi_3s_hop100\run_20260823_131435\three",
  [string]$CfThreeDir = "F:\Cyy\MI\code\train_lab\out\5090_alg_incr_3s_hop100_accpaper\conformer_openbmi_3s_hop100_balbatch_accpaper\openbmi_3s_hop100\run_20260823_135213\three",
  [switch]$SkipConformer,
  [switch]$SkipW,
  [switch]$NoConsole,
  [int]$CooldownSec = 180
)

$ErrorActionPreference = "Stop"
$Root = "F:\Cyy\MI"
$Pkg3 = Join-Path $Root "code\train_lab\src\step\5090_baselines_openbmi_3s_hop100_accpaper"
$Log = Join-Path $Pkg3 "24_continue.log"

function Log([string]$msg) {
  $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $msg
  Write-Host $line
  Add-Content -Path $Log -Value $line -Encoding utf8
}

function Invoke-Step([string]$name, [scriptblock]$block) {
  Log "BEGIN $name"
  & $block
  if ($LASTEXITCODE -ne 0) {
    Log "FAIL $name exit=$LASTEXITCODE"
    exit $LASTEXITCODE
  }
  Log "OK $name"
}

Log "SCHEME24 CONTINUE start"

$S3RunDir = (Resolve-Path $S3RunDir).Path
$EegRunDir = (Resolve-Path $EegRunDir).Path
$CfThreeDir = (Resolve-Path $CfThreeDir).Path

if (-not $SkipConformer) {
  $cfSummary = Join-Path $CfThreeDir "summary.json"
  if (-not (Test-Path $cfSummary)) {
    Invoke-Step "E conformer resume fold2-4" {
      Push-Location $Pkg3
      python baseline_conformer.py --three-only --max-folds 0 --num-workers 0 `
        --resume-three-dir $CfThreeDir
      Pop-Location
    }
    Start-Sleep -Seconds $CooldownSec
  } else {
    Log "SKIP conformer train ($cfSummary exists)"
  }
} else {
  Log "SKIP conformer (--SkipConformer)"
}

$cfSummary = Join-Path $CfThreeDir "summary.json"
if (-not (Test-Path $cfSummary)) {
  Log "ERROR: conformer summary missing — cannot run E fusion"
  exit 1
}

function Ensure-ProbDumps([string]$runDir, [string]$modelScript) {
  $has = Get-ChildItem $runDir -Recurse -Filter "prob_dump_three.csv" -ErrorAction SilentlyContinue
  if ($has) {
    Log "SKIP prob dump $runDir (already has prob_dump_three.csv)"
    return
  }
  Invoke-Step "dump probs $modelScript $runDir" {
    Push-Location $Pkg3
    python $modelScript --dump-probs --replay-run-dir $runDir --replay-stage three --num-workers 0
    Pop-Location
  }
}

Ensure-ProbDumps $S3RunDir "baseline_shallow.py"
Ensure-ProbDumps $EegRunDir "baseline_eegnet.py"
Ensure-ProbDumps $CfThreeDir "baseline_conformer.py"

$eFusion = Join-Path $S3RunDir "replay_e_fusion.json"
if (-not (Test-Path $eFusion)) {
  Invoke-Step "E fusion replay" {
    Push-Location $Pkg3
    python replay_e_fusion.py --shallow-run $S3RunDir --eegnet-run $EegRunDir --conformer-run $CfThreeDir
    Pop-Location
  }
} else {
  Log "SKIP E fusion ($eFusion exists)"
}

if (-not $SkipW) {
  $fullArgs = @(
    "-NoProfile", "-ExecutionPolicy", "Bypass",
    "-File", (Join-Path $Pkg3 "run_24_full_experiment.ps1"),
    "-S3RunDir", $S3RunDir,
    "-SkipV", "-SkipT", "-SkipE"
  )
  if ($NoConsole) { $fullArgs += "-NoConsole" }
  Invoke-Step "W chain (O3s_m + adaptive window)" {
    & powershell @fullArgs
  }
}

Log "SCHEME24 CONTINUE DONE"
exit 0
