# 方案 24 · 5090 全实验链（S3 权重就绪后）
#
# 前置：openbmi_3s_hop100 已 merge · S3 shallow 五折已完成
#
#   powershell -File .\run_24_full_experiment.ps1
#   powershell -File .\run_24_full_experiment.ps1 -S3RunDir "F:\...\run_20260823_...\three"
#   powershell -File .\run_24_full_experiment.ps1 -SkipW
param(
  [string]$S3RunDir = "",
  [switch]$SkipV,
  [switch]$SkipT,
  [switch]$SkipE,
  [switch]$SkipW,
  [switch]$NoConsole,
  [int]$CooldownSec = 180
)

$ErrorActionPreference = "Stop"
$Root = "F:\Cyy\MI"
$Pkg3 = Join-Path $Root "code\train_lab\src\step\5090_baselines_openbmi_3s_hop100_accpaper"
$PkgW = Join-Path $Root "code\train_lab\src\step\5090_mech_verify_accpaper"
$Out3 = Join-Path $Root "code\train_lab\out\5090_alg_incr_3s_hop100_accpaper"
$OutW = Join-Path $Root "code\train_lab\out\5090_mech_verify_accpaper"
$Log = Join-Path $Pkg3 "24_full_experiment.log"

# 方案 23 已完成 · W 腿 M1/M2
$O1Run = Join-Path $OutW "20260822_172743_O1s_m"
$O2Run = Join-Path $OutW "20260822_152136_O2s_m"

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

function Find-LatestTThree {
  $runs = Get-ChildItem $Out3 -Recurse -Directory -Filter "run_*" -ErrorAction SilentlyContinue |
    Where-Object {
      $s = Join-Path $_.FullName "three\summary.json"
      if (-not (Test-Path $s)) { return $false }
      $j = Get-Content $s -Raw -Encoding UTF8 | ConvertFrom-Json
      [double]$j.hparams.t0_weight_alpha -gt 0
    } |
    Sort-Object LastWriteTime -Descending
  if (-not $runs) { return $null }
  return Join-Path $runs[0].FullName "three"
}

function Find-LatestS3Three {
  $runs = Get-ChildItem $Out3 -Recurse -Directory -Filter "run_*" -ErrorAction SilentlyContinue |
    Where-Object {
      $s = Join-Path $_.FullName "three\summary.json"
      if (-not (Test-Path $s)) { return $false }
      $j = Get-Content $s -Raw -Encoding UTF8 | ConvertFrom-Json
      [double]$j.hparams.t0_weight_alpha -eq 0
    } |
    Sort-Object LastWriteTime -Descending
  if (-not $runs) { return $null }
  return Join-Path $runs[0].FullName "three"
}

function Get-TestAccPaper([string]$summaryPath) {
  $j = Get-Content $summaryPath -Raw -Encoding UTF8 | ConvertFrom-Json
  return [double]$j.test_acc_paper_mean
}

if (-not $S3RunDir) {
  $S3RunDir = Find-LatestS3Three
}
if (-not $S3RunDir -or -not (Test-Path (Join-Path $S3RunDir "summary.json"))) {
  Log "ERROR: S3 run not found. Pass -S3RunDir or finish S3 shallow first."
  exit 1
}
$S3RunDir = (Resolve-Path $S3RunDir).Path
$S3Root = Split-Path $S3RunDir -Parent
Log "SCHEME24 FULL START s3=$S3RunDir"

$s3Acc = Get-TestAccPaper (Join-Path $S3RunDir "summary.json")
Log "S3 calibration Three test_acc_paper=$([math]::Round($s3Acc,4)) gate=[0.584,0.591]"
if ($s3Acc -lt 0.584 -or $s3Acc -gt 0.591) {
  Log "WARN: S3 outside calibration gate — continuing per user request (cross-machine sanity 0.587)"
}

# ── V · 置信加权投票 ──
if (-not $SkipV) {
  $hasVDump = Get-ChildItem $S3RunDir -Recurse -Filter "prob_dump_three.csv" -ErrorAction SilentlyContinue
  if (-not $hasVDump) {
    Invoke-Step "V dump-probs" {
      Push-Location $Pkg3
      python baseline_shallow.py --dump-probs --replay-run-dir $S3RunDir --replay-stage three --num-workers 0
      Pop-Location
    }
  } else {
    Log "SKIP V dump-probs (prob_dump_three.csv exists)"
  }
  $vReplay = Join-Path $S3RunDir "replay_v_weighted_vote.json"
  if (-not (Test-Path $vReplay)) {
    Invoke-Step "V weighted vote replay" {
      Push-Location $Pkg3
      python replay_v_weighted_vote.py --run-dir $S3RunDir
      if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
      Pop-Location
    }
  } else {
    Log "SKIP V replay ($vReplay exists)"
  }
  Start-Sleep -Seconds $CooldownSec
}

# ── T · t0 软降权 α=0.6 ──
$TRunDir = Find-LatestTThree
if (-not $SkipT) {
  if ($TRunDir) {
    Log "SKIP T train (existing t0-weight run $TRunDir)"
  } else {
    Invoke-Step "T shallow t0-weight=0.6" {
      Push-Location $Pkg3
      python baseline_shallow.py --three-only --t0-weight 0.6 --max-folds 0 --num-workers 0
      if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
      Pop-Location
    }
    $TRunDir = Find-LatestTThree
  }
  if ($TRunDir) {
    $tAcc = Get-TestAccPaper (Join-Path $TRunDir "summary.json")
    Log "T alpha=0.6 test_acc_paper=$([math]::Round($tAcc,4)) delta_vs_S3=$([math]::Round(($tAcc-$s3Acc)*100,2))pp"
  }
  Start-Sleep -Seconds $CooldownSec
}

# ── E · eegnet + conformer ──
$EegRun = $null
$CfRun = $null
if (-not $SkipE) {
  $eegRuns = Get-ChildItem $Out3 -Recurse -Directory -Filter "run_*" |
    Where-Object { $_.FullName -match "eegnet" -and (Test-Path (Join-Path $_.FullName "three\summary.json")) } |
    Sort-Object LastWriteTime -Descending
  if (-not $eegRuns) {
    Invoke-Step "E eegnet five-fold" {
      Push-Location $Pkg3
      python baseline_eegnet.py --three-only --max-folds 0 --num-workers 0
      Pop-Location
    }
    Start-Sleep -Seconds $CooldownSec
    $eegRuns = Get-ChildItem $Out3 -Recurse -Directory -Filter "run_*" |
      Where-Object { $_.FullName -match "eegnet" -and (Test-Path (Join-Path $_.FullName "three\summary.json")) } |
      Sort-Object LastWriteTime -Descending
  } else {
    Log "SKIP E eegnet train (existing $($eegRuns[0].FullName))"
  }

  $cfRuns = Get-ChildItem $Out3 -Recurse -Directory -Filter "run_*" |
    Where-Object { $_.FullName -match "conformer" -and (Test-Path (Join-Path $_.FullName "three\summary.json")) } |
    Sort-Object LastWriteTime -Descending
  if (-not $cfRuns) {
    Invoke-Step "E conformer five-fold" {
      Push-Location $Pkg3
      python baseline_conformer.py --three-only --max-folds 0 --num-workers 0
      Pop-Location
    }
    Start-Sleep -Seconds $CooldownSec
    $cfRuns = Get-ChildItem $Out3 -Recurse -Directory -Filter "run_*" |
      Where-Object { $_.FullName -match "conformer" -and (Test-Path (Join-Path $_.FullName "three\summary.json")) } |
      Sort-Object LastWriteTime -Descending
  } else {
    Log "SKIP E conformer train (existing $($cfRuns[0].FullName))"
  }

  if ($eegRuns -and $cfRuns) {
    $EegRun = Join-Path $eegRuns[0].FullName "three"
    $CfRun = Join-Path $cfRuns[0].FullName "three"
    foreach ($pair in @(
      @{ run = $S3RunDir; script = "baseline_shallow.py" },
      @{ run = $EegRun; script = "baseline_eegnet.py" },
      @{ run = $CfRun; script = "baseline_conformer.py" }
    )) {
      $hasDump = Get-ChildItem $pair.run -Recurse -Filter "prob_dump_three.csv" -ErrorAction SilentlyContinue
      if ($hasDump) { Log "SKIP E dump $($pair.script) (prob_dump exists)"; continue }
      Invoke-Step "E dump $($pair.script)" {
        Push-Location $Pkg3
        python $($pair.script) --dump-probs --replay-run-dir $($pair.run) --replay-stage three --num-workers 0
        Pop-Location
      }
    }
    Invoke-Step "E fusion replay" {
      Push-Location $Pkg3
      python replay_e_fusion.py --shallow-run $S3RunDir --eegnet-run $EegRun --conformer-run $CfRun
      Pop-Location
    }
  }
  Start-Sleep -Seconds $CooldownSec
}

# ── W · O3s_m + 自适应窗回放 ──
$O3Run = $null
if (-not $SkipW) {
  if (-not (Test-Path $O1Run)) { Log "WARN missing O1 run $O1Run" }
  if (-not (Test-Path $O2Run)) { Log "WARN missing O2 run $O2Run" }

  Invoke-Step "W O3s_m five-fold" {
    Push-Location $PkgW
    powershell -NoProfile -File .\run_24_w_o3s_guarded.ps1 -MaxFolds 0 -NoConsole
    Pop-Location
  }
  $o3Runs = Get-ChildItem $OutW -Directory -Filter "*O3s_m" |
    Where-Object { Test-Path (Join-Path $_.FullName "summary.json") } |
    Sort-Object LastWriteTime -Descending
  if (-not $o3Runs) {
    Log "FAIL: O3s_m run not found"
    exit 1
  }
  $O3Run = $o3Runs[0].FullName

  foreach ($pair in @(
    @{ arm = "O1s_m"; dir = $O1Run },
    @{ arm = "O2s_m"; dir = $O2Run },
    @{ arm = "O3s_m"; dir = $O3Run }
  )) {
    if (-not (Test-Path $pair.dir)) { continue }
    $hasDump = Get-ChildItem $pair.dir -Recurse -Filter "prob_dump_three.csv" -ErrorAction SilentlyContinue
    if ($hasDump) {
      Log "SKIP dump $($pair.arm) (already has prob_dump)"
      continue
    }
    Invoke-Step "W dump $($pair.arm)" {
      Push-Location $PkgW
      python dump_probs_23.py --arm $($pair.arm) --run-dir $($pair.dir)
      Pop-Location
    }
  }

  Invoke-Step "W adaptive window replay" {
    Push-Location $PkgW
    python replay_w_adaptive_window.py --o1-run $O1Run --o2-run $O2Run --o3-run $O3Run
    Pop-Location
  }
}

Log "SCHEME24 FULL DONE"
Log "  S3=$S3RunDir acc=$s3Acc"
if ($TRunDir) { Log "  T=$TRunDir" }
if ($EegRun) { Log "  E eegnet=$EegRun conformer=$CfRun" }
if ($O3Run) { Log "  W O3s_m=$O3Run" }
Log "  Results: 资料/模型训练/24_.../总结/结果登记表.md"
exit 0
