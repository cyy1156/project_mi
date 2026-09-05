# Scheme 24 W finish: dump probs + adaptive replay (after O3s_m done)
#
#   powershell -File .\run_24_w_finish.ps1
#   powershell -File .\run_24_w_finish.ps1 -O3Run "F:\...\20260823_150623_O3s_m"
param(
  [string]$O1Run = "F:\Cyy\MI\code\train_lab\out\5090_mech_verify_accpaper\20260822_172743_O1s_m",
  [string]$O2Run = "F:\Cyy\MI\code\train_lab\out\5090_mech_verify_accpaper\20260822_152136_O2s_m",
  [string]$O3Run = "",
  [switch]$WaitForSummary,
  [int]$PollSec = 120,
  [int]$MaxWaitHours = 12
)

$ErrorActionPreference = "Stop"
$Root = "F:\Cyy\MI"
$PkgW = Join-Path $Root "code\train_lab\src\step\5090_mech_verify_accpaper"
$OutW = Join-Path $Root "code\train_lab\out\5090_mech_verify_accpaper"
$Log = Join-Path $PkgW "24_w_finish.log"

function Log([string]$msg) {
  $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $msg
  Write-Host $line
  Add-Content -Path $Log -Value $line -Encoding utf8
}

function Resolve-CyyPython {
  # powershell -File does NOT inherit conda activate
  $candidates = @(
    (Join-Path $env:USERPROFILE ".conda\envs\cyy\python.exe"),
    "C:\Users\yz5090-1\.conda\envs\cyy\python.exe",
    "D:\anaconda3\envs\cyy\python.exe"
  )
  foreach ($p in $candidates) {
    if ($p -and (Test-Path $p)) { return (Resolve-Path $p).Path }
  }
  if ($env:CONDA_PREFIX -and ($env:CONDA_PREFIX -match '\\envs\\cyy$')) {
    $p = Join-Path $env:CONDA_PREFIX "python.exe"
    if (Test-Path $p) { return (Resolve-Path $p).Path }
  }
  $cmdPy = Get-Command python -EA SilentlyContinue
  if ($cmdPy -and $cmdPy.Source -match '\\envs\\cyy\\') { return $cmdPy.Source }
  throw "cyy python not found. Run: conda activate cyy  OR install env at ~/.conda/envs/cyy"
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

$Py = Resolve-CyyPython
Log "python=$Py"

if (-not $O3Run) {
  $latest = Get-ChildItem $OutW -Directory -Filter "*O3s_m" |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1
  if (-not $latest) {
    Log "ERROR: no O3s_m run found"
    exit 1
  }
  $O3Run = $latest.FullName
}
$O3Run = (Resolve-Path $O3Run).Path
$O3Summary = Join-Path $O3Run "summary.json"

if ($WaitForSummary) {
  Log "WAIT O3s_m summary=$O3Summary"
  $deadline = (Get-Date).AddHours($MaxWaitHours)
  while ((Get-Date) -lt $deadline) {
    if (Test-Path $O3Summary) { break }
    $done = 0
    foreach ($i in 0..4) {
      if (Test-Path (Join-Path $O3Run "fold$i\metrics.json")) { $done++ }
    }
    Log "O3s_m progress $done/5 folds"
    Start-Sleep -Seconds $PollSec
  }
}

if (-not (Test-Path $O3Summary)) {
  Log "ERROR: O3s_m summary missing: $O3Summary"
  exit 1
}

foreach ($pair in @(
  @{ arm = "O1s_m"; dir = $O1Run },
  @{ arm = "O2s_m"; dir = $O2Run },
  @{ arm = "O3s_m"; dir = $O3Run }
)) {
  if (-not (Test-Path $pair.dir)) {
    Log "ERROR: missing $($pair.arm) run $($pair.dir)"
    exit 1
  }
  $hasDump = Get-ChildItem $pair.dir -Recurse -Filter "prob_dump_three.csv" -ErrorAction SilentlyContinue
  if ($hasDump) {
    Log "SKIP dump $($pair.arm) (prob_dump exists)"
    continue
  }
  Invoke-Step "dump $($pair.arm)" {
    Push-Location $PkgW
    & $Py dump_probs_23.py --arm $($pair.arm) --run-dir $($pair.dir)
    Pop-Location
  }
}

$replayOut = Join-Path $O3Run "replay_w_adaptive.json"
if (-not (Test-Path $replayOut)) {
  Invoke-Step "W adaptive replay" {
    Push-Location $PkgW
    & $Py replay_w_adaptive_window.py --o1-run $O1Run --o2-run $O2Run --o3-run $O3Run
    Pop-Location
  }
} else {
  Log "SKIP replay ($replayOut exists)"
}

$j = Get-Content $replayOut -Raw -Encoding UTF8 | ConvertFrom-Json
$o3 = Get-Content $O3Summary -Raw -Encoding UTF8 | ConvertFrom-Json
Log ("W DONE O3s_m={0:N4}+/-{1:N4} adaptive_test={2:N4} tau_conf={3:N2} delta_vs_o3={4:N2}pp" -f `
  $o3.test_acc_paper_mean, $o3.test_acc_paper_std, $j.test_acc_adaptive, $j.tau_conf_val, $j.delta_test_pp_vs_o3)
exit 0
