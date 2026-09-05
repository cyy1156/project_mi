# Scheme 25 · 5090 full chain (G first, A0 last)
#
#   powershell -File .\run_all_25_5090.ps1
#   powershell -File .\run_all_25_5090.ps1 -From g1_eval -G1Stamp run_YYYYMMDD_HHMMSS
#   powershell -File .\run_all_25_5090.ps1 -SkipG2 -SkipG3
param(
  [ValidateSet("stage0", "g1_train", "g1_eval", "g2", "g3", "a0")]
  [string]$From = "stage0",
  [string]$G1Stamp = "",
  [switch]$SkipG2,
  [switch]$SkipG3,
  [int]$CooldownSec = 180
)

$ErrorActionPreference = "Stop"
$Root = "F:\Cyy\MI"
$PY = "C:\Users\yz5090-1\.conda\envs\cyy\python.exe"
if (-not (Test-Path $PY)) {
  $cmdPy = Get-Command python -EA SilentlyContinue
  if ($cmdPy) { $PY = $cmdPy.Source } else { throw "conda env cyy not found" }
}

$Pkg = Join-Path $Root "code\train_lab\src\step\5070_aug_3s_accpaper"
$Log = Join-Path $Pkg "25_run_all_5090.log"
$Dev = "5090"
$G1OutBase = Join-Path $Root "code\train_lab\out\5090_aug_3s_accpaper\shallow_openbmi_3s_hop100_balbatch_accpaper\openbmi_3s_hop100"

function Log([string]$msg) {
  $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $msg
  Write-Host $line
  Add-Content -Path $Log -Value $line -Encoding utf8
}

function Invoke-Py([string]$name, [string[]]$pyArgs) {
  Log "BEGIN $name"
  Log ("  $PY " + ($pyArgs -join " "))
  & $PY @pyArgs 2>&1 | Tee-Object -FilePath $Log -Append
  if ($LASTEXITCODE -ne 0) {
    Log "FAIL $name exit=$LASTEXITCODE"
    exit $LASTEXITCODE
  }
  Log "OK $name"
  if ($CooldownSec -gt 0) {
    Log "cooldown ${CooldownSec}s ..."
    Start-Sleep -Seconds $CooldownSec
  }
}

function Resolve-G1Stamp {
  if ($G1Stamp) {
    $name = if ($G1Stamp.StartsWith("run_")) { $G1Stamp } else { "run_$G1Stamp" }
    return $name
  }
  $run = Get-ChildItem $G1OutBase -Directory -Filter "run_*" -EA SilentlyContinue |
    Sort-Object Name -Descending | Select-Object -First 1
  if (-not $run) { throw "G1 run dir not found under $G1OutBase" }
  return $run.Name
}

Set-Location $Pkg
Log "=== S25 5090 start From=$From ==="

$StNoz = Join-Path $Root "code\preprocess_lab\out\stieger_3s_hop100\stieger_X_noz.npy"
if (-not (Test-Path $StNoz)) { throw "missing stieger_X_noz.npy" }
Log "stieger data ok"

$steps = @("stage0", "g1_train", "g1_eval", "g2", "g3", "a0")
$startIdx = [array]::IndexOf($steps, $From)
if ($startIdx -lt 0) { throw "invalid From=$From" }

if ($startIdx -le 0) {
  Invoke-Py "verify_imports" @("verify_imports.py", "--train-device", $Dev)
  Invoke-Py "smoke_aug_test" @("smoke_aug_test.py")
}

if ($startIdx -le 1) {
  Invoke-Py "G1_train" @(
    "baseline_shallow_aug.py", "--aug", "g1", "--train-device", $Dev
  )
}

$stamp = Resolve-G1Stamp
Log "G1 stamp = $stamp"

if ($startIdx -le 2) {
  Invoke-Py "G1_openbmi_guard" @(
    "eval_openbmi_guard.py", "--train-device", $Dev, "--run-stamp", $stamp
  )
  Invoke-Py "G1_stieger_zeroshot" @(
    "eval_stieger.py", "--arm", "G1", "--train-device", $Dev, "--run-stamp", $stamp
  )
  Invoke-Py "G1_incremental" @(
    "incremental_ft.py", "--arm", "G1", "--train-device", $Dev, "--run-stamp", $stamp, "--skip-task"
  )
}

if ((-not $SkipG2) -and ($startIdx -le 3)) {
  Invoke-Py "G2_incremental" @(
    "incremental_ft.py", "--arm", "G2", "--train-device", $Dev,
    "--run-stamp", $stamp, "--replay-ratio", "0.15", "--skip-task"
  )
}

if ((-not $SkipG3) -and ($startIdx -le 4)) {
  Invoke-Py "G3_incremental" @(
    "incremental_ft.py", "--arm", "G3", "--train-device", $Dev,
    "--run-stamp", $stamp, "--aug", "g3", "--skip-task"
  )
}

if ($startIdx -le 5) {
  Invoke-Py "A0_incremental" @(
    "incremental_ft.py", "--arm", "A0", "--train-device", $Dev, "--skip-task"
  )
}

Log "=== S25 5090 done G1=$stamp ==="
