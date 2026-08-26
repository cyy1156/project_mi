# Scheme 28 · 5090 member economy replay (R0–R6 · zero training)
#
#   powershell -File .\run_all_28_5090.ps1
param(
  [ValidateSet("R0", "R1", "R2", "R3", "R4", "R5", "R6", "all")]
  [string]$From = "R0",
  [switch]$SkipSummary
)

$ErrorActionPreference = "Stop"
$Root = "F:\Cyy\MI"
$PY = "C:\Users\yz5090-1\.conda\envs\cyy\python.exe"
if (-not (Test-Path $PY)) {
  $cmdPy = Get-Command python -EA SilentlyContinue
  if ($cmdPy) { $PY = $cmdPy.Source } else { throw "conda env cyy not found" }
}

$Pkg = Join-Path $Root "code\train_lab\src\step\5090_ens_recipe_3s_hop100_accpaper"
$Log = Join-Path $Pkg "28_run_all_5090.log"
$RunLog = Join-Path $Pkg ("28_run_{0}.log" -f (Get-Date -Format "yyyyMMdd_HHmmss"))

$Arms = @("R0", "R1", "R2", "R3", "R4", "R5", "R6")

function Log([string]$msg) {
  $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $msg
  Write-Host $line
  foreach ($lp in @($RunLog, $Log)) {
    try { $line | Out-File -FilePath $lp -Append -Encoding utf8 -ErrorAction Stop } catch {}
  }
}

function Invoke-Py([string]$name, [string[]]$pyArgs) {
  Log "BEGIN $name"
  Log ("  $PY -u " + ($pyArgs -join " "))
  $env:PYTHONUNBUFFERED = "1"
  & $PY -u @pyArgs 2>&1 | ForEach-Object {
    $_ | Out-File -FilePath $RunLog -Append -Encoding utf8
    Write-Host $_
  }
  if ($LASTEXITCODE -ne 0) {
    Log "FAIL $name exit=$LASTEXITCODE"
    exit $LASTEXITCODE
  }
  Log "OK $name"
}

Set-Location $Pkg
Log "scheme28 start From=$From"

Invoke-Py "verify_r28_dumps" @(
  (Join-Path $Pkg "verify_r28_dumps.py")
)

if ($From -eq "all") {
  Invoke-Py "replay_r28_all" @(
    (Join-Path $Pkg "replay_r28.py"),
    "--arm", "all"
  )
  Log "scheme28 done (all)"
  exit 0
}

$start = $false
foreach ($arm in $Arms) {
  if (-not $start) {
    if ($arm -eq $From) { $start = $true } else { continue }
  }
  Invoke-Py "replay_$arm" @(
    (Join-Path $Pkg "replay_r28.py"),
    "--arm", $arm,
    "--out", (Join-Path $Pkg ("replay_{0}.json" -f $arm.ToLower()))
  )
}

if (-not $SkipSummary) {
  $need = @{}
  foreach ($arm in $Arms) { $need[$arm] = Join-Path $Pkg ("replay_{0}.json" -f $arm.ToLower()) }
  $missing = @($need.GetEnumerator() | Where-Object { -not (Test-Path $_.Value) })
  if ($missing.Count -eq 0) {
    Invoke-Py "replay_r28_summary" @(
      (Join-Path $Pkg "replay_r28.py"),
      "--summarize-only",
      "--summary-out", (Join-Path $Pkg "replay_r28_summary.json")
    )
  } else {
    Log "skip summary: missing $($missing.Count) arm json(s)"
  }
}

Log "scheme28 done"
