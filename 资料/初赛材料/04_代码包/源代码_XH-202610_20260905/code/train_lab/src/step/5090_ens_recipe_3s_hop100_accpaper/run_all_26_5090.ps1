# Scheme 26 · 5090 full chain (E1 → R1/R2/R3 → E2)
#
#   powershell -File .\run_all_26_5090.ps1
#   powershell -File .\run_all_26_5090.ps1 -From e1 -SkipE1
param(
  [ValidateSet("stage0", "e1", "r1", "r2", "r3", "e2b", "e2a", "final")]
  [string]$From = "stage0",
  [switch]$SkipE1,
  [switch]$SkipR2,
  [switch]$SkipR3,
  [switch]$SkipE2b,
  [switch]$SkipE2,
  [switch]$FourMemberE1,
  [int]$CooldownGpuSec = 180,
  [int]$CooldownCpuSec = 5
)

$ErrorActionPreference = "Stop"
$Root = "F:\Cyy\MI"
$PY = "C:\Users\yz5090-1\.conda\envs\cyy\python.exe"
if (-not (Test-Path $PY)) {
  $cmdPy = Get-Command python -EA SilentlyContinue
  if ($cmdPy) { $PY = $cmdPy.Source } else { throw "conda env cyy not found" }
}

$Pkg = Join-Path $Root "code\train_lab\src\step\5090_ens_recipe_3s_hop100_accpaper"
$Log = Join-Path $Pkg "26_run_all_5090.log"
$RunLog = Join-Path $Pkg ("26_run_{0}.log" -f (Get-Date -Format "yyyyMMdd_HHmmss"))
$M = @{
  shallow   = Join-Path $Root "code\train_lab\out\5090_alg_incr_3s_hop100_accpaper\shallow_openbmi_3s_hop100_balbatch_accpaper\openbmi_3s_hop100\run_20260823_095327\three"
  t_shallow = Join-Path $Root "code\train_lab\out\5090_alg_incr_3s_hop100_accpaper\shallow_openbmi_3s_hop100_balbatch_accpaper\openbmi_3s_hop100\run_20260823_123900\three"
  eegnet    = Join-Path $Root "code\train_lab\out\5090_alg_incr_3s_hop100_accpaper\eegnet_openbmi_3s_hop100_balbatch_accpaper\openbmi_3s_hop100\run_20260823_131435\three"
  conformer = Join-Path $Root "code\train_lab\out\5090_alg_incr_3s_hop100_accpaper\conformer_openbmi_3s_hop100_balbatch_accpaper\openbmi_3s_hop100\run_20260823_135213\three"
}

function Log([string]$msg) {
  $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $msg
  Write-Host $line
  foreach ($lp in @($RunLog, $Log)) {
    try { $line | Out-File -FilePath $lp -Append -Encoding utf8 -ErrorAction Stop } catch {}
  }
}

function Cooldown([int]$sec, [string]$tag) {
  if ($sec -gt 0) {
    Log "cooldown ${sec}s ($tag) ..."
    Start-Sleep -Seconds $sec
  }
}

function Wait-ForGpuFree([string]$reason) {
  while ($true) {
    $blockers = @(Get-CimInstance Win32_Process -EA SilentlyContinue | Where-Object {
        $_.CommandLine -and
        $_.CommandLine -match 'incremental_ft\.py' -and
        $_.CommandLine -notmatch '--help'
      })
    if ($blockers.Count -eq 0) {
      Log "GPU free ($reason)"
      return
    }
    $b = $blockers[0]
    Log "WAIT GPU ($reason): blocker pid=$($b.ProcessId) cmd=$($b.CommandLine.Substring(0, [Math]::Min(80, $b.CommandLine.Length)))"
    Start-Sleep -Seconds 120
  }
}

function Invoke-Py([string]$name, [string[]]$pyArgs, [switch]$NeedsGpu) {
  if ($NeedsGpu) { Wait-ForGpuFree $name }
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
  if ($NeedsGpu) { Cooldown $CooldownGpuSec $name } else { Cooldown $CooldownCpuSec $name }
}

Set-Location $Pkg
Log "=== S26 5090 start From=$From SkipE1=$SkipE1 runlog=$RunLog ==="

$steps = @("stage0", "e1", "r1", "r2", "r3", "e2b", "e2a", "final")
$startIdx = [array]::IndexOf($steps, $From)
if ($startIdx -lt 0) { throw "invalid From=$From" }

if ($startIdx -le 0) {
  Invoke-Py "verify_imports" @("verify_imports.py")
  Invoke-Py "smoke_s26" @("smoke_s26_test.py")
}

# E1a–E1d 三成员回放（CPU）
if (($startIdx -le 1) -and (-not $SkipE1)) {
  foreach ($arm in @("E1a", "E1b", "E1c", "E1d")) {
    Invoke-Py "replay_$arm" @("replay_e1.py", "--arm", $arm)
  }
} elseif ($SkipE1 -and ($startIdx -le 1)) {
  Log "SKIP E1a-E1d (already done; replay_e1a-d.json present)"
}

if ($startIdx -le 1) {
  if (-not (Test-Path $M.t_shallow)) {
    Log "WARN: T-shallow anchor missing; skip dump"
  } else {
    $dumped = Get-ChildItem $M.t_shallow -Recurse -Filter "prob_dump_three.csv" -EA SilentlyContinue
    if (-not $dumped) {
      Invoke-Py "dump_t_shallow" @("dump_member_probs.py", "--run-dir", $M.t_shallow) -NeedsGpu
    } else {
      Log "T-shallow prob dump already exists"
    }
  }
}

if ($startIdx -le 1) {
  foreach ($arm in @("E1e", "E1f")) {
    $td = Get-ChildItem $M.t_shallow -Recurse -Filter "prob_dump_three.csv" -EA SilentlyContinue
    if (-not $td) {
      Log "SKIP replay_$arm (T-shallow dump missing)"
      continue
    }
    Invoke-Py "replay_$arm" @("replay_e1.py", "--arm", $arm, "--four-member")
  }
}

if ($startIdx -le 2) {
  Invoke-Py "R1_train" @("baseline_shallow_r1.py", "--skip-task", "--three-only") -NeedsGpu
}

if ((-not $SkipR2) -and ($startIdx -le 3)) {
  Invoke-Py "R2_train" @("baseline_shallow_r2.py", "--skip-task", "--three-only") -NeedsGpu
}

if ((-not $SkipR3) -and ($startIdx -le 4)) {
  Invoke-Py "R3_train" @("baseline_conformer_r3.py", "--skip-task", "--three-only") -NeedsGpu
}

if ((-not $SkipE2b) -and (-not $SkipE2) -and ($startIdx -le 5)) {
  Invoke-Py "E2b_riemann" @("baseline_riemann_e2b.py")
}

if ((-not $SkipE2) -and ($startIdx -le 6)) {
  Invoke-Py "E2a_kan_fold0" @("baseline_kan_e2a.py", "--max-folds", "1") -NeedsGpu
  $e2aRun = Get-ChildItem (Join-Path $Root "code\train_lab\out\5090_ens_recipe_3s_hop100_accpaper\kan_bandpower_e2a\openbmi_3s_hop100") -Directory -Filter "run_*" -EA SilentlyContinue |
    Sort-Object Name -Descending | Select-Object -First 1
  if ($e2aRun) {
    $e2aThree = Join-Path $e2aRun.FullName "three"
    Invoke-Py "E2a_gate" @(
      "e2_fusion_gate.py", "--arm", "E2a", "--candidate-run", $e2aThree,
      "--four-member-base", "--out", (Join-Path $Pkg "e2_gate_e2a_fold0.json")
    )
    $gateJson = Join-Path $Pkg "e2_gate_e2a_fold0.json"
    $gateVerdict = "negative"
    if (Test-Path $gateJson) {
      $gate = Get-Content $gateJson -Raw -Encoding utf8 | ConvertFrom-Json
      $gateVerdict = $gate.verdict
      Log "E2a gate verdict=$gateVerdict delta_pp=$($gate.delta_pp)"
    }
    if ($gateVerdict -eq "adopt" -or $gateVerdict -eq "report") {
      Invoke-Py "E2a_kan_full" @("baseline_kan_e2a.py") -NeedsGpu
    } else {
      Log "SKIP E2a_kan_full (gate negative)"
    }
  }
}

if ($startIdx -le 7) {
  $td = Get-ChildItem $M.t_shallow -Recurse -Filter "prob_dump_three.csv" -EA SilentlyContinue
  if ($td) {
    Invoke-Py "replay_E1d_final" @("replay_e1.py", "--arm", "E1d", "--four-member")
  } else {
    Log "SKIP replay_E1d_final (T-shallow dump missing)"
  }
}

Log "=== S26 5090 done ==="
