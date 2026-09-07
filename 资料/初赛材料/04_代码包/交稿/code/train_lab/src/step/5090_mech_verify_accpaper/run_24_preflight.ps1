# 方案 24 · 5090 本机预检
#
#   powershell -File .\run_24_preflight.ps1
param()

$ErrorActionPreference = "Stop"
$Root = "F:\Cyy\MI"
$Pre = Join-Path $Root "code\preprocess_lab\out\openbmi_3s_hop100"
$Pf = Join-Path $Root "code\preprocess_lab\out\openbmi_2s_hop100_pf1000"
$Pkg3 = Join-Path $Root "code\train_lab\src\step\5090_baselines_openbmi_3s_hop100_accpaper"
$PkgW = Join-Path $Root "code\train_lab\src\step\5090_mech_verify_accpaper"

Write-Host "=== Scheme 24 preflight (5090) ==="

if (-not (Test-Path $Pf)) {
  Write-Warning "MISS pf1000 v3: $Pf"
} else { Write-Host "OK pf1000 v3" }

if (-not (Test-Path (Join-Path $Pre "openbmi_X.npy"))) {
  Write-Warning "MISS 3s hop100 data: $Pre (run preprocess openbmi_3s_hop100.yaml)"
} else { Write-Host "OK 3s hop100 data" }

Push-Location $Pkg3
python _smoke_local.py
if ($LASTEXITCODE -ne 0) { Pop-Location; exit 1 }
Pop-Location

Push-Location $PkgW
python _smoke_local.py
if ($LASTEXITCODE -ne 0) { Pop-Location; exit 1 }
python run_arm.py --arm O3s_m --dry-run
if ($LASTEXITCODE -ne 0) { Pop-Location; exit 1 }
Pop-Location

Write-Host "PREFLIGHT OK"
