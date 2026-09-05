# 方案25 · 域增广 + 增量 FT 配套
# 跑序：阶段0单测 → A0爬坡 → G1五折 → 三端评测 → G2/G3爬坡
$ErrorActionPreference = "Stop"
$PY = "C:\Users\yy\.conda\envs\cyy\python.exe"
if (-not (Test-Path $PY)) { throw "conda env cyy not found: $PY" }

$Repo = "D:\MI"
$Pkg = Join-Path $Repo "code\train_lab\src\step\5070_aug_3s_accpaper"
$Log = Join-Path $Pkg "_run_all_25.log"

function Log($msg) {
    $line = "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $msg"
    Write-Host $line
    Add-Content -Path $Log -Value $line -Encoding utf8
}

Set-Location $Pkg
Log "=== S25 aug+incremental_ft start ==="

$StNoz = Join-Path $Repo "code\preprocess_lab\out\stieger_3s_hop100\stieger_X_noz.npy"
if (-not (Test-Path $StNoz)) { throw "missing stieger_X_noz.npy - run stieger preprocess first" }
Log "data ok"

Log "verify_imports ..."
& $PY verify_imports.py 2>&1 | Tee-Object -FilePath $Log -Append
if ($LASTEXITCODE -ne 0) { throw "verify_imports failed" }

Log "stage0 smoke_aug_test ..."
& $PY smoke_aug_test.py 2>&1 | Tee-Object -FilePath $Log -Append
if ($LASTEXITCODE -ne 0) { throw "smoke_aug_test failed" }

Log "A0 incremental curve (S3 anchor) ..."
& $PY incremental_ft.py --arm A0 2>&1 | Tee-Object -FilePath $Log -Append
if ($LASTEXITCODE -ne 0) { throw "A0 incremental_ft failed" }

Log "G1 OpenBMI 5-fold train (--aug g1) ..."
& $PY baseline_shallow_aug.py --aug g1 --train-device 5070 2>&1 | Tee-Object -FilePath $Log -Append
if ($LASTEXITCODE -ne 0) { throw "G1 baseline_shallow_aug failed" }

# 5090 训练请用 run_g1_on_5090.ps1，后续评测加 --train-device 5090

$G1Base = Join-Path $Repo "code\train_lab\out\5070_aug_3s_accpaper\shallow_openbmi_3s_hop100_balbatch_accpaper\openbmi_3s_hop100"
$G1Run = Get-ChildItem $G1Base -Directory -Filter "run_*" | Sort-Object Name -Descending | Select-Object -First 1
if (-not $G1Run) { throw "G1 run dir not found under $G1Base" }
$G1Stamp = $G1Run.Name
Log "G1 run stamp = $G1Stamp"

Log "G1 OpenBMI guard ..."
& $PY eval_openbmi_guard.py --run-stamp $G1Stamp 2>&1 | Tee-Object -FilePath $Log -Append
if ($LASTEXITCODE -ne 0) { throw "eval_openbmi_guard failed" }

Log "G1 Stieger eval_half zeroshot ..."
& $PY eval_stieger.py --arm G1 --run-stamp $G1Stamp 2>&1 | Tee-Object -FilePath $Log -Append
if ($LASTEXITCODE -ne 0) { throw "eval_stieger G1 failed" }

Log "G1 incremental curve ..."
& $PY incremental_ft.py --arm G1 --run-stamp $G1Stamp 2>&1 | Tee-Object -FilePath $Log -Append
if ($LASTEXITCODE -ne 0) { throw "G1 incremental_ft failed" }

Log "G2 incremental curve (replay) ..."
& $PY incremental_ft.py --arm G2 --run-stamp $G1Stamp --replay-ratio 0.15 2>&1 | Tee-Object -FilePath $Log -Append
if ($LASTEXITCODE -ne 0) { throw "G2 incremental_ft failed" }

Log "G3 incremental curve (FT light aug) ..."
& $PY incremental_ft.py --arm G3 --run-stamp $G1Stamp --aug g3 2>&1 | Tee-Object -FilePath $Log -Append
if ($LASTEXITCODE -ne 0) { throw "G3 incremental_ft failed" }

Log "=== S25 aug+incremental_ft done ==="
