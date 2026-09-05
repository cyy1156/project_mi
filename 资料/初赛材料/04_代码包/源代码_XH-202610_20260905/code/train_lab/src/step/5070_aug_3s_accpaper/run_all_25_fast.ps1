# 方案25 · 加速版（主读数 three · 精简 k · 略短 FT）
# 相对 run_all_25.ps1：A0/G2/G3 约 3–4× 更快，判定线主看 k≤20 仍保留
$ErrorActionPreference = "Stop"
$PY = "C:\Users\yy\.conda\envs\cyy\python.exe"
if (-not (Test-Path $PY)) { throw "conda env cyy not found: $PY" }

$Repo = "D:\MI"
$Pkg = Join-Path $Repo "code\train_lab\src\step\5070_aug_3s_accpaper"
$Log = Join-Path $Pkg "_run_all_25_fast.log"

$KList = "0,10,20,-1"
$FtArgs = @("--skip-task", "--k-list", $KList, "--max-epochs", "100", "--patience", "10")

function Log($msg) {
    $line = "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $msg"
    Write-Host $line
    Add-Content -Path $Log -Value $line -Encoding utf8
}

Set-Location $Pkg
Log "=== S25 FAST start (k=$KList three-only max_epochs=100) ==="

$StNoz = Join-Path $Repo "code\preprocess_lab\out\stieger_3s_hop100\stieger_X_noz.npy"
if (-not (Test-Path $StNoz)) { throw "missing stieger_X_noz.npy" }

& $PY verify_imports.py 2>&1 | Tee-Object -FilePath $Log -Append
if ($LASTEXITCODE -ne 0) { throw "verify_imports failed" }

Log "A0 incremental (fast) ..."
& $PY incremental_ft.py --arm A0 @FtArgs 2>&1 | Tee-Object -FilePath $Log -Append
if ($LASTEXITCODE -ne 0) { throw "A0 failed" }

Log "G1 train ..."
& $PY baseline_shallow_aug.py --aug g1 2>&1 | Tee-Object -FilePath $Log -Append
if ($LASTEXITCODE -ne 0) { throw "G1 train failed" }

$G1Base = Join-Path $Repo "code\train_lab\out\5070_aug_3s_accpaper\shallow_openbmi_3s_hop100_balbatch_accpaper\openbmi_3s_hop100"
$G1Stamp = (Get-ChildItem $G1Base -Directory -Filter "run_*" | Sort-Object Name -Descending | Select-Object -First 1).Name
Log "G1 stamp = $G1Stamp"

& $PY eval_openbmi_guard.py --run-stamp $G1Stamp 2>&1 | Tee-Object -FilePath $Log -Append
& $PY eval_stieger.py --arm G1 --run-stamp $G1Stamp --tasks three 2>&1 | Tee-Object -FilePath $Log -Append
& $PY incremental_ft.py --arm G1 --run-stamp $G1Stamp @FtArgs 2>&1 | Tee-Object -FilePath $Log -Append
& $PY incremental_ft.py --arm G2 --run-stamp $G1Stamp @FtArgs --replay-ratio 0.15 2>&1 | Tee-Object -FilePath $Log -Append
& $PY incremental_ft.py --arm G3 --run-stamp $G1Stamp @FtArgs --aug g3 2>&1 | Tee-Object -FilePath $Log -Append

Log "=== S25 FAST done ==="
