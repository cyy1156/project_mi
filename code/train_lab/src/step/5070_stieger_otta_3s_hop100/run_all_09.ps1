# S09 OTTA full chain (EA + AdaBN) - Stieger 3s replay
# Prereq: stieger_3s_hop100 + 5060 S3 + S07 FT ckpt 20260822_153300
$ErrorActionPreference = "Stop"
$PY = "C:\Users\yy\.conda\envs\cyy\python.exe"
if (-not (Test-Path $PY)) { throw "conda env cyy not found: $PY" }

$Repo = "D:\MI"
$Pkg = Join-Path $Repo "code\train_lab\src\step\5070_stieger_otta_3s_hop100"
$Log = Join-Path $Pkg "_run_all_09.log"

function Log($msg) {
    $line = "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $msg"
    Write-Host $line
    Add-Content -Path $Log -Value $line -Encoding utf8
}

Set-Location $Pkg
Log "=== S09 OTTA chain start ==="

$StX = Join-Path $Repo "code\preprocess_lab\out\stieger_3s_hop100\stieger_X.npy"
$StNoz = Join-Path $Repo "code\preprocess_lab\out\stieger_3s_hop100\stieger_X_noz.npy"
if (-not (Test-Path $StX)) { throw "missing stieger_X.npy - run stieger preprocess first" }
if (-not (Test-Path $StNoz)) { throw "missing stieger_X_noz.npy - required for EA" }
Log "data ok"

Log "build EA ref=src cache ..."
& $PY -c "import _bootstrap; from ref_cov import load_ref_cov_src; load_ref_cov_src()" 2>&1 | Tee-Object -FilePath $Log -Append
if ($LASTEXITCODE -ne 0) { throw "EA ref cache failed" }

Log "eval_ab.py --arms A3,B3,B4"
& $PY eval_ab.py --arms A3,B3,B4 2>&1 | Tee-Object -FilePath $Log -Append
if ($LASTEXITCODE -ne 0) { throw "eval_ab failed" }

Log "eval_c1.py"
& $PY eval_c1.py 2>&1 | Tee-Object -FilePath $Log -Append
if ($LASTEXITCODE -ne 0) { throw "eval_c1 failed" }

Log "=== S09 OTTA chain done ==="
