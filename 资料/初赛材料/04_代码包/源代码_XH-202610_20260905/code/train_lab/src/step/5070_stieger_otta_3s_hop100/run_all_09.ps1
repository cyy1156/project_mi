# S09 OTTA v1.2 strict chain - unified noz pipeline + real EA(cal) + predict-first AdaBN
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
Log "=== S09 OTTA v1.2 strict chain start ==="

$StX = Join-Path $Repo "code\preprocess_lab\out\stieger_3s_hop100\stieger_X.npy"
$StNoz = Join-Path $Repo "code\preprocess_lab\out\stieger_3s_hop100\stieger_X_noz.npy"
if (-not (Test-Path $StX)) { throw "missing stieger_X.npy - run stieger preprocess first" }
if (-not (Test-Path $StNoz)) { throw "missing stieger_X_noz.npy - required for EA" }
Log "data ok"

Log "archive invalid / stopped runs ..."
& $PY archive_invalid_runs.py 2>&1 | Tee-Object -FilePath $Log -Append
if ($LASTEXITCODE -ne 0) { throw "archive_invalid_runs failed" }

Log "build EA ref=src cache (A-series) ..."
& $PY -c "import _bootstrap; from ref_cov import load_ref_cov_src; load_ref_cov_src()" 2>&1 | Tee-Object -FilePath $Log -Append
if ($LASTEXITCODE -ne 0) { throw "EA ref cache failed" }

Log "eval_ab.py --arms A0,A1,A2,A3"
& $PY eval_ab.py --arms A0,A1,A2,A3 2>&1 | Tee-Object -FilePath $Log -Append
if ($LASTEXITCODE -ne 0) { throw "eval_ab A-series failed" }

Log "eval_ab.py --arms B0,B1,B2,B3,B4"
& $PY eval_ab.py --arms B0,B1,B2,B3,B4 2>&1 | Tee-Object -FilePath $Log -Append
if ($LASTEXITCODE -ne 0) { throw "eval_ab B-series failed" }

Log "preflight C1 paired refs (A0/B3 v1.2 full 24 subj) ..."
& $PY -c "import _bootstrap; from paired_results import require_paired_summaries; require_paired_summaries(('A0','B3'))" 2>&1 | Tee-Object -FilePath $Log -Append
if ($LASTEXITCODE -ne 0) { throw "C1 preflight failed - run A0/B3 full eval first" }

Log "eval_c1.py (B3 v1.2 + pseudo-label FT)"
& $PY eval_c1.py 2>&1 | Tee-Object -FilePath $Log -Append
if ($LASTEXITCODE -ne 0) { throw "eval_c1 failed" }

Log "=== S09 OTTA v1.2 strict chain done ==="
