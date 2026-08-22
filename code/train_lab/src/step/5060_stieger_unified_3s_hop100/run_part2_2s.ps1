# S10 unified · 5060 · 3s repro + 2s window + OTTA v1.2
$ErrorActionPreference = "Stop"
$PY = "C:\Users\yy\.conda\envs\cyy\python.exe"
$Repo = "D:\MI"
$Pkg = Join-Path $Repo "code\train_lab\src\step\5060_stieger_unified_3s_hop100"
$Log = Join-Path $Pkg "_run_part2_2s.log"

function Log($msg) {
    $line = "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $msg"
    Write-Host $line
    Add-Content -Path $Log -Value $line -Encoding utf8
}

Set-Location $Pkg
Log "=== S10 Part II 2s window start ==="

$St2 = Join-Path $Repo "code\preprocess_lab\out\stieger_2s_hop100\stieger_X.npy"
if (-not (Test-Path $St2)) {
    Log "preprocess stieger_2s_hop100 ..."
    Set-Location (Join-Path $Repo "code\preprocess_lab")
    & $PY -m src.datasets.stieger.batch_2s_hop100 2>&1 | Tee-Object -FilePath $Log -Append
    Set-Location $Pkg
    if (-not (Test-Path $St2)) { throw "missing stieger_2s_hop100" }
}

Log "eval_zeroshot.py --tw 2s  (S10-01b)"
& $PY eval_zeroshot.py --tw 2s 2>&1 | Tee-Object -FilePath $Log -Append
if ($LASTEXITCODE -ne 0) { throw "S10-01b failed" }

Log "ft_half.py --tw 2s  (S10-02b)"
& $PY ft_half.py --tw 2s 2>&1 | Tee-Object -FilePath $Log -Append
if ($LASTEXITCODE -ne 0) { throw "S10-02b failed" }

Log "=== S10 Part II 2s done ==="
