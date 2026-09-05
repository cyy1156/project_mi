# 伪在线实验 07 · shallow 3s · 全链（本机 conda cyy）
# 前置：5060 S3 权重 run_20260821_190504 + stieger_3s_hop100 语料
$ErrorActionPreference = "Stop"
$PY = "C:\Users\yy\.conda\envs\cyy\python.exe"
if (-not (Test-Path $PY)) { throw "conda env cyy not found: $PY" }

$Repo = "D:\MI"
$Pre = Join-Path $Repo "code\preprocess_lab"
$Pkg = Join-Path $Repo "code\train_lab\src\step\stieger_pseudo_online_3s_hop100"
$Log = Join-Path $Pkg "_run_all_07.log"

function Log($msg) {
    $line = "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $msg"
    Write-Host $line
    Add-Content -Path $Log -Value $line -Encoding utf8
}

Set-Location $Pkg
Log "=== S07 full chain start ==="

# 1) Stieger 3s 预处理（须全量合并；仅 1 个 shard 的冒烟产物不算完成）
$StOut = Join-Path $Pre "out\stieger_3s_hop100\stieger_X.npy"
$NeedPre = $true
if (Test-Path $StOut) {
    $n = & $PY -c "import numpy as np; print(np.load(r'$StOut', mmap_mode='r').shape[0])"
    if ([int]$n -ge 50000) {
        $NeedPre = $false
        Log "skip preprocess (stieger_X n=$n)"
    } else {
        Log "stieger_X too small (n=$n) -> full preprocess --reset"
    }
}
if ($NeedPre) {
    Log "preprocess stieger_3s_hop100 (full) ..."
    Set-Location $Pre
    & $PY -m src.datasets.stieger.batch_3s_hop100 --reset 2>&1 | Tee-Object -FilePath (Join-Path $Pkg "_preprocess_stieger_3s.log") -Append
    if ($LASTEXITCODE -ne 0) { throw "stieger preprocess failed" }
    Set-Location $Pkg
}

# 2) A 零样本
Log "eval_zeroshot.py"
& $PY eval_zeroshot.py 2>&1 | Tee-Object -FilePath $Log -Append
if ($LASTEXITCODE -ne 0) { throw "eval_zeroshot failed" }

# 3) B 门控（零样本）
Log "eval_gated.py --mode zeroshot"
& $PY eval_gated.py --mode zeroshot 2>&1 | Tee-Object -FilePath $Log -Append
if ($LASTEXITCODE -ne 0) { throw "eval_gated zeroshot failed" }

# 4) Q0/Q1 表
Log "write_q0q1.py"
& $PY write_q0q1.py 2>&1 | Tee-Object -FilePath $Log -Append
if ($LASTEXITCODE -ne 0) { throw "write_q0q1 failed" }

# 5) C 前半 FT（最耗时）
Log "ft_half.py"
& $PY ft_half.py 2>&1 | Tee-Object -FilePath $Log -Append
if ($LASTEXITCODE -ne 0) { throw "ft_half failed" }

# 6) D FT+门控
Log "eval_gated.py --mode ft"
& $PY eval_gated.py --mode ft 2>&1 | Tee-Object -FilePath $Log -Append
if ($LASTEXITCODE -ne 0) { throw "eval_gated ft failed" }

Log "=== S07 full chain done ==="
