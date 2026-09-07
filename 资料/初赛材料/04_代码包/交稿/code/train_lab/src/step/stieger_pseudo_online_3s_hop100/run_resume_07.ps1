# 伪在线实验 07 · 从 S07-02 FT 续跑（01/03/04 已完成时）
$ErrorActionPreference = "Stop"
$PY = "C:\Users\yy\.conda\envs\cyy\python.exe"
if (-not (Test-Path $PY)) { throw "conda env cyy not found: $PY" }

$Pkg = "D:\MI\code\train_lab\src\step\stieger_pseudo_online_3s_hop100"
$Log = Join-Path $Pkg "_run_resume_07.log"

function Log($msg) {
    $line = "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $msg"
    Write-Host $line
    Add-Content -Path $Log -Value $line -Encoding utf8
}

Set-Location $Pkg
Log "=== S07 resume from ft_half ==="

Log "ft_half.py"
& $PY ft_half.py 2>&1 | Tee-Object -FilePath $Log -Append
if ($LASTEXITCODE -ne 0) { throw "ft_half failed" }

Log "eval_gated.py --mode ft"
& $PY eval_gated.py --mode ft 2>&1 | Tee-Object -FilePath $Log -Append
if ($LASTEXITCODE -ne 0) { throw "eval_gated ft failed" }

Log "=== S07 resume done ==="
