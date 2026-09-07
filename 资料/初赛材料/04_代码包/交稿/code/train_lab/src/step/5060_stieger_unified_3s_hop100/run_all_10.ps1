# S10 full chain: Part II 2s + Part III OTTA (Part I mount 07)
$ErrorActionPreference = "Stop"
$PY = "C:\Users\yy\.conda\envs\cyy\python.exe"
$Pkg = "D:\MI\code\train_lab\src\step\5060_stieger_unified_3s_hop100"
Set-Location $Pkg

& "$Pkg\run_part2_2s.ps1"
if ($LASTEXITCODE -ne 0) { throw "part2 failed" }

$Log = Join-Path $Pkg "_run_part3_otta.log"
function Log($m) { Write-Host $m; Add-Content $Log $m }

Log "eval_otta A0-A3 B0-B4"
& $PY eval_otta.py --arms A0,A1,A2,A3,B0,B1,B2,B3,B4 2>&1 | Tee-Object $Log -Append
& $PY eval_c1.py 2>&1 | Tee-Object $Log -Append
Log "done"
