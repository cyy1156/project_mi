# 方案25 · G1 在 5090 上训练（HP 对齐 5070 S3，独立 out 前缀）
# 评测仍在 5070 跑：eval / incremental_ft 加 --train-device 5090
$ErrorActionPreference = "Stop"
$PY = "C:\Users\yy\.conda\envs\cyy\python.exe"
$Repo = "D:\MI"
$Pkg = Join-Path $Repo "code\train_lab\src\step\5070_aug_3s_accpaper"
$Log = Join-Path $Pkg "_run_g1_5090.log"

Set-Location $Pkg
"[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] G1 train on 5090 (comparable HP)" | Tee-Object -FilePath $Log -Append
& $PY baseline_shallow_aug.py --aug g1 --train-device 5090 2>&1 | Tee-Object -FilePath $Log -Append
if ($LASTEXITCODE -ne 0) { throw "G1 5090 train failed" }

$G1Base = Join-Path $Repo "code\train_lab\out\5090_aug_3s_accpaper\shallow_openbmi_3s_hop100_balbatch_accpaper\openbmi_3s_hop100"
$G1Stamp = (Get-ChildItem $G1Base -Directory -Filter "run_*" | Sort-Object Name -Descending | Select-Object -First 1).Name
"[done] G1 stamp = $G1Stamp" | Tee-Object -FilePath $Log -Append
Write-Host "Next on 5070:"
Write-Host "  python eval_openbmi_guard.py --train-device 5090 --run-stamp $G1Stamp"
Write-Host "  python eval_stieger.py --arm G1 --train-device 5090 --run-stamp $G1Stamp"
Write-Host "  python incremental_ft.py --arm G1 --train-device 5090 --run-stamp $G1Stamp ..."
