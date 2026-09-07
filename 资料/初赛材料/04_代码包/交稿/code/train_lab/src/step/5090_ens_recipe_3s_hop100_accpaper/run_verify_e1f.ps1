# E1f 复现链一键校验（5070/5090 · 不重训）
#   powershell -File .\run_verify_e1f.ps1           # 快检 dump + replay_e1f.json
#   powershell -File .\run_verify_e1f.ps1 -Replay   # 重跑 E1f 融合 (~90min)

param([switch]$Replay)

$ErrorActionPreference = "Stop"
$Root = if ($env:MI_ROOT) { $env:MI_ROOT } else { "F:\Cyy\MI" }
$PY = "C:\Users\yz5090-1\.conda\envs\cyy\python.exe"
if (-not (Test-Path $PY)) {
  $cmdPy = Get-Command python -EA SilentlyContinue
  if ($cmdPy) { $PY = $cmdPy.Source } else { throw "python not found" }
}

$Pkg = Join-Path $Root "code\train_lab\src\step\5090_ens_recipe_3s_hop100_accpaper"
Set-Location $Pkg

$args = @("-u", (Join-Path $Pkg "verify_e1f_reproduce.py"))
if ($Replay) { $args += "--replay" }

Write-Host "[verify_e1f] $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') Replay=$Replay"
& $PY @args
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host "[verify_e1f] OK"
