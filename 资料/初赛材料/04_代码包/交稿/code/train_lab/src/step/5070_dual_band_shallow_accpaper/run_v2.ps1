# 方案19 · 预处理 μ/β + 训练 V2（5070）
param(
  [int]$Limit = 0,
  [switch]$SkipPreprocess,
  [switch]$SmokeOnly,
  [int]$MaxFolds = 0
)
$ErrorActionPreference = "Stop"
$Pre = "D:\MI\code\preprocess_lab"
$Pkg = "D:\MI\code\train_lab\src\step\5070_dual_band_shallow_accpaper"

conda activate cyy

if (-not $SkipPreprocess) {
  Set-Location $Pre
  $lim = @()
  if ($Limit -gt 0) { $lim = @("--limit", "$Limit") }
  Write-Host ">>> preprocess mu813"
  python -m src.datasets.openbmi.batch_2s_hop100 --band mu813 @lim --reset
  Write-Host ">>> preprocess beta1330"
  python -m src.datasets.openbmi.batch_2s_hop100 --band beta1330 @lim --reset
}

Set-Location $Pkg
python _smoke_local.py
if ($SmokeOnly) { exit 0 }

$mf = @()
if ($MaxFolds -gt 0) { $mf = @("--max-folds", "$MaxFolds") }
Write-Host ">>> train V2"
python train_kfold.py --arm V2 @mf
