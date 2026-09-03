# Exp34 一键流水线（5070 · conda cyy）
# 用法：在仓库根或本目录执行
#   powershell -File D:\MI\资料\模型训练\34_旁路_挑战杯官方集_59ch离线_openbmi协议_accpaper\run_exp34.ps1 -Stage preprocess
#   ... -Stage trainA_smoke | trainA | e1fA | trainB_smoke | trainB | e1fB | trackC

param(
  [ValidateSet("preprocess","trainA_smoke","trainA","e1fA","trainB_smoke","trainB","e1fB","trackC","all_smoke")]
  [string]$Stage = "all_smoke"
)

$ErrorActionPreference = "Stop"
$Py = "$env:USERPROFILE\.conda\envs\cyy\python.exe"
if (-not (Test-Path $Py)) { $Py = "python" }

$Pre = "D:\MI\code\preprocess_lab"
$A = "D:\MI\code\train_lab\src\step\5070_challenge_mi_59ch_accpaper"
$B = "D:\MI\code\train_lab\src\step\5070_challenge_mi_8ch_ft_accpaper"

function Invoke-Preprocess {
  Push-Location $Pre
  & $Py -m src.datasets.challenge_mi.batch_3s --mode 59
  & $Py -m src.datasets.challenge_mi.batch_3s --mode 8
  Pop-Location
}

function Invoke-TrainA([int]$maxFolds) {
  Push-Location $A
  & $Py baseline_shallow.py --max-folds $maxFolds
  & $Py baseline_shallow_b.py --max-folds $maxFolds
  & $Py baseline_eegnet.py --max-folds $maxFolds
  & $Py baseline_conformer.py --max-folds $maxFolds
  Pop-Location
}

function Invoke-E1fA {
  Push-Location $A
  & $Py fit_e1f_a59.py --auto-latest
  $j = Get-ChildItem "D:\MI\code\train_lab\out\5070_challenge_mi_59ch_accpaper\e1f_a59\e1f_*.json" |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1
  if ($j) { & $Py predict_e1f_submission.py --e1f-json $j.FullName }
  Pop-Location
}

function Invoke-TrainB([int]$maxFolds) {
  Push-Location $B
  & $Py baseline_shallow.py --max-folds $maxFolds
  & $Py baseline_shallow_b.py --max-folds $maxFolds
  & $Py baseline_eegnet.py --max-folds $maxFolds
  & $Py baseline_conformer.py --max-folds $maxFolds
  Pop-Location
}

function Invoke-E1fB {
  Push-Location $B
  & $Py fit_e1f_b8.py --auto-latest --arm ft
  $j = Get-ChildItem "D:\MI\code\train_lab\out\5070_challenge_mi_8ch_ft_accpaper\e1f_b8\e1f_ft_*.json" |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1
  if ($j) { & $Py predict_e1f_submission.py --e1f-json $j.FullName }
  Pop-Location
}

Write-Host "Exp34 stage=$Stage py=$Py"
switch ($Stage) {
  "preprocess" { Invoke-Preprocess }
  "trainA_smoke" { Invoke-TrainA 1 }
  "trainA" { Invoke-TrainA 0 }
  "e1fA" { Invoke-E1fA }
  "trainB_smoke" { Invoke-TrainB 1 }
  "trainB" { Invoke-TrainB 0 }
  "e1fB" { Invoke-E1fB }
  "trackC" {
    & $Py -m experiment_game.tools.run_track_c_leave_next --subject syj0828 --dry-run
  }
  "all_smoke" {
    Invoke-Preprocess
    Invoke-TrainA 1
    Invoke-TrainB 1
  }
}
Write-Host "done stage=$Stage"
