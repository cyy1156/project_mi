# 按被试分批跑 A0（单 GPU 串行，便于中断续跑）
# 用法：.\run_a0_by_subject.ps1
#       .\run_a0_by_subject.ps1 -Subjects "S1,S2,S3"
param(
    [string]$Subjects = "",
    [string]$KList = "0,10,20,40,80,-1",
    [switch]$SkipTask,
    [switch]$Fast
)

$ErrorActionPreference = "Stop"
$PY = "C:\Users\yy\.conda\envs\cyy\python.exe"
$Pkg = "D:\MI\code\train_lab\src\step\5070_aug_3s_accpaper"
Set-Location $Pkg

$args = @("incremental_ft.py", "--arm", "A0", "--k-list", $KList)
if ($SkipTask) { $args += "--skip-task" }
if ($Fast) { $args += @("--max-epochs", "100", "--patience", "10") }
if ($Subjects) { $args += @("--subjects", $Subjects) }

Write-Host "python $($args -join ' ')"
& $PY @args
