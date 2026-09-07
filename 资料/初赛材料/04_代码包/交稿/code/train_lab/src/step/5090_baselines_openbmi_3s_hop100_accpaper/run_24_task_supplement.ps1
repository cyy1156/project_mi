# 方案 24 · E1f 四成员补 Task 头（不重新训 Three）
#
#   powershell -File .\run_24_task_supplement.ps1
#   powershell -File .\run_24_task_supplement.ps1 -MaxFolds 1          # 冒烟 1 折
#   powershell -File .\run_24_task_supplement.ps1 -MemberFilter shallow      # 只跑 shallow
param(
  [ValidateSet("all", "shallow", "t_shallow", "eegnet", "conformer")]
  [string]$MemberFilter = "all",
  [int]$MaxFolds = 0,
  [int]$NumWorkers = 2,
  [switch]$SkipDump,
  [switch]$NoConsole
)

$ErrorActionPreference = "Stop"
$Root = "F:\Cyy\MI"
$Pkg = Join-Path $Root "code\train_lab\src\step\5090_baselines_openbmi_3s_hop100_accpaper"
$OutBase = Join-Path $Root "code\train_lab\out\5090_alg_incr_3s_hop100_accpaper"
$Py = "C:\Users\yz5090-1\.conda\envs\cyy\python.exe"
$Log = Join-Path $Pkg "24_task_supplement.log"

function Log([string]$msg) {
  $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $msg
  Write-Host $line
  Add-Content -Path $Log -Value $line -Encoding utf8
}

function Invoke-Step([string]$name, [scriptblock]$block) {
  Log "BEGIN $name"
  & $block
  if ($LASTEXITCODE -ne 0) {
    Log "FAIL $name exit=$LASTEXITCODE"
    exit $LASTEXITCODE
  }
  Log "OK $name"
}

$members = @(
  @{
    id = "shallow"
    script = "baseline_shallow.py"
    threeDir = Join-Path $OutBase "shallow_openbmi_3s_hop100_balbatch_accpaper\openbmi_3s_hop100\run_20260823_095327\three"
    extraArgs = @()
  },
  @{
    id = "t_shallow"
    script = "baseline_shallow.py"
    threeDir = Join-Path $OutBase "shallow_openbmi_3s_hop100_balbatch_accpaper\openbmi_3s_hop100\run_20260823_123900\three"
    extraArgs = @("--t0-weight", "0.6")
  },
  @{
    id = "eegnet"
    script = "baseline_eegnet.py"
    threeDir = Join-Path $OutBase "eegnet_openbmi_3s_hop100_balbatch_accpaper\openbmi_3s_hop100\run_20260823_131435\three"
    extraArgs = @()
  },
  @{
    id = "conformer"
    script = "baseline_conformer.py"
    threeDir = Join-Path $OutBase "conformer_openbmi_3s_hop100_balbatch_accpaper\openbmi_3s_hop100\run_20260823_135213\three"
    extraArgs = @()
  }
)

if ($MemberFilter -ne "all") {
  $members = @($members | Where-Object { $_.id -eq $MemberFilter })
}

Log "SCHEME24 TASK SUPPLEMENT start member=$MemberFilter max_folds=$MaxFolds"

foreach ($m in $members) {
  $threeDir = (Resolve-Path $m.threeDir).Path
  $runDir = Split-Path $threeDir -Parent
  $taskSummary = Join-Path $runDir "task\summary.json"

  if (Test-Path $taskSummary) {
    $acc = (Get-Content $taskSummary -Raw | ConvertFrom-Json).test_acc_paper_mean
    Log "SKIP $($m.id) task already done test_AccPaper=$acc"
    continue
  }

  $foldArg = @()
  if ($MaxFolds -gt 0) { $foldArg = @("--max-folds", "$MaxFolds") }

  Invoke-Step "task train $($m.id)" {
    Push-Location $Pkg
    & $Py $m.script --skip-three --resume-three-dir $threeDir --num-workers $NumWorkers @foldArg @($m.extraArgs)
    Pop-Location
  }
}

if (-not $SkipDump -and $MaxFolds -le 0) {
  foreach ($m in $members) {
    $threeDir = (Resolve-Path $m.threeDir).Path
    $runDir = Split-Path $threeDir -Parent
    $taskDir = Join-Path $runDir "task"
    $dump = Join-Path $taskDir "fold0\prob_dump_task.csv"
    if (-not (Test-Path (Join-Path $taskDir "summary.json"))) { continue }
    if (Test-Path $dump) {
      Log "SKIP dump $($m.id) ($dump exists)"
      continue
    }
    Invoke-Step "dump task probs $($m.id)" {
      Push-Location $Pkg
      & $Py $m.script --dump-probs --replay-run-dir $taskDir --replay-stage task --num-workers 0
      Pop-Location
    }
  }
}

Log "SCHEME24 TASK SUPPLEMENT DONE"

$allDone = $true
foreach ($m in $members) {
  $threeDir = (Resolve-Path $m.threeDir).Path
  $runDir = Split-Path $threeDir -Parent
  if (-not (Test-Path (Join-Path $runDir "task\summary.json"))) { $allDone = $false; break }
}
if ($allDone -and $MaxFolds -le 0) {
  Invoke-Step "register task results" {
    Push-Location $Pkg
    & $Py register_task_supplement.py
    Pop-Location
  }
}

exit 0
