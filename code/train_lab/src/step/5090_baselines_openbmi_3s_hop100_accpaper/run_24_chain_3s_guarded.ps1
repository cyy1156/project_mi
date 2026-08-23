# 方案 24 · 3s 腿链：S3 校准 → eegnet → conformer（5090 · V/E 训练段）
#
#   powershell -File .\run_24_chain_3s_guarded.ps1 -MaxFolds 0 -NoConsole
#   powershell -File .\run_24_chain_3s_guarded.ps1 -From shallow -MaxFolds 1
param(
  [ValidateSet("shallow", "eegnet", "conformer")]
  [string]$From = "shallow",
  [int]$MaxFolds = 0,
  [switch]$NoConsole,
  [int]$TimeoutSecPerArm = 86400,
  [int]$CooldownBetweenSec = 180
)

$ErrorActionPreference = "Continue"
$WorkDir = Join-Path (Split-Path $PSScriptRoot -Parent) "5090_baselines_openbmi_3s_hop100_accpaper"
$ChainLog = Join-Path $WorkDir "24_chain_3s_guarded.log"

$Arms = @(
  @{ id = "shallow"; script = "baseline_shallow.py" },
  @{ id = "eegnet"; script = "baseline_eegnet.py" },
  @{ id = "conformer"; script = "baseline_conformer.py" }
)

function Write-Chain([string]$msg) {
  $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $msg
  Write-Host $line
  Add-Content -Path $ChainLog -Value $line -Encoding utf8
}

$start = [array]::IndexOf(@($Arms.id), $From)
if ($start -lt 0) { throw "bad From=$From" }
$queue = $Arms[$start..($Arms.Count - 1)]

$foldLabel = if ($MaxFolds -le 0) { "5fold" } else { "fold0..$($MaxFolds-1)" }
Write-Chain "START scheme24-3s queue=$($queue.id -join ',') folds=$foldLabel three-only"

$fail = 0
foreach ($arm in $queue) {
  Write-Chain "RUN $($arm.id)"
  $py = Get-Command python -ErrorAction SilentlyContinue
  $pyPath = if ($py) { $py.Source } else { "python" }
  $script = Join-Path $WorkDir $arm.script
  $extra = "--three-only --max-folds $MaxFolds --num-workers 0"
  $cmd = "& `"$pyPath`" `"$script`" $extra"
  if ($NoConsole) {
    Push-Location $WorkDir
    Invoke-Expression $cmd
    $code = $LASTEXITCODE
    Pop-Location
  } else {
    Push-Location $WorkDir
    Invoke-Expression $cmd
    $code = $LASTEXITCODE
    Pop-Location
  }
  if ($code -ne 0) {
    Write-Chain "FAIL $($arm.id) exit=$code"
    $fail = $code
    break
  }
  Write-Chain "OK $($arm.id)"
  if ($arm -ne $queue[-1]) { Start-Sleep -Seconds $CooldownBetweenSec }
}

if ($fail -eq 0) { Write-Chain "ALL DONE scheme24-3s" }
exit $fail
