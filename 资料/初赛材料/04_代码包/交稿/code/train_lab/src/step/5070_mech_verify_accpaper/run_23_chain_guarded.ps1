# Scheme 23 · 5070 mechanism verification chain (mem guard)
param(
  [string]$FromArm = "O2s_m",
  [int]$MaxFolds = 1,
  [switch]$Tier2,
  [switch]$NoConsole
)
$ErrorActionPreference = "Continue"
$WorkDir = "F:\Cyy\MI\code\train_lab\src\step\5070_mech_verify_accpaper"
$Log = Join-Path $WorkDir "chain_23_guarded.log"
Set-Location $WorkDir
$args = @("chain_23_all.py", "--max-folds", "$MaxFolds")
if ($FromArm) { $args += @("--from", $FromArm) }
if ($Tier2) { $args += "--tier2" }
$line = "[{0}] START chain_23 from={1} max_folds={2} tier2={3}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $FromArm, $MaxFolds, $Tier2.IsPresent
Add-Content -Path $Log -Value $line -Encoding utf8
Write-Host $line
if ($NoConsole) {
  Start-Process -FilePath "python" -ArgumentList $args -WorkingDirectory $WorkDir -WindowStyle Hidden
} else {
  python @args
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
Write-Host "chain_23 done"
