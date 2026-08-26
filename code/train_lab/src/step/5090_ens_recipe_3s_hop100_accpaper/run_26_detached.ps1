#  detached launcher for S26 overnight chain
#  e.g. .\run_26_detached.ps1 -From e1 -SkipE1
param(
  [ValidateSet("stage0", "e1", "r1", "r2", "r3", "e2b", "e2a", "final")]
  [string]$From = "stage0",
  [switch]$SkipE1,
  [switch]$SkipR2,
  [switch]$SkipR3,
  [switch]$SkipE2b,
  [switch]$SkipE2
)
$Pkg = "F:\Cyy\MI\code\train_lab\src\step\5090_ens_recipe_3s_hop100_accpaper"
$script = Join-Path $Pkg "run_all_26_5090.ps1"
$args = @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $script, "-From", $From)
if ($SkipE1) { $args += "-SkipE1" }
if ($SkipR2) { $args += "-SkipR2" }
if ($SkipR3) { $args += "-SkipR3" }
if ($SkipE2b) { $args += "-SkipE2b" }
if ($SkipE2) { $args += "-SkipE2" }

Start-Process -FilePath "powershell.exe" `
  -ArgumentList $args `
  -WorkingDirectory $Pkg `
  -WindowStyle Minimized

Write-Host "S26 chain launched detached. Monitor: $Pkg\26_run_*.log"
