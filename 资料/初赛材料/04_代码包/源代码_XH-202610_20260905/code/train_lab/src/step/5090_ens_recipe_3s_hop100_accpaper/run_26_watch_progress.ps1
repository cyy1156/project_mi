# Scheme 26 progress watcher
param(
  [int]$IntervalSec = 60
)

$Root = "F:\Cyy\MI"
$Log = Join-Path $Root "code\train_lab\src\step\5090_ens_recipe_3s_hop100_accpaper\26_run_all_5090.log"
$Out = Join-Path $Root "code\train_lab\out\5090_ens_recipe_3s_hop100_accpaper"

while ($true) {
  Clear-Host
  Write-Host ("=== S26 watch {0} ===" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))
  if (Test-Path $Log) {
    Get-Content $Log -Tail 25 -Encoding utf8
  } else {
    Write-Host "log not found: $Log"
  }
  Write-Host "`n--- recent runs ---"
  if (Test-Path $Out) {
    Get-ChildItem $Out -Recurse -Filter "summary.json" -EA SilentlyContinue |
      Sort-Object LastWriteTime -Descending |
      Select-Object -First 8 |
      ForEach-Object { Write-Host $_.FullName }
  }
  Start-Sleep -Seconds $IntervalSec
}
