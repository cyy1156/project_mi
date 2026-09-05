# 每 15 分钟踢一脚 supervise_u_combo.ps1；链完成后退出
$ErrorActionPreference = "Continue"
$WorkDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$LoopLog = Join-Path $WorkDir "u_combo_supervise_loop.log"
$State = Join-Path $WorkDir "u_combo_chain_guarded_state.json"
$Sup = Join-Path $WorkDir "supervise_u_combo.ps1"

function Write-Loop([string]$msg) {
  $line = "[{0}] {1}" -f (Get-Date -Format o), $msg
  Add-Content -Path $LoopLog -Value $line -Encoding utf8
}

Write-Loop "U-combo supervise loop starting (15m) pid=$PID"
while ($true) {
  Write-Loop "tick"
  & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $Sup
  if (Test-Path $State) {
    try {
      $j = Get-Content $State -Raw -Encoding utf8 | ConvertFrom-Json
      if ($j.finished) {
        Write-Loop ("DONE finished={0}" -f $j.finished)
        exit 0
      }
    } catch {}
  }
  Start-Sleep -Seconds 900
}
