# LAN operator launcher (PowerShell fallback if .bat encoding fails)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$Py = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Py)) {
    $Py = Join-Path $env:USERPROFILE ".conda\envs\cyy\python.exe"
}
if (-not (Test-Path $Py)) {
    Write-Host "ERROR: Python not found." -ForegroundColor Red
    Write-Host "  $Root\.venv\Scripts\python.exe"
    Write-Host "  $env:USERPROFILE\.conda\envs\cyy\python.exe"
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""
Write-Host "=== Operator console (LAN) ==="
Write-Host "Repo root: $Root"
Write-Host "Python:    $Py"
Write-Host "Monitor:   http://<LAN-IP>:8080/operator.html"
Write-Host "Subject:   http://127.0.0.1:8080/"
Write-Host ""

& $Py -m experiment_game.tools.open_operator --host 0.0.0.0 @args
