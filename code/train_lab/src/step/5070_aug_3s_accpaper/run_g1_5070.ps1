# 方案25 · G1 单独重跑（5070 · bugfix 后）
# 先释放内存/清缓存，再跑回归测试 + 五折训练
$ErrorActionPreference = "Stop"
$PY = "C:\Users\yy\.conda\envs\cyy\python.exe"
if (-not (Test-Path $PY)) { throw "conda env cyy not found: $PY" }

$Repo = "D:\MI"
$Pkg = Join-Path $Repo "code\train_lab\src\step\5070_aug_3s_accpaper"
$Log = Join-Path $Pkg "_run_g1_5070.log"

function Log($msg) {
    $line = "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $msg"
    Write-Host $line
    Add-Content -Path $Log -Value $line -Encoding utf8
}

function Free-MemoryBestEffort {
    Log "memory: stop stray python ..."
    Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -match 'baseline_shallow_aug|task_runner|incremental_ft|run_all_25' } |
        ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }

    Log "memory: clear temp numpy memmap ..."
    Get-ChildItem $env:TEMP -Filter "*.npy" -ErrorAction SilentlyContinue |
        Remove-Item -Force -ErrorAction SilentlyContinue

    Log "memory: drop stale fold packs (float16 / no meta) ..."
    $OutRoot = Join-Path $Repo "code\train_lab\out\5070_aug_3s_accpaper"
    if (Test-Path $OutRoot) {
        Get-ChildItem $OutRoot -Recurse -Filter "_cache_*_X.npy*" -ErrorAction SilentlyContinue |
            Remove-Item -Force -ErrorAction SilentlyContinue
    }

    & $PY -c "import gc; gc.collect()" 2>$null | Out-Null
    $os = Get-CimInstance Win32_OperatingSystem
    $free = [math]::Round($os.FreePhysicalMemory / 1MB, 2)
    $total = [math]::Round($os.TotalVisibleMemorySize / 1MB, 2)
    Log "memory: free ${free}GB / ${total}GB"
}

Set-Location $Pkg
Log "=== S25 G1 rerun (5070) start ==="
Free-MemoryBestEffort

Log "regression test_s25_fixes ..."
& $PY test_s25_fixes.py 2>&1 | Tee-Object -FilePath $Log -Append
if ($LASTEXITCODE -ne 0) { throw "test_s25_fixes failed" }

Log "smoke_aug_test ..."
& $PY smoke_aug_test.py 2>&1 | Tee-Object -FilePath $Log -Append
if ($LASTEXITCODE -ne 0) { throw "smoke_aug_test failed" }

Log "G1 OpenBMI 5-fold train (--aug g1, memory-safe workers=0) ..."
& $PY baseline_shallow_aug.py --aug g1 --train-device 5070 --num-workers 0 2>&1 |
    Tee-Object -FilePath $Log -Append
if ($LASTEXITCODE -ne 0) { throw "G1 baseline_shallow_aug failed" }

$G1Base = Join-Path $Repo "code\train_lab\out\5070_aug_3s_accpaper\shallow_openbmi_3s_hop100_balbatch_accpaper\openbmi_3s_hop100"
$G1Run = Get-ChildItem $G1Base -Directory -Filter "run_*" -ErrorAction SilentlyContinue |
    Sort-Object Name -Descending | Select-Object -First 1
if (-not $G1Run) { throw "G1 run dir not found under $G1Base" }
Log "G1 done stamp = $($G1Run.Name)"
Log "=== S25 G1 rerun done ==="
