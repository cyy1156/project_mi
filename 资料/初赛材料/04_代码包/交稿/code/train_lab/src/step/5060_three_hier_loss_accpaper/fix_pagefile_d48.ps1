# Fix pagefile via registry (more reliable than New-CimInstance types).
# Run as Administrator, then REBOOT.
$ErrorActionPreference = 'Stop'
$log = 'D:\cyy\MI\code\train_lab\out\_ab_mem\pagefile_fix_log.txt'
New-Item -ItemType Directory -Force -Path (Split-Path $log) | Out-Null

function Write-Log([string]$msg) {
    $line = '{0} {1}' -f (Get-Date -Format o), $msg
    Add-Content -Path $log -Value $line -Encoding utf8
    Write-Host $line
}

Write-Log '==== pagefile fix v2 start ===='

$principal = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw 'Please run PowerShell as Administrator.'
}

$mmPath = 'HKLM:\SYSTEM\CurrentControlSet\Control\Session Manager\Memory Management'
$smPath = 'HKLM:\SYSTEM\CurrentControlSet\Control\Session Manager'

# 1) Disable automatic management
$cs = Get-CimInstance Win32_ComputerSystem
if ($cs.AutomaticManagedPagefile) {
    Write-Log 'Disable AutomaticManagedPagefile'
    Set-CimInstance -InputObject $cs -Property @{ AutomaticManagedPagefile = $false }
}

# 2) Remove WMI pagefile settings if any (ignore errors)
Get-CimInstance Win32_PageFileSetting -ErrorAction SilentlyContinue | ForEach-Object {
    Write-Log ('Remove PageFileSetting {0}' -f $_.Name)
    try { Remove-CimInstance -InputObject $_ -ErrorAction Stop } catch { Write-Log ('remove warn: {0}' -f $_.Exception.Message) }
}

# 3) Best-effort delete leftover pagefiles (optional)
foreach ($p in @('D:\pagefile.sys', 'C:\pagefile.sys', 'C:\swapfile.sys')) {
    if (Test-Path -LiteralPath $p) {
        try {
            Remove-Item -LiteralPath $p -Force -ErrorAction Stop
            Write-Log ('Deleted {0}' -f $p)
        } catch {
            Write-Log ('Could not delete {0}: {1}' -f $p, $_.Exception.Message)
            if ($p -eq 'D:\pagefile.sys') {
                $pending = @('\??\D:\pagefile.sys', '')
                $existing = @()
                try {
                    $existing = @((Get-ItemProperty -Path $smPath -Name PendingFileRenameOperations -ErrorAction Stop).PendingFileRenameOperations)
                } catch {}
                New-ItemProperty -Path $smPath -Name PendingFileRenameOperations -PropertyType MultiString -Value (@($existing + $pending)) -Force | Out-Null
                Write-Log 'Pending delete scheduled for D:\pagefile.sys'
            }
        }
    }
}

# 4) Write registry (this is what Windows reads at boot)
# C: 2GB safety + D: 48GB main
$pf = [string[]]@(
    'C:\pagefile.sys 2048 2048'
    'D:\pagefile.sys 49152 49152'
)
Set-ItemProperty -Path $mmPath -Name PagingFiles -Type MultiString -Value $pf
Set-ItemProperty -Path $mmPath -Name ExistingPageFiles -Type MultiString -Value ([string[]]@(
    '\??\C:\pagefile.sys'
    '\??\D:\pagefile.sys'
))

Write-Log ('PagingFiles = {0}' -f ([string]::Join(' ;; ', $pf)))
Write-Log 'ExistingPageFiles = C + D'

# 5) Also recreate WMI settings with explicit UInt32 (optional, for GUI consistency)
try {
    $initC = [UInt32]2048
    $maxC = [UInt32]2048
    $initD = [UInt32]49152
    $maxD = [UInt32]49152
    New-CimInstance -ClassName Win32_PageFileSetting -Property @{
        Name        = 'C:\pagefile.sys'
        InitialSize = $initC
        MaximumSize = $maxC
    } -ErrorAction Stop | Out-Null
    New-CimInstance -ClassName Win32_PageFileSetting -Property @{
        Name        = 'D:\pagefile.sys'
        InitialSize = $initD
        MaximumSize = $maxD
    } -ErrorAction Stop | Out-Null
    Write-Log 'WMI PageFileSetting created with UInt32 sizes'
} catch {
    Write-Log ('WMI create skipped/failed (registry already set): {0}' -f $_.Exception.Message)
}

# Verify registry
$mm = Get-ItemProperty -Path $mmPath
Write-Log ('VERIFY PagingFiles: {0}' -f ([string]::Join(' ;; ', @($mm.PagingFiles))))
Write-Log ('VERIFY ExistingPageFiles: {0}' -f ([string]::Join(' ;; ', @($mm.ExistingPageFiles))))
Write-Log ('VERIFY AutoManaged: {0}' -f (Get-CimInstance Win32_ComputerSystem).AutomaticManagedPagefile)
Write-Log '==== DONE. REBOOT NOW. ===='
Write-Host ''
Write-Host 'OK. Registry updated. Please REBOOT the computer now.' -ForegroundColor Green
Write-Host ('Log: {0}' -f $log)
