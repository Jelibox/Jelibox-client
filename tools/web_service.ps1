# Starts the Jelibox web server by itself when you sign in to Windows. No administrator rights are needed.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File tools\web_service.ps1 -Action install|remove|start|stop|status
#
#   install   puts a shortcut "Jelibox Web" in your Startup folder (Settings > Apps > Startup lists it) and starts the
#             server now. Run again after an update: the server is restarted with the new code.
#   remove    stops the server and deletes that shortcut.
#   start     starts the server if it is not running.   stop   stops it.   status   says whether it answers.
#
# The server runs hidden (pythonw.exe of Jelibox's own environment, no console window); its output is in
# temp\server.log. It starts when you SIGN IN, not at boot: after a restart of the PC somebody has to sign in once
# (or turn on automatic sign-in). Other devices on the network also need Windows Firewall to allow the port, see the
# note printed by "install". Nothing here touches the firewall, the registry or any other part of Windows.
#
# Optional environment variables:
#   JELIBOX_STARTUP_DIR  put the shortcut here instead of your real Startup folder (tests)
param(
    [ValidateSet('install', 'remove', 'start', 'stop', 'status')][string]$Action = 'status',
    [string]$Root = (Split-Path -Parent $PSScriptRoot)
)

$ErrorActionPreference = 'Stop'
$Root    = (Resolve-Path -LiteralPath $Root).Path
$python  = Join-Path $Root 'venv\Scripts\pythonw.exe'
$server  = Join-Path $Root 'utils\server.py'
$startup = if ($env:JELIBOX_STARTUP_DIR) { $env:JELIBOX_STARTUP_DIR } else { [Environment]::GetFolderPath('Startup') }
$link    = Join-Path $startup 'Jelibox Web.lnk'

function Say([string]$text) { Write-Host "[Jelibox] $text" }

function Get-Port {
    $port = 8420
    try {
        $cfg = Get-Content -Raw -LiteralPath (Join-Path $Root 'configs\_app.json') | ConvertFrom-Json
        if ($cfg.web_port) { $port = [int]$cfg.web_port }
    } catch { }
    return $port
}

# The processes running THIS install's server.py (pythonw, plus the python it was launched through).
function Get-ServerProcesses {
    Get-CimInstance Win32_Process -Filter "Name='pythonw.exe' OR Name='python.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -and ($_.CommandLine.IndexOf($server, [StringComparison]::OrdinalIgnoreCase) -ge 0) }
}

function Test-Port([int]$port) {
    $client = New-Object Net.Sockets.TcpClient
    try {
        $wait = $client.BeginConnect('127.0.0.1', $port, $null, $null)
        return ($wait.AsyncWaitHandle.WaitOne(500) -and $client.Connected)
    } catch { return $false } finally { $client.Close() }
}

function Stop-Server {
    $found = @(Get-ServerProcesses)
    foreach ($p in $found) { Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue }
    for ($i = 0; $i -lt 30 -and @(Get-ServerProcesses).Count -gt 0; $i++) { Start-Sleep -Milliseconds 200 }
    return $found.Count
}

function Start-Server {
    $port = Get-Port
    if (-not (Test-Path -LiteralPath $python)) { Say "Jelibox's Python was not found: $python"; return $false }
    if (@(Get-ServerProcesses).Count -eq 0) {
        Start-Process -FilePath $python -ArgumentList ('"' + $server + '"') -WorkingDirectory $Root -WindowStyle Hidden
    }
    for ($i = 0; $i -lt 120; $i++) {                      # up to 60 s: the first start imports a lot
        if (Test-Port $port) { return $true }
        if (@(Get-ServerProcesses).Count -eq 0) { break }  # it died: no point in waiting
        Start-Sleep -Milliseconds 500
    }
    return $false
}

function Show-Addresses {
    $port = Get-Port
    Say "The web version answers at:  http://localhost:$port"
    $ips = @()
    try { $ips = [Net.Dns]::GetHostAddresses([Net.Dns]::GetHostName()) | Where-Object { $_.AddressFamily -eq 'InterNetwork' -and -not $_.ToString().StartsWith('127.') } } catch { }
    foreach ($ip in $ips) { Say "From another device on the network:  http://${ip}:$port" }
}

switch ($Action) {
    'install' {
        if (-not (Test-Path -LiteralPath $server)) { Say "utils\server.py is missing in $Root"; exit 1 }
        New-Item -ItemType Directory -Force -Path $startup | Out-Null
        $shell = New-Object -ComObject WScript.Shell
        $sc = $shell.CreateShortcut($link)
        $sc.TargetPath = $python
        $sc.Arguments = '"' + $server + '"'
        $sc.WorkingDirectory = $Root
        $sc.WindowStyle = 7
        $sc.Description = 'Jelibox web server'
        $icon = Join-Path $Root 'assets\jelibox.ico'
        if (Test-Path -LiteralPath $icon) { $sc.IconLocation = $icon }
        $sc.Save()
        Say "Added to Windows Startup: $link"
        [void](Stop-Server)                                # an update must not keep running the old code
        if (Start-Server) {
            Say 'The Jelibox web server is running and will start every time you sign in to Windows.'
            Show-Addresses
            $port = Get-Port
            Say "For other devices, Windows Firewall must allow port $port. In PowerShell as administrator (once):"
            Say "    New-NetFirewallRule -DisplayName 'Jelibox Web' -Direction Inbound -Protocol TCP -LocalPort $port -Profile Private -Action Allow"
            exit 0
        }
        Say 'The server did not start. Look at temp\server.log in the Jelibox folder.'
        exit 1
    }
    'remove' {
        $n = Stop-Server
        if (Test-Path -LiteralPath $link) { Remove-Item -LiteralPath $link -Force; Say 'Removed from Windows Startup.' }
        if ($n -gt 0) { Say 'The web server was stopped.' }
        exit 0
    }
    'start' {
        if (Start-Server) { Show-Addresses; exit 0 }
        Say 'The server did not start. Look at temp\server.log in the Jelibox folder.'
        exit 1
    }
    'stop' {
        $n = Stop-Server
        Say $(if ($n -gt 0) { 'The web server was stopped.' } else { 'The web server was not running.' })
        exit 0
    }
    'status' {
        $port = Get-Port
        $up = Test-Port $port
        Say ('Web server: ' + $(if ($up) { "running on port $port" } else { 'not running' }))
        Say ('Starts at sign in: ' + $(if (Test-Path -LiteralPath $link) { 'yes' } else { 'no' }))
        exit $(if ($up) { 0 } else { 1 })
    }
}
