# Gives Jelibox its own Python 3.12 + virtual environment - without touching the Python you already have.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File tools\setup_python.ps1 [-Root <install folder>] [-Venv venv]
#
# What it does:
#   * a working environment already exists  -> keeps it (nothing is downloaded, nothing is deleted)
#   * otherwise it makes sure `uv` is installed (the official installer from astral.sh runs only when uv is
#     missing), lets uv fetch a standalone Python into  <Root>\.python,  and builds  <Root>\<Venv>  from it.
#   No administrator rights are needed. Python itself is never put on your PATH, never registered with
#   Windows (so `py -3.12` does not see it) and never installed outside the Jelibox folder.
#   The only thing outside the folder is uv: the official uv installer adds its own folder to your user PATH.
#   Uninstalling Jelibox = deleting its folder (and uv, if you do not want it, see https://docs.astral.sh/uv/).
#
# Exit code 0 = the environment is ready, anything else = it could not be set up.
#
# Optional environment variables:
#   JELIBOX_PYTHON_VERSION  exact Python to use              (default: 3.12.10)
#   JELIBOX_UV              path to an existing uv.exe to use instead of looking for / installing one
param(
    [string]$Root = (Split-Path -Parent $PSScriptRoot),
    [string]$Venv = 'venv'
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'     # the progress bar makes downloads many times slower in Windows PowerShell
try {
    [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
} catch { }

$pyVersion = if ($env:JELIBOX_PYTHON_VERSION) { $env:JELIBOX_PYTHON_VERSION } else { '3.12.10' }

$venvDir = Join-Path $Root $Venv
$venvPy  = Join-Path $venvDir 'Scripts\python.exe'

function Say([string]$text) { Write-Host "[*] $text" }

# "3.12" when the interpreter runs and has a working Tkinter, otherwise $null.
function Get-PythonVersion([string]$exe) {
    if (-not (Test-Path -LiteralPath $exe)) { return $null }
    try {
        $out = & $exe -c "import sys, tkinter; tkinter.Tcl(); print('%d.%d' % sys.version_info[:2])" 2>$null
        if ($LASTEXITCODE -eq 0 -and $out) { return ([string]($out | Select-Object -First 1)).Trim() }
    } catch { }
    return $null
}

# Path of an installed uv, or $null. Looks on PATH first, then where the official installer puts it
# (it may have just been installed, and this process's PATH does not know about it yet).
function Find-Uv {
    if ($env:JELIBOX_UV) {
        if (Test-Path -LiteralPath $env:JELIBOX_UV) { return $env:JELIBOX_UV }
        throw "JELIBOX_UV points to a file that does not exist: $env:JELIBOX_UV"
    }
    $onPath = Get-Command uv -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($onPath) { return $onPath.Source }
    $dirs = @($env:UV_INSTALL_DIR, $env:XDG_BIN_HOME)
    if ($env:USERPROFILE) { $dirs += (Join-Path $env:USERPROFILE '.local\bin'); $dirs += (Join-Path $env:USERPROFILE '.cargo\bin') }
    foreach ($dir in $dirs) {
        if (-not $dir) { continue }
        $candidate = Join-Path $dir 'uv.exe'
        if (Test-Path -LiteralPath $candidate) { return $candidate }
    }
    return $null
}

function Get-Uv {
    $uv = Find-Uv
    if ($uv) { Say "Using uv ($uv)."; return $uv }

    Say 'uv is not installed - installing it with the official installer (https://astral.sh/uv) ...'
    & powershell -NoProfile -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex" | Out-Host
    if ($LASTEXITCODE -ne 0) { throw 'The uv installer failed.' }
    $uv = Find-Uv
    if (-not $uv) { throw 'uv was installed but could not be found afterwards.' }
    return $uv
}

try {
    $found = Get-PythonVersion $venvPy
    if ($found) {
        if ($found -eq '3.12') { Say "Python environment is ready (Python $found)." }
        else { Say "Existing environment uses Python $found - keeping it." }
        exit 0
    }
    if (Test-Path -LiteralPath $venvDir) {
        Say 'The existing Python environment is broken - rebuilding it.'
        Remove-Item -LiteralPath $venvDir -Recurse -Force
    }

    $uv = Get-Uv
    # Keep Python inside the Jelibox folder, whatever uv settings this machine has, and leave no trace outside it.
    $env:UV_PYTHON_INSTALL_DIR = Join-Path $Root '.python'
    $env:UV_PYTHON_INSTALL_REGISTRY = '0'      # do not register it with Windows (`py -3.12` must not find it)

    Say "Getting a private Python $pyVersion (your own Python is not touched) ..."
    & $uv python install $pyVersion --no-bin --no-config
    if ($LASTEXITCODE -ne 0) { throw "uv could not download Python $pyVersion." }

    Say 'Creating the virtual environment ...'
    & $uv venv $venvDir --seed --python $pyVersion --managed-python --no-config
    if ($LASTEXITCODE -ne 0) { throw 'uv could not create the virtual environment.' }

    $made = Get-PythonVersion $venvPy
    if (-not $made) { throw 'The new environment does not run, or its Tkinter is missing.' }
    Say "Python $made is ready."
    exit 0
}
catch {
    Write-Host "[!] $($_.Exception.Message)"
    exit 1
}
