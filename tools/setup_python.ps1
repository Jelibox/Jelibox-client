# Gives Jelibox its own Python 3.12 + virtual environment - without touching the Python you already have.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File tools\setup_python.ps1 [-Root <install folder>] [-Venv venv]
#
# What it does:
#   * a working environment already exists  -> keeps it (nothing is downloaded, nothing is deleted)
#   * otherwise it downloads `uv` (one small program, checksum-verified) into  <Root>\.uv,
#     lets uv fetch a standalone Python into  <Root>\.python,  and builds  <Root>\<Venv>  from it.
#   Nothing is installed system-wide: no administrator rights, no PATH changes, no registry entries.
#   Uninstalling Jelibox = deleting its folder.
#
# Exit code 0 = the environment is ready, anything else = it could not be set up (the installer then
# falls back to a system-wide Python).
#
# Optional environment variables:
#   JELIBOX_PYTHON_VERSION  exact Python to use              (default: 3.12.10)
#   JELIBOX_UV_VERSION      uv release to download           (default: 0.12.23)
#   JELIBOX_UV              path to an existing uv.exe to use instead of downloading one
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
$uvVersion = if ($env:JELIBOX_UV_VERSION) { $env:JELIBOX_UV_VERSION } else { '0.12.23' }

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

function Get-Uv {
    if ($env:JELIBOX_UV) {
        if (Test-Path -LiteralPath $env:JELIBOX_UV) { return $env:JELIBOX_UV }
        throw "JELIBOX_UV points to a file that does not exist: $env:JELIBOX_UV"
    }
    $onPath = Get-Command uv -ErrorAction SilentlyContinue
    if ($onPath) { Say "Using the uv that is already installed ($($onPath.Source))."; return $onPath.Source }
    $own = Join-Path $Root '.uv\uv.exe'
    if (Test-Path -LiteralPath $own) { return $own }

    $isArm = ($env:PROCESSOR_ARCHITECTURE -eq 'ARM64') -or ($env:PROCESSOR_ARCHITEW6432 -eq 'ARM64')
    $target = if ($isArm) { 'aarch64-pc-windows-msvc' } else { 'x86_64-pc-windows-msvc' }
    $name = "uv-$target.zip"
    $base = "https://github.com/astral-sh/uv/releases/download/$uvVersion"

    $tmp = Join-Path ([IO.Path]::GetTempPath()) ('jelibox_uv_' + [Guid]::NewGuid().ToString('N'))
    New-Item -ItemType Directory -Path $tmp | Out-Null
    try {
        Say "Downloading uv $uvVersion (the tool that fetches Python) ..."
        $zip = Join-Path $tmp $name
        Invoke-WebRequest -UseBasicParsing -Uri "$base/$name" -OutFile $zip
        $sumFile = Join-Path $tmp "$name.sha256"
        Invoke-WebRequest -UseBasicParsing -Uri "$base/$name.sha256" -OutFile $sumFile    # .Content is raw bytes in Windows PowerShell 5.1
        $expected = ([string](Get-Content -LiteralPath $sumFile -TotalCount 1)).Trim().Split(' ')[0]
        $actual = (Get-FileHash -LiteralPath $zip -Algorithm SHA256).Hash
        if ($expected.ToLower() -ne $actual.ToLower()) { throw "The uv download is corrupted (checksum mismatch)." }

        Expand-Archive -LiteralPath $zip -DestinationPath (Join-Path $tmp 'x') -Force
        $exe = Get-ChildItem -LiteralPath (Join-Path $tmp 'x') -Recurse -Filter 'uv.exe' | Select-Object -First 1
        if (-not $exe) { throw 'uv.exe was not found inside the download.' }
        New-Item -ItemType Directory -Force -Path (Split-Path -Parent $own) | Out-Null
        Copy-Item -LiteralPath $exe.FullName -Destination $own -Force
        return $own
    }
    finally {
        Remove-Item -LiteralPath $tmp -Recurse -Force -ErrorAction SilentlyContinue
    }
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
    # Keep everything inside the Jelibox folder, whatever uv settings this machine has.
    $env:UV_PYTHON_INSTALL_DIR = Join-Path $Root '.python'

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
