# Jelibox one-line installer for Windows.
#
#   irm https://raw.githubusercontent.com/Jelibox/Jelibox-client/main/install.ps1 | iex
#
# What it does (read it - it is short):
#   1. finds the newest Jelibox release on GitHub (falls back to the main branch)
#   2. downloads that release as a .zip and unpacks it into  %LOCALAPPDATA%\Jelibox
#      (running it again updates the code and never touches your datasets, models or configs)
#   3. starts jelibox_windows_installation.bat, which installs Python 3.12 if needed, creates the
#      virtual environment, installs the dependencies and creates the shortcuts.
#      Windows will ask for administrator permission once for that step.
#
# Optional environment variables (set them before running the command):
#   JELIBOX_VERSION     install a specific release, e.g.  v0.1.0   (default: newest release)
#   JELIBOX_HOME        install somewhere else               (default: %LOCALAPPDATA%\Jelibox)
#   JELIBOX_NO_INSTALL  1 = only download and unpack, do not run the installer
#   JELIBOX_ARCHIVE     path to a local .zip instead of downloading (offline installs, tests)

function Install-Jelibox {
    $ErrorActionPreference = 'Stop'
    $ProgressPreference = 'SilentlyContinue'     # the progress bar makes downloads many times slower in Windows PowerShell
    try {
        [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
    } catch { }

    $repo    = 'Jelibox/Jelibox-client'
    $dest    = if ($env:JELIBOX_HOME) { $env:JELIBOX_HOME } else { Join-Path $env:LOCALAPPDATA 'Jelibox' }
    $version = $env:JELIBOX_VERSION
    $archive = $env:JELIBOX_ARCHIVE

    function Say([string]$text)  { Write-Host "[Jelibox] $text" -ForegroundColor Cyan }
    function Fail([string]$text) { Write-Host "[Jelibox] $text" -ForegroundColor Red; throw $text }

    if ($env:OS -ne 'Windows_NT') { Fail 'This installer is for Windows. On Linux use install.sh.' }

    $tmp = Join-Path ([IO.Path]::GetTempPath()) ('jelibox_' + [Guid]::NewGuid().ToString('N'))
    New-Item -ItemType Directory -Path $tmp | Out-Null
    try {
        if ($archive) {
            if (-not (Test-Path $archive)) { Fail "JELIBOX_ARCHIVE not found: $archive" }
            $zip = $archive
            Say "Using local archive $archive"
        } else {
            if (-not $version) {
                try {
                    $release = Invoke-RestMethod -UseBasicParsing "https://api.github.com/repos/$repo/releases/latest"
                    $version = $release.tag_name
                } catch {
                    Say 'No release found yet - using the latest development version (main).'
                }
            }
            if ($version) {
                $url = "https://github.com/$repo/archive/refs/tags/$version.zip"
                Say "Downloading Jelibox $version ..."
            } else {
                $url = "https://github.com/$repo/archive/refs/heads/main.zip"
                Say 'Downloading Jelibox (main) ...'
            }
            $zip = Join-Path $tmp 'jelibox.zip'
            Invoke-WebRequest -UseBasicParsing -Uri $url -OutFile $zip
        }

        Say 'Unpacking ...'
        $extract = Join-Path $tmp 'src'
        Expand-Archive -LiteralPath $zip -DestinationPath $extract -Force
        $top = Get-ChildItem -LiteralPath $extract -Directory | Select-Object -First 1
        if (-not $top) { Fail 'The downloaded archive is empty.' }

        New-Item -ItemType Directory -Force -Path $dest | Out-Null
        # Copy over the existing install. Nothing is deleted, so datasets/models/configs/venv survive an update.
        Copy-Item -Path (Join-Path $top.FullName '*') -Destination $dest -Recurse -Force
        Say "Installed files are in $dest"

        if ($env:JELIBOX_NO_INSTALL -eq '1') {
            Say 'JELIBOX_NO_INSTALL=1 - stopping before the dependency installer.'
            return
        }

        $bat = Join-Path $dest 'jelibox_windows_installation.bat'
        if (-not (Test-Path $bat)) { Fail "Installer script missing: $bat" }
        New-Item -ItemType File -Force -Path (Join-Path $dest '.install-yes') | Out-Null    # skip the Y/N question

        Say 'Starting the installer. Approve the Windows permission prompt when it appears.'
        Say 'It opens in its own window and shows its progress there.'
        Start-Process -FilePath 'cmd.exe' -ArgumentList '/c', "`"$bat`"" -WorkingDirectory $dest
        Say "When it says JELIBOX IS READY, open  Jelibox  from your Desktop or Start menu."
    }
    finally {
        Remove-Item -LiteralPath $tmp -Recurse -Force -ErrorAction SilentlyContinue
    }
}

Install-Jelibox
