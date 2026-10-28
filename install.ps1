# Jelibox one-line installer for Windows.
#
#   irm https://raw.githubusercontent.com/Jelibox/Jelibox-client/main/install.ps1 | iex
#
# What it does (read it - it is short):
#   1. asks where to install (Enter = %USERPROFILE%\Jelibox, B = browse) and looks for an existing Jelibox there
#   2. finds the newest Jelibox release on GitHub (falls back to the main branch)
#   3. not installed yet   -> installs the newest version fresh
#      installed, older    -> updates it (your datasets, models and configs are never touched)
#      installed, current  -> says so and stops
#   4. asks what to install: the app + the web version (a server you open in a browser, also from other devices, started
#      for you every time you sign in to Windows - a shortcut in your Startup folder, no administrator rights), or the
#      app only. The answer is asked again on every install and update, so it can be changed.
#   5. starts jelibox_windows_installation.bat, which installs uv if it is missing (official installer, user level),
#      gives Jelibox its own private Python 3.12 (downloaded by uv into the Jelibox folder - your own Python is not
#      touched, nothing is put on PATH), creates the virtual environment, installs the dependencies and creates the
#      shortcuts. No administrator rights are needed, except to install the Visual C++ runtime when it is missing:
#      then only that installer is started elevated and Windows shows its permission prompt.
#
# Optional environment variables (set them before running the command):
#   JELIBOX_VERSION     install a specific release, e.g.  v0.1.0   (default: newest release)
#                       main = the latest development version (always re-installed)
#   JELIBOX_HOME        install here without being asked     (default: you are asked; Enter = %USERPROFILE%\Jelibox)
#   JELIBOX_NO_INSTALL  1 = only download and unpack, do not run the installer
#   JELIBOX_FORCE       1 = reinstall even when this version is already installed
#   JELIBOX_GPU         nvidia | cpu = which PyTorch to install without being asked (default: you are asked)
#   JELIBOX_WEB         yes | no = app + web version, or the app only, without being asked (default: you are asked;
#                       Enter = what this folder had before, otherwise app + web)
#   JELIBOX_ARCHIVE     path to a local .zip instead of downloading (offline installs, tests)

function Install-Jelibox {
    $ErrorActionPreference = 'Stop'
    $ProgressPreference = 'SilentlyContinue'     # the progress bar makes downloads many times slower in Windows PowerShell
    try {
        [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
    } catch { }

    $repo    = 'Jelibox/Jelibox-client'
    $version = $env:JELIBOX_VERSION
    $archive = $env:JELIBOX_ARCHIVE

    function Say([string]$text)  { Write-Host "[Jelibox] $text" -ForegroundColor Cyan }
    function Fail([string]$text) { Write-Host "[Jelibox] $text" -ForegroundColor Red; throw $text }

    if ($env:OS -ne 'Windows_NT') { Fail 'This installer is for Windows. On Linux use install.sh.' }

    # Where to install: JELIBOX_HOME if set, otherwise ask. Enter = the default (a "Jelibox" folder next to Downloads,
    # Documents, Pictures), B = pick a folder in a window, or type a path. A "Jelibox" folder is created inside the
    # chosen folder (a path that already ends in "Jelibox" is used as-is).
    $dest = $env:JELIBOX_HOME
    if (-not $dest) {
        # Earlier versions of this installer always used %LOCALAPPDATA%\Jelibox - keep updating that one instead of asking.
        $legacy = Join-Path $env:LOCALAPPDATA 'Jelibox'
        if ((Test-Path -LiteralPath (Join-Path $legacy '.jelibox-version')) -or (Test-Path -LiteralPath (Join-Path $legacy 'utils\AnnotationGUI.py'))) {
            $dest = $legacy
            Say "Found an existing Jelibox in $dest - it will be updated there. (To relocate it, use Move Jelibox inside the app.)"
        }
    }
    if (-not $dest) {
        $suggested = Join-Path $env:USERPROFILE 'Jelibox'
        $answer = ''
        try {
            Write-Host ''
            Write-Host '[Jelibox] Where should Jelibox be installed?' -ForegroundColor Cyan
            Write-Host "          [Enter]  default: $suggested"
            Write-Host '          [B]      browse - pick a folder in a window'
            Write-Host '          or type a folder path'
            $answer = Read-Host '          Your choice'
        } catch {
            Say 'No keyboard available - using the default location.'
        }
        $answer = "$answer".Trim().Trim('"')
        if ($answer -ieq 'b') {
            $answer = ''
            try {
                Add-Type -AssemblyName System.Windows.Forms
                $owner = New-Object System.Windows.Forms.Form -Property @{ TopMost = $true }   # keeps the window in front of the console
                $dialog = New-Object System.Windows.Forms.FolderBrowserDialog
                $dialog.Description = 'Choose where Jelibox should be installed (a "Jelibox" folder is created inside it)'
                $dialog.SelectedPath = $env:USERPROFILE
                if ($dialog.ShowDialog($owner) -eq [System.Windows.Forms.DialogResult]::OK) { $answer = $dialog.SelectedPath }
                $owner.Dispose()
            } catch {
                Say 'Could not open the folder window.'
            }
            if (-not $answer) { Say 'No folder chosen - using the default location.' }
        }
        if ($answer) {
            $answer = [Environment]::ExpandEnvironmentVariables($answer)
            if ($answer -eq '~' -or $answer.StartsWith('~\') -or $answer.StartsWith('~/')) { $answer = $env:USERPROFILE + $answer.Substring(1) }
            $answer = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($answer)
            $dest = if ((Split-Path -Leaf $answer) -ieq 'Jelibox') { $answer } else { Join-Path $answer 'Jelibox' }
        } else {
            $dest = $suggested
        }
    }

    # Is Jelibox already here? (.jelibox-version is written by this script; older installs only have the code.)
    if (Test-Path -LiteralPath (Join-Path $dest '.git')) {
        Say "$dest is a git checkout, so it is updated with git rather than this installer:"
        Say "    cd `"$dest`"; git pull"
        return
    }
    $existing = $false
    $installed = ''
    $versionFile = Join-Path $dest '.jelibox-version'
    if ((Test-Path -LiteralPath $versionFile) -or (Test-Path -LiteralPath (Join-Path $dest 'utils\AnnotationGUI.py'))) {
        $existing = $true
        if (Test-Path -LiteralPath $versionFile) { $installed = ([string](Get-Content -LiteralPath $versionFile -TotalCount 1)).Trim() }
    }

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
            if ($version -eq 'main') {
                $url = "https://github.com/$repo/archive/refs/heads/main.zip"
                Say 'Downloading Jelibox (main) ...'
            } elseif ($version) {
                $url = "https://github.com/$repo/archive/refs/tags/$version.zip"
                Say "Downloading Jelibox $version ..."
            } else {
                $url = "https://github.com/$repo/archive/refs/heads/main.zip"
                Say 'Downloading Jelibox (main) ...'
            }
            $zip = Join-Path $tmp 'jelibox.zip'
        }

        # Fresh install, update, or nothing to do?
        $target = if ($version) { $version } elseif ($archive) { 'local archive' } else { 'main' }
        if ($existing) {
            if ($version -and ($version -ne 'main') -and ($installed -eq $version) -and (Test-Path -LiteralPath (Join-Path $dest '.jelibox-ready')) -and ($env:JELIBOX_FORCE -ne '1')) {
                Say "Jelibox $version is already installed and up to date ($dest)."
                Say 'Nothing to do. (JELIBOX_FORCE=1 reinstalls it anyway.)'
                return
            }
            $was = if ($installed) { $installed } else { '(unknown version)' }
            Say "Found Jelibox $was in $dest - updating to $target."
        } else {
            Say "Jelibox is not installed yet ($dest) - installing $target."
        }

        # Discrete NVIDIA GPU or not? This decides which PyTorch is downloaded (CUDA build or the smaller CPU build).
        # JELIBOX_GPU=nvidia|cpu skips the question; the installer script asks by itself if it is started without it.
        if ($env:JELIBOX_NO_INSTALL -ne '1') {
            $gpu = "$env:JELIBOX_GPU".Trim().ToLower()
            if ($gpu -and $gpu -ne 'nvidia' -and $gpu -ne 'cpu') { Fail "JELIBOX_GPU must be 'nvidia' or 'cpu' (got '$gpu')." }
            if (-not $gpu) {
                $detected = $false
                try {
                    if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) { & nvidia-smi -L *> $null; if ($LASTEXITCODE -eq 0) { $detected = $true } }
                    if (-not $detected) { $detected = [bool](Get-CimInstance Win32_VideoController -ErrorAction Stop | Where-Object { $_.Name -match 'NVIDIA' }) }
                } catch { }
                $isArm = ($env:PROCESSOR_ARCHITECTURE -eq 'ARM64') -or ($env:PROCESSOR_ARCHITEW6432 -eq 'ARM64')
                if ($isArm) {
                    $gpu = 'cpu'
                    Say 'Windows on ARM: the CPU setup is used (no CUDA PyTorch for ARM).'
                } else {
                    $answer = ''
                    try {
                        Write-Host ''
                        Write-Host '[Jelibox] Does this computer have a discrete NVIDIA graphics card (GeForce / RTX / GTX / Quadro)?' -ForegroundColor Cyan
                        Write-Host '          Yes = the CUDA build of PyTorch (about 2.5 GB, needs the NVIDIA driver). No = the much smaller CPU build.'
                        Write-Host '          AMD and Intel graphics cannot use CUDA - answer No for them.'
                        Write-Host ("          Detected: " + $(if ($detected) { 'an NVIDIA GPU' } else { 'no NVIDIA GPU' }))
                        $default = if ($detected) { 'Y' } else { 'N' }
                        $answer = Read-Host "          NVIDIA GPU? [Y/N, Enter = $default]"
                    } catch {
                        Say 'No keyboard available - using what was detected.'
                    }
                    $answer = "$answer".Trim()
                    $gpu = if ($answer -match '^(y|yes)$') { 'nvidia' } elseif ($answer -match '^(n|no)$') { 'cpu' } elseif ($detected) { 'nvidia' } else { 'cpu' }
                }
            }
            $env:JELIBOX_GPU = $gpu
            Say ("PyTorch build: " + $(if ($gpu -eq 'nvidia') { 'NVIDIA GPU (CUDA)' } else { 'CPU only' }))
        }

        # App + web, or the app only? Asked on every install and update. Enter = what this folder had before, else both.
        # JELIBOX_WEB=yes|no skips the question; the installer script asks by itself if it is started without it.
        if ($env:JELIBOX_NO_INSTALL -ne '1') {
            $web = "$env:JELIBOX_WEB".Trim().ToLower()
            if ($web -and $web -ne 'yes' -and $web -ne 'no') { Fail "JELIBOX_WEB must be 'yes' or 'no' (got '$web')." }
            if (-not $web) {
                $before = ''
                try { $before = ([string](Get-Content -LiteralPath (Join-Path $dest '.jelibox-web') -TotalCount 1)).Trim() } catch { }
                $default = if ($before -eq 'no') { '2' } else { '1' }
                $answer = ''
                try {
                    Write-Host ''
                    Write-Host '[Jelibox] What do you want to install?' -ForegroundColor Cyan
                    Write-Host '          1  App + web   the desktop app, plus the web version: a server you open in a browser (also from'
                    Write-Host '                         another computer on your network). It starts by itself every time you sign in to'
                    Write-Host '                         Windows (a shortcut in your Startup folder, no administrator rights).'
                    Write-Host '          2  App only    just the desktop app. An earlier web server of this folder is stopped and removed.'
                    $answer = Read-Host "          Your choice [1/2, Enter = $default]"
                } catch {
                    Say 'No keyboard available - using the default.'
                }
                $answer = "$answer".Trim()
                $web = if ($answer -eq '1') { 'yes' } elseif ($answer -eq '2') { 'no' } elseif ($default -eq '2') { 'no' } else { 'yes' }
            }
            $env:JELIBOX_WEB = $web
            Say ("Installing: " + $(if ($web -eq 'yes') { 'app + web version' } else { 'app only' }))
        }

        if (-not $archive) {
            Invoke-WebRequest -UseBasicParsing -Uri $url -OutFile $zip
        }

        Say 'Unpacking ...'
        $extract = Join-Path $tmp 'src'
        Expand-Archive -LiteralPath $zip -DestinationPath $extract -Force
        $top = Get-ChildItem -LiteralPath $extract -Directory | Select-Object -First 1
        if (-not $top) { Fail 'The downloaded archive is empty.' }

        # An update must not run under the old server: stop it before any file or package is replaced.
        $service = Join-Path $dest 'tools\web_service.ps1'
        if ($existing -and (Test-Path -LiteralPath $service)) {
            try { & powershell -NoProfile -ExecutionPolicy Bypass -File $service -Action stop -Root $dest | Out-Null } catch { }
        }

        New-Item -ItemType Directory -Force -Path $dest | Out-Null
        # Copy over the existing install. Nothing is deleted, so datasets/models/configs/venv survive an update.
        Copy-Item -Path (Join-Path $top.FullName '*') -Destination $dest -Recurse -Force
        Set-Content -LiteralPath $versionFile -Value $target -Encoding ASCII
        Remove-Item -LiteralPath (Join-Path $dest '.jelibox-ready') -Force -ErrorAction SilentlyContinue   # set again by the installer below once everything worked
        Say "Installed files are in $dest"

        if ($env:JELIBOX_NO_INSTALL -eq '1') {
            Say 'JELIBOX_NO_INSTALL=1 - stopping before the dependency installer.'
            return
        }

        $bat = Join-Path $dest 'jelibox_windows_installation.bat'
        if (-not (Test-Path $bat)) { Fail "Installer script missing: $bat" }
        New-Item -ItemType File -Force -Path (Join-Path $dest '.install-yes') | Out-Null    # skip the Y/N question

        Say 'Setting up Python and the dependencies. Windows asks for permission only if the Visual C++ runtime is missing.'
        Say 'It opens in its own window and shows its progress there.'
        Start-Process -FilePath 'cmd.exe' -ArgumentList '/c', "`"$bat`"" -WorkingDirectory $dest
        Say "When it says JELIBOX IS READY, open the Jelibox shortcut on your Desktop."
        if ($env:JELIBOX_WEB -eq 'yes') { Say 'The web server is started there too; its address is shown in that window.' }
    }
    finally {
        Remove-Item -LiteralPath $tmp -Recurse -Force -ErrorAction SilentlyContinue
    }
}

Install-Jelibox
