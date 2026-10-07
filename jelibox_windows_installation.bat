@echo off
setlocal EnableDelayedExpansion

title Jelibox Installer
cd /d "%~dp0"

:: =========================================================
:: ADMINISTRATOR RIGHTS
:: A normal install needs none: Python lives inside this folder. Windows is only asked for permission
:: when a step really needs it (the Visual C++ runtime, or the system-wide Python fallback).
:: =========================================================
set "SELF=%~f0"

set FAILED=0
set TORCH_STATUS=SUCCESS
set CLIP_STATUS=SUCCESS
set GPU_TYPE=CPU
set ARCH=x64

set PYTHON_VERSION=3.12.6
set PYTHON_FOLDER=Python312
set PYTHON_INSTALLER=%TEMP%\python_installer.exe

:: The one-line installer (install.ps1) drops this marker so nobody has to answer Y/N.
set ASSUME_YES=0
if /i "%~1"=="/yes" set ASSUME_YES=1
if exist "%~dp0.install-yes" (
    set ASSUME_YES=1
    del "%~dp0.install-yes" >nul 2>&1
)

cls
echo ==========================================
echo   Welcome to Jelibox, Local annotation tool
echo         Thanks for choosing us
echo ==========================================
echo System is preparing your environment...
echo ==========================================

if "%ASSUME_YES%"=="1" goto CONFIRMED
choice /c YN /m "Continue installation?"
if errorlevel 2 (
    echo.
    echo Installation cancelled.
    timeout /t 2 >nul
    exit
)
:CONFIRMED

:: =========================================================
:: 0. INTERNET CHECK
:: =========================================================
echo.
echo [0/6] Checking internet connection...
ping google.com -n 1 -w 3000 >nul
if errorlevel 1 (
    ping 1.1.1.1 -n 1 -w 3000 >nul
    if errorlevel 1 (
        echo [ERROR] No internet connection detected.
        set FAILED=1
        goto END
    )
)
echo [OK] Internet connection detected.

:: =========================================================
:: 1. SYSTEM SCAN
:: =========================================================
echo.
echo [1/6] Scanning system...

echo %PROCESSOR_ARCHITECTURE% | findstr /i "ARM" >nul
if %ERRORLEVEL% equ 0 (
    set ARCH=ARM
    echo [!] Windows ARM detected.
)

where nvidia-smi >nul 2>&1
if not errorlevel 1 (
    nvidia-smi -L >nul 2>&1
    if not errorlevel 1 (
        set GPU_TYPE=NVIDIA
        echo [OK] NVIDIA GPU detected through nvidia-smi.
        nvidia-smi --query-gpu=name --format=csv,noheader,nounits 2>nul
    ) else (
        echo [!] nvidia-smi is installed but no usable NVIDIA GPU was reported.
        echo [*] CPU mode will be used.
    )
) else (
    :: Fallback for systems where the NVIDIA utility is unavailable or not on PATH.
    powershell -NoProfile -Command "$ErrorActionPreference = 'Stop'; $gpu = Get-CimInstance Win32_VideoController; if($gpu.Name -match 'NVIDIA|RTX|GTX|TESLA'){ exit 0 } else { exit 1 }"
    if not errorlevel 1 (
        set GPU_TYPE=NVIDIA
        echo [OK] NVIDIA GPU detected through Windows device information.
    ) else (
        echo [!] NVIDIA GPU not detected.
        echo [*] CPU mode will be used.
    )
)

:: =========================================================
:: 1b. VISUAL C++ RUNTIME (required by PyTorch)
:: =========================================================
if "%ARCH%"=="x64" (
    reg query "HKLM\SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64" /v Installed 2>nul | find "0x1" >nul
    if errorlevel 1 (
        if exist "%~dp0VC_redist\VC_redist.x64.exe" (
            echo [*] The Microsoft Visual C++ runtime is missing and needs administrator permission to install.
            call :REQUIRE_ADMIN
            if errorlevel 1 (
                set FAILED=1
                goto END
            )
            echo [*] Installing Microsoft Visual C++ runtime...
            "%~dp0VC_redist\VC_redist.x64.exe" /install /quiet /norestart
        )
    ) else (
        echo [OK] Visual C++ runtime already installed.
    )
)

:: =========================================================
:: 2. PYTHON
:: By default Jelibox gets its own private Python 3.12, downloaded with uv into this folder: nothing is
:: installed system-wide and the Python you already have is left alone. Set JELIBOX_PYTHON=system to
:: use the old behaviour (a system-wide Python from python.org), which is also the fallback.
:: =========================================================
echo.
echo [2/6] Preparing Python 3.12...

if /i "%JELIBOX_PYTHON%"=="system" goto PYTHON_SYSTEM
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\setup_python.ps1" -Root "%~dp0."
if %ERRORLEVEL% EQU 0 goto PYTHON_READY
echo [!] The private Python could not be set up - falling back to the system Python.

:PYTHON_SYSTEM
py -3.12 --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [!] Python 3.12 not found.
    echo [*] Installing it system-wide needs administrator permission.
    call :REQUIRE_ADMIN
    if errorlevel 1 (
        set FAILED=1
        goto END
    )
    echo [*] Downloading Python %PYTHON_VERSION% installer...
    
    powershell -NoProfile -Command "Invoke-WebRequest -Uri 'https://www.python.org/ftp/python/%PYTHON_VERSION%/python-%PYTHON_VERSION%-amd64.exe' -OutFile '%PYTHON_INSTALLER%'"
    
    if not exist "%PYTHON_INSTALLER%" (
        echo [ERROR] Failed to download Python installer.
        set FAILED=1
        goto END
    )

    echo [*] Installing Python silently...
    start /wait "" "%PYTHON_INSTALLER%" /quiet InstallAllUsers=1 PrependPath=1 Include_test=0 Include_launcher=1 Include_tcltk=1

    if !ERRORLEVEL! NEQ 0 (
        echo [ERROR] Python installation failed.
        set FAILED=1
        goto END
    )
    echo [OK] Python installed successfully.
)

set "PATH=%PATH%;C:\Program Files\%PYTHON_FOLDER%;C:\Program Files\%PYTHON_FOLDER%\Scripts"

py -3.12 --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python still not detected.
    set FAILED=1
    goto END
)
echo [OK] Python detected.

if not exist "%~dp0venv\" (
    echo [*] Creating virtual environment...
    py -3.12 -m venv "%~dp0venv"
    if !ERRORLEVEL! NEQ 0 (
        echo [ERROR] Failed to create virtual environment.
        set FAILED=1
        goto END
    )
)

:PYTHON_READY
:: =========================================================
:: 3. VIRTUAL ENVIRONMENT
:: =========================================================
echo.
echo [3/6] Preparing virtual environment...

if not exist "%~dp0venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment is corrupted.
    set FAILED=1
    goto END
)

call "%~dp0venv\Scripts\activate.bat"
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to activate virtual environment.
    set FAILED=1
    goto END
)
echo [OK] Virtual environment ready.

:: =========================================================
:: 4. UPDATE PIP
:: =========================================================
echo.
echo [4/6] Updating package manager...
python -m pip install --upgrade pip setuptools wheel --retries 5 --timeout 30
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to update pip.
    set FAILED=1
    goto END
)

:: =========================================================
:: [5/6] INSTALL PYTORCH
:: =========================================================
echo.
echo [5/6] Installing AI engine...

if "%ARCH%"=="ARM" goto TORCH_ARM
if "%GPU_TYPE%"=="NVIDIA" goto TORCH_NVIDIA

:: Default CPU fallback
echo [*] Installing PyTorch (CPU version)...
python -m pip install torch torchvision torchaudio --retries 5 --timeout 60
goto CHECK_TORCH

:TORCH_NVIDIA
echo [*] Installing PyTorch with CUDA 12.1 support...
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --retries 5 --timeout 60
goto CHECK_TORCH

:TORCH_ARM
echo [!] ARM detected.
echo [!] PyTorch installation skipped.
set TORCH_STATUS=SKIPPED_ARM
goto TORCH_DONE

:CHECK_TORCH
if errorlevel 1 (
    echo [WARNING] PyTorch installation failed.
    set TORCH_STATUS=FAILED
)

:TORCH_DONE

:: =========================================================
:: 6. INSTALL DEPENDENCIES
:: =========================================================
echo.
echo [6/6] Installing Jelibox dependencies...
python -m pip install ultralytics pyinstaller streamlit yt-dlp --retries 5 --timeout 30

if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Dependency installation failed.
    set FAILED=1
    goto END
)
echo [OK] Dependencies installed successfully.

:: CLIP powers the YOLO-World Label Assistant. Installed from a zip archive so
:: git is not required. Non-fatal: without it only YOLO-World is unavailable.
echo [*] Installing CLIP (required by YOLO-World)...
python -m pip install ftfy regex tqdm https://github.com/ultralytics/CLIP/archive/refs/heads/main.zip --retries 5 --timeout 60
if %ERRORLEVEL% NEQ 0 (
    echo [WARNING] CLIP installation failed. YOLO-World Label Assistant will be unavailable.
    set CLIP_STATUS=FAILED
) else (
    echo [OK] CLIP installed successfully.
)

:: =========================================================
:: CREATE SHORTCUTS
:: =========================================================
echo.
echo [*] Creating Jelibox shortcuts...

set "PS_SCRIPT=%TEMP%\jelibox_shortcut.ps1"

> "%PS_SCRIPT%" echo $rootPath = Split-Path -Parent "%~f0"
>> "%PS_SCRIPT%" echo $desktop = [Environment]::GetFolderPath('Desktop')
>> "%PS_SCRIPT%" echo $pythonExe = Join-Path $rootPath 'venv\Scripts\python.exe'
>> "%PS_SCRIPT%" echo $scriptPath = Join-Path $rootPath 'utils\Annotator.py'
>> "%PS_SCRIPT%" echo $iconPath = Join-Path $rootPath 'assets\jelibox.ico'
>> "%PS_SCRIPT%" echo $rootShortcut = Join-Path $rootPath 'Jelibox Launcher.lnk'
>> "%PS_SCRIPT%" echo $desktopShortcut = Join-Path $desktop 'Jelibox.lnk'
>> "%PS_SCRIPT%" echo $ws = New-Object -ComObject WScript.Shell

>> "%PS_SCRIPT%" echo $sc1 = $ws.CreateShortcut($rootShortcut)
>> "%PS_SCRIPT%" echo $sc1.TargetPath = $pythonExe
>> "%PS_SCRIPT%" echo $sc1.Arguments = '"' + $scriptPath + '"'
>> "%PS_SCRIPT%" echo $sc1.WorkingDirectory = $rootPath
>> "%PS_SCRIPT%" echo $sc1.Description = 'Local Annotation Tool'
>> "%PS_SCRIPT%" echo if (Test-Path $iconPath^) { $sc1.IconLocation = $iconPath }
>> "%PS_SCRIPT%" echo $sc1.Save(^)

>> "%PS_SCRIPT%" echo $sc2 = $ws.CreateShortcut($desktopShortcut)
>> "%PS_SCRIPT%" echo $sc2.TargetPath = $rootShortcut
>> "%PS_SCRIPT%" echo $sc2.Description = 'Local Annotation Tool'
>> "%PS_SCRIPT%" echo if (Test-Path $iconPath^) { $sc2.IconLocation = $iconPath }
>> "%PS_SCRIPT%" echo $sc2.Save(^)

>> "%PS_SCRIPT%" echo Write-Host "[OK] Root launcher created:"
>> "%PS_SCRIPT%" echo Write-Host $rootShortcut
>> "%PS_SCRIPT%" echo Write-Host "[OK] Desktop shortcut created:"
>> "%PS_SCRIPT%" echo Write-Host $desktopShortcut

powershell -NoProfile -ExecutionPolicy Bypass -File "%PS_SCRIPT%"

del "%PS_SCRIPT%" >nul 2>&1

:: =========================================================
:: SUCCESS
:: =========================================================
echo.
echo =========================================================
if %FAILED% equ 1 (
    echo                    INSTALLATION FAILED
) else (
    echo                     JELIBOX IS READY
)
echo =========================================================

:: Tells the one-line installer (install.ps1) that this install finished, so it can
:: say "already up to date" next time instead of repeating the whole setup.
if %FAILED% neq 1 echo done> "%~dp0.jelibox-ready"

if "%TORCH_STATUS%"=="SKIPPED_ARM" (
    echo [NOTICE] ARM platform detected.
    echo [NOTICE] AI acceleration disabled.
)

if "%GPU_TYPE%"=="CPU" (
    echo [NOTICE] Running in CPU mode.
)

if "%TORCH_STATUS%"=="FAILED" (
    echo [WARNING] PyTorch installation failed.
)

if "%CLIP_STATUS%"=="FAILED" (
    echo [WARNING] CLIP installation failed - YOLO-World Label Assistant is unavailable.
    echo [WARNING] Retry with: venv\Scripts\python -m pip install ftfy regex tqdm https://github.com/ultralytics/CLIP/archive/refs/heads/main.zip
)

echo.
echo Desktop shortcut created:
echo.
echo   Jelibox.lnk
echo.
echo Thank you for using Jelibox.
echo =========================================================
goto FINISH

:: =========================================================
:: ERROR HANDLER
:: =========================================================
:END
echo.
echo =========================================================
echo                    INSTALLATION FAILED
echo =========================================================
echo Please check your internet connection or permissions.
echo =========================================================

:FINISH
echo.
pause
exit

:: =========================================================
:: REQUIRE_ADMIN - continue when already elevated, otherwise restart this installer elevated.
:: Returns 1 when Windows permission was refused. When the restart works this window closes
:: and the elevated copy takes over (it skips the Y/N question: /yes).
:: =========================================================
:REQUIRE_ADMIN
net session >nul 2>&1
if not errorlevel 1 exit /b 0
echo [*] Windows will ask you to approve administrator permission...
if "%ASSUME_YES%"=="1" (
    powershell -NoProfile -Command "Start-Process -FilePath '%SELF%' -ArgumentList '/yes' -WorkingDirectory '%CD%' -Verb RunAs"
) else (
    powershell -NoProfile -Command "Start-Process -FilePath '%SELF%' -WorkingDirectory '%CD%' -Verb RunAs"
)
if errorlevel 1 (
    echo [ERROR] Administrator permission was not granted.
    exit /b 1
)
exit