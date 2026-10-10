@echo off
setlocal EnableDelayedExpansion

title Jelibox Installer
cd /d "%~dp0"

:: =========================================================
:: ADMINISTRATOR RIGHTS
:: This installer does NOT run as administrator. Python, PyTorch and everything else live inside this
:: folder. The one step that needs administrator permission is installing the Microsoft Visual C++
:: runtime (PyTorch needs it), and only when it is missing: then just that installer is started
:: elevated and Windows shows its permission prompt.
:: =========================================================
set "VCREDIST=%~dp0VC_redist\VC_redist.x64.exe"
set "VENV_PY=%~dp0venv\Scripts\python.exe"

set FAILED=0
set TORCH_STATUS=SUCCESS
set CLIP_STATUS=SUCCESS
set GPU_TYPE=CPU
set ARCH=x64

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
    )
) else (
    :: Fallback for systems where the NVIDIA utility is unavailable or not on PATH.
    powershell -NoProfile -Command "$ErrorActionPreference = 'Stop'; $gpu = Get-CimInstance Win32_VideoController; if($gpu.Name -match 'NVIDIA|RTX|GTX|TESLA'){ exit 0 } else { exit 1 }"
    if not errorlevel 1 (
        set GPU_TYPE=NVIDIA
        echo [OK] NVIDIA GPU detected through Windows device information.
    ) else (
        echo [!] NVIDIA GPU not detected.
    )
)

:: =========================================================
:: 1a. DISCRETE NVIDIA GPU OR NOT?
:: Decides which PyTorch is downloaded: the CUDA build or the much smaller CPU build. install.ps1 asks this
:: question itself and passes the answer in JELIBOX_GPU (nvidia or cpu); started on its own, this asks it.
:: The detection above is only the suggestion. Windows on ARM has no CUDA PyTorch, so nothing is asked there.
:: =========================================================
if "%ARCH%"=="ARM" (
    set GPU_TYPE=CPU
    goto GPU_DONE
)
if defined JELIBOX_GPU goto GPU_PRESET
echo.
if "%GPU_TYPE%"=="NVIDIA" (
    echo [*] Detected: an NVIDIA GPU.
) else (
    echo [*] Detected: no NVIDIA GPU.
)
echo     Yes = the CUDA build of PyTorch, about 2.5 GB, needs the NVIDIA driver.
echo     No  = the much smaller CPU build. AMD and Intel graphics cannot use CUDA, answer No for them.
choice /c YN /m "Does this computer have a discrete NVIDIA graphics card"
if errorlevel 2 (
    set GPU_TYPE=CPU
) else (
    set GPU_TYPE=NVIDIA
)
goto GPU_DONE

:GPU_PRESET
if /i "%JELIBOX_GPU%"=="nvidia" (
    set GPU_TYPE=NVIDIA
) else if /i "%JELIBOX_GPU%"=="cpu" (
    set GPU_TYPE=CPU
) else (
    echo [ERROR] JELIBOX_GPU must be nvidia or cpu, not "%JELIBOX_GPU%".
    set FAILED=1
    goto END
)

:GPU_DONE
if "%GPU_TYPE%"=="NVIDIA" (
    echo [OK] PyTorch build: NVIDIA GPU ^(CUDA^).
    where nvidia-smi >nul 2>&1
    if errorlevel 1 echo [WARNING] nvidia-smi was not found. PyTorch needs the NVIDIA driver to use the GPU: https://www.nvidia.com/drivers
) else (
    echo [OK] PyTorch build: CPU only.
)

:: =========================================================
:: 1b. VISUAL C++ RUNTIME (required by PyTorch)
:: =========================================================
if "%ARCH%"=="x64" (
    reg query "HKLM\SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64" /v Installed 2>nul | find "0x1" >nul
    if errorlevel 1 (
        if exist "%VCREDIST%" (
            echo [*] The Microsoft Visual C++ runtime ^(needed by PyTorch^) is missing.
            echo [*] Installing it needs administrator permission - Windows will ask you to approve it.
            call :INSTALL_VCREDIST
            if errorlevel 1 (
                set FAILED=1
                goto END
            )
        ) else (
            echo [WARNING] The Visual C++ runtime is missing and VC_redist\VC_redist.x64.exe was not found.
        )
    ) else (
        echo [OK] Visual C++ runtime already installed.
    )
)

:: =========================================================
:: 2. PYTHON
:: Jelibox gets its own Python 3.12, downloaded by uv into this folder. It is never installed system-wide,
:: never added to PATH and never registered with Windows, so the Python you already have is left alone.
:: uv itself is installed (official installer) only if it is not there yet. No administrator rights needed.
:: =========================================================
echo.
echo [2/6] Preparing Jelibox's private Python 3.12...

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\setup_python.ps1" -Root "%~dp0."
if errorlevel 1 (
    echo [ERROR] Could not set up Jelibox's private Python. Check your internet connection and try again.
    set FAILED=1
    goto END
)

:: =========================================================
:: 3. VIRTUAL ENVIRONMENT
:: The environment is never "activated": every command below calls its python.exe directly, so nothing
:: here can change which Python the rest of your system finds on PATH.
:: =========================================================
echo.
echo [3/6] Checking virtual environment...

if not exist "%VENV_PY%" (
    echo [ERROR] Virtual environment is corrupted.
    set FAILED=1
    goto END
)
echo [OK] Virtual environment ready.

:: =========================================================
:: 4. UPDATE PIP
:: =========================================================
echo.
echo [4/6] Updating package manager...
"%VENV_PY%" -m pip install --upgrade pip setuptools wheel --retries 5 --timeout 30
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
"%VENV_PY%" -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu --retries 5 --timeout 60
goto CHECK_TORCH

:TORCH_NVIDIA
echo [*] Installing PyTorch with CUDA 12.1 support...
"%VENV_PY%" -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --retries 5 --timeout 60
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
:: requirements.txt holds everything for all three modes (YOLO-World, LocateAnything, custom head models),
:: including the version limits that keep them compatible: ultralytics>=8.4.68, transformers==4.57.6.
echo [6/6] Installing Jelibox dependencies...
"%VENV_PY%" -m pip install -r "%~dp0requirements.txt" --retries 5 --timeout 30

if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Dependency installation failed.
    set FAILED=1
    goto END
)
echo [OK] Dependencies installed successfully.

:: CLIP powers the YOLO-World Label Assistant. Installed from a zip archive so
:: git is not required. Non-fatal: without it only YOLO-World is unavailable.
echo [*] Installing CLIP (required by YOLO-World)...
"%VENV_PY%" -m pip install ftfy regex tqdm https://github.com/ultralytics/CLIP/archive/refs/heads/main.zip --retries 5 --timeout 60
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
:: INSTALL_VCREDIST - run only the Visual C++ runtime installer elevated (Windows shows its permission
:: prompt) and wait for it. The rest of this installer keeps running as the normal user.
:: Returns 0 on success, 1 when it failed or permission was refused.
:: 1638 = a newer runtime is already installed, 3010 = installed, a restart is recommended.
:: =========================================================
:INSTALL_VCREDIST
powershell -NoProfile -Command "try { $p = Start-Process -FilePath $env:VCREDIST -ArgumentList '/install','/quiet','/norestart' -Verb RunAs -Wait -PassThru; exit $p.ExitCode } catch { exit 1223 }"
set "VC_RC=!ERRORLEVEL!"
if "!VC_RC!"=="0" exit /b 0
if "!VC_RC!"=="1638" exit /b 0
if "!VC_RC!"=="3010" (
    echo [NOTICE] The Visual C++ runtime was installed. Restart Windows if Jelibox reports a missing DLL.
    exit /b 0
)
if "!VC_RC!"=="1223" (
    echo [ERROR] Administrator permission was not granted, so the Visual C++ runtime was not installed.
) else (
    echo [ERROR] The Visual C++ runtime installer failed ^(code !VC_RC!^).
)
exit /b 1
