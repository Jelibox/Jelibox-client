#!/bin/bash
set -Eeuo pipefail

# ==========================================
#    Jelibox Universal Installer
# ==========================================

echo "=========================================="
echo "  Welcome to Jelibox, Local annotation tool"
echo "        Thanks for choosing us"
echo "=========================================="
echo "System is preparing your environment..."
echo "=========================================="
sleep 1

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
APP_PATH="$SCRIPT_DIR"
PYTHON_VERSION_PREFIX="3.12"
VENV_DIR="$APP_PATH/jelibox"
VENV_PY="$VENV_DIR/bin/python"

# ──────────────────────────────────────────
# PYTHON
# Jelibox gets its own Python 3.12, downloaded by uv into this folder (uv itself is installed with its
# official installer if it is missing). Nothing is installed system-wide, no sudo is needed, Python is
# never put on your PATH and the Python you already have is left alone.
# ──────────────────────────────────────────
echo "[*] Preparing Jelibox's private Python $PYTHON_VERSION_PREFIX (no sudo needed)..."
if ! bash "$SCRIPT_DIR/tools/setup_python.sh" "$APP_PATH" "$(basename "$VENV_DIR")"; then
    echo "[ERROR] Could not set up Jelibox's private Python. Check your internet connection and try again."
    exit 1
fi

# The environment is never "activated": every command below calls its python directly, so nothing here
# can change which Python the rest of your system finds on PATH.
"$VENV_PY" -c 'import tkinter' || {
    echo "[ERROR] Tkinter is unavailable in Jelibox's Python."
    exit 1
}

# ──────────────────────────────────────────
# INSTALL PYTHON PACKAGES
# ──────────────────────────────────────────
echo "[*] Upgrading pip..."
"$VENV_PY" -m pip install --upgrade pip setuptools wheel
"$VENV_PY" -m pip install pycocotools
MACHINE_ARCH=$(uname -m)

# ──────────────────────────────────────────
# DISCRETE NVIDIA GPU OR NOT?
# Decides which PyTorch is downloaded: the CUDA build (with its NVIDIA libraries, several GB) or the much
# smaller CPU build. install.sh asks this itself and passes the answer in JELIBOX_GPU (nvidia or cpu);
# started on its own, this script asks. The detection is only the suggestion. On ARM nothing is asked.
# ──────────────────────────────────────────
GPU_TYPE="${JELIBOX_GPU:-}"
GPU_TYPE="$(printf '%s' "$GPU_TYPE" | tr '[:upper:]' '[:lower:]')"
case "$GPU_TYPE" in
    ""|nvidia|cpu) ;;
    *) echo "[ERROR] JELIBOX_GPU must be 'nvidia' or 'cpu', not '$GPU_TYPE'."; exit 1 ;;
esac
if [[ "$MACHINE_ARCH" != "aarch64" && -z "$GPU_TYPE" ]]; then
    GPU_DETECTED=0
    if command -v nvidia-smi &> /dev/null && nvidia-smi -L &> /dev/null; then
        GPU_DETECTED=1
    elif command -v lspci &> /dev/null && lspci 2>/dev/null | grep -Ei 'vga|3d|display' | grep -qi nvidia; then
        GPU_DETECTED=1
    fi
    GPU_REPLY=""
    if [ -t 0 ]; then
        echo ""
        if [ "$GPU_DETECTED" = 1 ]; then echo "[*] Detected: an NVIDIA GPU."; else echo "[*] Detected: no NVIDIA GPU."; fi
        echo "    Yes = the CUDA build of PyTorch (about 2.5 GB, needs the NVIDIA driver)."
        echo "    No  = the much smaller CPU build. AMD and Intel graphics cannot use CUDA - answer No for them."
        read -r -p "Does this computer have a discrete NVIDIA graphics card? [Y/N, Enter = $([ "$GPU_DETECTED" = 1 ] && echo Y || echo N)]: " GPU_REPLY || GPU_REPLY=""
    else
        echo "[*] No keyboard available - using what was detected."
    fi
    case "$GPU_REPLY" in
        [Yy]|[Yy][Ee][Ss]) GPU_TYPE="nvidia" ;;
        [Nn]|[Nn][Oo])     GPU_TYPE="cpu" ;;
        *) if [ "$GPU_DETECTED" = 1 ]; then GPU_TYPE="nvidia"; else GPU_TYPE="cpu"; fi ;;
    esac
fi

if [[ "$MACHINE_ARCH" == "aarch64" ]]; then
    echo "[!] ARM detected - installing the standard PyTorch build"
    "$VENV_PY" -m pip install torch torchvision torchaudio

elif [[ "$GPU_TYPE" == "nvidia" ]]; then
    echo "[OK] PyTorch build: NVIDIA GPU (CUDA)"
    if ! command -v nvidia-smi &> /dev/null; then
        echo "[WARNING] nvidia-smi was not found. PyTorch needs the NVIDIA driver to use the GPU."
    fi
    "$VENV_PY" -m pip install torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu121

else
    # The default Linux wheel on PyPI is the CUDA build and pulls in several GB of NVIDIA libraries - not wanted here.
    echo "[OK] PyTorch build: CPU only"
    "$VENV_PY" -m pip install torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cpu
fi

# requirements.txt holds everything for all three modes (YOLO-World, LocateAnything, custom head models),
# including the version limits that keep them compatible: ultralytics>=8.4.68, transformers==4.57.6.
echo "[*] Installing application dependencies..."
"$VENV_PY" -m pip install -r "$APP_PATH/requirements.txt"

# CLIP powers the YOLO-World Label Assistant. Installed from a zip archive so
# git is not required. Non-fatal: without it only YOLO-World is unavailable.
echo "[*] Installing CLIP (required by YOLO-World)..."
"$VENV_PY" -m pip install ftfy regex tqdm https://github.com/ultralytics/CLIP/archive/refs/heads/main.zip ||     echo "[WARNING] CLIP installation failed. YOLO-World Label Assistant will be unavailable."

# ──────────────────────────────────────────
# SYSTEM LIBRARIES (the only step that can need sudo)
# OpenCV needs the system graphics library (libGL) and GLib. They are normally present on a desktop. When
# they are missing this is the one thing the installer cannot do without administrator rights: it asks
# for your sudo password (or tells you the exact command when sudo is not available).
# ──────────────────────────────────────────
if ! "$VENV_PY" -c "import cv2" 2>/dev/null; then
    echo "[*] OpenCV could not load - a system graphics library is missing."
    SYSTEM_PACKAGES=""
    INSTALL_CMD=""
    OS_ID=""
    if [ -f /etc/os-release ]; then
        OS_ID="$(. /etc/os-release && echo "${ID:-} ${ID_LIKE:-}")"
    fi
    case " $OS_ID " in
        *" debian "*|*" ubuntu "*|*" linuxmint "*) SYSTEM_PACKAGES="libgl1 libglib2.0-0"; INSTALL_CMD="apt-get install -y" ;;
        *" fedora "*|*" rhel "*|*" centos "*)      SYSTEM_PACKAGES="mesa-libGL glib2";     INSTALL_CMD="dnf install -y" ;;
        *" arch "*|*" manjaro "*)                  SYSTEM_PACKAGES="mesa glib2";           INSTALL_CMD="pacman -S --noconfirm --needed" ;;
    esac
    if [ -z "$INSTALL_CMD" ]; then
        echo "[WARNING] Unknown Linux distribution. Install your distribution's libGL and GLib packages, then start Jelibox."
    else
        SUDO=""
        if [ "$(id -u)" != "0" ]; then SUDO="sudo"; fi
        if [ -n "$SUDO" ] && ! command -v sudo >/dev/null 2>&1; then
            echo "[WARNING] sudo is not available. As an administrator run:  $INSTALL_CMD $SYSTEM_PACKAGES"
        else
            echo "[*] Installing $SYSTEM_PACKAGES needs administrator permission."
            if [ -n "$SUDO" ]; then sudo -v || echo "[WARNING] Sudo authentication failed."; fi
            if { [ -z "$SUDO" ] || sudo -n true 2>/dev/null; } && $SUDO $INSTALL_CMD $SYSTEM_PACKAGES; then
                echo "[OK] System libraries installed."
            else
                echo "[WARNING] Could not install them. As an administrator run:  $INSTALL_CMD $SYSTEM_PACKAGES"
            fi
        fi
    fi
    "$VENV_PY" -c "import cv2" 2>/dev/null || echo "[WARNING] OpenCV still cannot load - Jelibox will not start until that is fixed."
fi

# ──────────────────────────────────────────
# CREATE DESKTOP ENTRY
# ──────────────────────────────────────────
LAUNCHER_PATH="$APP_PATH/Jelibox-launcher.sh"
DESKTOP_FILE="$APP_PATH/Jelibox.desktop"

cat <<EOF > "$LAUNCHER_PATH"
#!/bin/bash
cd "$APP_PATH"
exec "$VENV_DIR/bin/python" -u "$APP_PATH/utils/Annotator.py"
EOF

chmod +x "$LAUNCHER_PATH"

cat <<EOF > "$DESKTOP_FILE"
[Desktop Entry]
Name=Jelibox
Comment=Annotate dataset with Jelibox
Exec=$LAUNCHER_PATH
Icon=$APP_PATH/assets/jelibox.png
Type=Application
Path=$APP_PATH
Terminal=true
Categories=Development;
EOF

chmod +x "$DESKTOP_FILE"

USER_APP_DIR="$HOME/.local/share/applications"
mkdir -p "$USER_APP_DIR"
cp "$DESKTOP_FILE" "$USER_APP_DIR/Jelibox.desktop"

if command -v update-desktop-database &> /dev/null; then
    update-desktop-database "$USER_APP_DIR" 2>/dev/null || true
fi
echo "[OK] Jelibox added to your Application Menu (App Drawer)."

DESKTOP_DIR="$(xdg-user-dir DESKTOP 2>/dev/null || true)"
if [[ -n "$DESKTOP_DIR" && -d "$DESKTOP_DIR" ]]; then
    cp "$DESKTOP_FILE" "$DESKTOP_DIR/Jelibox.desktop"
    chmod +x "$DESKTOP_DIR/Jelibox.desktop"
    if command -v gio &> /dev/null; then
        gio set "$DESKTOP_DIR/Jelibox.desktop" metadata::trusted true 2>/dev/null || true
    fi
    echo "[OK] Desktop shortcut created at $DESKTOP_DIR/Jelibox.desktop"
fi

# ──────────────────────────────────────────
# DONE
# ──────────────────────────────────────────
# Tells the one-line installer (install.sh) that this install finished, so it can
# say "already up to date" next time instead of repeating the whole setup.
date > "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.jelibox-ready" 2>/dev/null || true

echo ""
echo "=========================================="
echo "      INSTALLATION COMPLETED!"
echo "=========================================="
echo "You can now run Jelibox from your Application Menu"
echo "or via the Desktop shortcut!"
echo "=========================================="
