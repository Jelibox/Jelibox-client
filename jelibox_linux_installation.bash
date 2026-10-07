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
PYTHON_BIN="python3.12"
VENV_DIR="$APP_PATH/jelibox"

# ──────────────────────────────────────────
# SYSTEM-WIDE PYTHON (fallback)
# Only used when JELIBOX_PYTHON=system is set, or when the private Python below could not be set up.
# ──────────────────────────────────────────
use_system_python() {
    # Ask for the sudo password before changing the system.
    echo "[*] Sudo access is required to install system packages."
    sudo -v || {
        echo "[ERROR] Sudo authentication failed."
        exit 1
    }

    # DETECT OS
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        OS=$ID
    else
        echo "[ERROR] Unsupported OS"
        exit 1
    fi

    echo "[*] Detecting OS: $OS"

    # DEBIAN / UBUNTU / MINT
    if [[ "$OS" == "ubuntu" || "$OS" == "debian" || "$OS" == "linuxmint" ]]; then
        echo "[*] Installing Python $PYTHON_VERSION_PREFIX and Tkinter packages..."
        sudo apt update
        sudo apt install -y software-properties-common

        if [[ "$OS" == "ubuntu" || "$OS" == "linuxmint" ]]; then
            sudo add-apt-repository -y ppa:deadsnakes/ppa
            sudo apt update
        fi

        sudo apt install -y \
            python3.12 \
            python3.12-venv \
            python3.12-dev \
            python3.12-tk
    fi

    # ARCH / MANJARO
    if [[ "$OS" == "arch" || "$OS" == "manjaro" ]]; then
        echo "[*] Installing Python $PYTHON_VERSION_PREFIX, Tkinter, and virtual environment dependencies..."
        # Force python312 so every machine matches; if it is missing from the official repo it may need the AUR (yay -S python312)
        sudo pacman -S --noconfirm python312 tk mesa libcanberra || {
            echo "[!] Failed to install python312 via pacman. Make sure python312 is available or use the AUR (for example: yay -S python312)"
            exit 1
        }
    fi

    # FEDORA / RHEL / CENTOS
    if [[ "$OS" == "fedora" || "$OS" == "rhel" || "$OS" == "centos" ]]; then
        echo "[*] Installing Python $PYTHON_VERSION_PREFIX and Tkinter packages..."
        sudo dnf install -y \
            python3.12 \
            python3.12-devel \
            python3.12-tkinter
        sudo dnf install -y mesa-libGL libglvnd-glx
    fi

    if ! command -v "$PYTHON_BIN" &> /dev/null; then
        echo "[ERROR] Python interpreter $PYTHON_BIN was not found."
        exit 1
    fi

    PYTHON_ACTUAL_VERSION="$($PYTHON_BIN -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')"
    echo "[OK] Python $PYTHON_ACTUAL_VERSION detected."

    # Check that the Python version starts with "3.12"
    if [[ "$PYTHON_ACTUAL_VERSION" != ${PYTHON_VERSION_PREFIX}.* ]]; then
        echo "[ERROR] Jelibox requires Python $PYTHON_VERSION_PREFIX.x exactly."
        echo "[ERROR] The detected interpreter is $PYTHON_ACTUAL_VERSION."
        exit 1
    fi

    # CREATE VENV
    echo "[*] Creating Virtual Environment with $PYTHON_BIN..."

    $PYTHON_BIN -m venv "$VENV_DIR" || {
        echo "[ERROR] Failed creating venv"
        exit 1
    }

    if [[ ! -x "$VENV_DIR/bin/python" ]]; then
        echo "[ERROR] Virtual environment was not created correctly."
        exit 1
    fi
}

# ──────────────────────────────────────────
# PYTHON
# By default Jelibox gets its own private Python 3.12 (downloaded with uv into this folder), so nothing
# is installed system-wide, no sudo is needed and the Python you already have is left alone.
# ──────────────────────────────────────────
USE_SYSTEM_PYTHON=0
if [[ "${JELIBOX_PYTHON:-}" == "system" ]]; then
    USE_SYSTEM_PYTHON=1
else
    echo "[*] Preparing Python $PYTHON_VERSION_PREFIX (private copy - no sudo needed)..."
    if ! bash "$SCRIPT_DIR/tools/setup_python.sh" "$APP_PATH" "$(basename "$VENV_DIR")"; then
        echo "[!] The private Python could not be set up - falling back to the system Python."
        USE_SYSTEM_PYTHON=1
    fi
fi
if [[ "$USE_SYSTEM_PYTHON" == "1" ]]; then
    use_system_python
fi

source "$VENV_DIR/bin/activate"

python -c 'import tkinter' || {
    echo "[ERROR] Tkinter is unavailable. Install the matching python3.12-tk / python3.12-tkinter package."
    exit 1
}

# ──────────────────────────────────────────
# INSTALL PYTHON PACKAGES
# ──────────────────────────────────────────
echo "[*] Upgrading pip..."
python -m pip install --upgrade pip setuptools wheel
echo "[*] Installing Streamlit..."
python -m pip install streamlit yt-dlp
python -m pip install pycocotools
MACHINE_ARCH=$(uname -m)

if [[ "$MACHINE_ARCH" == "aarch64" ]]; then
    echo "[!] ARM detected"
    pip install torch torchvision torchaudio

else
    if command -v nvidia-smi &> /dev/null; then
        echo "[OK] NVIDIA GPU detected"
        pip install torch torchvision torchaudio \
            --index-url https://download.pytorch.org/whl/cu121
    else
        echo "[!] Installing CPU version"
        pip install torch torchvision torchaudio
    fi
fi

echo "[*] Installing application dependencies..."
pip install ultralytics pyinstaller

# CLIP powers the YOLO-World Label Assistant. Installed from a zip archive so
# git is not required. Non-fatal: without it only YOLO-World is unavailable.
echo "[*] Installing CLIP (required by YOLO-World)..."
pip install ftfy regex tqdm https://github.com/ultralytics/CLIP/archive/refs/heads/main.zip ||     echo "[WARNING] CLIP installation failed. YOLO-World Label Assistant will be unavailable."

# OpenCV needs the system graphics library (libGL). The private Python does not install system packages, so say so
# instead of letting the app fail later with a confusing "libGL.so.1" error.
python -c "import cv2" 2>/dev/null || {
    echo "[WARNING] OpenCV could not load - it usually needs the system graphics library:"
    echo "          Debian/Ubuntu: sudo apt install libgl1    Fedora: sudo dnf install mesa-libGL    Arch: sudo pacman -S mesa"
}

# ──────────────────────────────────────────
# CREATE DESKTOP ENTRY
# ──────────────────────────────────────────
LAUNCHER_PATH="$APP_PATH/Jelibox-launcher.sh"
DESKTOP_FILE="$APP_PATH/Jelibox.desktop"

cat <<EOF > "$LAUNCHER_PATH"
#!/bin/bash
cd "$APP_PATH"
source "$VENV_DIR/bin/activate"
exec python -u "$APP_PATH/utils/Annotator.py"
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
