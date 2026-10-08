#!/usr/bin/env bash
# Jelibox one-line installer for Linux.
#
#   curl -fsSL https://raw.githubusercontent.com/Jelibox/Jelibox-client/main/install.sh | bash
#
# What it does (read it - it is short):
#   1. asks where to install (Enter = ~/jelibox, B = browse) and looks for an existing Jelibox there
#   2. finds the newest Jelibox release on GitHub (falls back to the main branch)
#   3. not installed yet   -> installs the newest version fresh
#      installed, older    -> updates it (your datasets, models and configs are never touched)
#      installed, current  -> says so and stops
#   4. runs jelibox_linux_installation.bash, which installs uv if it is missing (official installer, user level),
#      gives Jelibox its own private Python 3.12 (downloaded by uv into the Jelibox folder - your own Python is
#      not touched, nothing is put on PATH), creates the virtual environment, installs the dependencies and adds
#      Jelibox to your application menu. sudo is asked for only if the system graphics libraries OpenCV needs
#      (libGL, GLib) are missing.
#
# Optional environment variables (set them before the command):
#   JELIBOX_VERSION     install a specific release, e.g.  v0.1.0   (default: newest release)
#                       main = the latest development version (always re-installed)
#   JELIBOX_HOME        install here without being asked     (default: you are asked; Enter = ~/jelibox)
#   JELIBOX_NO_INSTALL  1 = only download and unpack, do not run the installer
#   JELIBOX_FORCE       1 = reinstall even when this version is already installed
#   JELIBOX_GPU         nvidia | cpu = which PyTorch to install without being asked (default: you are asked)
#   JELIBOX_ARCHIVE     path to a local .tar.gz instead of downloading (offline installs, tests)

# Everything lives in one function so bash has read the whole script before running any of it.
# (When piped into bash, a command that reads stdin could otherwise swallow the rest of the script.)
main() {
    set -euo pipefail

    local repo="Jelibox/Jelibox-client"
    local dest="${JELIBOX_HOME:-}"
    local version="${JELIBOX_VERSION:-}"
    local archive="${JELIBOX_ARCHIVE:-}"
    local installed="" existing=0 url=""

    say()  { printf '\033[1;36m[Jelibox]\033[0m %s\n' "$*"; }
    fail() { printf '\033[1;31m[Jelibox] %s\033[0m\n' "$*" >&2; exit 1; }

    command -v tar >/dev/null 2>&1 || fail "'tar' is required but was not found."

    download() {   # download <url> <output file>
        if command -v curl >/dev/null 2>&1; then
            curl -fsSL "$1" -o "$2"
        elif command -v wget >/dev/null 2>&1; then
            wget -q -O "$2" "$1"
        else
            fail "Neither 'curl' nor 'wget' is installed. Install one of them (for example: sudo apt install curl) and run this again."
        fi
    }

    # Where to install: JELIBOX_HOME if set, otherwise ask on the terminal (stdin is the piped script, so read /dev/tty).
    # Enter = the default (a "jelibox" folder in your home folder), B = pick a folder in a window (needs zenity or kdialog),
    # or type a path. A "jelibox" folder is created inside the chosen folder (a path already ending in jelibox is used as-is).
    if [ -z "$dest" ]; then
        # Earlier versions of this installer always used ~/.local/share/jelibox - keep updating that one instead of asking.
        local legacy="$HOME/.local/share/jelibox"
        if [ -f "$legacy/.jelibox-version" ] || [ -f "$legacy/utils/AnnotationGUI.py" ]; then
            dest="$legacy"
            say "Found an existing Jelibox in $dest - it will be updated there. (To relocate it, use Move Jelibox inside the app.)"
        fi
    fi
    if [ -z "$dest" ]; then
        local suggested="$HOME/jelibox" answer=""
        if [ -r /dev/tty ] && [ -w /dev/tty ]; then
            {
                printf '\n\033[1;36m[Jelibox]\033[0m Where should Jelibox be installed?\n'
                printf '          [Enter]  default: %s\n' "$suggested"
                printf '          [B]      browse - pick a folder in a window\n'
                printf '          or type a folder path\n'
                printf '          Your choice: '
            } > /dev/tty
            read -r answer < /dev/tty || answer=""
            if [ "$answer" = "b" ] || [ "$answer" = "B" ]; then
                answer=""
                if [ -n "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ] && command -v zenity >/dev/null 2>&1; then
                    answer="$(zenity --file-selection --directory --title="Where should Jelibox be installed?" --filename="$HOME/" 2>/dev/null || true)"
                elif [ -n "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ] && command -v kdialog >/dev/null 2>&1; then
                    answer="$(kdialog --getexistingdirectory "$HOME" 2>/dev/null || true)"
                else
                    say "No folder window available (install zenity or kdialog) - using the default location." > /dev/tty
                fi
                [ -n "$answer" ] || say "No folder chosen - using the default location." > /dev/tty
            fi
        fi
        if [ -n "$answer" ]; then
            case "$answer" in
                "~"|"~/"*) answer="$HOME${answer#"~"}" ;;
                /*) ;;
                *) answer="$PWD/$answer" ;;
            esac
            answer="${answer%/}"
            if [ "$(basename "$answer" | tr '[:upper:]' '[:lower:]')" = "jelibox" ]; then
                dest="$answer"
            else
                dest="$answer/jelibox"
            fi
        else
            dest="$suggested"
        fi
    fi

    # Is Jelibox already here? (.jelibox-version is written by this script; older installs only have the code.)
    if [ -d "$dest/.git" ]; then
        say "$dest is a git checkout, so it is updated with git rather than this installer:"
        say "    cd \"$dest\" && git pull"
        return 0
    fi
    if [ -f "$dest/.jelibox-version" ] || [ -f "$dest/utils/AnnotationGUI.py" ]; then
        existing=1
        installed="$(head -n 1 "$dest/.jelibox-version" 2>/dev/null | tr -d '\r' || true)"
    fi

    local tmp
    tmp="$(mktemp -d)"
    trap "rm -rf '$tmp'" EXIT     # expanded now: tmp is a local variable and is gone when the trap fires

    if [ -n "$archive" ]; then
        [ -f "$archive" ] || fail "JELIBOX_ARCHIVE not found: $archive"
        say "Using local archive $archive"
    else
        if [ -z "$version" ]; then
            if download "https://api.github.com/repos/$repo/releases/latest" "$tmp/latest.json" 2>/dev/null; then
                version="$(sed -n 's/.*"tag_name"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' "$tmp/latest.json" | head -n 1)"
            fi
            [ -n "$version" ] || say "No release found yet - using the latest development version (main)."
        fi
        if [ "$version" = "main" ]; then
            url="https://github.com/$repo/archive/refs/heads/main.tar.gz"
            say "Downloading Jelibox (main) ..."
        elif [ -n "$version" ]; then
            url="https://github.com/$repo/archive/refs/tags/$version.tar.gz"
            say "Downloading Jelibox $version ..."
        else
            url="https://github.com/$repo/archive/refs/heads/main.tar.gz"
            say "Downloading Jelibox (main) ..."
        fi
        archive="$tmp/jelibox.tar.gz"
    fi

    # Fresh install, update, or nothing to do?
    local target="${version:-main}"
    [ -z "${JELIBOX_ARCHIVE:-}" ] || target="${version:-local archive}"
    if [ "$existing" = "1" ]; then
        if [ -n "$version" ] && [ "$version" != "main" ] && [ "$installed" = "$version" ] && [ -f "$dest/.jelibox-ready" ] \
                && [ "${JELIBOX_FORCE:-}" != "1" ]; then
            say "Jelibox $version is already installed and up to date ($dest)."
            say "Nothing to do. (JELIBOX_FORCE=1 reinstalls it anyway.)"
            return 0
        fi
        say "Found Jelibox ${installed:-(unknown version)} in $dest - updating to $target."
    else
        say "Jelibox is not installed yet ($dest) - installing $target."
    fi

    # Discrete NVIDIA GPU or not? This decides which PyTorch is downloaded (CUDA build or the smaller CPU build).
    # JELIBOX_GPU=nvidia|cpu skips the question; the installer script asks by itself if it is started without it.
    if [ "${JELIBOX_NO_INSTALL:-}" != "1" ]; then
        local gpu="${JELIBOX_GPU:-}"
        gpu="$(printf '%s' "$gpu" | tr '[:upper:]' '[:lower:]')"
        case "$gpu" in
            ""|nvidia|cpu) ;;
            *) fail "JELIBOX_GPU must be 'nvidia' or 'cpu' (got '$gpu')." ;;
        esac
        if [ -z "$gpu" ]; then
            if [ "$(uname -m)" = "aarch64" ]; then
                gpu="cpu"
                say "ARM: the standard PyTorch build is used (no separate CUDA choice)."
            else
                local detected=0 reply=""
                if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
                    detected=1
                elif command -v lspci >/dev/null 2>&1 && lspci 2>/dev/null | grep -Ei 'vga|3d|display' | grep -qi nvidia; then
                    detected=1
                fi
                if [ -r /dev/tty ] && [ -w /dev/tty ]; then
                    {
                        printf '\n\033[1;36m[Jelibox]\033[0m Does this computer have a discrete NVIDIA graphics card (GeForce / RTX / GTX / Quadro)?\n'
                        printf '          Yes = the CUDA build of PyTorch (about 2.5 GB, needs the NVIDIA driver). No = the much smaller CPU build.\n'
                        printf '          AMD and Intel graphics cannot use CUDA - answer No for them.\n'
                        printf '          Detected: %s\n' "$([ "$detected" = 1 ] && echo 'an NVIDIA GPU' || echo 'no NVIDIA GPU')"
                        printf '          NVIDIA GPU? [Y/N, Enter = %s]: ' "$([ "$detected" = 1 ] && echo Y || echo N)"
                    } > /dev/tty
                    read -r reply < /dev/tty || reply=""
                else
                    say "No keyboard available - using what was detected."
                fi
                case "$reply" in
                    [Yy]|[Yy][Ee][Ss]) gpu="nvidia" ;;
                    [Nn]|[Nn][Oo])     gpu="cpu" ;;
                    *) if [ "$detected" = 1 ]; then gpu="nvidia"; else gpu="cpu"; fi ;;
                esac
            fi
        fi
        export JELIBOX_GPU="$gpu"
        say "PyTorch build: $([ "$gpu" = "nvidia" ] && echo 'NVIDIA GPU (CUDA)' || echo 'CPU only')"
    fi

    if [ -z "${JELIBOX_ARCHIVE:-}" ]; then
        download "$url" "$archive"
    fi

    say "Unpacking ..."
    mkdir -p "$tmp/src"
    tar -xzf - -C "$tmp/src" < "$archive"
    local top
    top="$(find "$tmp/src" -mindepth 1 -maxdepth 1 -type d | head -n 1)"
    [ -n "$top" ] || fail "The downloaded archive is empty."

    mkdir -p "$dest"
    # Copy over the existing install. Nothing is deleted, so datasets/models/configs/venv survive an update.
    cp -a "$top/." "$dest/"
    printf '%s\n' "$target" > "$dest/.jelibox-version"
    rm -f "$dest/.jelibox-ready"          # set again by the installer below once everything worked
    say "Installed files are in $dest"

    if [ "${JELIBOX_NO_INSTALL:-}" = "1" ]; then
        say "JELIBOX_NO_INSTALL=1 - stopping before the dependency installer."
        return 0
    fi

    local installer="$dest/jelibox_linux_installation.bash"
    [ -f "$installer" ] || fail "Installer script missing: $installer"
    chmod +x "$installer"

    say "Setting up Python and the dependencies (sudo is only asked for if system graphics libraries are missing)."
    if [ -r /dev/tty ]; then
        bash "$installer" < /dev/tty      # keep the keyboard available for the sudo prompt
    else
        bash "$installer"
    fi
}

main "$@"
