#!/usr/bin/env bash
# Jelibox one-line installer for Linux.
#
#   curl -fsSL https://raw.githubusercontent.com/Jelibox/Jelibox-client/main/install.sh | bash
#
# What it does (read it - it is short):
#   1. looks for an existing Jelibox in ~/.local/share/jelibox
#   2. finds the newest Jelibox release on GitHub (falls back to the main branch)
#   3. not installed yet   -> installs the newest version fresh
#      installed, older    -> updates it (your datasets, models and configs are never touched)
#      installed, current  -> says so and stops
#   4. runs jelibox_linux_installation.bash, which installs Python 3.12 + Tkinter (it asks for your
#      sudo password), creates the virtual environment, installs the dependencies and adds
#      Jelibox to your application menu.
#
# Optional environment variables (set them before the command):
#   JELIBOX_VERSION     install a specific release, e.g.  v0.1.0   (default: newest release)
#   JELIBOX_HOME        install somewhere else               (default: ~/.local/share/jelibox)
#   JELIBOX_NO_INSTALL  1 = only download and unpack, do not run the installer
#   JELIBOX_FORCE       1 = reinstall even when this version is already installed
#   JELIBOX_ARCHIVE     path to a local .tar.gz instead of downloading (offline installs, tests)

# Everything lives in one function so bash has read the whole script before running any of it.
# (When piped into bash, a command that reads stdin could otherwise swallow the rest of the script.)
main() {
    set -euo pipefail

    local repo="Jelibox/Jelibox-client"
    local dest="${JELIBOX_HOME:-$HOME/.local/share/jelibox}"
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
        if [ -n "$version" ]; then
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
        if [ -n "$version" ] && [ "$installed" = "$version" ] && [ -f "$dest/.jelibox-ready" ] \
                && [ "${JELIBOX_FORCE:-}" != "1" ]; then
            say "Jelibox $version is already installed and up to date ($dest)."
            say "Nothing to do. (JELIBOX_FORCE=1 reinstalls it anyway.)"
            return 0
        fi
        say "Found Jelibox ${installed:-(unknown version)} in $dest - updating to $target."
    else
        say "Jelibox is not installed yet ($dest) - installing $target."
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

    say "Setting up Python and the dependencies. It will ask for your sudo password."
    if [ -r /dev/tty ]; then
        bash "$installer" < /dev/tty      # keep the keyboard available for the sudo prompt
    else
        bash "$installer"
    fi
}

main "$@"
