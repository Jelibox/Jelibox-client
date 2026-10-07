#!/usr/bin/env bash
# Gives Jelibox its own Python 3.12 + virtual environment - without touching the Python you already have.
#
#   bash tools/setup_python.sh [install folder] [venv folder name]
#
# What it does:
#   * a working environment already exists  -> keeps it (nothing is downloaded, nothing is deleted)
#   * otherwise it downloads `uv` (one small program, checksum-verified) into  <folder>/.uv,
#     lets uv fetch a standalone Python into  <folder>/.python,  and builds  <folder>/<venv>  from it.
#   Nothing is installed system-wide: no sudo, no apt/dnf/pacman, no PATH changes.
#   Uninstalling Jelibox = deleting its folder.
#
# Exit code 0 = the environment is ready, anything else = it could not be set up (the installer then
# falls back to a system-wide Python).
#
# Optional environment variables:
#   JELIBOX_PYTHON_VERSION  exact Python to use              (default: 3.12.10)
#   JELIBOX_UV_VERSION      uv release to download           (default: 0.12.23)
#   JELIBOX_UV              path to an existing uv to use instead of downloading one

main() {
    set -euo pipefail

    local here root venv_name py_version uv_version venv_dir venv_py
    here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
    root="${1:-$(dirname "$here")}"
    venv_name="${2:-jelibox}"
    py_version="${JELIBOX_PYTHON_VERSION:-3.12.10}"
    uv_version="${JELIBOX_UV_VERSION:-0.12.23}"
    venv_dir="$root/$venv_name"
    venv_py="$venv_dir/bin/python"

    say() { printf '[*] %s\n' "$*"; }
    oops() { printf '[!] %s\n' "$*" >&2; exit 1; }

    # Prints "3.12" when the interpreter runs and has a working Tkinter; prints nothing otherwise.
    python_version() {
        [ -x "$1" ] || return 0
        "$1" -c 'import sys, tkinter; tkinter.Tcl(); print("%d.%d" % sys.version_info[:2])' 2>/dev/null || true
    }

    download() {   # download <url> <output file>
        if command -v curl >/dev/null 2>&1; then
            curl -fsSL "$1" -o "$2"
        elif command -v wget >/dev/null 2>&1; then
            wget -q -O "$2" "$1"
        else
            oops "Neither 'curl' nor 'wget' is installed."
        fi
    }

    sha256_of() {
        if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1
        elif command -v shasum >/dev/null 2>&1; then shasum -a 256 "$1" | cut -d' ' -f1
        else oops "No sha256sum/shasum available to verify the download."; fi
    }

    find_uv() {
        if [ -n "${JELIBOX_UV:-}" ]; then
            [ -x "$JELIBOX_UV" ] || oops "JELIBOX_UV points to something that is not executable: $JELIBOX_UV"
            echo "$JELIBOX_UV"; return 0
        fi
        if command -v uv >/dev/null 2>&1; then
            say "Using the uv that is already installed ($(command -v uv))." >&2
            command -v uv; return 0
        fi
        if [ -x "$root/.uv/uv" ]; then echo "$root/.uv/uv"; return 0; fi

        local target name base tmp expected actual bin
        case "$(uname -m)" in
            x86_64|amd64)  target="x86_64-unknown-linux-gnu" ;;
            aarch64|arm64) target="aarch64-unknown-linux-gnu" ;;
            *) oops "No uv download for this CPU ($(uname -m))." ;;
        esac
        name="uv-$target.tar.gz"
        base="https://github.com/astral-sh/uv/releases/download/$uv_version"
        command -v tar >/dev/null 2>&1 || oops "'tar' is required but was not found."

        tmp="$(mktemp -d)"
        say "Downloading uv $uv_version (the tool that fetches Python) ..." >&2
        download "$base/$name" "$tmp/$name" || { rm -rf "$tmp"; oops "Could not download uv."; }
        download "$base/$name.sha256" "$tmp/$name.sha256" || { rm -rf "$tmp"; oops "Could not download the uv checksum."; }
        expected="$(cut -d' ' -f1 < "$tmp/$name.sha256" | head -n 1)"
        actual="$(sha256_of "$tmp/$name")"
        if [ "$expected" != "$actual" ]; then rm -rf "$tmp"; oops "The uv download is corrupted (checksum mismatch)."; fi

        mkdir -p "$tmp/x"
        tar -xzf "$tmp/$name" -C "$tmp/x"
        bin="$(find "$tmp/x" -type f -name uv | head -n 1)"
        [ -n "$bin" ] || { rm -rf "$tmp"; oops "uv was not found inside the download."; }
        mkdir -p "$root/.uv"
        cp "$bin" "$root/.uv/uv"
        chmod +x "$root/.uv/uv"
        rm -rf "$tmp"
        echo "$root/.uv/uv"
    }

    local found
    found="$(python_version "$venv_py")"
    if [ -n "$found" ]; then
        if [ "$found" = "3.12" ]; then say "Python environment is ready (Python $found)."
        else say "Existing environment uses Python $found - keeping it."; fi
        return 0
    fi
    if [ -e "$venv_dir" ]; then
        say "The existing Python environment is broken - rebuilding it."
        rm -rf "$venv_dir"
    fi

    local uv
    uv="$(find_uv)"
    # Keep everything inside the Jelibox folder, whatever uv settings this machine has.
    export UV_PYTHON_INSTALL_DIR="$root/.python"

    say "Getting a private Python $py_version (your own Python is not touched) ..."
    "$uv" python install "$py_version" --no-bin --no-config || oops "uv could not download Python $py_version."

    say "Creating the virtual environment ..."
    "$uv" venv "$venv_dir" --seed --python "$py_version" --managed-python --no-config \
        || oops "uv could not create the virtual environment."

    found="$(python_version "$venv_py")"
    [ -n "$found" ] || oops "The new environment does not run, or its Tkinter is missing."
    say "Python $found is ready."
}

main "$@"
