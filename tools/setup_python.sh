#!/usr/bin/env bash
# Gives Jelibox its own Python 3.12 + virtual environment - without touching the Python you already have.
#
#   bash tools/setup_python.sh [install folder] [venv folder name]
#
# What it does:
#   * a working environment already exists  -> keeps it (nothing is downloaded, nothing is deleted)
#   * otherwise it makes sure `uv` is installed (the official installer from astral.sh runs only when uv is
#     missing), lets uv fetch a standalone Python into  <folder>/.python,  and builds  <folder>/<venv>  from it.
#   No sudo is needed. Python itself is never put on your PATH and never installed outside the Jelibox folder.
#   The only thing outside the folder is uv: the official uv installer puts it in ~/.local/bin and adds that
#   folder to your shell profile's PATH.
#   Uninstalling Jelibox = deleting its folder (and uv, if you do not want it, see https://docs.astral.sh/uv/).
#
# Exit code 0 = the environment is ready, anything else = it could not be set up.
#
# Optional environment variables:
#   JELIBOX_PYTHON_VERSION  exact Python to use              (default: 3.12.10)
#   JELIBOX_UV              path to an existing uv to use instead of looking for / installing one

main() {
    set -euo pipefail

    local here root venv_name py_version venv_dir venv_py
    here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
    root="${1:-$(dirname "$here")}"
    venv_name="${2:-jelibox}"
    py_version="${JELIBOX_PYTHON_VERSION:-3.12.10}"
    venv_dir="$root/$venv_name"
    venv_py="$venv_dir/bin/python"

    say() { printf '[*] %s\n' "$*"; }
    oops() { printf '[!] %s\n' "$*" >&2; exit 1; }

    # Prints "3.12" when the interpreter runs and has a working Tkinter; prints nothing otherwise.
    python_version() {
        [ -x "$1" ] || return 0
        "$1" -c 'import sys, tkinter; tkinter.Tcl(); print("%d.%d" % sys.version_info[:2])' 2>/dev/null || true
    }

    # Path of an installed uv, or nothing. Looks on PATH first, then where the official installer puts it
    # (it may have just been installed, and this process's PATH does not know about it yet).
    find_uv() {
        if [ -n "${JELIBOX_UV:-}" ]; then
            [ -x "$JELIBOX_UV" ] || oops "JELIBOX_UV points to something that is not executable: $JELIBOX_UV"
            echo "$JELIBOX_UV"; return 0
        fi
        if command -v uv >/dev/null 2>&1; then command -v uv; return 0; fi
        local dir
        for dir in "${UV_INSTALL_DIR:-}" "${XDG_BIN_HOME:-}" "$HOME/.local/bin" "$HOME/.cargo/bin"; do
            if [ -n "$dir" ] && [ -x "$dir/uv" ]; then echo "$dir/uv"; return 0; fi
        done
        return 0
    }

    get_uv() {
        local uv
        uv="$(find_uv)"
        if [ -n "$uv" ]; then say "Using uv ($uv)." >&2; echo "$uv"; return 0; fi

        case "$(uname -s)" in
            Linux|Darwin) ;;
            *) oops "uv cannot be installed automatically on $(uname -s). Install it from https://docs.astral.sh/uv/ and run this again." ;;
        esac
        say "uv is not installed - installing it with the official installer (https://astral.sh/uv) ..." >&2
        # the installer's own output goes to stderr so that only the path below ends up on stdout
        if command -v curl >/dev/null 2>&1; then
            curl -LsSf https://astral.sh/uv/install.sh | sh >&2 || oops "The uv installer failed."
        elif command -v wget >/dev/null 2>&1; then
            wget -qO- https://astral.sh/uv/install.sh | sh >&2 || oops "The uv installer failed."
        else
            oops "Neither 'curl' nor 'wget' is installed, so uv cannot be downloaded."
        fi
        uv="$(find_uv)"
        [ -n "$uv" ] || oops "uv was installed but could not be found afterwards."
        echo "$uv"
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
    uv="$(get_uv)"
    # Keep Python inside the Jelibox folder, whatever uv settings this machine has.
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
