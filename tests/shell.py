"""Locate a bash that actually works.

On Windows, `bash` on PATH is often the WSL launcher (C:\\Windows\\System32\\bash.exe), which fails with
"Windows Subsystem for Linux has no installed distributions" when no distro is set up - as on GitHub's
Windows runners. Git for Windows ships a real bash, so look there too and verify by running it."""
import os
import shutil
import subprocess

_cache = {}


def _works(path):
    try:
        r = subprocess.run([path, "-c", "echo jelibox-ok"], capture_output=True, timeout=30)
        return r.returncode == 0 and b"jelibox-ok" in r.stdout
    except (OSError, subprocess.SubprocessError):
        return False


def find_bash():
    """Path of a working bash, or None."""
    if "bash" in _cache:
        return _cache["bash"]
    candidates = []
    if os.name == "nt":
        for base in (os.environ.get("ProgramFiles"), os.environ.get("ProgramFiles(x86)"), os.environ.get("LOCALAPPDATA")):
            if base:
                candidates.append(os.path.join(base, "Git", "bin", "bash.exe"))
                candidates.append(os.path.join(base, "Programs", "Git", "bin", "bash.exe"))
    which = shutil.which("bash")
    if which:
        candidates.append(which)
    _cache["bash"] = next((c for c in candidates if os.path.isfile(c) and _works(c)), None)
    return _cache["bash"]
