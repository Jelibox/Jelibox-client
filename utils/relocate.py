"""
Move the whole Jelibox install to a folder of the user's choice.

The picker's "Move Jelibox" button calls validate_target() and spawn_helper().
The helper is a copy of this very file, started with the *base* Python (not the
install's venv - a running venv's python.exe can't be deleted on Windows) once
the picker has exited. It:

  1. creates <chosen folder>/Jelibox
  2. builds a brand-new virtual environment there and reinstalls the exact same
     package versions the old one had (pip freeze -> pip install -r)
  3. moves everything else (datasets, models, configs, code, ...) across
  4. deletes the old venv and the old Jelibox folder
  5. recreates the launcher / desktop shortcuts and offers to start Jelibox

Order matters: the slow, failure-prone part (venv + pip) runs before anything is
moved, so a failure there leaves the old install completely untouched. A failure
while moving rolls the already-moved items back.

Standard library only - it has to run with a bare interpreter.
"""

import argparse
import os
import queue
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
import time

IS_WIN = os.name == "nt"
FOLDER_NAME = "Jelibox"
MARKER = os.path.join("utils", "Annotator.py")      # proves a folder really is a Jelibox install
CREATE_NO_WINDOW = 0x08000000
DETACHED_PROCESS = 0x00000008
CREATE_NEW_PROCESS_GROUP = 0x00000200
STEPS = ["Preparing", "Creating virtual environment", "Installing packages",
         "Moving files", "Removing the old install", "Creating shortcuts"]


class RelocateError(Exception):
    pass


# ----------------------------------------------------------------------
#  Pure helpers (also used by the picker and by the tests)
# ----------------------------------------------------------------------
def _real(path):
    return os.path.normcase(os.path.realpath(path))


def _is_inside(child, parent):
    c, p = _real(child), _real(parent)
    return c == p or c.startswith(p.rstrip(os.sep) + os.sep)


def target_for(parent_folder):
    return os.path.join(os.path.abspath(parent_folder), FOLDER_NAME)


def validate_target(src, parent_folder):
    """None if <parent_folder>/Jelibox is a usable destination, otherwise why not."""
    if not parent_folder or not os.path.isdir(parent_folder):
        return "That is not a folder."
    target = target_for(parent_folder)
    if _real(target) == _real(src):
        return "Jelibox is already in that location."
    if _is_inside(target, src):
        return "The new location can't be inside the current Jelibox folder."
    if _is_inside(src, target):
        return "The new location can't contain the current Jelibox folder."
    if os.path.exists(target) and (not os.path.isdir(target) or os.listdir(target)):
        return f'"{target}" already exists and is not empty.'
    try:
        probe = tempfile.mkdtemp(dir=parent_folder, prefix=".jelibox_write_test_")
        os.rmdir(probe)
    except OSError:
        return "Jelibox can't write to that folder - pick another one."
    return None


def default_venv_name():
    return "venv" if IS_WIN else "jelibox"


def venv_python(venv_dir):
    return os.path.join(venv_dir, "Scripts", "python.exe") if IS_WIN else os.path.join(venv_dir, "bin", "python")


def running_venv(root):
    """The venv this interpreter runs from, if it lives inside `root`."""
    if sys.prefix != sys.base_prefix and _is_inside(sys.prefix, root):
        return os.path.abspath(sys.prefix)
    return None


def base_python(gui=False):
    """The interpreter the venv was made from. gui=True prefers pythonw (no console window)."""
    exe = getattr(sys, "_base_executable", None) or sys.executable
    if IS_WIN and gui:
        w = os.path.join(os.path.dirname(exe), "pythonw.exe")
        if os.path.exists(w):
            return w
    return exe


def clean_requirements(freeze_output):
    """Keep only lines pip can reinstall from an index / URL (no editable or local-path installs)."""
    keep = []
    for line in freeze_output.splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "-e")) or "@ file:" in line:
            continue
        keep.append(line)
    return keep


def torch_index_url(requirements):
    """Local-version torch builds (torch==2.5.1+cu121) only exist on PyTorch's own index."""
    for line in requirements:
        m = re.match(r"(?i)torch(?:vision|audio)?==[\w.]+\+(cu\d+|cpu|rocm[\d.]+)", line)
        if m:
            return f"https://download.pytorch.org/whl/{m.group(1)}"
    return None


def move_tree(src, dest, skip_names, log):
    """Move every top-level entry of src into dest (except skip_names); roll back on failure."""
    moved = []
    current = None
    try:
        for name in sorted(os.listdir(src)):
            if name in skip_names:
                continue
            current = (os.path.join(src, name), os.path.join(dest, name))
            if os.path.lexists(current[1]):
                raise RelocateError(f'"{current[1]}" already exists in the new location.')
            log(f"  moving {name}")
            shutil.move(*current)
            moved.append(current)
            current = None
    except Exception as e:
        log(f"Move failed ({e}) - restoring what was already moved...")
        if current and os.path.lexists(current[0]) and os.path.lexists(current[1]):
            _rmtree(current[1])                 # half-copied leftover; the source is still intact
        for s, d in reversed(moved):
            shutil.move(d, s)
        raise RelocateError(f"Could not move files: {e}") from e
    return moved


def _rmtree(path):
    def on_error(func, p, exc_info):
        try:
            os.chmod(p, stat.S_IWRITE)
            func(p)
        except OSError:
            pass
    if sys.version_info >= (3, 12):
        shutil.rmtree(path, onexc=lambda f, p, e: on_error(f, p, e))
    else:
        shutil.rmtree(path, onerror=on_error)


def remove_tree(path, log, attempts=8, delay=1.5):
    """Delete a folder, retrying while Windows releases locks on files of the just-closed app."""
    for i in range(attempts):
        if not os.path.lexists(path):
            return True
        _rmtree(path)
        if not os.path.lexists(path):
            return True
        time.sleep(delay)
    log(f"Could not fully delete {path} - remove it by hand.")
    return False


# ----------------------------------------------------------------------
#  Starting the helper from the picker
# ----------------------------------------------------------------------
def spawn_helper(src, target, venv_dir):
    """Start the detached helper (a temp copy of this file). The caller must exit right after."""
    tmp = tempfile.mkdtemp(prefix="jelibox_move_")
    script = os.path.join(tmp, "relocate.py")
    shutil.copy2(os.path.abspath(__file__), script)
    cmd = [base_python(gui=True), script, "--src", src, "--dest", target,
           "--pid", str(os.getpid()), "--freeze-python", sys.executable]
    if venv_dir:
        cmd += ["--venv", venv_dir]
    env = {k: v for k, v in os.environ.items() if k not in ("JELIBOX_PARENT", "VIRTUAL_ENV")}
    kwargs = {"cwd": tmp, "env": env, "close_fds": True}
    if IS_WIN:
        kwargs["creationflags"] = DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    subprocess.Popen(cmd, **kwargs)


# ----------------------------------------------------------------------
#  The relocation itself (runs inside the helper)
# ----------------------------------------------------------------------
def _pid_alive(pid):
    if IS_WIN:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(0x1000, False, pid)     # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
            return code.value == 259                           # STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def wait_for_exit(pid, timeout=60):
    deadline = time.time() + timeout
    while pid and _pid_alive(pid) and time.time() < deadline:
        time.sleep(0.3)
    time.sleep(1.0)         # let the venv launcher process holding python.exe finish too


def run_command(cmd, log, cwd=None):
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PIP_DISABLE_PIP_VERSION_CHECK="1")
    kwargs = {"creationflags": CREATE_NO_WINDOW} if IS_WIN else {}
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            encoding="utf-8", errors="replace", cwd=cwd, env=env, **kwargs)
    for line in proc.stdout:
        line = line.rstrip()
        if line:
            log(line)
    proc.stdout.close()
    if proc.wait() != 0:
        raise RelocateError(f"Command failed ({proc.returncode}): {' '.join(cmd[:4])} ...")


def freeze(python, log):
    kwargs = {"creationflags": CREATE_NO_WINDOW} if IS_WIN else {}
    r = subprocess.run([python, "-m", "pip", "freeze"], capture_output=True, text=True,
                       encoding="utf-8", errors="replace", **kwargs)
    if r.returncode != 0:
        raise RelocateError(f"Could not read the installed packages: {r.stderr.strip()}")
    return clean_requirements(r.stdout)


def create_shortcuts(root, venv_dir, log):
    entry = os.path.join(root, "utils", "Annotator.py")
    icon = os.path.join(root, "assets", "jelibox.ico")
    py = venv_python(venv_dir)
    if IS_WIN:
        q = lambda s: "'" + s.replace("'", "''") + "'"
        lines = [
            f"$rootPath = {q(root)}", f"$pythonExe = {q(py)}", f"$scriptPath = {q(entry)}",
            f"$iconPath = {q(icon)}",
            "$desktop = [Environment]::GetFolderPath('Desktop')",
            "$rootShortcut = Join-Path $rootPath 'Jelibox Launcher.lnk'",
            "$desktopShortcut = Join-Path $desktop 'Jelibox.lnk'",
            "$ws = New-Object -ComObject WScript.Shell",
            "$sc1 = $ws.CreateShortcut($rootShortcut)",
            "$sc1.TargetPath = $pythonExe", "$sc1.Arguments = '\"' + $scriptPath + '\"'",
            "$sc1.WorkingDirectory = $rootPath", "$sc1.Description = 'Local Annotation Tool'",
            "if (Test-Path $iconPath) { $sc1.IconLocation = $iconPath }", "$sc1.Save()",
            "$sc2 = $ws.CreateShortcut($desktopShortcut)",
            "$sc2.TargetPath = $rootShortcut", "$sc2.Description = 'Local Annotation Tool'",
            "if (Test-Path $iconPath) { $sc2.IconLocation = $iconPath }", "$sc2.Save()",
        ]
        ps1 = os.path.join(tempfile.gettempdir(), "jelibox_move_shortcuts.ps1")
        with open(ps1, "w", encoding="utf-8-sig") as f:       # BOM: Windows PowerShell 5.1 reads UTF-8 only with it
            f.write("\n".join(lines) + "\n")
        try:
            run_command(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", ps1], log)
        finally:
            try:
                os.remove(ps1)
            except OSError:
                pass
        return
    launcher = os.path.join(root, "Jelibox-launcher.sh")
    desktop_file = os.path.join(root, "Jelibox.desktop")
    with open(launcher, "w", encoding="utf-8", newline="\n") as f:
        f.write(f'#!/bin/bash\ncd "{root}"\nsource "{venv_dir}/bin/activate"\nexec python -u "{entry}"\n')
    os.chmod(launcher, 0o755)
    with open(desktop_file, "w", encoding="utf-8", newline="\n") as f:
        f.write("[Desktop Entry]\nName=Jelibox\nComment=Annotate dataset with Jelibox\n"
                f"Exec={launcher}\nIcon={os.path.join(root, 'assets', 'jelibox.png')}\n"
                f"Type=Application\nPath={root}\nTerminal=true\nCategories=Development;\n")
    os.chmod(desktop_file, 0o755)
    apps = os.path.join(os.path.expanduser("~"), ".local", "share", "applications")
    os.makedirs(apps, exist_ok=True)
    shutil.copy2(desktop_file, os.path.join(apps, "Jelibox.desktop"))
    desktop_dir = os.path.join(os.path.expanduser("~"), "Desktop")
    if os.path.isdir(desktop_dir):
        shutil.copy2(desktop_file, os.path.join(desktop_dir, "Jelibox.desktop"))


def launch_jelibox(root, venv_dir):
    if IS_WIN:
        lnk = os.path.join(root, "Jelibox Launcher.lnk")
        if os.path.exists(lnk):
            os.startfile(lnk)       # same as double-clicking it
            return
    launcher = os.path.join(root, "Jelibox-launcher.sh")
    cmd = [launcher] if os.path.exists(launcher) else [venv_python(venv_dir), os.path.join(root, "utils", "Annotator.py")]
    subprocess.Popen(cmd, cwd=root, start_new_session=not IS_WIN)


def relocate(src, dest, venv_dir, freeze_python, wait_pid, log, step):
    """Run the whole move. Raises RelocateError (after cleaning up) if something fails.
    Returns (new venv dir, leftovers that could not be deleted)."""
    step(0)
    src = os.path.abspath(src)
    dest = os.path.abspath(dest)
    if not os.path.isfile(os.path.join(src, MARKER)):
        raise RelocateError(f"{src} does not look like a Jelibox install.")
    log("Waiting for Jelibox to close...")
    wait_for_exit(wait_pid)

    err = validate_target(src, os.path.dirname(dest))
    if err:
        raise RelocateError(err)
    dest_existed = os.path.isdir(dest)
    os.makedirs(dest, exist_ok=True)
    if venv_dir is None or not _is_inside(venv_dir, src):
        old_venv = None
        new_venv_name = default_venv_name()
    else:
        old_venv = venv_dir
        new_venv_name = os.path.basename(os.path.normpath(venv_dir))
    new_venv = os.path.join(dest, new_venv_name)
    moved_something = False

    def cleanup():
        if moved_something:
            return          # data now lives in dest - never delete that
        if os.path.lexists(new_venv):
            _rmtree(new_venv)
        if not dest_existed:
            try:
                os.rmdir(dest)
            except OSError:
                pass

    try:
        log("Reading the installed packages of the current environment...")
        requirements = freeze(freeze_python, log)
        log(f"{len(requirements)} package(s) will be reinstalled.")

        step(1)
        log("Creating the new virtual environment...")
        run_command([base_python(), "-m", "venv", new_venv], log)
        new_py = venv_python(new_venv)

        step(2)
        if requirements:
            req_file = os.path.join(dest, ".jelibox_requirements.txt")
            with open(req_file, "w", encoding="utf-8") as f:
                f.write("\n".join(requirements) + "\n")
            cmd = [new_py, "-m", "pip", "install", "-r", req_file, "--retries", "5", "--timeout", "60"]
            index = torch_index_url(requirements)
            if index:
                cmd += ["--extra-index-url", index]
            try:
                run_command(cmd, log)
            finally:
                try:
                    os.remove(req_file)
                except OSError:
                    pass
        else:
            log("Nothing to install.")

        step(3)
        log("Moving files...")
        moved_something = True          # from here on, a half-move is rolled back by move_tree itself
        try:
            move_tree(src, dest, {new_venv_name} if old_venv else set(), log)
        except RelocateError:
            moved_something = False
            raise
    except Exception as e:
        cleanup()
        if isinstance(e, RelocateError):
            raise
        raise RelocateError(str(e)) from e

    leftovers = []
    step(4)
    log("Removing the old install...")
    if old_venv and not remove_tree(old_venv, log):
        leftovers.append(old_venv)
    if not remove_tree(src, log):
        leftovers.append(src)

    step(5)
    log("Creating shortcuts...")
    try:
        create_shortcuts(dest, new_venv, log)
    except Exception as e:      # shortcuts are a convenience; the move itself succeeded
        log(f"Could not create shortcuts: {e}")
    return new_venv, leftovers


# ----------------------------------------------------------------------
#  Helper window
# ----------------------------------------------------------------------
def _run_gui(args):
    import tkinter as tk
    from tkinter import ttk

    BG, PANEL, TXT, MUTED, ACCENT, RED, GREEN = "#0b0c10", "#14161d", "#e8e9ee", "#8b8fa3", "#6c7bff", "#ff5d6c", "#3ddc97"
    root = tk.Tk()
    root.title("Jelibox — Moving")
    root.configure(bg=BG)
    root.geometry("640x420")
    root.minsize(520, 340)
    root.protocol("WM_DELETE_WINDOW", lambda: None if state["running"] else root.destroy())

    tk.Label(root, text="Moving Jelibox", bg=BG, fg=TXT, font=("Segoe UI", 14, "bold")).pack(anchor="w", padx=20, pady=(18, 0))
    tk.Label(root, text=f"to  {args.dest}", bg=BG, fg=MUTED, font=("Segoe UI", 9), wraplength=600,
             justify="left").pack(anchor="w", padx=20)
    status = tk.StringVar(value="Starting...")
    tk.Label(root, textvariable=status, bg=BG, fg=TXT, font=("Segoe UI", 10)).pack(anchor="w", padx=20, pady=(14, 4))
    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure("J.Horizontal.TProgressbar", troughcolor=PANEL, background=ACCENT, bordercolor=PANEL,
                    lightcolor=ACCENT, darkcolor=ACCENT)
    bar = ttk.Progressbar(root, style="J.Horizontal.TProgressbar", mode="determinate", maximum=len(STEPS))
    bar.pack(fill="x", padx=20)
    text = tk.Text(root, bg=PANEL, fg=MUTED, relief="flat", font=("Consolas", 8), wrap="word",
                   state="disabled", height=10)
    text.pack(fill="both", expand=True, padx=20, pady=14)
    buttons = tk.Frame(root, bg=BG)
    buttons.pack(fill="x", padx=20, pady=(0, 16))

    events = queue.Queue()
    state = {"running": True, "venv": None}

    def append(line):
        text.config(state="normal")
        text.insert("end", line + "\n")
        text.see("end")
        text.config(state="disabled")

    def button(label, command, primary=False):
        tk.Button(buttons, text=label, command=command, relief="flat", cursor="hand2", font=("Segoe UI", 9, "bold"),
                  bg=ACCENT if primary else PANEL, fg="#ffffff" if primary else TXT,
                  activebackground=ACCENT if primary else PANEL, borderwidth=0).pack(side="right", padx=(8, 0), ipadx=14, ipady=6)

    def worker():
        try:
            venv, leftovers = relocate(args.src, args.dest, args.venv, args.freeze_python or sys.executable, args.pid,
                                       lambda m: events.put(("log", m)), lambda i: events.put(("step", i)))
            events.put(("done", True, (venv, leftovers)))
        except Exception as e:
            events.put(("done", False, str(e)))

    def pump():
        try:
            while True:
                kind, *rest = events.get_nowait()
                if kind == "log":
                    append(rest[0])
                elif kind == "step":
                    status.set(f"Step {rest[0] + 1} of {len(STEPS)}: {STEPS[rest[0]]}...")
                    bar["value"] = rest[0]
                else:
                    finish(*rest)
                    return
        except queue.Empty:
            pass
        root.after(100, pump)

    def finish(ok, payload):
        state["running"] = False
        if ok:
            venv, leftovers = payload
            state["venv"] = venv
            bar["value"] = len(STEPS)
            status.set("Done - Jelibox now lives in the new location.")
            append("\nDone.")
            if leftovers:
                append("Could not delete (remove by hand): " + ", ".join(leftovers))
            button("Launch Jelibox", lambda: (launch_jelibox(args.dest, venv), root.destroy()), primary=True)
        else:
            status.set("The move failed - your original Jelibox is untouched.")
            append(f"\nERROR: {payload}")
        button("Close", root.destroy)

    threading.Thread(target=worker, daemon=True).start()
    pump()
    root.mainloop()


def main(argv=None):
    p = argparse.ArgumentParser(description="Move a Jelibox install (internal helper).")
    p.add_argument("--src", required=True)
    p.add_argument("--dest", required=True)
    p.add_argument("--venv")
    p.add_argument("--pid", type=int, default=0)
    p.add_argument("--freeze-python")
    p.add_argument("--no-gui", action="store_true", help="print progress instead of opening a window")
    args = p.parse_args(argv)
    if args.no_gui:
        try:
            relocate(args.src, args.dest, args.venv, args.freeze_python or sys.executable, args.pid,
                     print, lambda i: print(f"== Step {i + 1}/{len(STEPS)}: {STEPS[i]}"))
        except RelocateError as e:
            print(f"ERROR: {e}")
            return 1
        return 0
    _run_gui(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
