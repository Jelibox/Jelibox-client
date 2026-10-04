#!/usr/bin/env python3
"""
Main Entry Point for Object Detection Annotation Tool

Run with no arguments to show the workspace picker (a VS Code-style "no
folder opened" screen listing the workspaces found in datasetsInput). Run
with a dataset folder path as the first argument to open the annotation GUI
directly for that folder - this is how the picker launches the annotation
window (as a separate process), and how its "Back to Workspace" button
returns control to the picker.

All classes have been modularized into separate files in the utils folder:
  - TkTerminalRedirector → utils/TkTerminalRedirector.py
  - TrainingConfigDialog → utils/TrainingConfigDialog.py
  - AnnotationGUI → utils/AnnotationGUI.py
  - WorkspacePicker → utils/WorkspacePicker.py
"""

import sys
import os
import tkinter as tk

# Add parent directory to path so utils is recognized as a package
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# Run from the project root regardless of how the script was launched, since
# the GUI loads assets (e.g. assets/jelibox.png) via relative paths.
os.chdir(parent_dir)


def _show_loading_splash(root):
    """A small 'Preparing workspace...' window with a progress bar.

    Opening a workspace is normally instant, but the first time a dataset
    annotated before YOLOdataset/ existed gets opened, config.load_workspace
    backfills a YOLO .txt label for every VOC XML that doesn't have one yet -
    which can take a while on a large dataset. This keeps that from looking
    like a frozen window.
    """
    from utils.theme import C_BASE, C_PANEL, C_ACCENT, C_TXT1, C_TXT2

    splash = tk.Toplevel(root)
    splash.title("Jelibox")
    splash.configure(bg=C_BASE)
    splash.overrideredirect(True)
    splash.resizable(False, False)

    width, height = 420, 130
    x = (splash.winfo_screenwidth() // 2) - (width // 2)
    y = (splash.winfo_screenheight() // 2) - (height // 2)
    splash.geometry(f"{width}x{height}+{x}+{y}")

    tk.Label(splash, text="Jelibox", bg=C_BASE, fg=C_TXT1,
             font=('Segoe UI', 14, 'bold')).pack(pady=(22, 4))
    status_var = tk.StringVar(value="Preparing workspace...")
    tk.Label(splash, textvariable=status_var, bg=C_BASE, fg=C_TXT2,
             font=('Segoe UI', 9)).pack()

    bar_w = width - 60
    bar_bg = tk.Frame(splash, bg=C_PANEL, width=bar_w, height=6)
    bar_bg.pack(pady=18)
    bar_bg.pack_propagate(False)
    bar_fill = tk.Frame(bar_bg, bg=C_ACCENT, width=0, height=6)
    bar_fill.place(x=0, y=0, relheight=1)

    splash.update()

    state = {'last': 0.0}

    def on_progress(done, total):
        import time
        now = time.time()
        if done < total and now - state['last'] < 0.05:
            return
        state['last'] = now
        status_var.set(f"Syncing YOLO labels... {done}/{total}")
        frac = (done / total) if total else 1.0
        bar_fill.place(width=int(bar_w * frac))
        splash.update()

    return splash, on_progress


RESTART_EXIT_CODE = 75      # "relaunch me": the workspace picker restarts the window on this code
PARENT_ENV = "JELIBOX_PARENT"   # set by the picker so the GUI knows someone will relaunch it


def _restart_self(args):
    """Relaunch this script with `args` (used when there is no picker above us)."""
    import subprocess
    subprocess.Popen([sys.executable, os.path.abspath(__file__)] + list(args))


def _resume_image_index(dataset_folder):
    """After a restart (e.g. theme change) reopen on the image the user was on."""
    from utils import app_settings
    resume = app_settings.get("resume")
    if not resume:
        return None
    app_settings.set("resume", None)
    if os.path.normcase(os.path.normpath(resume.get("folder", ""))) != \
            os.path.normcase(os.path.normpath(dataset_folder)):
        return None
    images = sorted(f for f in os.listdir(dataset_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg')))
    try:
        return images.index(resume.get("image"))
    except ValueError:
        return None


def run_annotation_gui(dataset_folder):
    """Load the given dataset folder as the active workspace and open the
    annotation GUI for it. Must run before utils.AnnotationGUI is imported,
    since it (and the modules it pulls in) read workspace globals off
    utils.config at import time."""
    from utils import config

    from utils import window_style
    window_style.set_app_id()
    root = tk.Tk()
    window_style.setup(root)
    root.withdraw()

    from utils import ScreenGuard
    ok, reason, sw, sh = ScreenGuard.check_screen(root)
    if not ok:
        ScreenGuard.show_unsupported_dialog(root, reason, sw, sh)
        root.destroy()
        return

    splash, on_progress = _show_loading_splash(root)

    config.load_workspace(dataset_folder, progress_cb=on_progress)

    resume_index = _resume_image_index(dataset_folder)
    if resume_index is not None:
        config.state.current_index = resume_index

    from utils.AnnotationGUI import AnnotationGUI

    # Build the whole window while still hidden, so nothing but the loading
    # splash is ever visible - no blank window flashes in before it's ready.
    AnnotationGUI(root)

    splash.destroy()
    if root.winfo_exists():
        ScreenGuard.maximize(root)
        root.deiconify()
        root.lift()
        root.focus_force()

    root.mainloop()

    if getattr(root, "_jelibox_restart", False):
        if os.environ.get(PARENT_ENV):
            sys.exit(RESTART_EXIT_CODE)           # the picker relaunches us
        _restart_self([dataset_folder])


def run_workspace_picker():
    from utils.WorkspacePicker import WorkspacePickerApp

    from utils import window_style
    window_style.set_app_id()
    root = tk.Tk()
    window_style.setup(root)
    WorkspacePickerApp(root, entry_script=os.path.abspath(__file__))
    root.mainloop()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_annotation_gui(sys.argv[1])
    else:
        run_workspace_picker()
