"""
Export helper for Jelibox using Ultralytics YOLO model export API.
Provides `export_model` which runs `YOLO(model_path).export(...)` with common options.
"""
import os
import re
import shutil
import traceback

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


def export_model(model_path: str,
                 fmt: str = 'onnx',
                 imgsz: int = 640,
                 optimize: bool = False,
                 keras: bool = False,
                 half: bool = False,
                 int8: bool = False,
                 dynamic: bool = False,
                 simplify: bool = False,
                 end2end: bool = False,
                 save_dir: str = None):
    """Export the model using ultralytics.YOLO.export.

    Returns (success: bool, message: str, out_path: str|None)
    """
    if YOLO is None:
        return False, "ultralytics package not installed", None

    if not os.path.exists(model_path):
        return False, f"Model file not found: {model_path}", None

    try:
        model = YOLO(model_path)

        kwargs = {
            'format': fmt,
            'imgsz': imgsz,
            'optimize': optimize,
            'keras': keras,
            'half': half,
            'int8': int8,
            'dynamic': dynamic,
            'simplify': simplify,
            'end2end': end2end,
        }

        # Remove keys with None or False defaults are fine to pass
        # ultralytics.export expects keyword names similar to these
        out = model.export(**kwargs)

        # `export` may return path or list of paths; normalize to string
        out_path = None
        if isinstance(out, (list, tuple)) and len(out) > 0:
            out_path = out[0]
        elif isinstance(out, str):
            out_path = out

        if save_dir and out_path:
            try:
                os.makedirs(save_dir, exist_ok=True)
                dest = os.path.join(save_dir, os.path.basename(out_path))
                os.replace(out_path, dest)
                out_path = dest
            except Exception:
                # ignore
                pass

        return True, "Export completed", out_path

    except Exception as e:
        tb = traceback.format_exc()
        return False, f"Export failed: {str(e)}\n{tb}", None


# ---- Export Dataset: where the dataset is saved -----------------------------------------------------------------

def next_version_name(folder, workspace):
    """`<workspace>-v<N>`: N is one more than the highest `<workspace>-v<number>` folder already inside `folder`
    (1 when there is none, or the folder does not exist yet)."""
    pattern = re.compile(re.escape(workspace) + r"-v(\d+)$")
    highest = 0
    try:
        for name in os.listdir(folder):
            m = pattern.match(name)
            if m and os.path.isdir(os.path.join(folder, name)):
                highest = max(highest, int(m.group(1)))
    except OSError:
        pass
    return f"{workspace}-v{highest + 1}"


def plan_export_target(default_folder, chosen_folder, workspace, fmt=None):
    """(target folder, replace): where an export goes.

    No folder chosen: the default folder inside Jelibox, and the previous export of the workspace there is replaced.
    A folder chosen (an archive on another drive, say): a NEW folder `<workspace>-v<N>` inside it, N counting up
    from what is already there, so an archive is never overwritten and nothing in the chosen folder is deleted."""
    if not chosen_folder:
        return default_folder, True
    return os.path.join(chosen_folder, next_version_name(chosen_folder, workspace)), False


def files_size(paths):
    """Total size in bytes of the files that exist."""
    total = 0
    for path in paths:
        try:
            total += os.path.getsize(path)
        except OSError:
            pass
    return total


def check_destination(folder, needed_bytes=0):
    """A sentence saying why `folder` cannot take the export (missing drive, no write permission, not enough free
    space for at least `needed_bytes`), or None when it can."""
    try:
        os.makedirs(folder, exist_ok=True)
        probe = os.path.join(folder, f".jelibox_write_test_{os.getpid()}")
        with open(probe, "wb") as f:
            f.write(b"ok")
        os.remove(probe)
    except OSError as exc:
        return f"Cannot save to this folder (is the drive connected?): {exc}"
    try:
        free = shutil.disk_usage(folder).free
    except OSError:
        return None
    if needed_bytes and free < needed_bytes:
        gb = 1024 ** 3
        return (f"Not enough free space there: the images alone need about {needed_bytes / gb:.1f} GB "
                f"(augmented copies need more) and only {free / gb:.1f} GB is free.")
    return None
