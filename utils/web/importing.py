"""
What Add Workspace needs on the server side, without any web framework (so it is easy to test):

  Uploads       files sent from the browser land in a staging folder (temp/uploads/<id>/), zip / rar / 7z archives (split ones too) are
                unpacked there, and the importer then reads that folder like any other. Every name that comes from the browser
                is checked: nothing can be written outside the staging folder.
  list_folder   the folder browser of the page (folders of THIS computer, the server).
  summarize     what a folder holds (pictures, videos, annotation format, classes), shown before Create.

The import itself is utils/dataset_import.py, unchanged.
"""
import os
import secrets
import shutil
import string
import sys
import time

from .. import dataset_import as di
from .. import workspace_manager as wm
from ..archive_support import free_bytes, safe_parts, unpack_all     # free_bytes: the upload routes use it from here

STALE_SECONDS = 24 * 3600


class Uploads:
    def __init__(self, root):
        self.root = root
        self.owners = {}

    def create(self, user):
        os.makedirs(self.root, exist_ok=True)
        self.forget_stale()
        upload_id = secrets.token_hex(8)
        os.makedirs(self.folder(upload_id))
        self.owners[upload_id] = user
        return upload_id

    def folder(self, upload_id):
        if not upload_id or not all(c in string.hexdigits for c in upload_id) or len(upload_id) != 16:
            raise ValueError("Unknown upload.")
        return os.path.join(self.root, upload_id)

    def check(self, upload_id, user):
        """The staging folder of an upload that exists and belongs to `user`."""
        folder = self.folder(upload_id)
        if not os.path.isdir(folder) or self.owners.get(upload_id) not in (user, None):
            raise ValueError("Unknown upload.")
        return folder

    def target(self, upload_id, user, relative):
        folder = self.check(upload_id, user)
        path = os.path.join(folder, *safe_parts(relative))
        if os.path.commonpath([os.path.realpath(path), os.path.realpath(folder)]) != os.path.realpath(folder):
            raise ValueError("Not an allowed file name.")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path

    def remove(self, upload_id):
        try:
            shutil.rmtree(self.folder(upload_id), ignore_errors=True)
        except ValueError:
            pass
        self.owners.pop(upload_id, None)

    def forget_stale(self):
        now = time.time()
        try:
            names = os.listdir(self.root)
        except OSError:
            return
        for name in names:
            path = os.path.join(self.root, name)
            try:
                if now - os.path.getmtime(path) > STALE_SECONDS:
                    shutil.rmtree(path, ignore_errors=True)
                    self.owners.pop(name, None)
            except OSError:
                pass

    def unpack_archives(self, folder):
        """Unpack every .zip / .rar / .7z (split ones too) found in `folder` next to itself, then delete the archive
        files. Returns how many folders were made."""
        return len(unpack_all(folder, remove=True))


# ------------------------------------------------------------------ folder browser
def list_folder(path="", ext=""):
    """{"path", "parent", "folders": [names], "files": [names], "roots": [...]} for the folder browser. An empty path =
    the start places: the user's home folder and (Windows) the drives, or the file system root (Linux). With `ext`
    (e.g. ".pt") the files of that type are listed too (choosing a model file)."""
    roots = [os.path.expanduser("~")]
    if sys.platform == "win32":
        roots += [f"{c}:\\" for c in string.ascii_uppercase if os.path.exists(f"{c}:\\")]
    else:
        roots.append("/")
    if not path:
        return {"path": "", "parent": None, "folders": [], "files": [], "roots": roots}
    path = os.path.abspath(path)
    if not os.path.isdir(path):
        raise ValueError("That folder does not exist.")
    try:
        entries = [n for n in os.listdir(path) if not n.startswith(".")]
        names = sorted((n for n in entries if os.path.isdir(os.path.join(path, n))), key=str.lower)
        files = sorted((n for n in entries if ext and n.lower().endswith(ext.lower())
                        and os.path.isfile(os.path.join(path, n))), key=str.lower)
    except PermissionError:
        raise ValueError("This folder cannot be read by Jelibox.")
    parent = os.path.dirname(path)
    return {"path": path, "parent": parent if parent != path else "", "folders": names, "files": files, "roots": roots}


# ------------------------------------------------------------------ what a folder holds
def summarize(folder, workspace=None):
    """The facts Add Workspace shows after a folder is chosen. {"error": text} when it cannot be imported."""
    if not folder or not os.path.isdir(folder):
        return {"error": "That folder does not exist."}
    scan = di.scan_dataset_folder(folder)
    n_images, n_videos = len(scan["images"]), len(scan["videos"])
    model = None
    if scan["model_folders"]:
        marker = scan["model_folders"][0][1]
        model = {"epoch": marker.get("epoch"), "epochs": marker.get("epochs"), "classes": list(marker.get("classes") or []),
                 "workspace": marker.get("workspace"), "status": marker.get("status")}
    if not n_images and not n_videos:
        if model is None:
            return {"error": "No images or videos found in this folder (every subfolder was searched)."}
        if not workspace or workspace not in wm.list_workspaces():
            return {"error": "The upload holds a saved model but no pictures. Add the dataset's pictures in the same upload, "
                             "or use Add Instance on an existing workspace."}
        return {"error": None, "images": 0, "videos": 0, "duplicates": 0, "format": None, "annotation_files": 0, "classes": {},
                "yolo_unnamed": False, "video_frames": 0, "locked_classes": wm.existing_classes_for_workspace(workspace),
                "skipped_classes": [], "model": model}
    try:
        fmt = di.detect_dataset_format(scan) if n_images else None
    except ValueError as exc:
        return {"error": str(exc)}
    found = di.scan_classes(scan, fmt) if fmt else {}
    out = {
        "error": None, "images": n_images, "videos": n_videos, "duplicates": scan["duplicates"], "format": fmt,
        "annotation_files": len(scan[fmt]) if fmt else 0, "classes": found,
        "yolo_unnamed": bool(fmt == "yolo" and not di.yolo_classes_resolved(scan)),
        "video_frames": sum(p["frames"] for p in di.probe_videos(scan["videos"])) if n_videos else 0,
        "locked_classes": None, "skipped_classes": [], "model": model,
    }
    locked = wm.existing_classes_for_workspace(workspace) if workspace else None
    if locked:
        out["locked_classes"] = locked
        out["skipped_classes"] = [c for c in found if c not in locked]
    return out
