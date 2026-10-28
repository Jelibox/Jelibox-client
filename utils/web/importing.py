"""
What Add Workspace needs on the server side, without any web framework (so it is easy to test):

  Uploads       files sent from the browser land in a staging folder (temp/uploads/<id>/), zip files are unpacked
                there, and the importer then reads that folder like any other. Every name that comes from the browser
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
import zipfile

from .. import dataset_import as di
from .. import workspace_manager as wm

RESERVED = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}
SPARE_BYTES = 1 << 30                     # always leave 1 GB free on the disk
MAX_ZIP_FILES = 300_000
STALE_SECONDS = 24 * 3600


def safe_parts(path):
    """The folders and file name of a relative path from the browser, or ValueError. No '..', no drive letters, no
    control characters, no Windows device names."""
    parts = []
    for part in str(path or "").replace("\\", "/").split("/"):
        if part in ("", "."):
            continue
        if part == ".." or ":" in part or any(ord(c) < 32 for c in part) or len(part) > 200:
            raise ValueError(f"Not an allowed file name: {path!r}")
        if part.split(".")[0].upper() in RESERVED or part != part.rstrip(" ."):
            raise ValueError(f"Not an allowed file name: {path!r}")
        parts.append(part)
    if not parts:
        raise ValueError("The file has no name.")
    return parts


def free_bytes(folder):
    try:
        return shutil.disk_usage(folder).free - SPARE_BYTES
    except OSError:
        return 0


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

    def unpack_zips(self, folder):
        """Unpack every .zip found in `folder` next to itself (then delete it). Returns how many were unpacked."""
        zips = [os.path.join(d, f) for d, _dirs, files in os.walk(folder) for f in files if f.lower().endswith(".zip")]
        for path in zips:
            _unpack(path)
        return len(zips)


def _unpack(path):
    stem = os.path.splitext(path)[0]
    target = stem
    while os.path.exists(target):
        target += "_x"
    try:
        archive = zipfile.ZipFile(path)
    except zipfile.BadZipFile:
        raise ValueError(f"{os.path.basename(path)} is not a valid zip file.")
    with archive:
        infos = [i for i in archive.infolist() if not i.is_dir()]
        if len(infos) > MAX_ZIP_FILES:
            raise ValueError(f"{os.path.basename(path)} holds too many files.")
        if sum(i.file_size for i in infos) > free_bytes(os.path.dirname(path)):
            raise ValueError(f"Not enough free disk space to unpack {os.path.basename(path)}.")
        root = os.path.realpath(target)
        for info in infos:
            try:
                out = os.path.join(target, *safe_parts(info.filename))
            except ValueError:
                continue                                    # a name that is not allowed is skipped, never written
            if os.path.commonpath([os.path.realpath(out), root]) != root:
                continue
            os.makedirs(os.path.dirname(out), exist_ok=True)
            with archive.open(info) as src, open(out, "wb") as dst:
                shutil.copyfileobj(src, dst)
    os.remove(path)


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
    if not n_images and not n_videos:
        return {"error": "No images or videos found in this folder (every subfolder was searched)."}
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
        "locked_classes": None, "skipped_classes": [],
    }
    locked = wm.existing_classes_for_workspace(workspace) if workspace else None
    if locked:
        out["locked_classes"] = locked
        out["skipped_classes"] = [c for c in found if c not in locked]
    return out
