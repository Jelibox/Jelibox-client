"""How many pictures each workspace has, kept in `configs/_stats.json`.

Counting means listing the dataset folders and looking at every label file: seconds on a big workspace, and a minute
on a cold disk. The Train windows used to do it every time they opened. Now the numbers are written down here and
recounted ONLY when asked: the picker's **Refresh** button (every workspace; the ones that are gone are forgotten) or
the Refresh button of a Train window (that workspace). Nothing counts by itself. The training run still reads the
real files when it starts, so a stale number here never changes what is trained.

Per workspace: images (all pictures in its dataset folders, and per folder), labelled (pictures with at least one
box), background (pictures saved with nothing in them), boxes (per class), counted (when).
"""
import json
import os
import threading
import time

from . import head_data
from . import workspace_manager as wm

_lock = threading.Lock()
FILE = "_stats.json"


def _path():
    return os.path.join(wm.CONFIGS_ROOT, FILE)


def _read():
    try:
        with open(_path(), "r", encoding="utf-8") as f:
            data = json.load(f)
        found = data.get("workspaces") if isinstance(data, dict) else None
        return found if isinstance(found, dict) else {}
    except (OSError, ValueError):
        return {}


def _write(workspaces):
    path = _path()
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path + ".tmp", "w", encoding="utf-8") as f:
            json.dump({"version": 1, "workspaces": workspaces}, f, indent=1)
        os.replace(path + ".tmp", path)
    except OSError:
        pass                                         # a read-only folder: the numbers are just not remembered


def labels_folder(workspace):
    return os.path.join(wm.BASE_DIR, "YOLOdataset", workspace, "labels")


def compute(workspace, instances=None):
    """Count one workspace now (reads its folders and label files)."""
    instances = wm.list_workspaces().get(workspace, []) if instances is None else instances
    per_instance = {name: wm.count_images(name) for name in instances}
    folders = [wm.instance_path(name) for name in instances]
    pairs, empty = head_data.scan_labels(folders, labels_folder(workspace))
    classes = wm.existing_classes_for_workspace(workspace) or []
    return {"images": sum(per_instance.values()), "instances": per_instance, "labelled": len(pairs),
            "background": len(empty), "boxes": head_data.class_counts(pairs, len(classes)), "counted": time.time()}


def get(workspace):
    """The stored numbers of a workspace, or None when it was never counted."""
    entry = _read().get(workspace)
    return entry if isinstance(entry, dict) and "counted" in entry else None


EMPTY = {"images": 0, "instances": {}, "labelled": 0, "background": 0, "boxes": [], "counted": None}


def get_or_empty(workspace):
    """The stored numbers, or zeros marked "counted": None when there are none. Nothing is ever counted here: the
    Train windows must open at once, so counting is only done on request (Refresh)."""
    return get(workspace) or dict(EMPTY)


def recount(workspace):
    stats = compute(workspace)
    with _lock:
        data = _read()
        data[workspace] = stats
        _write(data)
    return stats


def recount_all(progress_cb=None):
    """Count every workspace that exists and write the result: workspaces that are gone disappear from the file.
    progress_cb(done, total, text) is called before each workspace."""
    groups = wm.list_workspaces()
    fresh = {}
    for done, (name, instances) in enumerate(groups.items()):
        if progress_cb:
            progress_cb(done, len(groups), f"Counting {name} ...")
        fresh[name] = compute(name, instances)
    with _lock:
        _write(fresh)
    if progress_cb:
        progress_cb(len(groups), len(groups), "Done")
    return fresh


def instance_images(workspace, instance):
    """Pictures in one dataset folder for the picker's list: the stored count, or None when it was not counted yet
    (no folder is listed to find out)."""
    stats = _read().get(workspace) or {}
    return (stats.get("instances") or {}).get(instance)
