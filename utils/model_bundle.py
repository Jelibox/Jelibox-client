"""
A model folder: files that belong in models/<workspace>, carried from one Jelibox to another (a Kaggle session ends and
the next one starts empty).

The folder holds a small marker file, `jelibox-model.json`, and the files laid out like they are inside
models/<workspace>. When a folder or archive with that marker is uploaded through Add Workspace, Jelibox does not
read it as a dataset: it puts the files in models/<workspace> and the Train window offers to go on from there.

Today it carries one thing, the checkpoint of a full YOLO training (see run_state.py):

    <workspace>-checkpoint/
        jelibox-model.json                       the marker (and what the folder is: workspace, epoch, classes)
        yolo_run/resume_point.pt                 the weights + optimizer after the last saved epoch
        yolo_run/run_state.json                  status, split (which pictures validate), settings
        yolo_run/train_run/results.csv           the table of the epochs so far
        yolo_run/train_run/weights/best.pt       the best model so far

Only these four names are ever installed, whatever else a folder contains. No torch / tkinter / web in here.
"""
import json
import os
import shutil
import time

from . import run_state

MARKER = "jelibox-model.json"
KIND = "jelibox-model-folder"
SCHEMA = 1
FILES = (
    "yolo_run/resume_point.pt",
    "yolo_run/run_state.json",
    "yolo_run/train_run/results.csv",
    "yolo_run/train_run/weights/best.pt",
)


def _native(rel):
    return os.path.join(*rel.split("/"))


def build(model_folder, workspace, dest_parent):
    """Make the checkpoint folder of the run in models/<workspace>/yolo_run inside `dest_parent`; returns its path.
    ValueError when there is nothing to carry yet (no epoch has been saved)."""
    run_dir = os.path.join(model_folder, "yolo_run")
    state = run_state.read(run_dir)
    if not state or not os.path.isfile(run_state.resume_path(run_dir)):
        raise ValueError("There is nothing to save yet: a full training has to finish its first epoch before it has a checkpoint.")
    folder = os.path.join(dest_parent, f"{workspace}-checkpoint")
    shutil.rmtree(folder, ignore_errors=True)
    for rel in FILES:
        source = os.path.join(model_folder, _native(rel))
        if os.path.isfile(source):
            target = os.path.join(folder, _native(rel))
            os.makedirs(os.path.dirname(target), exist_ok=True)
            shutil.copyfile(source, target)
    marker = {"jelibox": KIND, "schema": SCHEMA, "workspace": workspace, "put_in": f"models/{workspace}",
              "what": "checkpoint of a full YOLO training", "status": state.get("status"),
              "epoch": state.get("epoch"), "epochs": state.get("epochs"), "classes": list(state.get("classes") or []),
              "created": time.time()}
    with open(os.path.join(folder, MARKER), "w", encoding="utf-8") as f:       # last: a folder with a marker is complete
        json.dump(marker, f, indent=2)
    return folder


def read_marker(folder):
    """The marker of a model folder as a dict, or None when `folder` has none or it is not ours."""
    try:
        with open(os.path.join(folder, MARKER), "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) and data.get("jelibox") == KIND else None


def find(roots):
    """[(folder, marker)] of the model folders under `roots` (a folder or a list of them), most advanced epoch first.
    A model folder is not searched further."""
    if isinstance(roots, (str, os.PathLike)):
        roots = [roots]
    found = []
    for root in roots:
        for dirpath, dirnames, _files in os.walk(root):
            marker = read_marker(dirpath)
            if marker is not None:
                found.append((dirpath, marker))
                dirnames[:] = []
    found.sort(key=lambda item: int(item[1].get("epoch") or 0), reverse=True)
    return found


def install(folder, model_folder, workspace):
    """Copy the files of the model folder `folder` into models/<workspace> (`model_folder`). The run that is there is
    replaced. Returns {"epoch", "epochs", "classes", "files"}. ValueError when the folder has no resume point."""
    if not os.path.isfile(os.path.join(folder, _native(FILES[0]))):
        raise ValueError("That model folder has no checkpoint in it (yolo_run/resume_point.pt is missing).")
    marker = read_marker(folder) or {}
    run_dir = os.path.join(model_folder, "yolo_run")
    run_state.forget(run_dir)
    shutil.rmtree(os.path.join(run_dir, "train_run"), ignore_errors=True)
    copied = 0
    for rel in FILES:
        source = os.path.join(folder, _native(rel))
        if not os.path.isfile(source) or os.path.islink(source):
            continue
        target = os.path.join(model_folder, _native(rel))
        os.makedirs(os.path.dirname(target), exist_ok=True)
        tmp = target + ".part"
        shutil.copyfile(source, tmp)
        os.replace(tmp, target)
        copied += 1
    state = run_state.read(run_dir)
    if state is not None:
        run_state.write(run_dir, workspace=workspace)
    else:                                                       # a folder without its state: the marker knows enough
        run_state.write(run_dir, status="paused", epoch=int(marker.get("epoch") or 0), epochs=int(marker.get("epochs") or 0),
                        classes=list(marker.get("classes") or []), workspace=workspace)
    state = run_state.read(run_dir)
    return {"epoch": state.get("epoch"), "epochs": state.get("epochs"), "classes": list(state.get("classes") or []),
            "files": copied, "status": state.get("status")}
