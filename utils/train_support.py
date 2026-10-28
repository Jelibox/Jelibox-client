"""
What a training needs besides a window: the job files (`build_job`, `build_yolo_job`), the process that runs them
(`TrainProcess`), installing the best model of a run and pointing the Label Assistant at it. No tkinter and no
`utils.config` globals: the desktop's HeadTrainDialog and the web server both use this.
"""
import json
import os
import shutil
import subprocess
import sys

from . import process_guard
from . import workspace_config as wcfg

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

PREFIX = "@@ "
POLL_MS = 200
DEFAULT_WEIGHTS = "yolov8m.pt"
STOCK_WEIGHTS = ["yolov8m.pt", "yolov8s.pt", "yolov8l.pt", "yolov9c.pt", "yolo11m.pt", "yolo11s.pt", "yolo26m.pt"]
DEFAULT_IMGSZ = (384, 640)           # height x width: 16:9-ish keeps the picture's shape (standing vs sitting)
HEAD_FILE = "head_best.pt"
MODEL_FILE = "modelAssistant.pt"     # what models/<workspace>/ holds for YOLO-World's "My trained model"
RUN_NAME = "train_run"               # Ultralytics' run folder inside yolo_run/
DEFAULT_YOLO = {"detect": "yolo11s.pt", "seg": "yolo11s-seg.pt"}
SIZES = ("n", "s", "m", "l", "x")
# Every stock model the full training can start from (Ultralytics downloads the one chosen the first time): the whole
# YOLOv8, v9, v10, 11, 12 and 26 families for boxes, v8, v9, 11 and 26 for polygons. The newest first.
STOCK_YOLO = {
    "detect": ([f"yolo26{s}.pt" for s in SIZES] + [f"yolo12{s}.pt" for s in SIZES] + [f"yolo11{s}.pt" for s in SIZES]
               + [f"yolov10{s}.pt" for s in SIZES] + [f"yolov9{s}.pt" for s in ("t", "s", "m", "c", "e")]
               + [f"yolov8{s}.pt" for s in SIZES]),
    "seg": ([f"yolo26{s}-seg.pt" for s in SIZES] + [f"yolo11{s}-seg.pt" for s in SIZES]
            + [f"yolov9{s}-seg.pt" for s in ("c", "e")] + [f"yolov8{s}-seg.pt" for s in SIZES]),
}
HEAD_COLUMNS = (("epoch", "Epoch", 56), ("loss", "Loss", 70), ("acc", "Accuracy", 80), ("f1", "Macro-F1", 80),
                ("time", "Time", 60))
YOLO_COLUMNS = (("epoch", "Epoch", 52), ("box", "Box", 56), ("cls", "Cls", 56), ("dfl", "DFL", 56),
                ("p", "Precision", 70), ("r", "Recall", 62), ("map50", "mAP50", 62), ("map5095", "mAP50-95", 76),
                ("time", "Time", 52))


# ------------------------------------------------------------------ pieces without any window (tested alone)
def default_command(job_path):
    """The training process for this job file: the head trainer, or full YOLO training when the job says `kind: yolo`."""
    module = "utils.head_train"
    try:
        with open(job_path, "r", encoding="utf-8") as f:
            if json.load(f).get("kind") == "yolo":
                module = "utils.yolo_train_job"
    except (OSError, ValueError):
        pass
    return [sys.executable, "-u", "-m", module, job_path]


def stop_file_for(out_dir):
    """The file the window creates to say "Finish early"; the training process looks for it after every batch."""
    return os.path.join(out_dir, "finish_early.flag")


def build_job(*, images_folders, labels_folder, classes, weights, taps, imgsz, epochs, batch, lr, class_weights,
              hflip, val_fraction, out_dir, base_dir=BASE_DIR, workers=2):
    return {
        "images_folders": list(images_folders), "labels_folder": labels_folder, "classes": list(classes),
        "weights": weights, "detectors_dir": os.path.join(base_dir, "models", "_detectors"),
        "taps": list(taps), "imgsz": list(imgsz), "epochs": int(epochs), "batch": int(batch), "lr": float(lr),
        "workers": workers, "class_weights": bool(class_weights), "hflip": bool(hflip),
        "val_fraction": float(val_fraction), "out_dir": out_dir, "stop_file": stop_file_for(out_dir),
    }


def build_yolo_job(*, images_folders, labels_folder, classes, init_weights, imgsz, epochs, batch, lr, freeze,
                   val_fraction, out_dir, workers=1, weight_decay=0.0005, augment=None, test_fraction=0.0, shuffle=False,
                   split_seed=None, background=False):
    """The job file of a full YOLO training (see utils/yolo_train_job.py). `augment` = an AugmentConfig as a dict
    (augmentation.config_to_dict): the training pictures get those augmented copies before Ultralytics starts."""
    job = {
        "kind": "yolo", "images_folders": list(images_folders), "labels_folder": labels_folder,
        "classes": list(classes), "init_weights": init_weights, "imgsz": int(imgsz), "epochs": int(epochs),
        "batch": int(batch), "lr": float(lr), "freeze": int(freeze), "val_fraction": float(val_fraction),
        "weight_decay": float(weight_decay), "out_dir": out_dir, "scratch_dir": os.path.join(out_dir, "dataset"),
        "workers": workers, "stop_file": stop_file_for(out_dir),
    }
    if augment:
        job["augment"] = augment
    if background:
        job["background"] = True
    if test_fraction:
        job["test_fraction"] = float(test_fraction)
    if shuffle:
        job["shuffle"] = True
        job["split_seed"] = split_seed
    return job


def install_head(run_dir, model_folder):
    """Copy the best head of a run to models/<workspace>/head_best.pt; the head that was there is kept as
    head_best.prev.pt. Returns the new path, or None when the run produced no head."""
    best = os.path.join(run_dir, HEAD_FILE)
    if not os.path.isfile(best):
        return None
    os.makedirs(model_folder, exist_ok=True)
    target = os.path.join(model_folder, HEAD_FILE)
    if os.path.exists(target):
        shutil.copy2(target, os.path.join(model_folder, "head_best.prev.pt"))
    shutil.copy2(best, target)
    return target


def yolo_best_path(run_dir):
    return os.path.join(run_dir, RUN_NAME, "weights", "best.pt")


def install_yolo(best, target):
    """Copy the best model of a full training to `target` (models/<workspace>/modelAssistant.pt); the model that
    was there is kept next to it as modelAssistant.prev.pt. Returns `target`, or None when there is no best.pt."""
    if not best or not os.path.isfile(best):
        return None
    os.makedirs(os.path.dirname(target), exist_ok=True)
    if os.path.exists(target) and os.path.abspath(target) != os.path.abspath(best):
        stem, ext = os.path.splitext(target)
        shutil.copy2(target, stem + ".prev" + ext)
    shutil.copy2(best, target)
    return target


def select_in_assistant(workspace, head_path, weights, base_dir=BASE_DIR):
    """Point Label Assistant at the new head: head path (relative to the Jelibox folder, so Move Jelibox keeps
    working), the detector it was trained on (only when that is a file on disk) and an empty class map (the head's
    classes are the workspace's classes, so they map by name)."""
    a = wcfg.get_assistant(workspace)
    try:
        stored = os.path.relpath(head_path, base_dir)
    except ValueError:
        stored = head_path
    if stored.startswith(".."):
        stored = head_path
    a["custom_head"]["head_path"] = stored.replace("\\", "/")
    a["custom_head"]["detector_weights"] = weights if os.path.isfile(weights) else ""
    a["custom_head"]["class_map"] = {}
    wcfg.set_assistant(workspace, a)


def select_trained_model(workspace):
    """Make YOLO-World mode use the workspace's own trained model (models/<workspace>/modelAssistant.pt)."""
    a = wcfg.get_assistant(workspace)
    a["provider"] = wcfg.PROVIDER_CUSTOM
    a["custom_model"]["path"] = ""
    a["custom_model"]["class_map"] = {}
    wcfg.set_assistant(workspace, a)


def parse_taps(text):
    try:
        taps = [int(t) for t in text.replace(",", " ").split()]
    except ValueError:
        raise ValueError("Taps are three layer numbers, e.g. 15, 18, 21.")
    if len(taps) != 3 or any(t < 0 for t in taps):
        raise ValueError("Taps are three layer numbers, e.g. 15, 18, 21.")
    return taps


class TrainProcess:
    """The training process. Its output goes to a log file (which also stays for the user to read) and Tk polls the
    file: no reader thread, because Tk variables that get garbage-collected on another thread crash Python 3.10."""

    def __init__(self, cmd, cwd, log_path):
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
        self.log_path = log_path
        with open(log_path, "wb") as out:
            self.proc = subprocess.Popen(cmd, cwd=cwd, stdout=out, stderr=subprocess.STDOUT, env=env,
                                         creationflags=flags, **process_guard.popen_extra())
        process_guard.bind_to_parent(self.proc)         # the worker ends with Jelibox, however Jelibox ends
        self._reader = open(log_path, "rb")
        self._partial = b""

    def read_new(self):
        """(new complete lines, ended). `ended` is True once the process has exited and its output is all read."""
        exited = self.proc.poll() is not None
        data = self._reader.read()                              # after the exit check, so nothing can be missed
        lines = []
        if data:
            parts = (self._partial + data).split(b"\n")
            self._partial = parts.pop()
            lines = [p.decode("utf-8", "replace").rstrip("\r") for p in parts]
        if exited and self._partial:
            lines.append(self._partial.decode("utf-8", "replace").rstrip("\r"))
            self._partial = b""
        return lines, exited

    def close(self):
        try:
            self._reader.close()
        except OSError:
            pass

    def running(self):
        return self.proc.poll() is None

    def stop(self):
        if self.proc.poll() is not None:
            return
        if os.name == "nt":                                     # also takes the data-loader workers with it
            # DEVNULL, not capture_output: capturing starts reader threads, and a Tk variable garbage-collected
            # on one of them crashes Python 3.10 (Tcl_AsyncDelete)
            subprocess.run(["taskkill", "/PID", str(self.proc.pid), "/T", "/F"], stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        else:
            self.proc.terminate()


def _fmt_time(seconds):
    seconds = int(max(0, seconds))
    return f"{seconds // 3600}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}" if seconds >= 3600 \
        else f"{seconds // 60}:{seconds % 60:02d}"
