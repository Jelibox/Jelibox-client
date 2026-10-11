"""
The Train button of the Custom Model mode. Two kinds of training, chosen at the top of the window:

  Head only   - fit a small head on a frozen detector (utils/head_train.py). The head is saved as
                models/<workspace>/head_best.pt and selected in Label Assistant, so G and Auto-annotate all use it.
  Full YOLO   - train every layer of a YOLO model with Ultralytics, like the plain Train button
                (utils/yolo_train_job.py). The best model is saved as models/<workspace>/modelAssistant.pt and
                selected as "My trained model", which YOLO-World mode uses.

Setup window (settings) -> the same window turns into a progress window with a per-epoch table while the training
process runs. Import this module only after config.load_workspace() has run.
"""
import json
import os
import shutil
import subprocess
import sys
import time
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

from . import head_data
from . import workspace_config as wcfg
from .ModeDialogs import _Dialog
from .config import workspaceName, BASE_DIR
from .theme import (C_BASE, C_CARD, C_CARD2, C_BORDER, C_ACCENT, C_AMBER, C_GREEN, C_RED, C_TXT1, C_TXT2, C_TXT3,
                    C_ON_ACCENT)

PREFIX = "@@ "
POLL_MS = 200
DEFAULT_WEIGHTS = "yolov8m.pt"
STOCK_WEIGHTS = ["yolov8m.pt", "yolov8s.pt", "yolov8l.pt", "yolov9c.pt", "yolo11m.pt", "yolo11s.pt", "yolo26m.pt"]
DEFAULT_IMGSZ = (384, 640)           # height x width: 16:9-ish keeps the picture's shape (standing vs sitting)
HEAD_FILE = "head_best.pt"
MODEL_FILE = "modelAssistant.pt"     # what models/<workspace>/ holds for YOLO-World's "My trained model"
RUN_NAME = "train_run"               # Ultralytics' run folder inside yolo_run/
DEFAULT_YOLO = {"detect": "yolo11s.pt", "seg": "yolo11s-seg.pt"}
STOCK_YOLO = {"detect": ["yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov9c.pt"],
              "seg": ["yolo11n-seg.pt", "yolo11s-seg.pt", "yolo11m-seg.pt", "yolov8n-seg.pt", "yolov8s-seg.pt",
                      "yolov8m-seg.pt", "yolov9c-seg.pt"]}
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
                   val_fraction, out_dir, workers=1, weight_decay=0.0005):
    """The job file of a full YOLO training (see utils/yolo_train_job.py)."""
    return {
        "kind": "yolo", "images_folders": list(images_folders), "labels_folder": labels_folder,
        "classes": list(classes), "init_weights": init_weights, "imgsz": int(imgsz), "epochs": int(epochs),
        "batch": int(batch), "lr": float(lr), "freeze": int(freeze), "val_fraction": float(val_fraction),
        "weight_decay": float(weight_decay), "out_dir": out_dir, "scratch_dir": os.path.join(out_dir, "dataset"),
        "workers": workers, "stop_file": stop_file_for(out_dir),
    }


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
                                         creationflags=flags)
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


HEAD_INTRO = ("Trains a head on this workspace's labels: the box is what the detector finds (a person, a bottle ...) "
              "and its class is what the head learns (a behavior, a brand ...). The detector itself is never changed.")
YOLO_INTRO = ("Trains a whole YOLO model on this workspace's labels with Ultralytics, like the plain Train button. "
              "Every layer learns your classes and boxes, and the result is a normal YOLO model.")


# ------------------------------------------------------------------ the window
class HeadTrainDialog(_Dialog):
    TITLE = "TRAIN CUSTOM MODEL"
    WIDTH = 700
    SAVE_TEXT = "Start training"

    def __init__(self, parent, images_folders, labels_folder, model_folder, command=None, on_done=None,
                 model_path=None, task="detect", preset=None, title=None):
        """`preset` = a ready full-training job (the plain Train button's settings): the setup part is skipped, the
        window goes straight to its progress view and does not hold the annotation window while it runs."""
        if title:
            self.TITLE = title
        self.preset = preset
        self.images_folders, self.labels_folder, self.model_folder = list(images_folders), labels_folder, model_folder
        self.run_dir = os.path.join(model_folder, "head_run")
        self.yolo_dir = os.path.join(model_folder, "yolo_run")
        self.model_path = model_path or os.path.join(model_folder, MODEL_FILE)
        self.task = task if task in DEFAULT_YOLO else "detect"
        self.command = command or default_command
        self.on_done = on_done
        self.proc = None
        self.running = False
        self.finished = False
        self.stopping = False                # Stop now: the process was killed
        self.finishing = False               # Finish early: the process was asked to wrap up
        self.result = None
        self.error = None
        self.active_kind = "head"            # what the run in progress (or the last one) is
        self.active_dir = self.run_dir
        self.pairs = head_data.labelled_pairs(self.images_folders, self.labels_folder)
        super().__init__(parent)
        self._built = True
        self.win.title("Train model" if preset else "Train custom model")
        self.win.bind('<Escape>', lambda e: self._close())
        self.win.protocol("WM_DELETE_WINDOW", self._close)
        if preset:
            self._start_preset()

    def _start_preset(self):
        job = self.preset
        os.makedirs(self.yolo_dir, exist_ok=True)
        shutil.rmtree(os.path.join(self.yolo_dir, RUN_NAME), ignore_errors=True)      # an older run must not pass for this one
        self._launch(job, "yolo", self.yolo_dir, int(job["epochs"]))
        if self.proc is None:                                    # could not start: the error box said why
            self.win.destroy()
            return
        self.win.grab_release()                                  # hours of training must not lock the annotation window

    # ---------------------------------------------------------- setup view
    def build(self, body):
        h = self.cfg["custom_head"]
        self.intro = self.label(body, HEAD_INTRO, pady=(0, 8))

        card = self.card(body)
        self.label(card, "TRAINING TYPE", size=8, bold=True)
        self.kind_var = tk.StringVar(value="head")
        for value, text, note in (
                ("head", "Head only (fast)",
                 "Fits a small head on a frozen detector. Minutes on a CPU. Used by the Custom Model mode."),
                ("yolo", "Full YOLO training (every layer)",
                 "Trains all the weights, with Ultralytics' own augmentation and loss. Much slower: a GPU is "
                 "strongly recommended. The model is used by YOLO-World mode as My trained model.")):
            tk.Radiobutton(card, text=text, value=value, variable=self.kind_var, command=self._apply_kind, bg=C_CARD,
                           fg=C_TXT1, selectcolor=C_BASE, activebackground=C_CARD, activeforeground=C_ACCENT,
                           font=('Segoe UI', 10, 'bold'), bd=0, highlightthickness=0, cursor='hand2',
                           anchor='w').pack(anchor='w', pady=(4, 0))
            self.label(card, note, size=8, fg=C_TXT3, padx=(24, 0)).config(wraplength=self.WIDTH - 110)

        card = self.card(body)
        self.label(card, "DATA", size=8, bold=True)
        self.data_info = self.label(card, "", size=9, fg=C_TXT1, pady=(4, 0))
        row = tk.Frame(card, bg=C_CARD)
        row.pack(fill=tk.X, pady=(6, 0))
        tk.Label(row, text="Validation share (last images)", bg=C_CARD, fg=C_TXT2, font=('Segoe UI', 9), width=28,
                 anchor='w').pack(side=tk.LEFT)
        self.val_var = tk.IntVar(value=20)
        self.spin(row, self.val_var, 5, 50).pack(side=tk.LEFT)
        tk.Label(row, text="%", bg=C_CARD, fg=C_TXT3, font=('Segoe UI', 9)).pack(side=tk.LEFT, padx=4)
        self.val_var.trace_add("write", lambda *_: self._update_data_info())
        self.label(card, "The split is by position, not random: video frames are near-duplicates, and a random split "
                         "would score the model on pictures it has practically seen.", size=8, fg=C_TXT3, pady=(4, 0))
        self.classes_note = self.label(card, "", size=8, fg=C_AMBER, pady=(2, 0))
        self._update_data_info()

        # the model card: its parts change with the training type, so they are packed again in _apply_kind
        card = self.card(body)
        self.det_title = self.label(card, "DETECTOR", size=8, bold=True)
        row = tk.Frame(card, bg=C_CARD)
        self.weights_var = tk.StringVar(value=h["detector_weights"] or DEFAULT_WEIGHTS)
        self.base_var = tk.StringVar(value=DEFAULT_YOLO[self.task])
        self.weights_box = ttk.Combobox(row, textvariable=self.weights_var, values=STOCK_WEIGHTS, font=('Segoe UI', 10),
                                        width=46, style='Jelibox.TCombobox')
        self.weights_box.pack(side=tk.LEFT, ipady=2)
        self.button(row, "Browse custom model ...", self._browse_weights).pack(side=tk.LEFT, padx=8, ipadx=8, ipady=3)
        self.det_note = tk.Label(card, text="", bg=C_CARD, fg=C_TXT3, wraplength=self.WIDTH - 70, justify=tk.LEFT,
                                 anchor='w', font=('Segoe UI', 8))
        self.taps_row = tk.Frame(card, bg=C_CARD)
        tk.Label(self.taps_row, text="Taps (neck layers)", bg=C_CARD, fg=C_TXT2, font=('Segoe UI', 9), width=28,
                 anchor='w').pack(side=tk.LEFT)
        self.taps_var = tk.StringVar()
        self.entry(self.taps_row, self.taps_var, 12).pack(side=tk.LEFT, ipady=3)
        self.taps_hint = tk.Label(self.taps_row, text="", bg=C_CARD, fg=C_TXT3, font=('Segoe UI', 8), anchor='w')
        self.taps_hint.pack(side=tk.LEFT, padx=10)
        self.weights_var.trace_add("write", lambda *_: self._update_taps())
        self._update_taps()
        self.continue_var = tk.BooleanVar(value=False)
        self.continue_chk = self.check(card, "Continue from the model trained here before (keeps what it learned)",
                                       self.continue_var)
        if not os.path.isfile(self.model_path):
            self.continue_chk.config(state=tk.DISABLED)
        self._model_parts = [          # (widget, shown for, pack options), in order
            (self.det_title, None, dict(anchor='w')),
            (row, None, dict(fill=tk.X, pady=(4, 0))),
            (self.det_note, None, dict(fill=tk.X, pady=(4, 0))),
            (self.taps_row, "head", dict(fill=tk.X, pady=(6, 0))),
            (self.continue_chk, "yolo", dict(anchor='w', pady=(6, 0))),
        ]

        card = self.card(body)
        self.label(card, "TRAINING", size=8, bold=True)
        grid = tk.Frame(card, bg=C_CARD)
        grid.pack(fill=tk.X, pady=(4, 0))
        self.h_var, self.w_var = tk.IntVar(value=DEFAULT_IMGSZ[0]), tk.IntVar(value=DEFAULT_IMGSZ[1])
        self.sq_var = tk.IntVar(value=640)
        self.freeze_var = tk.IntVar(value=0)
        self.epochs_var, self.batch_var = tk.IntVar(value=50), tk.IntVar(value=16)
        self.lr_var = tk.StringVar(value="0.001")
        self.cw_var, self.flip_var = tk.BooleanVar(value=True), tk.BooleanVar(value=True)
        self._head_cells, self._yolo_cells = [], []

        def row_label(r, text, kind=None):
            lbl = tk.Label(grid, text=text, bg=C_CARD, fg=C_TXT2, font=('Segoe UI', 9), width=28, anchor='w')
            lbl.grid(row=r, column=0, sticky='w', pady=3)
            if kind:
                (self._head_cells if kind == "head" else self._yolo_cells).append(lbl)

        row_label(0, "Image size (height × width)", "head")
        size = tk.Frame(grid, bg=C_CARD)
        size.grid(row=0, column=1, sticky='w')
        self.spin(size, self.h_var, 32, 2048, 32).pack(side=tk.LEFT)
        tk.Label(size, text="×", bg=C_CARD, fg=C_TXT2, font=('Segoe UI', 10)).pack(side=tk.LEFT, padx=6)
        self.spin(size, self.w_var, 32, 2048, 32).pack(side=tk.LEFT)
        tk.Label(size, text="multiples of 32", bg=C_CARD, fg=C_TXT3, font=('Segoe UI', 8)).pack(side=tk.LEFT, padx=8)
        self._head_cells.append(size)
        row_label(0, "Image size (square)", "yolo")
        sq = tk.Frame(grid, bg=C_CARD)
        sq.grid(row=0, column=1, sticky='w')
        self.spin(sq, self.sq_var, 32, 2048, 32).pack(side=tk.LEFT)
        tk.Label(sq, text="multiple of 32", bg=C_CARD, fg=C_TXT3, font=('Segoe UI', 8)).pack(side=tk.LEFT, padx=8)
        self._yolo_cells.append(sq)
        row_label(1, "Epochs")
        self.spin(grid, self.epochs_var, 1, 1000).grid(row=1, column=1, sticky='w')
        row_label(2, "Batch size")
        self.spin(grid, self.batch_var, 1, 128).grid(row=2, column=1, sticky='w')
        row_label(3, "Learning rate")
        self.entry(grid, self.lr_var, 10).grid(row=3, column=1, sticky='w', ipady=3)
        cw = self.check(grid, "Class weights (rare classes count more)", self.cw_var)
        cw.grid(row=4, column=0, columnspan=2, sticky='w', pady=(4, 0))
        flip = self.check(grid, "Flip images horizontally (turn off when logos or text matter)", self.flip_var)
        flip.grid(row=5, column=0, columnspan=2, sticky='w')
        self._head_cells += [cw, flip]
        row_label(4, "Freeze first layers", "yolo")
        fr = tk.Frame(grid, bg=C_CARD)
        fr.grid(row=4, column=1, sticky='w')
        self.spin(fr, self.freeze_var, 0, 100).pack(side=tk.LEFT)
        tk.Label(fr, text="0 = train every layer, 10 = keep the backbone", bg=C_CARD, fg=C_TXT3,
                 font=('Segoe UI', 8)).pack(side=tk.LEFT, padx=8)
        self._yolo_cells.append(fr)
        self.size_note = self.label(card, "", size=8, fg=C_TXT3, pady=(6, 0))
        self.device_hint = self.label(card, "", size=8, fg=C_AMBER, pady=(2, 0))
        self._device_hint()
        self._apply_kind()

    def _apply_kind(self):
        """Show the parts that belong to the chosen training type."""
        yolo = self.kind_var.get() == "yolo"
        kind = "yolo" if yolo else "head"
        self.intro.config(text=YOLO_INTRO if yolo else HEAD_INTRO)
        self.det_title.config(text="BASE MODEL" if yolo else "DETECTOR")
        self.weights_box.config(textvariable=self.base_var if yolo else self.weights_var,
                                values=STOCK_YOLO[self.task] if yolo else STOCK_WEIGHTS)
        self.det_note.config(text=(
            "A stock name is downloaded the first time. Training starts from these pretrained weights and changes "
            "all of them; pick a -seg model for polygons (this workspace is in "
            f"{'polygon' if self.task == 'seg' else 'box'} mode)." if yolo else
            "A stock name is downloaded the first time. A model you choose must be the detector you will annotate "
            "with: the head reads its features."))
        for widget, _, _ in self._model_parts:
            widget.pack_forget()
        for widget, only, opts in self._model_parts:
            if only in (None, kind):
                widget.pack(**opts)
        for widget in self._head_cells:
            widget.grid() if not yolo else widget.grid_remove()
        for widget in self._yolo_cells:
            widget.grid() if yolo else widget.grid_remove()
        self.size_note.config(text=(
            "Ultralytics resizes every picture to a square of this size (640 is the usual choice) and uses its own "
            "augmentation (mosaic, flips, colour shifts)." if yolo else
            "COCO detectors are trained on 640×640 squares, but the head reads features at any size. A wider size "
            "such as 384×640 keeps people's proportions (standing vs sitting) instead of squeezing a 16:9 picture "
            "into a square."))
        if hasattr(self, "scroll") and getattr(self, "_built", False):
            self.fit()

    def spin(self, parent, var, low, high, step=1):
        return tk.Spinbox(parent, from_=low, to=high, increment=step, textvariable=var, width=7, font=('Segoe UI', 10),
                          bg=C_CARD2, fg=C_TXT1, buttonbackground=C_CARD2, relief=tk.FLAT, highlightthickness=1,
                          highlightbackground=C_BORDER, highlightcolor=C_ACCENT, insertbackground=C_ACCENT)

    def check(self, parent, text, var):
        return tk.Checkbutton(parent, text=text, variable=var, bg=C_CARD, fg=C_TXT1, selectcolor=C_BASE,
                              activebackground=C_CARD, activeforeground=C_ACCENT, font=('Segoe UI', 9), bd=0,
                              highlightthickness=0, cursor='hand2')

    def _device_hint(self):
        try:
            import torch
            on_gpu = torch.cuda.is_available()
        except Exception:
            return
        if not on_gpu:
            self.device_hint.config(text="No NVIDIA GPU is available to PyTorch here, so training runs on the CPU: "
                                         "expect minutes per epoch for the head, and much longer for a full model. "
                                         "Try fewer epochs first.")

    def _update_data_info(self):
        n = len(self.pairs)
        try:
            share = max(5, min(50, int(self.val_var.get())))
        except (tk.TclError, ValueError):
            share = 20
        if n >= 2:
            train, val = head_data.split_contiguous(self.pairs, share / 100)
            text = f"{n} labelled images: {len(train)} to train, {len(val)} to validate."
        else:
            text = f"{n} labelled images."
        counts = head_data.class_counts(self.pairs, len(self.classes))
        used = [f"{c} {k}" for c, k in zip(self.classes, counts) if k]
        self.data_info.config(text=text + ("\nBoxes per class: " + ", ".join(used) if used else ""))
        notes = []
        if n < head_data.MIN_LABELLED:
            notes.append(f"At least {head_data.MIN_LABELLED} labelled images are needed.")
        unused = [c for c, k in zip(self.classes, counts) if not k]
        if unused and used:
            notes.append("No examples yet for: " + ", ".join(unused) + " (it will not learn them).")
        self.classes_note.config(text="  ".join(notes))

    def _update_taps(self):
        taps, known = head_data.taps_for(self.weights_var.get())
        self.taps_var.set(", ".join(str(t) for t in taps))
        self.taps_hint.config(text="" if known else "model family not recognised - check these")

    def _browse_weights(self):
        yolo = self.kind_var.get() == "yolo"
        path = filedialog.askopenfilename(parent=self.win, title="Choose the base model (.pt)" if yolo
                                          else "Choose the detector (.pt)",
                                          initialdir=os.path.join(BASE_DIR, "models"),
                                          filetypes=[("YOLO model", "*.pt"), ("All files", "*.*")])
        if path:
            (self.base_var if yolo else self.weights_var).set(path)

    # ---------------------------------------------------------- start
    def _read_common(self):
        """(epochs, batch, lr, validation share) from the shared fields; ValueError when one is out of range."""
        epochs, batch = int(self.epochs_var.get()), int(self.batch_var.get())
        if not 1 <= epochs <= 1000:
            raise ValueError("Epochs must be between 1 and 1000.")
        if not 1 <= batch <= 128:
            raise ValueError("Batch size must be between 1 and 128.")
        lr = float(self.lr_var.get())
        if not 1e-5 <= lr <= 0.1:
            raise ValueError("Learning rate must be between 0.00001 and 0.1.")
        share = int(self.val_var.get())
        if not 5 <= share <= 50:
            raise ValueError("The validation share must be between 5 and 50 %.")
        return epochs, batch, lr, share

    def _warn(self, exc):
        messagebox.showwarning("Train custom model", str(exc) if isinstance(exc, ValueError) else "Check the numbers.",
                               parent=self.win)

    def save(self):
        """The footer's main button: validate the settings and start."""
        if self.kind_var.get() == "yolo":
            self._start_yolo()
            return
        try:
            weights = self.weights_var.get().strip()
            if not weights:
                raise ValueError("Choose the detector weights.")
            if (os.path.isabs(weights) or os.sep in weights or "/" in weights) and not os.path.isfile(weights):
                raise ValueError(f"The detector file was not found:\n{weights}")
            taps = parse_taps(self.taps_var.get())
            imgsz = (int(self.h_var.get()), int(self.w_var.get()))
            if any(v < 32 or v % 32 for v in imgsz):
                raise ValueError("The image size must be two multiples of 32, e.g. 384 × 640.")
            epochs, batch, lr, share = self._read_common()
        except (ValueError, tk.TclError) as exc:
            self._warn(exc)
            return
        if len(self.pairs) < head_data.MIN_LABELLED:
            messagebox.showwarning("Train custom model", f"Only {len(self.pairs)} images have labels. At least "
                                                 f"{head_data.MIN_LABELLED} are needed.", parent=self.win)
            return
        if os.path.exists(os.path.join(self.model_folder, HEAD_FILE)) and not messagebox.askyesno(
                "Train custom model", f"This workspace already has a trained head ({HEAD_FILE}).\n\nTrain a new one? The "
                              f"current head is kept as head_best.prev.pt.", parent=self.win):
            return

        self.weights_used = weights
        job = build_job(images_folders=self.images_folders, labels_folder=self.labels_folder, classes=self.classes,
                        weights=weights, taps=taps, imgsz=imgsz, epochs=epochs, batch=batch, lr=lr,
                        class_weights=self.cw_var.get(), hflip=self.flip_var.get(), val_fraction=share / 100,
                        out_dir=self.run_dir)
        os.makedirs(self.run_dir, exist_ok=True)
        for old in ("head_best.pt", "head_last.pt"):               # an older run must not pass for this one
            try:
                os.remove(os.path.join(self.run_dir, old))
            except OSError:
                pass
        self._launch(job, "head", self.run_dir, epochs)

    def _start_yolo(self):
        try:
            base = self.base_var.get().strip()
            if not base:
                raise ValueError("Choose the base model.")
            if (os.path.isabs(base) or os.sep in base or "/" in base) and not os.path.isfile(base):
                raise ValueError(f"The model file was not found:\n{base}")
            size = int(self.sq_var.get())
            if size < 32 or size % 32:
                raise ValueError("The image size must be a multiple of 32, e.g. 640.")
            freeze = int(self.freeze_var.get())
            if not 0 <= freeze <= 100:
                raise ValueError("Freeze must be between 0 and 100 layers.")
            epochs, batch, lr, share = self._read_common()
        except (ValueError, tk.TclError) as exc:
            self._warn(exc)
            return
        if len(self.pairs) < head_data.MIN_LABELLED:
            messagebox.showwarning("Train custom model", f"Only {len(self.pairs)} images have labels. At least "
                                                 f"{head_data.MIN_LABELLED} are needed.", parent=self.win)
            return
        resume = self.continue_var.get() and os.path.isfile(self.model_path)
        if not resume and os.path.exists(self.model_path) and not messagebox.askyesno(
                "Train custom model", f"This workspace already has a trained model ({os.path.basename(self.model_path)})."
                                      f"\n\nTrain a new one from {os.path.basename(base)}? The current model is kept as "
                                      f"{os.path.basename(os.path.splitext(self.model_path)[0])}.prev.pt.",
                parent=self.win):
            return
        job = build_yolo_job(images_folders=self.images_folders, labels_folder=self.labels_folder,
                             classes=self.classes, init_weights=self.model_path if resume else base, imgsz=size,
                             epochs=epochs, batch=batch, lr=lr, freeze=freeze, val_fraction=share / 100,
                             out_dir=self.yolo_dir)
        os.makedirs(self.yolo_dir, exist_ok=True)
        shutil.rmtree(os.path.join(self.yolo_dir, RUN_NAME), ignore_errors=True)   # an older run must not pass for this one
        self._launch(job, "yolo", self.yolo_dir, epochs)

    def _remove_flag(self):
        try:
            os.remove(stop_file_for(self.active_dir))
        except OSError:
            pass

    def _launch(self, job, kind, run_dir, epochs):
        self.active_dir = run_dir
        self._remove_flag()                                      # a flag left by an older run must not end this one
        job_path = os.path.join(run_dir, "job.json")
        with open(job_path, "w", encoding="utf-8") as f:
            json.dump(job, f, indent=2)
        try:
            self.proc = TrainProcess(self.command(job_path), BASE_DIR, os.path.join(run_dir, "train.log"))
        except OSError as exc:
            messagebox.showerror("Train custom model", f"Could not start the training process:\n{exc}", parent=self.win)
            return
        self.active_kind, self.active_dir = kind, run_dir
        self.running = True
        self.started_at = time.time()
        self.epoch_seconds = []
        self.best_epoch = 0
        self.total_epochs = epochs
        self._show_progress()
        self.win.after(POLL_MS, self._poll)

    # ---------------------------------------------------------- progress view
    def _show_progress(self):
        self.scroll.frame.pack_forget()
        self.footer.pack_forget()
        yolo = self.active_kind == "yolo"
        style = ttk.Style(self.win)
        style.configure('Jelibox.Horizontal.TProgressbar', troughcolor=C_CARD, background=C_ACCENT,
                        bordercolor=C_BORDER, lightcolor=C_ACCENT, darkcolor=C_ACCENT)
        style.configure('Jelibox.Treeview', background=C_CARD2, fieldbackground=C_CARD2, foreground=C_TXT1,
                        rowheight=20, borderwidth=0, font=('Consolas', 9))
        style.configure('Jelibox.Treeview.Heading', background=C_CARD, foreground=C_TXT2, relief='flat',
                        font=('Segoe UI', 8, 'bold'))
        style.map('Jelibox.Treeview', background=[('selected', C_CARD)], foreground=[('selected', C_TXT1)])
        view = self.view = tk.Frame(self.win, bg=C_BASE)
        view.pack(fill=tk.BOTH, expand=True, padx=20, pady=(14, 16))
        self.title_label = tk.Label(view, text="Training the model ..." if yolo else "Training the head ...",
                                    bg=C_BASE, fg=C_TXT1, font=('Segoe UI', 12, 'bold'), anchor='w')
        self.title_label.pack(fill=tk.X)
        self.status_label = tk.Label(view, text="Starting ...", bg=C_BASE, fg=C_TXT2, font=('Segoe UI', 9),
                                     anchor='w', justify=tk.LEFT, wraplength=self.WIDTH - 60)
        self.status_label.pack(fill=tk.X, pady=(4, 8))
        self.bar = ttk.Progressbar(view, mode='determinate', maximum=self.total_epochs, length=self.WIDTH - 60,
                                   style='Jelibox.Horizontal.TProgressbar')
        self.bar.pack(fill=tk.X)
        self.stats_label = tk.Label(view, text="\n\n ", bg=C_BASE, fg=C_TXT1, font=('Segoe UI', 10), anchor='w',
                                    justify=tk.LEFT, wraplength=self.WIDTH - 60)
        self.stats_label.pack(fill=tk.X, pady=(10, 4))
        # one row per finished epoch
        columns = YOLO_COLUMNS if yolo else HEAD_COLUMNS
        table_box = tk.Frame(view, bg=C_BORDER)
        table_box.pack(fill=tk.X, pady=(4, 6))
        small_screen = self.win.winfo_screenheight() < 900             # a 12-inch laptop: shorter table and log
        self.table = ttk.Treeview(table_box, columns=[c[0] for c in columns], show='headings',
                                  height=6 if small_screen else 8,
                                  style='Jelibox.Treeview', selectmode='none')
        for key, title, width in columns:
            self.table.heading(key, text=title)
            self.table.column(key, width=width, anchor='center', stretch=True)
        self.table.tag_configure('best', foreground=C_GREEN)
        table_scroll = tk.Scrollbar(table_box, command=self.table.yview, width=10, bg=C_BORDER, troughcolor=C_CARD2,
                                    relief=tk.FLAT)
        self.table.configure(yscrollcommand=table_scroll.set)
        table_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.table.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=1, pady=1)
        box = tk.Frame(view, bg=C_BORDER)
        box.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        self.log = tk.Text(box, height=4 if small_screen else 6, bg=C_CARD2, fg=C_TXT2, font=('Consolas', 8), relief=tk.FLAT, wrap='none',
                           state=tk.DISABLED, highlightthickness=0)
        scroll = tk.Scrollbar(box, command=self.log.yview, width=10, bg=C_BORDER, troughcolor=C_CARD2, relief=tk.FLAT)
        self.log.configure(yscrollcommand=scroll.set)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.log.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=1, pady=1)
        buttons = tk.Frame(view, bg=C_BASE)
        buttons.pack(fill=tk.X)
        self.finish_btn = tk.Button(buttons, text="Finish early", command=self._finish_early, bg=C_ACCENT,
                                    fg=C_ON_ACCENT, font=('Segoe UI', 9, 'bold'), relief=tk.FLAT, cursor='hand2',
                                    borderwidth=0, activebackground=C_ACCENT)
        self.finish_btn.pack(side=tk.RIGHT, ipadx=16, ipady=4)
        self.stop_btn = tk.Button(buttons, text="Stop now", command=self._stop, bg=C_AMBER, fg=C_ON_ACCENT,
                                  font=('Segoe UI', 9, 'bold'), relief=tk.FLAT, cursor='hand2', borderwidth=0)
        self.stop_btn.pack(side=tk.RIGHT, padx=(0, 8), ipadx=16, ipady=4)
        self.close_btn = tk.Button(buttons, text="Close", command=self._close, bg=C_CARD2, fg=C_TXT1,
                                   font=('Segoe UI', 9), relief=tk.FLAT, cursor='hand2', borderwidth=0,
                                   state=tk.DISABLED)
        self.close_btn.pack(side=tk.RIGHT, padx=8, ipadx=16, ipady=4)
        self.win.geometry("")                                  # follow what the content asks for, as its text changes
        self.win.minsize(self.WIDTH, 1)
        self.win.update_idletasks()
        x, y = self.win.winfo_x(), self.win.winfo_y()
        y = max(0, min(y, self.win.winfo_screenheight() - self.win.winfo_reqheight() - 40))
        self.win.geometry(f"+{x}+{y}")                         # a position only: the size stays automatic

    def _log(self, text):
        self.log.config(state=tk.NORMAL)
        self.log.insert(tk.END, text + "\n")
        self.log.see(tk.END)
        self.log.config(state=tk.DISABLED)

    def _poll(self):
        if not self.win.winfo_exists():
            return
        lines, ended = self.proc.read_new()
        for line in lines:
            self._handle(line)
        if ended:
            self._ended()
            return
        self.win.after(POLL_MS, self._poll)

    def _handle(self, line):
        if not line.startswith(PREFIX):
            self._log(line)
            return
        try:
            ev = json.loads(line[len(PREFIX):])
        except ValueError:
            self._log(line)
            return
        kind = ev.get("event")
        if kind == "status":
            self.status_label.config(text=ev.get("text", ""))
        elif kind == "start":
            self.total_epochs = ev["epochs"]
            self.bar.config(maximum=ev["epochs"])
            if self.active_kind == "yolo":
                self.status_label.config(
                    text=f"{ev['train_images']} images to train, {ev['val_images']} to validate  -  "
                         f"{ev['device'].upper()}  -  batch {ev['batch']}  -  {ev['imgsz'][0]}×{ev['imgsz'][1]}  -  "
                         f"{ev.get('model', '')}  -  every layer" + (f" except the first {ev['freeze']}" if ev.get("freeze")
                                                                    else ""))
            else:
                self.status_label.config(
                    text=f"{ev['train_images']} images to train, {ev['val_images']} to validate  -  "
                         f"{ev['device'].upper()}  -  batch {ev['batch']}  -  {ev['imgsz'][0]}×{ev['imgsz'][1]}  -  "
                         f"head {ev['head_params_m']} M parameters")
            self._log(f"classes: {', '.join(ev['classes'])}   boxes: {ev['counts']}")
        elif kind == "epoch":
            self._epoch(ev)
        elif kind == "done":
            self.result = ev
        elif kind == "error":
            self.error = ev.get("message", "unknown error")
            self._log("ERROR: " + self.error)

    def _epoch(self, ev):
        self.epoch_seconds.append(ev["seconds"])
        if ev["best"]:
            self.best_epoch = ev["epoch"]
        self.bar.config(value=ev["epoch"])
        left = (ev["epochs"] - ev["epoch"]) * (sum(self.epoch_seconds) / len(self.epoch_seconds))
        timing = f"{_fmt_time(time.time() - self.started_at)} elapsed, about {_fmt_time(left)} left"
        tags = ('best',) if ev["best"] else ()
        if self.active_kind == "yolo":
            self.stats_label.config(
                text=f"Epoch {ev['epoch']} / {ev['epochs']}      box loss {ev['box_loss']:.3f}      cls loss "
                     f"{ev['cls_loss']:.3f}      dfl loss {ev['dfl_loss']:.3f}\n"
                     f"Validation: precision {ev['precision']:.3f}      recall {ev['recall']:.3f}      mAP50 "
                     f"{ev['map50']:.3f}      mAP50-95 {ev['map5095']:.3f}      best epoch so far {self.best_epoch}\n"
                     + timing)
            self.table.insert('', tk.END, tags=tags, values=(
                ev["epoch"], f"{ev['box_loss']:.3f}", f"{ev['cls_loss']:.3f}", f"{ev['dfl_loss']:.3f}",
                f"{ev['precision']:.3f}", f"{ev['recall']:.3f}", f"{ev['map50']:.3f}", f"{ev['map5095']:.3f}",
                _fmt_time(ev["seconds"])))
            self._log(f"epoch {ev['epoch']:3d}/{ev['epochs']}  box {ev['box_loss']:.3f}  cls {ev['cls_loss']:.3f}  "
                      f"dfl {ev['dfl_loss']:.3f}  P {ev['precision']:.3f}  R {ev['recall']:.3f}  "
                      f"mAP50 {ev['map50']:.3f}  mAP50-95 {ev['map5095']:.3f}  ({ev['seconds']:.0f}s)")
        else:
            self.stats_label.config(
                text=f"Epoch {ev['epoch']} / {ev['epochs']}      training loss {ev['loss']:.3f}\n"
                     f"Validation: accuracy {ev['acc']:.1%}      macro-F1 {ev['f1']:.3f}      best epoch so far "
                     f"{self.best_epoch}\n" + timing)
            self.table.insert('', tk.END, tags=tags, values=(
                ev["epoch"], f"{ev['loss']:.3f}", f"{ev['acc']:.1%}", f"{ev['f1']:.3f}", _fmt_time(ev["seconds"])))
            self._log(f"epoch {ev['epoch']:3d}/{ev['epochs']}  loss {ev['loss']:.4f}  acc {ev['acc']:.3f}  "
                      f"F1 {ev['f1']:.3f}  recall {[round(r, 2) for r in ev['recall']]}  ({ev['seconds']:.0f}s)")
        self.table.yview_moveto(1.0)

    def _ended(self):
        """The training process closed its output: show how it ended."""
        self.running = False
        self.finished = True
        self.proc.proc.wait()
        self.proc.close()
        self.stop_btn.config(state=tk.DISABLED)
        self.finish_btn.config(state=tk.DISABLED)
        self.close_btn.config(state=tk.NORMAL)
        self._remove_flag()
        if self.active_kind == "yolo":
            self._ended_yolo()
        else:
            self._ended_head()

    def _ended_head(self):
        error = self.error
        installed = None
        if error is None and (self.result is not None or self.stopping):
            installed = install_head(self.run_dir, self.model_folder)
        if installed:
            select_in_assistant(workspaceName, installed, self.weights_used)
        if error is not None:
            self.title_label.config(text="Training failed", fg=C_RED)
            self.status_label.config(text=error, fg=C_RED)
        elif installed and self.result and self.finishing and not self.stopping:
            self.title_label.config(text="Finished early", fg=C_GREEN)
            self.bar.config(value=self.result["epochs_run"])
            self.status_label.config(
                text=f"Stopped during epoch {self.result['epochs_run']}; best head: epoch {self.result['best_epoch']}, "
                     f"macro-F1 {self.result['best_f1']:.3f}.\n"
                     f"Saved as {os.path.relpath(installed, BASE_DIR)} and selected in Label Assistant - G and "
                     f"Auto-annotate all use it now.", fg=C_TXT2)
        elif installed and self.result and not self.result["stopped"]:
            self.title_label.config(text="Training finished", fg=C_GREEN)
            self.bar.config(value=self.total_epochs)
            self.status_label.config(
                text=f"Best head: epoch {self.result['best_epoch']}, macro-F1 {self.result['best_f1']:.3f}.\n"
                     f"Saved as {os.path.relpath(installed, BASE_DIR)} and selected in Label Assistant - G and "
                     f"Auto-annotate all use it now.", fg=C_TXT2)
        elif installed:
            self.title_label.config(text="Stopped - the best head so far was kept", fg=C_AMBER)
            self.status_label.config(text=f"Saved as {os.path.relpath(installed, BASE_DIR)} and selected in Label "
                                          f"Assistant.", fg=C_TXT2)
        elif self.stopping:
            self.title_label.config(text="Stopped", fg=C_AMBER)
            self.status_label.config(text="Stopped before the first epoch finished, so nothing was saved.", fg=C_TXT2)
        else:
            self.title_label.config(text="Training ended unexpectedly", fg=C_RED)
            self.status_label.config(text="The log below has the details (also in head_run/train.log). Common causes: "
                                          "PyTorch or Ultralytics missing, or not enough memory.", fg=C_RED)
        if self.on_done:
            self.on_done(installed)

    def _ended_yolo(self):
        shutil.rmtree(os.path.join(self.yolo_dir, "dataset"), ignore_errors=True)    # the scratch copy of the images
        error = self.error
        installed = None
        if error is None and (self.result is not None or self.stopping):
            best = (self.result or {}).get("weights") or yolo_best_path(self.yolo_dir)
            installed = install_yolo(best if os.path.isfile(best) else yolo_best_path(self.yolo_dir), self.model_path)
        if installed:
            select_trained_model(workspaceName)
        shown = os.path.relpath(installed, BASE_DIR) if installed else ""
        from . import assistant_modes
        if self.preset and assistant_modes.get_mode() == assistant_modes.MODE_YOLO_WORLD:
            use = f"Saved as {shown} and selected as My trained model: G and the full-dataset run use it now."
        else:
            use = (f"Saved as {shown} and selected as My trained model. YOLO-World mode uses it: press Mode, choose "
                   f"YOLO-World, then press G.")
        if error is not None:
            self.title_label.config(text="Training failed", fg=C_RED)
            self.status_label.config(text=error, fg=C_RED)
        elif installed and self.result and self.finishing and not self.stopping:
            self.title_label.config(text="Finished early", fg=C_GREEN)
            self.bar.config(value=self.result["epochs_run"])
            self.status_label.config(
                text=f"Stopped during epoch {self.result['epochs_run']}, then evaluated. Best epoch "
                     f"{self.result['best_epoch']}; last epoch: mAP50 {self.result.get('map50') or 0:.3f}, mAP50-95 "
                     f"{self.result.get('map5095') or 0:.3f}.\n" + use, fg=C_TXT2)
        elif installed and self.result and not self.result.get("stopped"):
            self.title_label.config(text="Training finished", fg=C_GREEN)
            self.bar.config(value=self.total_epochs)
            self.status_label.config(
                text=f"Best epoch {self.result['best_epoch']}: mAP50 {self.result.get('map50') or 0:.3f}, mAP50-95 "
                     f"{self.result.get('map5095') or 0:.3f} (of the last epoch).\n" + use, fg=C_TXT2)
        elif installed:
            self.title_label.config(text="Stopped - the best model so far was kept", fg=C_AMBER)
            self.status_label.config(text=use, fg=C_TXT2)
        elif self.stopping:
            self.title_label.config(text="Stopped", fg=C_AMBER)
            self.status_label.config(text="Stopped before the first epoch finished, so nothing was saved.", fg=C_TXT2)
        else:
            self.title_label.config(text="Training ended unexpectedly", fg=C_RED)
            self.status_label.config(text="The log below has the details (also in yolo_run/train.log). Common causes: "
                                          "PyTorch or Ultralytics missing, or not enough memory.", fg=C_RED)
        if self.on_done:
            self.on_done(installed)

    def _finish_early(self):
        """Ask the training to stop now but end properly: validate, save and run its final evaluation."""
        if not (self.proc and self.running) or self.finishing or self.stopping:
            return
        try:
            with open(stop_file_for(self.active_dir), "w", encoding="utf-8") as f:
                f.write("finish early\n")
        except OSError as exc:
            messagebox.showerror("Train custom model", f"Could not ask the training to finish:\n{exc}", parent=self.win)
            return
        self.finishing = True
        self.finish_btn.config(state=tk.DISABLED, text="Finishing ...")
        self.status_label.config(text="Finishing early: the epoch in progress is validated and saved, then the final "
                                      "evaluation runs. Stop now ends it at once instead.")

    def _stop(self):
        if self.proc and self.running:
            self.stopping = True
            self.stop_btn.config(state=tk.DISABLED, text="Stopping ...")
            self.finish_btn.config(state=tk.DISABLED)
            self.proc.stop()

    def _close(self):
        if self.running:
            kept = "model" if self.active_kind == "yolo" else "head"
            if not messagebox.askyesno("Train custom model", f"Training is running. Stop it?\n\nThe best {kept} so far "
                                                             f"is kept.", parent=self.win):
                return
            self._stop()
            return                                                  # Close turns on once the process has ended
        self.win.destroy()


def open_model_training(parent, images_folders, labels_folder, model_folder, job, model_path, command=None,
                        on_done=None):
    """The plain Train button (YOLO-World and the other modes): run `job` (see build_yolo_job) and show its progress
    in a window. Returns the window at once; training goes on while the user keeps annotating."""
    return HeadTrainDialog(parent, images_folders, labels_folder, model_folder, command, on_done,
                           model_path=model_path, preset=job, title="TRAIN MODEL")


def open_head_training(parent, images_folders, labels_folder, model_folder, command=None, on_done=None,
                       model_path=None, task="detect"):
    dlg = HeadTrainDialog(parent, images_folders, labels_folder, model_folder, command, on_done, model_path, task)
    parent.wait_window(dlg.win)
    return dlg
