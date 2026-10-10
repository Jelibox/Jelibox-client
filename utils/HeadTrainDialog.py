"""
The Train button of the Custom head mode: train a head on the labels of this workspace.

Setup window (detector, taps, image size, epochs ...) -> the same window turns into a progress window while
`utils/head_train.py` runs as a separate process -> on success the head is saved as models/<workspace>/head_best.pt
and selected in Label Assistant, so G and Auto-annotate all use it right away.

Import this module only after config.load_workspace() has run.
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


# ------------------------------------------------------------------ pieces without any window (tested alone)
def default_command(job_path):
    return [sys.executable, "-u", "-m", "utils.head_train", job_path]


def build_job(*, images_folders, labels_folder, classes, weights, taps, imgsz, epochs, batch, lr, class_weights,
              hflip, val_fraction, out_dir, base_dir=BASE_DIR, workers=2):
    return {
        "images_folders": list(images_folders), "labels_folder": labels_folder, "classes": list(classes),
        "weights": weights, "detectors_dir": os.path.join(base_dir, "models", "_detectors"),
        "taps": list(taps), "imgsz": list(imgsz), "epochs": int(epochs), "batch": int(batch), "lr": float(lr),
        "workers": workers, "class_weights": bool(class_weights), "hflip": bool(hflip),
        "val_fraction": float(val_fraction), "out_dir": out_dir,
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


# ------------------------------------------------------------------ the window
class HeadTrainDialog(_Dialog):
    TITLE = "TRAIN CUSTOM MODEL"
    WIDTH = 700
    SAVE_TEXT = "Start training"

    def __init__(self, parent, images_folders, labels_folder, model_folder, command=None, on_done=None):
        self.images_folders, self.labels_folder, self.model_folder = list(images_folders), labels_folder, model_folder
        self.run_dir = os.path.join(model_folder, "head_run")
        self.command = command or default_command
        self.on_done = on_done
        self.proc = None
        self.running = False
        self.finished = False
        self.stopping = False
        self.result = None
        self.error = None
        self.pairs = head_data.labelled_pairs(self.images_folders, self.labels_folder)
        super().__init__(parent)
        self.win.title("Train custom model")
        self.win.bind('<Escape>', lambda e: self._close())
        self.win.protocol("WM_DELETE_WINDOW", self._close)

    # ---------------------------------------------------------- setup view
    def build(self, body):
        h = self.cfg["custom_head"]
        self.label(body, "Trains a head on this workspace's labels: the box is what the detector finds (a person, a "
                         "bottle ...) and its class is what the head learns (a behavior, a brand ...). The detector "
                         "itself is never changed.", pady=(0, 8))

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
                         "would score the head on pictures it has practically seen.", size=8, fg=C_TXT3, pady=(4, 0))
        self.classes_note = self.label(card, "", size=8, fg=C_AMBER, pady=(2, 0))
        self._update_data_info()

        card = self.card(body)
        self.label(card, "DETECTOR", size=8, bold=True)
        row = tk.Frame(card, bg=C_CARD)
        row.pack(fill=tk.X, pady=(4, 0))
        self.weights_var = tk.StringVar(value=h["detector_weights"] or DEFAULT_WEIGHTS)
        self.weights_box = ttk.Combobox(row, textvariable=self.weights_var, values=STOCK_WEIGHTS, font=('Segoe UI', 10),
                                        width=46, style='Jelibox.TCombobox')
        self.weights_box.pack(side=tk.LEFT, ipady=2)
        self.button(row, "Browse custom model ...", self._browse_weights).pack(side=tk.LEFT, padx=8, ipadx=8, ipady=3)
        self.label(card, "A stock name is downloaded the first time. A model you choose must be the detector you will "
                         "annotate with: the head reads its features.", size=8, fg=C_TXT3, pady=(4, 0))
        row = tk.Frame(card, bg=C_CARD)
        row.pack(fill=tk.X, pady=(6, 0))
        tk.Label(row, text="Taps (neck layers)", bg=C_CARD, fg=C_TXT2, font=('Segoe UI', 9), width=28,
                 anchor='w').pack(side=tk.LEFT)
        self.taps_var = tk.StringVar()
        self.entry(row, self.taps_var, 12).pack(side=tk.LEFT, ipady=3)
        self.taps_hint = tk.Label(row, text="", bg=C_CARD, fg=C_TXT3, font=('Segoe UI', 8), anchor='w')
        self.taps_hint.pack(side=tk.LEFT, padx=10)
        self.weights_var.trace_add("write", lambda *_: self._update_taps())
        self._update_taps()

        card = self.card(body)
        self.label(card, "TRAINING", size=8, bold=True)
        grid = tk.Frame(card, bg=C_CARD)
        grid.pack(fill=tk.X, pady=(4, 0))
        self.h_var, self.w_var = tk.IntVar(value=DEFAULT_IMGSZ[0]), tk.IntVar(value=DEFAULT_IMGSZ[1])
        self.epochs_var, self.batch_var = tk.IntVar(value=50), tk.IntVar(value=16)
        self.lr_var = tk.StringVar(value="0.001")
        self.cw_var, self.flip_var = tk.BooleanVar(value=True), tk.BooleanVar(value=True)

        def row_label(r, text):
            tk.Label(grid, text=text, bg=C_CARD, fg=C_TXT2, font=('Segoe UI', 9), width=28, anchor='w'
                     ).grid(row=r, column=0, sticky='w', pady=3)

        row_label(0, "Image size (height × width)")
        size = tk.Frame(grid, bg=C_CARD)
        size.grid(row=0, column=1, sticky='w')
        self.spin(size, self.h_var, 32, 2048, 32).pack(side=tk.LEFT)
        tk.Label(size, text="×", bg=C_CARD, fg=C_TXT2, font=('Segoe UI', 10)).pack(side=tk.LEFT, padx=6)
        self.spin(size, self.w_var, 32, 2048, 32).pack(side=tk.LEFT)
        tk.Label(size, text="multiples of 32", bg=C_CARD, fg=C_TXT3, font=('Segoe UI', 8)).pack(side=tk.LEFT, padx=8)
        row_label(1, "Epochs")
        self.spin(grid, self.epochs_var, 1, 1000).grid(row=1, column=1, sticky='w')
        row_label(2, "Batch size")
        self.spin(grid, self.batch_var, 1, 128).grid(row=2, column=1, sticky='w')
        row_label(3, "Learning rate")
        self.entry(grid, self.lr_var, 10).grid(row=3, column=1, sticky='w', ipady=3)
        self.check(grid, "Class weights (rare classes count more)", self.cw_var).grid(
            row=4, column=0, columnspan=2, sticky='w', pady=(4, 0))
        self.check(grid, "Flip images horizontally (turn off when logos or text matter)", self.flip_var).grid(
            row=5, column=0, columnspan=2, sticky='w')
        self.label(card, "COCO detectors are trained on 640×640 squares, but the head reads features at any size. A "
                         "wider size such as 384×640 keeps people's proportions (standing vs sitting) instead of "
                         "squeezing a 16:9 picture into a square.", size=8, fg=C_TXT3, pady=(6, 0))
        self.device_hint = self.label(card, "", size=8, fg=C_AMBER, pady=(2, 0))
        self._device_hint()

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
                                         "expect minutes per epoch. Try fewer epochs first.")

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
            notes.append("No examples yet for: " + ", ".join(unused) + " (the head will not learn them).")
        self.classes_note.config(text="  ".join(notes))

    def _update_taps(self):
        taps, known = head_data.taps_for(self.weights_var.get())
        self.taps_var.set(", ".join(str(t) for t in taps))
        self.taps_hint.config(text="" if known else "model family not recognised - check these")

    def _browse_weights(self):
        path = filedialog.askopenfilename(parent=self.win, title="Choose the detector (.pt)",
                                          initialdir=os.path.join(BASE_DIR, "models"),
                                          filetypes=[("YOLO model", "*.pt"), ("All files", "*.*")])
        if path:
            self.weights_var.set(path)

    # ---------------------------------------------------------- start
    def save(self):
        """The footer's main button: validate the settings and start."""
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
        except (ValueError, tk.TclError) as exc:
            messagebox.showwarning("Train custom model", str(exc) if isinstance(exc, ValueError) else "Check the numbers.",
                                   parent=self.win)
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
        job_path = os.path.join(self.run_dir, "job.json")
        with open(job_path, "w", encoding="utf-8") as f:
            json.dump(job, f, indent=2)
        try:
            self.proc = TrainProcess(self.command(job_path), BASE_DIR, os.path.join(self.run_dir, "train.log"))
        except OSError as exc:
            messagebox.showerror("Train custom model", f"Could not start the training process:\n{exc}", parent=self.win)
            return
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
        style = ttk.Style(self.win)
        style.configure('Jelibox.Horizontal.TProgressbar', troughcolor=C_CARD, background=C_ACCENT,
                        bordercolor=C_BORDER, lightcolor=C_ACCENT, darkcolor=C_ACCENT)
        view = self.view = tk.Frame(self.win, bg=C_BASE)
        view.pack(fill=tk.BOTH, expand=True, padx=20, pady=(14, 16))
        self.title_label = tk.Label(view, text="Training the head ...", bg=C_BASE, fg=C_TXT1,
                                    font=('Segoe UI', 12, 'bold'), anchor='w')
        self.title_label.pack(fill=tk.X)
        self.status_label = tk.Label(view, text="Starting ...", bg=C_BASE, fg=C_TXT2, font=('Segoe UI', 9),
                                     anchor='w', justify=tk.LEFT, wraplength=self.WIDTH - 60)
        self.status_label.pack(fill=tk.X, pady=(4, 8))
        self.bar = ttk.Progressbar(view, mode='determinate', maximum=self.total_epochs, length=self.WIDTH - 60,
                                   style='Jelibox.Horizontal.TProgressbar')
        self.bar.pack(fill=tk.X)
        self.stats_label = tk.Label(view, text="", bg=C_BASE, fg=C_TXT1, font=('Segoe UI', 10), anchor='w',
                                    justify=tk.LEFT, wraplength=self.WIDTH - 60)
        self.stats_label.pack(fill=tk.X, pady=(10, 4))
        box = tk.Frame(view, bg=C_BORDER)
        box.pack(fill=tk.BOTH, expand=True, pady=(4, 10))
        self.log = tk.Text(box, height=11, bg=C_CARD2, fg=C_TXT2, font=('Consolas', 8), relief=tk.FLAT, wrap='none',
                           state=tk.DISABLED, highlightthickness=0)
        scroll = tk.Scrollbar(box, command=self.log.yview, width=10, bg=C_BORDER, troughcolor=C_CARD2, relief=tk.FLAT)
        self.log.configure(yscrollcommand=scroll.set)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.log.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=1, pady=1)
        buttons = tk.Frame(view, bg=C_BASE)
        buttons.pack(fill=tk.X)
        self.stop_btn = tk.Button(buttons, text="Stop", command=self._stop, bg=C_AMBER, fg=C_ON_ACCENT,
                                  font=('Segoe UI', 9, 'bold'), relief=tk.FLAT, cursor='hand2', borderwidth=0)
        self.stop_btn.pack(side=tk.RIGHT, ipadx=22, ipady=4)
        self.close_btn = tk.Button(buttons, text="Close", command=self._close, bg=C_CARD2, fg=C_TXT1,
                                   font=('Segoe UI', 9), relief=tk.FLAT, cursor='hand2', borderwidth=0,
                                   state=tk.DISABLED)
        self.close_btn.pack(side=tk.RIGHT, padx=8, ipadx=16, ipady=4)
        self.scroll.fit(self.win, self.WIDTH)

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
            self.status_label.config(text=f"{ev['train_images']} images to train, {ev['val_images']} to validate  -  "
                                          f"{ev['device'].upper()}  -  batch {ev['batch']}  -  "
                                          f"{ev['imgsz'][0]}×{ev['imgsz'][1]}  -  head {ev['head_params_m']} M parameters")
            self._log(f"classes: {', '.join(ev['classes'])}   boxes: {ev['counts']}")
        elif kind == "epoch":
            self.epoch_seconds.append(ev["seconds"])
            if ev["best"]:
                self.best_epoch = ev["epoch"]
            self.bar.config(value=ev["epoch"])
            left = (ev["epochs"] - ev["epoch"]) * (sum(self.epoch_seconds) / len(self.epoch_seconds))
            self.stats_label.config(
                text=f"Epoch {ev['epoch']} / {ev['epochs']}      training loss {ev['loss']:.3f}\n"
                     f"Validation: accuracy {ev['acc']:.1%}      macro-F1 {ev['f1']:.3f}      best epoch so far "
                     f"{self.best_epoch}\n"
                     f"{_fmt_time(time.time() - self.started_at)} elapsed, about {_fmt_time(left)} left")
            self._log(f"epoch {ev['epoch']:3d}/{ev['epochs']}  loss {ev['loss']:.4f}  acc {ev['acc']:.3f}  "
                      f"F1 {ev['f1']:.3f}  recall {[round(r, 2) for r in ev['recall']]}  ({ev['seconds']:.0f}s)")
        elif kind == "done":
            self.result = ev
        elif kind == "error":
            self.error = ev.get("message", "unknown error")
            self._log("ERROR: " + self.error)

    def _ended(self):
        """The training process closed its output: show how it ended."""
        self.running = False
        self.finished = True
        self.proc.proc.wait()
        self.proc.close()
        self.stop_btn.config(state=tk.DISABLED)
        self.close_btn.config(state=tk.NORMAL)
        error = self.error
        installed = None
        if error is None and (self.result is not None or self.stopping):
            installed = install_head(self.run_dir, self.model_folder)
        if installed:
            select_in_assistant(workspaceName, installed, self.weights_used)
        if error is not None:
            self.title_label.config(text="Training failed", fg=C_RED)
            self.status_label.config(text=error, fg=C_RED)
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

    def _stop(self):
        if self.proc and self.running:
            self.stopping = True
            self.stop_btn.config(state=tk.DISABLED, text="Stopping ...")
            self.proc.stop()

    def _close(self):
        if self.running:
            if not messagebox.askyesno("Train custom model", "Training is running. Stop it?\n\nThe best head so far is kept.",
                                       parent=self.win):
                return
            self._stop()
            return                                                  # Close turns on once the process has ended
        self.win.destroy()


def open_head_training(parent, images_folders, labels_folder, model_folder, command=None, on_done=None):
    dlg = HeadTrainDialog(parent, images_folders, labels_folder, model_folder, command, on_done)
    parent.wait_window(dlg.win)
    return dlg
