"""
Training for the web server: the same two trainings as the desktop's Train windows (full YOLO and the Custom Model head),
the same job files and the same subprocess (`train_support.TrainProcess`), but the window is a page.

ONE training at a time on the whole server (the GPU is shared: `process_guard` keeps `temp/training.lock`, which the
desktop window uses too). The process belongs to the server, not to a browser: closing the tab does not stop it, the
page just asks `state()` again later. The process ends with the server (process_guard binds it).

A monitor thread reads the process's log (lines `@@ {json}` are events, the rest is plain log text), keeps the rows of
the epoch table and, when the process is over, does what the desktop window does: install the best model, point the
Label Assistant at it and say how it ended.
"""
import json
import os
import shutil
import threading
import time
from collections import deque

from .. import augmentation as aug
from .. import head_data, process_guard, workspace_counts
from .. import workspace_manager as wm
from ..train_support import (DEFAULT_IMGSZ, DEFAULT_WEIGHTS, DEFAULT_YOLO, HEAD_FILE, MODEL_FILE, PREFIX, RUN_NAME,
                             STOCK_WEIGHTS, STOCK_YOLO, TrainProcess, build_job, build_yolo_job, default_command,
                             install_head, install_yolo, parse_taps, select_in_assistant, select_trained_model,
                             stop_file_for, yolo_best_path)
from . import exporting
from .editing import EditError

POLL_SECONDS = 0.3
LOG_LINES = 300
MIN_LABELLED = head_data.MIN_LABELLED


class NeedsConfirm(EditError):
    """Something the person must say yes to first (a download, replacing a trained model)."""

    def __init__(self, title, message):
        super().__init__(message)
        self.title = title


class Busy(EditError):
    """Another training is running."""


def _gpu_probe():
    """True / False / None (unknown) - importing torch takes seconds, so it is asked once, aside."""
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return None


class Run:
    def __init__(self, workspace, kind, user, epochs, run_dir, model_folder, model_path, task):
        self.id = str(int(time.time() * 1000))
        self.workspace, self.kind, self.user = workspace, kind, user
        self.total_epochs, self.run_dir, self.model_folder, self.model_path, self.task = epochs, run_dir, model_folder, model_path, task
        self.started = time.time()
        self.state = "starting"                 # starting | running | ended
        self.title, self.status = "Starting ...", "Starting ..."
        self.bar = {"done": 0, "total": epochs, "phase": "epochs"}
        self.stats = ""
        self.rows, self.log = [], deque(maxlen=LOG_LINES)
        self.best_epoch = 0
        self.result = self.error = self.start_info = None
        self.stopping = self.finishing = False
        self.final = None
        self.weights_used = ""
        self.proc = None

    def public(self, rows_from=0):
        return {"id": self.id, "workspace": self.workspace, "kind": self.kind, "user": self.user, "state": self.state,
                "title": self.title, "status": self.status, "bar": self.bar, "stats": self.stats,
                "rows": self.rows[rows_from:], "rows_total": len(self.rows), "log": list(self.log)[-80:],
                "best_epoch": self.best_epoch, "result": self.result, "error": self.error, "final": self.final,
                "start_info": self.start_info, "stopping": self.stopping, "finishing": self.finishing,
                "seconds": round(time.time() - self.started, 1), "total_epochs": self.total_epochs}


class TrainingManager:
    def __init__(self, base_dir, command=None):
        self.base_dir = base_dir
        self.command = command or default_command
        self.run = None
        self.lock = threading.RLock()
        self._gpu = "unknown"
        self._gpu_thread = None

    # ---------------------------------------------------------------- what the setup page shows
    def gpu(self):
        if self._gpu == "unknown" and self._gpu_thread is None:
            def probe():
                self._gpu = _gpu_probe()
            self._gpu_thread = threading.Thread(target=probe, daemon=True)
            self._gpu_thread.start()
        return None if self._gpu == "unknown" else self._gpu

    def info(self, ws, polygon=False):
        counts = workspace_counts.get_or_empty(ws.name)
        task = "seg" if polygon else "detect"
        model_path = os.path.join(ws.base_dir, "models", ws.name, MODEL_FILE)
        classes = ws.classes.get_classes()
        return {
            "workspace": ws.name, "task": task, "classes": classes, "counts": counts, "min_labelled": MIN_LABELLED,
            "has_model": os.path.isfile(model_path),
            "has_head": os.path.isfile(os.path.join(ws.base_dir, "models", ws.name, HEAD_FILE)),
            "defaults": {"base": DEFAULT_YOLO[task], "detector": DEFAULT_WEIGHTS, "imgsz": 640,
                         "head_size": list(DEFAULT_IMGSZ), "epochs": 50, "batch": 16, "lr": 0.001},
            "stock": {"yolo": STOCK_YOLO[task], "detector": STOCK_WEIGHTS},
            "taps": {w: list(head_data.taps_for(w)[0]) for w in STOCK_WEIGHTS},
            "gpu": self.gpu(), "running": self.is_running(),
        }

    def is_running(self):
        with self.lock:
            return bool(self.run and self.run.state != "ended")

    # ---------------------------------------------------------------- start
    def start(self, ws, req, user, confirmed=False):
        """Validate like the desktop window, then start the process. Raises EditError with a message to show,
        NeedsConfirm for what needs a yes, Busy when a training already runs."""
        kind = req.get("kind")
        if kind not in ("yolo", "head"):
            raise EditError("Choose the training type.")
        task = "seg" if req.get("polygon") else "detect"
        model_folder = os.path.join(ws.base_dir, "models", ws.name)
        model_path = os.path.join(model_folder, MODEL_FILE)
        counts = workspace_counts.get_or_empty(ws.name)
        if counts.get("counted") and counts["labelled"] < MIN_LABELLED:
            raise EditError(f"Only {counts['labelled']} images have labels. At least {MIN_LABELLED} are needed.")
        epochs, batch, lr = self._common(req)
        folders = [wm.instance_path(i) for i in wm.list_workspaces().get(ws.name, [])]
        if kind == "yolo":
            job, run_dir, init = self._yolo_job(ws, req, folders, model_folder, model_path, epochs, batch, lr, confirmed)
            weights_used = init
        else:
            job, run_dir, weights_used = self._head_job(ws, req, folders, model_folder, epochs, batch, lr, confirmed)
        with self.lock:
            other = process_guard.running_training(self.base_dir)
            if other or self.is_running():
                who = (other or {}).get("workspace") or self.run.workspace
                raise Busy(f"A training is already running (workspace \"{who}\"). Only one fits on the GPU at a time: "
                           f"wait for it, or stop it, then start this one.")
            os.makedirs(run_dir, exist_ok=True)
            if kind == "yolo":
                shutil.rmtree(os.path.join(run_dir, RUN_NAME), ignore_errors=True)    # an older run must not pass for this one
            else:
                for old in ("head_best.pt", "head_last.pt"):
                    try:
                        os.remove(os.path.join(run_dir, old))
                    except OSError:
                        pass
            self._remove_flag(run_dir)
            job_path = os.path.join(run_dir, "job.json")
            with open(job_path, "w", encoding="utf-8") as f:
                json.dump(job, f, indent=2)
            run = Run(ws.name, kind, user, epochs, run_dir, model_folder, model_path, task)
            run.weights_used = weights_used
            try:
                run.proc = TrainProcess(self.command(job_path), self.base_dir, os.path.join(run_dir, "train.log"))
            except OSError as exc:
                raise EditError(f"Could not start the training process:\n{exc}")
            process_guard.claim(self.base_dir, ws.name, run.proc.proc.pid)
            run.state = "running"
            run.title = "Training the model ..." if kind == "yolo" else "Training the head ..."
            self.run = run
        threading.Thread(target=self._monitor, args=(run,), name="train-monitor", daemon=True).start()
        return run.public()

    @staticmethod
    def _common(req):
        epochs = exporting._int(req.get("epochs"), "Epochs", 1, 1000)
        batch = exporting._int(req.get("batch"), "Batch size", 1, 128)
        try:
            lr = float(req.get("lr"))
        except (TypeError, ValueError):
            raise EditError("Learning rate must be a number.")
        if not 1e-5 <= lr <= 0.1:
            raise EditError("Learning rate must be between 0.00001 and 0.1.")
        return epochs, batch, lr

    def _yolo_job(self, ws, req, folders, model_folder, model_path, epochs, batch, lr, confirmed):
        from .. import yolo_train_job
        base = str(req.get("base") or "").strip()
        if not base:
            raise EditError("Choose the base model.")
        if (os.path.isabs(base) or os.sep in base or "/" in base) and not os.path.isfile(base):
            raise EditError(f"The model file was not found:\n{base}")
        size = exporting._int(req.get("imgsz"), "The image size", 32, 2048)
        if size % 32:
            raise EditError("The image size must be a multiple of 32, e.g. 640.")
        freeze = exporting._int(req.get("freeze", 0), "Freeze", 0, 100)
        try:
            weight_decay = float(req.get("weight_decay", 0.0005))
        except (TypeError, ValueError):
            raise EditError("Weight decay must be a number between 0 and 0.1.")
        if not 0 <= weight_decay <= 0.1:
            raise EditError("Weight decay must be between 0 and 0.1.")
        valid = exporting._int(req.get("valid_pct"), "The validation share", 0, 100)
        test = exporting._int(req.get("test_pct", 0), "The test share", 0, 100)
        if valid < 1:
            raise EditError("Give the validation part at least 1 %: the training needs pictures to score itself on.")
        if 100 - valid - test < 1:
            raise EditError("Give the training part at least 1 %.")
        resume = bool(req.get("continue")) and os.path.isfile(model_path)
        if not resume and os.path.exists(model_path) and not confirmed:
            raise NeedsConfirm("Train custom model", f"This workspace already has a trained model ({MODEL_FILE}).\n\nTrain a new "
                               f"one from {os.path.basename(base)}? The current model is kept as modelAssistant.prev.pt.")
        if not resume and yolo_train_job.needs_download(base, ws.base_dir) and not confirmed:
            raise NeedsConfirm("Train custom model - first use",
                               f"The base model {base} is not on the server yet.\n\nJelibox will download it once from "
                               f"github.com (Ultralytics' model files) into\n{yolo_train_job.STOCK_DIR}\n\n"
                               f"Download it when the training starts?")
        config = exporting.build_augment(req.get("augment"))
        counts = workspace_counts.get_or_empty(ws.name)
        run_dir = os.path.join(model_folder, "yolo_run")
        job = build_yolo_job(images_folders=folders, labels_folder=ws.labels, classes=ws.classes.get_classes(),
                             init_weights=model_path if resume else base, imgsz=size, epochs=epochs, batch=batch, lr=lr,
                             freeze=freeze, weight_decay=weight_decay, val_fraction=valid / 100, test_fraction=test / 100,
                             shuffle=bool(req.get("shuffle")), split_seed=int(time.time()),
                             background=bool(counts.get("background")) and bool(req.get("background")),
                             out_dir=run_dir, augment=aug.config_to_dict(config) if config is not None else None)
        return job, run_dir, (model_path if resume else base)

    def _head_job(self, ws, req, folders, model_folder, epochs, batch, lr, confirmed):
        weights = str(req.get("detector") or "").strip()
        if not weights:
            raise EditError("Choose the detector weights.")
        if (os.path.isabs(weights) or os.sep in weights or "/" in weights) and not os.path.isfile(weights):
            raise EditError(f"The detector file was not found:\n{weights}")
        try:
            taps = parse_taps(str(req.get("taps") or ""))
        except ValueError as exc:
            raise EditError(str(exc))
        h, w = exporting._int(req.get("height"), "The image height", 32, 2048), exporting._int(req.get("width"), "The image width", 32, 2048)
        if h % 32 or w % 32:
            raise EditError("The image size must be two multiples of 32, e.g. 384 × 640.")
        share = exporting._int(req.get("val_share", 20), "The validation share", 5, 50)
        if os.path.exists(os.path.join(model_folder, HEAD_FILE)) and not confirmed:
            raise NeedsConfirm("Train custom model", f"This workspace already has a trained head ({HEAD_FILE}).\n\nTrain a new one? "
                               f"The current head is kept as head_best.prev.pt.")
        run_dir = os.path.join(model_folder, "head_run")
        job = build_job(images_folders=folders, labels_folder=ws.labels, classes=ws.classes.get_classes(), weights=weights,
                        taps=taps, imgsz=(h, w), epochs=epochs, batch=batch, lr=lr,
                        class_weights=bool(req.get("class_weights", True)), hflip=bool(req.get("hflip", True)),
                        val_fraction=share / 100, out_dir=run_dir, base_dir=self.base_dir)
        return job, run_dir, weights

    # ---------------------------------------------------------------- the live run
    def state(self, rows_from=0):
        with self.lock:
            return {"run": self.run.public(rows_from) if self.run else None, "running": self.is_running()}

    def _remove_flag(self, run_dir):
        try:
            os.remove(stop_file_for(run_dir))
        except OSError:
            pass

    def finish_early(self, user=None):
        with self.lock:
            run = self.run
            if not (run and run.state == "running") or run.finishing or run.stopping:
                raise EditError("There is no training to finish.")
            with open(stop_file_for(run.run_dir), "w", encoding="utf-8") as f:
                f.write("finish early\n")
            run.finishing = True
            run.status = ("Finishing early: the epoch in progress is validated and saved, then the final evaluation "
                          "runs. Stop now ends it at once instead.")
            return run.public()

    def stop(self, user=None):
        with self.lock:
            run = self.run
            if not (run and run.state == "running"):
                raise EditError("There is no training to stop.")
            run.stopping = True
            run.proc.stop()
            return run.public()

    def dismiss(self):
        """Forget a finished run (the page shows the next setup)."""
        with self.lock:
            if self.run and self.run.state == "ended":
                self.run = None

    def stop_and_wait(self, timeout=30):
        """For tests and shutdown: stop the run and wait until it is over."""
        with self.lock:
            run = self.run
        if run and run.state == "running":
            try:
                self.stop()
            except EditError:
                pass
            end = time.time() + timeout
            while run.state != "ended" and time.time() < end:
                time.sleep(0.1)

    # ---------------------------------------------------------------- monitor
    def _monitor(self, run):
        try:
            while True:
                lines, ended = run.proc.read_new()
                with self.lock:
                    for line in lines:
                        self._handle(run, line)
                if ended:
                    break
                time.sleep(POLL_SECONDS)
            with self.lock:
                self._ended(run)
        except Exception as exc:                                # the page must still get an answer
            with self.lock:
                run.error = run.error or f"The training monitor failed: {exc}"
                run.state = "ended"
                run.final = {"kind": "error", "title": "Training failed", "text": run.error, "installed": ""}

    def _handle(self, run, line):
        if not line.startswith(PREFIX):
            run.log.append(line)
            return
        try:
            ev = json.loads(line[len(PREFIX):])
        except ValueError:
            run.log.append(line)
            return
        kind = ev.get("event")
        if kind == "status":
            run.status = ev.get("text", "")
        elif kind == "download":
            run.title, run.status = "Downloading the base model ...", ev.get("text", "")
            run.bar = {"done": 0, "total": 0, "phase": "download"}
        elif kind == "prepare":
            run.title, run.status = "Preparing the data ...", ev.get("text", "")
            run.bar = {"done": int(ev.get("done") or 0), "total": int(ev.get("total") or 0), "phase": "prepare"}
        elif kind == "start":
            run.start_info = ev
            run.total_epochs = ev["epochs"]
            run.bar = {"done": 0, "total": ev["epochs"], "phase": "epochs"}
            run.title = "Training the model ..." if run.kind == "yolo" else "Training the head ..."
            device = str(ev["device"]).upper()
            if run.kind == "yolo":
                run.status = (f"{ev['train_images']} images to train, {ev['val_images']} to validate  -  {device}  -  batch "
                              f"{ev['batch']}  -  {ev['imgsz'][0]}×{ev['imgsz'][1]}  -  {ev.get('model', '')}  -  every layer"
                              + (f" except the first {ev['freeze']}" if ev.get("freeze") else ""))
            else:
                run.status = (f"{ev['train_images']} images to train, {ev['val_images']} to validate  -  {device}  -  batch "
                              f"{ev['batch']}  -  {ev['imgsz'][0]}×{ev['imgsz'][1]}  -  head {ev['head_params_m']} M parameters")
            run.log.append(f"classes: {', '.join(ev['classes'])}   boxes: {ev['counts']}")
        elif kind == "epoch":
            run.rows.append(ev)
            if ev.get("best"):
                run.best_epoch = ev["epoch"]
            run.bar = {"done": ev["epoch"], "total": ev["epochs"], "phase": "epochs"}
        elif kind == "done":
            run.result = ev
        elif kind == "error":
            run.error = ev.get("message", "unknown error")
            run.log.append("ERROR: " + run.error)

    def _ended(self, run):
        """The process has closed its output: install the best model and say how it ended."""
        run.proc.proc.wait()
        run.proc.close()
        process_guard.release(self.base_dir, run.proc.proc.pid)
        self._remove_flag(run.run_dir)
        if run.kind == "yolo":
            run.final = self._ended_yolo(run)
        else:
            run.final = self._ended_head(run)
        run.title = run.final["title"]
        run.status = run.final["text"]
        run.state = "ended"
        if run.result and run.result.get("epochs_run"):
            run.bar = {"done": run.result["epochs_run"], "total": run.total_epochs, "phase": "epochs"}

    def _ended_head(self, run):
        installed = None
        if run.error is None and (run.result is not None or run.stopping):
            installed = install_head(run.run_dir, run.model_folder)
        if installed:
            select_in_assistant(run.workspace, installed, run.weights_used, self.base_dir)
        shown = os.path.relpath(installed, self.base_dir).replace("\\", "/") if installed else ""
        r = run.result
        tail = f"Saved as {shown} and selected in Label Assistant - G and Auto-annotate all use it now."
        if run.error is not None:
            return {"kind": "error", "title": "Training failed", "text": run.error, "installed": ""}
        if installed and r and run.finishing and not run.stopping:
            return {"kind": "ok", "title": "Finished early", "installed": shown,
                    "text": f"Stopped during epoch {r['epochs_run']}; best head: epoch {r['best_epoch']}, macro-F1 {r['best_f1']:.3f}.\n{tail}"}
        if installed and r and not r["stopped"]:
            return {"kind": "ok", "title": "Training finished", "installed": shown,
                    "text": f"Best head: epoch {r['best_epoch']}, macro-F1 {r['best_f1']:.3f}.\n{tail}"}
        if installed:
            return {"kind": "warn", "title": "Stopped - the best head so far was kept", "installed": shown,
                    "text": f"Saved as {shown} and selected in Label Assistant."}
        if run.stopping:
            return {"kind": "warn", "title": "Stopped", "installed": "", "text": "Stopped before the first epoch finished, so nothing was saved."}
        return {"kind": "error", "title": "Training ended unexpectedly", "installed": "",
                "text": "The log has the details (also in head_run/train.log). Common causes: PyTorch or Ultralytics missing, "
                        "or not enough memory."}

    def _ended_yolo(self, run):
        shutil.rmtree(os.path.join(run.run_dir, "dataset"), ignore_errors=True)       # the scratch copy of the images
        installed = None
        if run.error is None and (run.result is not None or run.stopping):
            best = (run.result or {}).get("weights") or yolo_best_path(run.run_dir)
            installed = install_yolo(best if os.path.isfile(best) else yolo_best_path(run.run_dir), run.model_path)
        if installed:
            select_trained_model(run.workspace)
        shown = os.path.relpath(installed, self.base_dir).replace("\\", "/") if installed else ""
        r = run.result
        use = (f"Saved as {shown} and selected as My trained model. YOLO-World mode uses it: choose that mode on the "
               f"workspace page, then press G.")
        if r and r.get("test_map50") is not None:
            use = (f"Test ({r.get('test_images')} pictures never used in training): mAP50 {r['test_map50']:.3f}, "
                   f"mAP50-95 {r.get('test_map5095') or 0:.3f}.\n" + use)
        if run.error is not None:
            return {"kind": "error", "title": "Training failed", "text": run.error, "installed": ""}
        if installed and r and run.finishing and not run.stopping:
            return {"kind": "ok", "title": "Finished early", "installed": shown,
                    "text": f"Stopped during epoch {r['epochs_run']}, then evaluated. Best epoch {r['best_epoch']}; last epoch: mAP50 "
                            f"{r.get('map50') or 0:.3f}, mAP50-95 {r.get('map5095') or 0:.3f}.\n" + use}
        if installed and r and not r.get("stopped"):
            return {"kind": "ok", "title": "Training finished", "installed": shown,
                    "text": f"Best epoch {r['best_epoch']}: mAP50 {r.get('map50') or 0:.3f}, mAP50-95 {r.get('map5095') or 0:.3f} "
                            f"(of the last epoch).\n" + use}
        if installed:
            return {"kind": "warn", "title": "Stopped - the best model so far was kept", "installed": shown, "text": use}
        if run.stopping:
            return {"kind": "warn", "title": "Stopped", "installed": "", "text": "Stopped before the first epoch finished, so nothing was saved."}
        return {"kind": "error", "title": "Training ended unexpectedly", "installed": "",
                "text": "The log has the details (also in yolo_run/train.log). Common causes: PyTorch or Ultralytics missing, "
                        "or not enough memory."}
