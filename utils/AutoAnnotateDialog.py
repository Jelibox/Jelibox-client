"""
Progress windows for the Label Assistant.

  - run_blocking(...)     : a small modal "please wait" window for one long job (loading LocateAnything, running a
                            heavy model on the current image). The job runs on a worker thread, so the window
                            stays alive and can show what is happening.
  - open_auto_annotate()  : the "Auto-annotate all images" window - runs the configured provider over every image
                            of the dataset folder, with a progress bar, ETA and a Stop button.

Import this module only after config.load_workspace() has run.
"""
import os
import time
import tkinter as tk
from tkinter import ttk, messagebox

from . import workspace_config as wcfg
from . import assistant_modes
from . import file_handler
from . import inferenceObjectDetection as inf
from .assistant_providers import AssistantError
from .auto_annotate import BatchWorker, run_batch
from .config import input_folder, workspaceName
from .theme import (C_BASE, C_PANEL, C_CARD, C_CARD2, C_BORDER, C_ACCENT, C_TXT1, C_TXT2, C_TXT3,
                    C_ON_ACCENT, C_AMBER, C_GREEN)

POLL_MS = 150


def _style(win):
    style = ttk.Style(win)
    style.theme_use('clam')
    style.configure('Jelibox.Horizontal.TProgressbar', troughcolor=C_CARD, background=C_ACCENT,
                    bordercolor=C_BORDER, lightcolor=C_ACCENT, darkcolor=C_ACCENT)


def _center(win, parent, w, h):
    parent.update_idletasks()
    x = parent.winfo_x() + parent.winfo_width() // 2 - w // 2
    y = parent.winfo_y() + parent.winfo_height() // 2 - h // 2
    win.geometry(f"{w}x{h}+{max(x, 0)}+{max(y, 0)}")


def _fmt_time(seconds):
    seconds = int(max(0, seconds))
    return f"{seconds // 60}:{seconds % 60:02d}"


def confirm_download_if_needed(parent, mode=None):
    """LocateAnything downloads its 7 GB model from Hugging Face the first time. Jelibox works offline by
    default, so ask before anything leaves the machine. True = go ahead."""
    if inf.current_provider(mode) != wcfg.PROVIDER_LOCATE:
        return True
    from . import locate_anything as la
    problem = la.check_requirements()
    if problem:
        messagebox.showwarning("LocateAnything", problem, parent=parent)
        return False
    if la.is_cached():
        return True
    return messagebox.askyesno(
        "LocateAnything - first use",
        f"LocateAnything-3B is not on this computer yet.\n\n"
        f"Jelibox will download it from huggingface.co (about {la.DOWNLOAD_GB} GB) into\n{la.HF_HOME}\n\n"
        f"and run the model's own code from that repository (pinned to a fixed version). "
        f"This happens once. Download now?", parent=parent)


def run_blocking(parent, title, work, first_line="Working ..."):
    """Run work(status, progress, cancel_flag) on a worker thread behind a modal window.
    Returns (result, error); error is the exception raised by `work`, or None."""
    win = tk.Toplevel(parent)
    win.title(title)
    win.configure(bg=C_BASE)
    win.transient(parent)
    win.resizable(False, False)
    win.protocol("WM_DELETE_WINDOW", lambda: None)             # not closable while it works
    _style(win)
    tk.Label(win, text=title, bg=C_BASE, fg=C_TXT1, font=('Segoe UI', 11, 'bold')).pack(padx=22, pady=(18, 4), anchor='w')
    text = tk.Label(win, text=first_line, bg=C_BASE, fg=C_TXT2, font=('Segoe UI', 9), wraplength=420, justify=tk.LEFT)
    text.pack(padx=22, anchor='w')
    bar = ttk.Progressbar(win, mode='indeterminate', length=420, style='Jelibox.Horizontal.TProgressbar')
    bar.pack(padx=22, pady=(12, 20))
    bar.start(12)
    _center(win, parent, 464, 150)
    win.grab_set()

    worker = BatchWorker(work)
    worker.start()
    out = {}

    def poll():
        done, status, _progress, result, error = worker.snapshot()
        if status:
            text.config(text=status)
        if done:
            out["result"], out["error"] = result, error
            bar.stop()
            win.grab_release()
            win.destroy()
            return
        win.after(POLL_MS, poll)

    win.after(POLL_MS, poll)
    parent.wait_window(win)
    return out.get("result"), out.get("error")


class AutoAnnotateDialog:
    def __init__(self, parent, images, mode=None, before_start=None, after_done=None):
        self.parent, self.images = parent, list(images)
        self.mode = mode or assistant_modes.get_mode()
        self.before_start, self.after_done = before_start, after_done
        self.cfg = wcfg.get_assistant(workspaceName)
        self.worker = None
        self.started_at = 0.0
        self.finished = False

        self.win = tk.Toplevel(parent)
        self.win.title("Auto-annotate all images")
        self.win.configure(bg=C_BASE)
        self.win.transient(parent)
        self.win.resizable(False, False)
        self.win.protocol("WM_DELETE_WINDOW", self._close)
        _style(self.win)
        self._build()
        self.win.update_idletasks()
        _center(self.win, parent, 520, self.win.winfo_reqheight())
        self.win.bind('<Escape>', lambda e: self._close())
        self.win.grab_set()
        self.win.focus_force()

    # ------------------------------------------------------------------ layout
    def _build(self):
        provider = inf.current_provider(self.mode)
        names = {wcfg.PROVIDER_YOLO_WORLD: "YOLO-World", wcfg.PROVIDER_CUSTOM: "your trained model",
                 wcfg.PROVIDER_LOCATE: "LocateAnything-3B", wcfg.PROVIDER_HEAD: "your custom head model"}
        tk.Label(self.win, text="AUTO-ANNOTATE ALL IMAGES", bg=C_BASE, fg=C_TXT1,
                 font=('Segoe UI', 12, 'bold')).pack(padx=22, pady=(18, 2), anchor='w')
        tk.Label(self.win, bg=C_BASE, fg=C_TXT2, font=('Segoe UI', 9), wraplength=476, justify=tk.LEFT,
                 text=f"Mode: {assistant_modes.title(self.mode)} - annotating with {names.get(provider, provider)}.\n"
                      f"{len(self.images)} image(s) in {os.path.basename(os.path.normpath(input_folder))}. "
                      f"New annotations are merged into what is already there; existing ones are never changed."
                 ).pack(padx=22, anchor='w')

        self.only_var = tk.BooleanVar(value=self.cfg["batch"]["only_unlabeled"])
        self.only_check = tk.Checkbutton(
            self.win, text="Only images without annotations", variable=self.only_var,
            bg=C_BASE, fg=C_TXT1, selectcolor=C_CARD, activebackground=C_BASE, activeforeground=C_TXT1,
            font=('Segoe UI', 9), highlightthickness=0, bd=0, cursor='hand2')
        self.only_check.pack(padx=22, pady=(12, 0), anchor='w')

        self.bar = ttk.Progressbar(self.win, mode='determinate', length=476, maximum=max(1, len(self.images)),
                                   style='Jelibox.Horizontal.TProgressbar')
        self.bar.pack(padx=22, pady=(14, 4))
        self.count_label = tk.Label(self.win, text="Ready.", bg=C_BASE, fg=C_TXT1, font=('Segoe UI', 9, 'bold'), anchor='w')
        self.count_label.pack(padx=22, fill=tk.X)
        self.status_label = tk.Label(self.win, text="", bg=C_BASE, fg=C_TXT2, font=('Segoe UI', 8), anchor='w',
                                     wraplength=476, justify=tk.LEFT)
        self.status_label.pack(padx=22, fill=tk.X)

        footer = tk.Frame(self.win, bg=C_BASE)
        footer.pack(fill=tk.X, padx=22, pady=(14, 18))
        self.start_btn = tk.Button(footer, text="Start", command=self._start, bg=C_ACCENT, fg=C_ON_ACCENT,
                                   font=('Segoe UI', 9, 'bold'), relief=tk.FLAT, cursor='hand2',
                                   activebackground=C_ACCENT, borderwidth=0)
        self.start_btn.pack(side=tk.RIGHT, ipadx=22, ipady=4)
        self.close_btn = tk.Button(footer, text="Close", command=self._close, bg=C_CARD2, fg=C_TXT1,
                                   font=('Segoe UI', 9), relief=tk.FLAT, cursor='hand2',
                                   activebackground=C_CARD2, activeforeground=C_TXT1, borderwidth=0)
        self.close_btn.pack(side=tk.RIGHT, padx=8, ipadx=16, ipady=4)

    # ------------------------------------------------------------------ run
    def _start(self):
        if not self.images:
            return
        if not confirm_download_if_needed(self.win, self.mode):
            return
        if self.before_start:
            self.before_start()
        self.cfg["batch"]["only_unlabeled"] = bool(self.only_var.get())
        wcfg.set_assistant(workspaceName, self.cfg)

        self.only_check.config(state=tk.DISABLED)
        self.start_btn.config(text="Stop", command=self._stop, bg=C_AMBER)
        self.close_btn.config(state=tk.DISABLED)
        self.started_at = time.time()
        only_unlabeled = bool(self.only_var.get())
        mode = self.mode
        classes = inf.current_classes()

        def work(status, progress, cancel_flag):
            predict = inf.build_predictor(mode, status)
            status("Annotating ...")
            return run_batch(
                self.images,
                read_image=lambda name: inf.read_image(os.path.join(input_folder, name)),
                predict=predict,
                load_annotations=file_handler.load_annotation_local,
                save_annotations=lambda name, shape, b, p: file_handler.save_annotations(name, shape, b, p, classes),
                classes=classes, only_unlabeled=only_unlabeled,
                progress=progress, should_cancel=cancel_flag.is_set)

        self.worker = BatchWorker(work)
        self.worker.start()
        self.count_label.config(text="Preparing the model ...")
        self.win.after(POLL_MS, self._poll)

    def _stop(self):
        if self.worker:
            self.worker.cancel()
        self.start_btn.config(state=tk.DISABLED, text="Stopping ...")
        self.status_label.config(text="Finishing the image in progress ...")

    def _poll(self):
        done, status, (n, total, name), result, error = self.worker.snapshot()
        if total:
            self.bar.config(value=n)
            elapsed = time.time() - self.started_at
            eta = f", about {_fmt_time(elapsed / n * (total - n))} left" if n else ""
            self.count_label.config(text=f"{n} / {total} images  -  {_fmt_time(elapsed)} elapsed{eta}")
        self.status_label.config(text=(f"{name}" if name and not done else status))
        if not done:
            self.win.after(POLL_MS, self._poll)
            return

        self.finished = True
        self.start_btn.config(state=tk.NORMAL, text="Start", command=self._start, bg=C_ACCENT)
        self.close_btn.config(state=tk.NORMAL)
        self.only_check.config(state=tk.NORMAL)
        if error is not None:
            self.count_label.config(text="Stopped.")
            self.status_label.config(text="")
            if self.after_done:
                self.after_done()
            messagebox.showwarning("Auto-annotate", str(error) if isinstance(error, AssistantError)
                                   else f"Auto-annotate failed:\n{error}", parent=self.win)
            return
        self.count_label.config(text="Done." if not result.cancelled else "Stopped.", fg=C_GREEN)
        self.status_label.config(text=result.summary())
        if self.after_done:
            self.after_done()
        detail = ""
        if result.failed:
            detail = "\n\nFailed images:\n" + "\n".join(f"  {n}: {m}" for n, m in result.failed[:8])
            if len(result.failed) > 8:
                detail += f"\n  ... and {len(result.failed) - 8} more"
        messagebox.showinfo("Auto-annotate", result.summary() + detail, parent=self.win)

    def _close(self):
        if self.worker and not self.finished:
            return                                           # still running: use Stop
        self.win.destroy()


def open_auto_annotate(parent, images, before_start=None, after_done=None):
    dlg = AutoAnnotateDialog(parent, images, before_start=before_start, after_done=after_done)
    parent.wait_window(dlg.win)
