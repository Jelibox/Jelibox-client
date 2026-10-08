"""
Label Assistant settings for the two newer modes.

  - LocateAnythingDialog : categories (what to look for -> which workspace class) and how the model runs
  - CustomHeadDialog     : which trained head to use, which detector classes it relabels, head class -> workspace class

The first mode (YOLO-World) keeps its own window in LabelAssistantDialog.py. Settings are stored per workspace in
configs/<workspace>.json (see workspace_config.py). Import this module only after config.load_workspace() has run.
"""
import copy
import gc
import os
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

from PIL import Image, ImageTk

from . import workspace_config as wcfg
from . import assistant_providers as providers
from .dialog_scroll import ScrollBody
from .detector_classes import COCO_NAMES, parse_class_ids, format_class_ids
from .config import class_manager, workspaceName, BASE_DIR
from .theme import (C_BASE, C_PANEL, C_CARD, C_CARD2, C_BORDER, C_ACCENT, C_AMBER, C_RED, C_GREEN, C_TXT1, C_TXT2,
                    C_TXT3, C_ON_ACCENT, C_ON_RED, C_DANGER_FG)

SKIP = "(skip)"
MAX_VISIBLE_ROWS = 5
ROW_HEIGHT = 40


class _Dialog:
    """Window shell shared by both dialogs: header, scrolling body, Save / Cancel footer (always visible)."""
    TITLE = ""
    WIDTH = 680
    SAVE_TEXT = "Save"

    def __init__(self, parent):
        self.parent = parent
        self.cfg = copy.deepcopy(wcfg.get_assistant(workspaceName))
        self.classes = list(class_manager.get_classes())
        self.win = tk.Toplevel(parent)
        self.win.title("Label Assistant")
        self.win.configure(bg=C_BASE)
        self.win.transient(parent)
        self.win.resizable(False, False)
        self._style()
        self._header()
        self._footer()                          # packed first so a tall body can never push it off the screen
        self.scroll = ScrollBody(self.win, C_BASE)
        self.scroll.pack(fill=tk.BOTH, expand=True, padx=20, pady=(14, 8))
        self.body = self.scroll.inner
        self.build(self.body)
        self.win.update_idletasks()
        self.fit()
        self.win.bind('<Escape>', lambda e: self.win.destroy())
        self.win.bind('<Destroy>', self._destroyed)
        self.win.grab_set()
        self.win.focus_force()

    def _destroyed(self, event):
        # Tk variables must not be garbage-collected on a worker thread (Python 3.10 aborts with Tcl_AsyncDelete),
        # so the dialog's leftovers are collected here, on the Tk thread, as soon as the window is gone
        if event.widget is self.win:
            try:
                self.parent.after_idle(gc.collect)
            except tk.TclError:
                pass

    def fit(self):
        self.scroll.fit(self.win, self.WIDTH, self.parent)

    def _style(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('Jelibox.TCombobox', fieldbackground=C_CARD, background=C_CARD2, foreground=C_TXT1,
                        selectbackground=C_CARD, selectforeground=C_TXT1, bordercolor=C_BORDER,
                        darkcolor=C_CARD2, lightcolor=C_CARD2, arrowcolor=C_ACCENT, padding=4)
        style.map('Jelibox.TCombobox', fieldbackground=[('disabled', C_PANEL), ('readonly', C_CARD)],
                  foreground=[('disabled', C_TXT3)], arrowcolor=[('disabled', C_TXT3)])
        self.win.option_add('*TCombobox*Listbox.background', C_CARD)
        self.win.option_add('*TCombobox*Listbox.foreground', C_TXT1)
        self.win.option_add('*TCombobox*Listbox.selectBackground', C_ACCENT)
        self.win.option_add('*TCombobox*Listbox.selectForeground', C_ON_ACCENT)

    def _header(self):
        header = tk.Frame(self.win, bg=C_PANEL, height=54)
        header.pack(fill=tk.X)
        header.pack_propagate(False)
        try:
            logo = Image.open("assets/jelibox.png").resize((32, 32))
            self._logo = ImageTk.PhotoImage(logo)
            tk.Label(header, image=self._logo, bg=C_PANEL).pack(side=tk.LEFT, padx=(14, 8))
        except Exception:
            pass
        tk.Label(header, text=self.TITLE, bg=C_PANEL, fg=C_TXT1, font=('Segoe UI', 12, 'bold')).pack(side=tk.LEFT)
        tk.Frame(self.win, bg=C_ACCENT, height=2).pack(fill=tk.X)

    def _footer(self):
        footer = self.footer = tk.Frame(self.win, bg=C_BASE)
        footer.pack(side=tk.BOTTOM, fill=tk.X, padx=20, pady=(4, 16))
        tk.Button(footer, text=self.SAVE_TEXT, command=self.save, bg=C_ACCENT, fg=C_ON_ACCENT, font=('Segoe UI', 9, 'bold'),
                  relief=tk.FLAT, cursor='hand2', activebackground=C_ACCENT, borderwidth=0
                  ).pack(side=tk.RIGHT, ipadx=22, ipady=4)
        tk.Button(footer, text="Cancel", command=self.win.destroy, bg=C_CARD2, fg=C_TXT1, font=('Segoe UI', 9),
                  relief=tk.FLAT, cursor='hand2', activebackground=C_CARD2, activeforeground=C_TXT1, borderwidth=0
                  ).pack(side=tk.RIGHT, padx=8, ipadx=16, ipady=4)

    # -- small widgets
    def card(self, parent, pady=(0, 10)):
        frame = tk.Frame(parent, bg=C_CARD, highlightbackground=C_BORDER, highlightthickness=1)
        frame.pack(fill=tk.X, pady=pady)
        inner = tk.Frame(frame, bg=C_CARD)
        inner.pack(fill=tk.X, padx=14, pady=10)
        return inner

    def label(self, parent, text, fg=C_TXT2, size=9, bold=False, **pack):
        lbl = tk.Label(parent, text=text, bg=parent.cget('bg'), fg=fg, wraplength=self.WIDTH - 70,
                       font=('Segoe UI', size, 'bold' if bold else 'normal'), justify=tk.LEFT, anchor='w')
        lbl.pack(**({'anchor': 'w'} | pack))
        return lbl

    def entry(self, parent, var, width=30, **kw):
        return tk.Entry(parent, textvariable=var, width=width, font=('Segoe UI', 10), bg=C_CARD2, fg=C_TXT1,
                        insertbackground=C_ACCENT, relief=tk.FLAT, highlightthickness=1,
                        highlightbackground=C_BORDER, highlightcolor=C_ACCENT, **kw)

    def combo(self, parent, var, values, width=18):
        return ttk.Combobox(parent, textvariable=var, values=values, state='readonly', font=('Segoe UI', 10),
                            width=width, style='Jelibox.TCombobox')

    def button(self, parent, text, command, accent=False):
        return tk.Button(parent, text=text, command=command, bg=C_ACCENT if accent else C_CARD2,
                         fg=C_ON_ACCENT if accent else C_TXT1, font=('Segoe UI', 8, 'bold'), relief=tk.FLAT,
                         cursor='hand2', activebackground=C_ACCENT if accent else C_CARD2, borderwidth=0)

    def slider(self, parent, text, var, low, high, step, fmt="{:.2f}"):
        row = tk.Frame(parent, bg=parent.cget('bg'))
        row.pack(fill=tk.X, pady=(4, 0))
        tk.Label(row, text=text, bg=row.cget('bg'), fg=C_TXT2, font=('Segoe UI', 9), width=22, anchor='w').pack(side=tk.LEFT)
        value = tk.Label(row, text=fmt.format(var.get()), bg=row.cget('bg'), fg=C_ACCENT,
                         font=('Segoe UI', 9, 'bold'), width=5)
        value.pack(side=tk.RIGHT)
        tk.Scale(row, variable=var, from_=low, to=high, resolution=step, orient='horizontal', showvalue=False,
                 command=lambda _: value.config(text=fmt.format(var.get())), bg=row.cget('bg'), fg=C_TXT1,
                 troughcolor=C_CARD2, highlightthickness=0, activebackground=C_ACCENT, bd=0, length=260
                 ).pack(side=tk.RIGHT, padx=10)

    def build(self, body):
        raise NotImplementedError

    def save(self):
        raise NotImplementedError


# ====================================================================== LocateAnything
class LocateAnythingDialog(_Dialog):
    TITLE = "LABEL ASSISTANT  -  LOCATEANYTHING"
    WIDTH = 700

    def build(self, body):
        la = self.cfg["locate_anything"]
        self.rows = []
        self.label(body, "NVIDIA LocateAnything-3B finds every instance of the categories you list, in one pass "
                         "over the whole folder.", pady=(0, 8))

        # --- categories
        card = self.card(body)
        head = tk.Frame(card, bg=C_CARD)
        head.pack(fill=tk.X)
        tk.Label(head, text="CATEGORIES", bg=C_CARD, fg=C_TXT2, font=('Segoe UI', 8, 'bold')).pack(side=tk.LEFT)
        self.add_btn = self.button(head, "＋  Add category", lambda: self._add_row(focus=True), accent=True)
        self.add_btn.pack(side=tk.RIGHT, ipadx=8, ipady=2)
        cols = tk.Frame(card, bg=C_CARD, height=20)
        cols.pack(fill=tk.X, pady=(8, 2))
        tk.Label(cols, text="Look for (letters & spaces, English or Chinese)", bg=C_CARD, fg=C_TXT2,
                 font=('Segoe UI', 8)).place(x=0, y=0)
        tk.Label(cols, text="Save as workspace class", bg=C_CARD, fg=C_TXT2, font=('Segoe UI', 8)).place(x=300, y=0)

        wrap = tk.Frame(card, bg=C_CARD)
        wrap.pack(fill=tk.X)
        self.list_canvas = tk.Canvas(wrap, bg=C_CARD, highlightthickness=0, height=ROW_HEIGHT)
        self.list_scroll = tk.Scrollbar(wrap, orient='vertical', command=self.list_canvas.yview, bg=C_BORDER,
                                        troughcolor=C_CARD, relief=tk.FLAT, width=8)
        self.list_canvas.configure(yscrollcommand=self.list_scroll.set)
        self.list_canvas.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.rows_frame = tk.Frame(self.list_canvas, bg=C_CARD)
        window = self.list_canvas.create_window((0, 0), window=self.rows_frame, anchor='nw')
        self.rows_frame.bind('<Configure>', lambda e: self.list_canvas.configure(scrollregion=self.list_canvas.bbox('all')))
        self.list_canvas.bind('<Configure>', lambda e: self.list_canvas.itemconfig(window, width=e.width))
        self.hint = self.label(card, "", fg=C_AMBER, size=8, pady=(4, 0))
        for t in la["target_classes"]:
            self._add_row(t["prompt"], t["map_to"])
        if not self.rows:
            self._add_row()

        # --- how it runs
        card = self.card(body)
        self.label(card, "HOW IT RUNS", fg=C_TXT2, size=8, bold=True)
        grid = tk.Frame(card, bg=C_CARD)
        grid.pack(fill=tk.X, pady=(6, 0))
        self.mode_var = tk.StringVar(value=la["generation_mode"])
        self.device_var = tk.StringVar(value=la["device"])
        self.side_var = tk.IntVar(value=la["short_side"])
        self.passes_var = tk.IntVar(value=la["passes"])
        self.temp_var = tk.DoubleVar(value=la["temperature"])
        for r, (text, widget) in enumerate([
            ("Decoding", self.combo(grid, self.mode_var, wcfg.LOCATE_GENERATION_MODES, 12)),
            ("Device", self.combo(grid, self.device_var, wcfg.LOCATE_DEVICES, 12)),
            ("Image short side (px)", tk.Spinbox(grid, from_=256, to=2048, increment=128, textvariable=self.side_var,
                                                 width=8, font=('Segoe UI', 10), bg=C_CARD2, fg=C_TXT1,
                                                 relief=tk.FLAT, buttonbackground=C_CARD2)),
            ("Passes (merged by agreement)", tk.Spinbox(grid, from_=1, to=8, textvariable=self.passes_var, width=8,
                                                        font=('Segoe UI', 10), bg=C_CARD2, fg=C_TXT1,
                                                        relief=tk.FLAT, buttonbackground=C_CARD2)),
        ]):
            tk.Label(grid, text=text, bg=C_CARD, fg=C_TXT2, font=('Segoe UI', 9), width=26, anchor='w').grid(row=r, column=0, pady=3, sticky='w')
            widget.grid(row=r, column=1, pady=3, sticky='w')
        self.label(card, "Decoding: hybrid = best recall, fast = quickest, slow = steadiest on long answers.",
                   fg=C_TXT3, size=8, pady=(2, 0))
        self.slider(card, "Temperature (passes > 1)", self.temp_var, 0.0, 1.0, 0.05)

        # --- status
        card = self.card(body, pady=(0, 4))
        self.status = self.label(card, self._status_text(), size=8, fg=C_TXT2)
        self._update_rows_layout()

    def _status_text(self):
        from . import locate_anything as la
        lines = []
        problem = la.check_requirements()
        lines.append("Packages: ready" if not problem else "Packages: missing - " + problem.split("\n")[0])
        lines.append(f"Model: {'downloaded' if not problem and la.is_cached() else f'not downloaded yet (about {la.DOWNLOAD_GB} GB, asked before it happens)'}")
        try:
            import torch
            dev = f"NVIDIA GPU - {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "CPU only (slow, needs ~9 GB free RAM)"
        except Exception:
            dev = "unknown"
        lines.append(f"Device: {dev}")
        return "\n".join(lines)

    # -- category rows
    def _add_row(self, prompt="", map_to="", focus=False):
        frame = tk.Frame(self.rows_frame, bg=C_CARD, height=ROW_HEIGHT)
        frame.pack(fill=tk.X)
        frame.pack_propagate(False)
        prompt_var = tk.StringVar(value=prompt)
        cls_var = tk.StringVar(value=map_to if map_to in self.classes else "")
        vcmd = (self.win.register(lambda proposed: wcfg.is_valid_prompt_text(proposed)), '%P')
        entry = self.entry(frame, prompt_var, 30, validate='key', validatecommand=vcmd)
        entry.pack(side=tk.LEFT, ipady=4, pady=6)
        tk.Label(frame, text="→", bg=C_CARD, fg=C_ACCENT, font=('Segoe UI', 11, 'bold')).pack(side=tk.LEFT, padx=8)
        combo = self.combo(frame, cls_var, self.classes, 18)
        combo.pack(side=tk.LEFT, pady=6)
        row = {'frame': frame, 'prompt': prompt_var, 'cls': cls_var}
        tk.Button(frame, text="✕", command=lambda r=row: self._remove_row(r), bg=C_CARD, fg=C_DANGER_FG,
                  font=('Segoe UI', 10, 'bold'), relief=tk.FLAT, cursor='hand2', activebackground=C_RED,
                  activeforeground=C_ON_RED, borderwidth=0, width=3).pack(side=tk.RIGHT, padx=(6, 4), pady=6)
        self.rows.append(row)
        self._update_rows_layout()
        if focus:
            entry.focus_set()
            self.list_canvas.update_idletasks()
            self.list_canvas.yview_moveto(1.0)

    def _remove_row(self, row):
        row['frame'].destroy()
        self.rows.remove(row)
        self._update_rows_layout()

    def _update_rows_layout(self):
        visible = max(1, min(len(self.rows), MAX_VISIBLE_ROWS))
        self.list_canvas.configure(height=visible * ROW_HEIGHT)
        if len(self.rows) > MAX_VISIBLE_ROWS:
            self.list_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        else:
            self.list_scroll.pack_forget()
        self.hint.config(text="" if self.rows else "No category yet - click “Add category”.")
        if self.win.winfo_exists():
            self.fit()

    def save(self):
        targets, error = wcfg.validate_targets((r['prompt'].get(), r['cls'].get()) for r in self.rows)
        if error:
            messagebox.showwarning("Label Assistant", error, parent=self.win)
            return
        if not targets:
            messagebox.showwarning(
                "Label Assistant", "LocateAnything can't work without a category.\n\nYour settings will be saved, "
                "but G and Auto-annotate will not annotate anything until you add one.", parent=self.win)
        try:
            side, passes = int(self.side_var.get()), int(self.passes_var.get())
        except (tk.TclError, ValueError):
            messagebox.showwarning("Label Assistant", "Image short side and passes must be whole numbers.", parent=self.win)
            return
        la = self.cfg["locate_anything"]
        la.update(target_classes=targets, generation_mode=self.mode_var.get(), device=self.device_var.get(),
                  short_side=side, passes=passes, temperature=round(float(self.temp_var.get()), 2),
                  min_votes=min(la["min_votes"], passes))
        self.cfg["provider"] = wcfg.PROVIDER_LOCATE
        wcfg.set_assistant(workspaceName, self.cfg)
        print(f"[Assistant] Saved LocateAnything: {len(targets)} categories, {la['generation_mode']}, {la['device']}")
        self.win.destroy()


# ====================================================================== Custom head
class CustomHeadDialog(_Dialog):
    TITLE = "LABEL ASSISTANT  -  CUSTOM HEAD"
    WIDTH = 720

    def build(self, body):
        h = self.cfg["custom_head"]
        self.head_names = []
        self.map_vars = {}
        self.detector_names = list(COCO_NAMES)          # class names of the detector (COCO unless a model is browsed)
        self._names_for = ""                            # the weights text detector_names was read for
        self.label(body, "A YOLO detector finds the boxes (person, bottle, ...). Your trained head gives each box "
                         "one of your own classes.", pady=(0, 8))

        card = self.card(body)
        self.label(card, "HEAD CHECKPOINT (head_best.pt)", size=8, bold=True)
        row = tk.Frame(card, bg=C_CARD)
        row.pack(fill=tk.X, pady=(4, 0))
        self.path_var = tk.StringVar(value=h["head_path"])
        self.entry(row, self.path_var, 54).pack(side=tk.LEFT, ipady=4)
        self.button(row, "Browse head ...", self._browse_head).pack(side=tk.LEFT, padx=8, ipadx=8, ipady=3)
        self.head_info = self.label(card, "", size=8, fg=C_TXT3, pady=(4, 0))

        card = self.card(body)
        self.label(card, "DETECTOR", size=8, bold=True)
        grid = tk.Frame(card, bg=C_CARD)
        grid.pack(fill=tk.X, pady=(4, 0))
        self.weights_var = tk.StringVar(value=h["detector_weights"])
        self._load_detector_names(h["detector_weights"], quiet=True)
        self.classes_var = tk.StringVar(value=self._class_text(h["detector_classes"]))
        label_kw = dict(bg=C_CARD, fg=C_TXT2, font=('Segoe UI', 9), width=24, anchor='w')
        hint_kw = dict(bg=C_CARD, fg=C_TXT3, font=('Segoe UI', 8), anchor='w')
        tk.Label(grid, text="Detect these classes", **label_kw).grid(row=0, column=0, sticky='w', pady=(3, 0))
        self.entry(grid, self.classes_var, 26).grid(row=0, column=1, sticky='w', ipady=3, pady=(3, 0))
        self.classes_hint = tk.Label(grid, text="", wraplength=400, justify=tk.LEFT, **hint_kw)
        self.classes_hint.grid(row=1, column=1, columnspan=2, sticky='w')
        tk.Label(grid, text="Detector weights (optional)", **label_kw).grid(row=2, column=0, sticky='w', pady=(6, 0))
        self.entry(grid, self.weights_var, 26).grid(row=2, column=1, sticky='w', ipady=3, pady=(6, 0))
        self.button(grid, "Browse custom model ...", self._browse_detector).grid(row=2, column=2, padx=8, pady=(6, 0),
                                                                                  ipadx=8, ipady=3)
        tk.Label(grid, text="Empty = the detector the head was trained on. A model you choose here must be that "
                            "detector (same architecture, same layers): the head reads its features.",
                 wraplength=400, justify=tk.LEFT, **hint_kw).grid(row=3, column=1, columnspan=2, sticky='w')
        self._update_classes_hint()
        self.conf_var = tk.DoubleVar(value=self.cfg["confidence"])
        self.iou_var = tk.DoubleVar(value=h["iou"])
        self.min_head_var = tk.DoubleVar(value=h["min_head_conf"])
        self.slider(card, "Detector confidence", self.conf_var, 0.05, 0.95, 0.05)
        self.slider(card, "Detector NMS IoU", self.iou_var, 0.1, 0.9, 0.05)
        self.slider(card, "Minimum head confidence", self.min_head_var, 0.0, 0.95, 0.05)

        card = self.card(body, pady=(0, 4))
        self.label(card, "HEAD CLASS  →  WORKSPACE CLASS", size=8, bold=True)
        self.map_frame = tk.Frame(card, bg=C_CARD)
        self.map_frame.pack(fill=tk.X, pady=(6, 0))
        self.map_hint = self.label(card, "", size=8, fg=C_AMBER, pady=(4, 0))
        if h["head_path"]:
            self._load_head(h["head_path"], quiet=True)
        else:
            self._render_map()

    def _class_text(self, ids):
        return format_class_ids(ids, self.detector_names)

    def _update_classes_hint(self):
        names = [f"{i} {n}" for i, n in enumerate(self.detector_names) if n]
        shown = ", ".join(names[:10]) + (f", ... ({len(names)} classes)" if len(names) > 10 else "")
        self.classes_hint.config(text=f"Names or numbers, e.g. {self.detector_names[0] or 0}   -   this detector: {shown}")

    def _load_detector_names(self, weights, quiet=False):
        """Class names of the chosen detector: COCO for the stock models, the file's own names for a browsed one."""
        weights = (weights or "").strip()
        if weights and os.path.isfile(providers.resolve_head_path(weights, BASE_DIR)):
            try:
                self.detector_names = providers.model_class_names(providers.resolve_head_path(weights, BASE_DIR))
                self._names_for = weights
                return True
            except providers.AssistantError as exc:
                if not quiet:
                    messagebox.showwarning("Label Assistant", str(exc), parent=self.win)
                return False
        self.detector_names = list(COCO_NAMES)
        self._names_for = weights
        return True

    def _browse_detector(self):
        start = os.path.join(BASE_DIR, "models", workspaceName)
        path = filedialog.askopenfilename(
            parent=self.win, title="Choose the custom detector model",
            filetypes=[("YOLO model", "*.pt"), ("All files", "*.*")],
            initialdir=start if os.path.isdir(start) else BASE_DIR)
        if not path:
            return
        old_names = self.detector_names
        self.win.config(cursor='watch')
        self.win.update_idletasks()
        try:
            ok = self._load_detector_names(path)
        finally:
            self.win.config(cursor='')
        if not ok:
            self.detector_names = old_names
            return
        self.weights_var.set(os.path.normpath(path))
        self._names_for = self.weights_var.get().strip()
        try:                                    # keep the chosen classes when they still exist in this detector
            parse_class_ids(self.classes_var.get(), self.detector_names)
        except ValueError:
            self.classes_var.set(self._class_text([0]))
        self._update_classes_hint()
        self.fit()

    def _browse_head(self):
        start = os.path.join(BASE_DIR, "models", workspaceName)
        path = filedialog.askopenfilename(
            parent=self.win, title="Choose the head checkpoint", filetypes=[("Head checkpoint", "*.pt"), ("All files", "*.*")],
            initialdir=start if os.path.isdir(start) else BASE_DIR)
        if path:
            self.path_var.set(path)
            self._load_head(path)

    def _load_head(self, path, quiet=False):
        """Read the class names out of the checkpoint so the mapping table can list them."""
        full = providers.resolve_head_path(path, BASE_DIR)
        self.win.config(cursor='watch')
        self.win.update_idletasks()
        try:
            from . import custom_head as ch
            ck = ch.read_checkpoint(full)
            self.head_names = list(ck["names"])
            cfg = ck["cfg"]
            self.head_info.config(fg=C_GREEN, text=f"{len(self.head_names)} classes: {', '.join(self.head_names)}   |   "
                                                   f"detector {cfg['weights']}, {cfg['imgsz'][0]}x{cfg['imgsz'][1]}")
        except Exception as exc:
            self.head_names = []
            self.head_info.config(fg=C_AMBER, text=str(exc).split("\n")[0])
            if not quiet:
                messagebox.showwarning("Label Assistant", str(exc), parent=self.win)
        finally:
            self.win.config(cursor='')
        self._render_map()

    def _render_map(self):
        for child in self.map_frame.winfo_children():
            child.destroy()
        saved = self.cfg["custom_head"]["class_map"]
        auto = providers.map_head_classes(self.head_names, saved, self.classes)
        self.map_vars = {}
        for r, name in enumerate(self.head_names):
            tk.Label(self.map_frame, text=name, bg=C_CARD, fg=C_TXT1, font=('Segoe UI', 10), width=24, anchor='w').grid(row=r, column=0, pady=2, sticky='w')
            tk.Label(self.map_frame, text="→", bg=C_CARD, fg=C_ACCENT, font=('Segoe UI', 11, 'bold')).grid(row=r, column=1, padx=8)
            var = tk.StringVar(value=auto.get(name) or SKIP)
            self.combo(self.map_frame, var, [SKIP] + self.classes, 22).grid(row=r, column=2, pady=2, sticky='w')
            self.map_vars[name] = var
        self.map_hint.config(text="" if self.head_names else "Choose a head checkpoint to map its classes.")
        self.fit()

    def save(self):
        path = self.path_var.get().strip()
        if not path:
            messagebox.showwarning("Label Assistant", "Choose the head checkpoint (head_best.pt) first.", parent=self.win)
            return
        if not os.path.isfile(providers.resolve_head_path(path, BASE_DIR)):
            messagebox.showwarning("Label Assistant", f"The head checkpoint was not found:\n{path}", parent=self.win)
            return
        weights = self.weights_var.get().strip()
        looks_like_path = bool(weights) and (os.path.isabs(weights) or "/" in weights or "\\" in weights)
        if looks_like_path and not os.path.isfile(providers.resolve_head_path(weights, BASE_DIR)):
            messagebox.showwarning("Label Assistant", f"The detector model was not found:\n{weights}", parent=self.win)
            return
        if weights != self._names_for and not self._load_detector_names(weights):     # typed by hand: read its names
            return
        try:
            ids = parse_class_ids(self.classes_var.get(), self.detector_names)
        except ValueError as exc:
            messagebox.showwarning("Label Assistant", str(exc), parent=self.win)
            return
        class_map = {n: v.get() for n, v in self.map_vars.items() if v.get() != SKIP}
        if self.head_names and not class_map:
            messagebox.showwarning("Label Assistant", "Map at least one head class to a workspace class.", parent=self.win)
            return
        h = self.cfg["custom_head"]
        h.update(head_path=path, detector_weights=self.weights_var.get().strip(), detector_classes=ids,
                 iou=round(float(self.iou_var.get()), 2), min_head_conf=round(float(self.min_head_var.get()), 2),
                 class_map=class_map)
        self.cfg["confidence"] = round(float(self.conf_var.get()), 2)
        self.cfg["provider"] = wcfg.PROVIDER_HEAD
        wcfg.set_assistant(workspaceName, self.cfg)
        print(f"[Assistant] Saved custom head: {os.path.basename(path)}, detector classes {ids}, {len(class_map)} mapped")
        self.win.destroy()


def open_locate_anything(parent):
    dlg = LocateAnythingDialog(parent)
    parent.wait_window(dlg.win)


def open_custom_head(parent):
    dlg = CustomHeadDialog(parent)
    parent.wait_window(dlg.win)
