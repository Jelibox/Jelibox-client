"""The Export Dataset window: format and destination, the train / valid / test split, Preprocessing (Auto-Orient,
Resize) and the Augmentation picker, with the size of every split shown before anything is written.

The exporters themselves live on AnnotationGUI (`_export_yolo_dataset` ...); this file only asks and reports.
"""
import os
import random
import shutil
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from . import app_settings
from . import export as export_module
from . import export_preprocess as pre
from .AugmentationPanel import AugmentationPanel, _flat_button
from .augmentation import Augmenter
from .dialog_scroll import ScrollBody
from .theme import (C_BASE, C_PANEL, C_CARD, C_CARD2, C_BORDER, C_ACCENT, C_AMBER, C_GREEN, C_TXT1, C_TXT2, C_TXT3,
                    C_ON_ACCENT)

FORMATS = ["YOLO", "Pascal VOC (XML)", "COCO"]
IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
WIDTH = 940
BAR_W = 820
SPLIT_COLORS = (C_ACCENT, C_AMBER, C_GREEN)


def split_counts(total, train_pct, valid_pct):
    """(train, valid, test) pictures for a split in percent: the same rounding the exporters use."""
    train = int(round(total * (train_pct / 100.0)))
    valid = int(round(total * (valid_pct / 100.0)))
    if train + valid > total:
        valid = max(0, total - train)
    return train, valid, total - train - valid


def to_int(text):
    digits = ''.join(ch for ch in str(text or '') if ch.isdigit())
    return int(digits) if digits else 0


class ExportDatasetDialog:
    def __init__(self, gui, workspace, default_folder, labels_folder):
        self.gui, self.root = gui, gui.root
        self.workspace, self.default_folder, self.labels_folder = workspace, default_folder, labels_folder
        self.active = False                                    # an export is running
        self._busy = False
        self._total = 0
        self.locks = []                                        # widgets to switch off meanwhile

        self.dialog = dlg = tk.Toplevel(self.root)
        dlg.title("Export Dataset")
        dlg.configure(bg=C_BASE)
        dlg.transient(self.root)
        dlg.resizable(False, False)
        dlg.protocol("WM_DELETE_WINDOW", self.close)

        head = tk.Frame(dlg, bg=C_PANEL, height=56)
        head.pack(fill=tk.X)
        head.pack_propagate(False)
        tk.Label(head, text="Export Dataset", bg=C_PANEL, fg=C_TXT1, font=('Segoe UI', 13, 'bold')).pack(
            side=tk.LEFT, padx=22, pady=12)
        self.count_label = tk.Label(head, text="", bg=C_PANEL, fg=C_TXT2, font=('Segoe UI', 10))
        self.count_label.pack(side=tk.RIGHT, padx=22)
        tk.Frame(dlg, bg=C_ACCENT, height=2).pack(fill=tk.X)

        self.footer = tk.Frame(dlg, bg=C_BASE)
        self.footer.pack(side=tk.BOTTOM, fill=tk.X)
        self.scroll = ScrollBody(dlg)
        self.scroll.pack(fill=tk.BOTH, expand=True)
        self.body = tk.Frame(self.scroll.inner, bg=C_BASE)
        self.body.pack(fill=tk.X, padx=18, pady=(14, 6))

        self._build_target()
        self._build_split()
        self._build_preprocess()
        self._build_augmentation()
        self._build_footer()
        self.recount()
        self.scroll.fit(dlg, WIDTH, self.root)
        dlg.grab_set()

    # ------------------------------------------------------------ pieces
    def _section(self, number, title, hint=None):
        card = tk.Frame(self.body, bg=C_CARD, highlightbackground=C_BORDER, highlightthickness=1)
        card.pack(fill=tk.X, pady=(0, 12))
        top = tk.Frame(card, bg=C_CARD)
        top.pack(fill=tk.X, padx=16, pady=(12, 0))
        if number:
            tk.Label(top, text=str(number), bg=C_ACCENT, fg=C_ON_ACCENT, width=2, font=('Segoe UI', 10, 'bold')
                     ).pack(side=tk.LEFT)
        tk.Label(top, text=title, bg=C_CARD, fg=C_TXT1, font=('Segoe UI', 11, 'bold')).pack(side=tk.LEFT, padx=10)
        if hint:
            tk.Label(card, text=hint, bg=C_CARD, fg=C_TXT3, font=('Segoe UI', 9), anchor='w', justify=tk.LEFT,
                     wraplength=WIDTH - 90).pack(fill=tk.X, padx=16, pady=(2, 0))
        inner = tk.Frame(card, bg=C_CARD)
        inner.pack(fill=tk.X, padx=16, pady=(8, 14))
        return inner

    def _check(self, parent, text, var, command=None):
        cb = tk.Checkbutton(parent, text=text, variable=var, command=command, bg=C_CARD, fg=C_TXT1,
                            selectcolor=C_BASE, activebackground=C_CARD, activeforeground=C_ACCENT,
                            font=('Segoe UI', 10), bd=0, highlightthickness=0, cursor='hand2', anchor='w')
        self.locks.append(cb)
        return cb

    # ------------------------------------------------------------ format + destination
    def _build_target(self):
        inner = self._section(None, "Format and destination")
        row = tk.Frame(inner, bg=C_CARD)
        row.pack(fill=tk.X)
        tk.Label(row, text="Format", bg=C_CARD, fg=C_TXT2, font=('Segoe UI', 9)).pack(side=tk.LEFT)
        self.fmt = ttk.Combobox(row, values=FORMATS, state='readonly', font=('Segoe UI', 10), width=20)
        self.fmt.set('YOLO')
        self.fmt.pack(side=tk.LEFT, padx=10)
        self.fmt.bind('<<ComboboxSelected>>', lambda _e: self.recount())

        tk.Label(inner, text="Save to", bg=C_CARD, fg=C_TXT2, font=('Segoe UI', 9)).pack(anchor='w', pady=(10, 0))
        dest_row = tk.Frame(inner, bg=C_CARD)
        dest_row.pack(fill=tk.X, pady=(2, 0))
        saved = app_settings.get('export_dir') or ''
        # A remembered folder that is gone (an unplugged drive, say) is not used: the default takes over, and the
        # setting stays so the folder is picked up again when the drive is back.
        self.missing = saved if saved and not os.path.isdir(saved) else ''
        self.dest = '' if self.missing else saved
        self.dest_shown = tk.StringVar()
        tk.Label(dest_row, textvariable=self.dest_shown, bg=C_CARD2, fg=C_TXT1, anchor='w', padx=8, pady=5,
                 font=('Segoe UI', 9)).pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.browse_btn = tk.Button(dest_row, text='Browse...', command=self.browse, bg=C_CARD2, fg=C_TXT1,
                                    relief=tk.FLAT, cursor='hand2')
        self.browse_btn.pack(side=tk.LEFT, padx=(6, 0), ipadx=6)
        self.default_btn = tk.Button(dest_row, text='Default', command=self.use_default, bg=C_CARD2, fg=C_TXT1,
                                     relief=tk.FLAT, cursor='hand2')
        self.default_btn.pack(side=tk.LEFT, padx=(6, 0), ipadx=6)
        self.locks += [self.browse_btn, self.default_btn]
        self.dest_hint = tk.Label(inner, text='', bg=C_CARD, fg=C_TXT3, anchor='w', justify=tk.LEFT,
                                  wraplength=WIDTH - 90, font=('Segoe UI', 8))
        self.dest_hint.pack(fill=tk.X, pady=(2, 0))
        self.refresh_dest()

    def refresh_dest(self):
        if self.dest:
            self.dest_shown.set(self.dest)
            nxt = export_module.next_version_name(self.dest, self.workspace)
            self.dest_hint.config(text=f'The next export goes into a new folder "{nxt}" inside it; '
                                       f'nothing already there is touched.', fg=C_TXT3)
        elif self.missing:
            self.dest_shown.set(f'Jelibox folder  ({self.default_folder})')
            self.dest_hint.config(text=f'The folder you chose before was not found ({self.missing} - is the drive '
                                       f'connected?), so the default folder is used. Connect it and reopen this '
                                       f'window to use it again, or press Browse... for another one.', fg=C_AMBER)
        else:
            self.dest_shown.set(f'Jelibox folder  ({self.default_folder})')
            self.dest_hint.config(text='The previous export of this workspace in that folder is replaced. '
                                       'Choose another folder to keep every export.', fg=C_TXT3)

    def browse(self):
        if self.active:
            return
        chosen = filedialog.askdirectory(title="Save the exported dataset in...", parent=self.dialog,
                                         initialdir=self.dest if os.path.isdir(self.dest) else None)
        if chosen:
            self.missing = ''
            self.dest = os.path.normpath(chosen)
            app_settings.set('export_dir', self.dest)
            self.refresh_dest()

    def use_default(self):
        self.missing = ''
        self.dest = ''
        app_settings.set('export_dir', None)
        self.refresh_dest()

    # ------------------------------------------------------------ split
    def _build_split(self):
        inner = self._section(1, "Train / Valid / Test split",
                              "Drag the two handles, or type the percentages. Pictures are picked at random.")
        self.bar = tk.Canvas(inner, width=BAR_W, height=46, bg=C_CARD, highlightthickness=0)
        self.bar.pack(anchor='w', pady=(6, 0))
        self.train_pct, self.valid_pct = 70, 20
        self.train_var = tk.StringVar(value='70')
        self.valid_var = tk.StringVar(value='20')
        self.test_var = tk.StringVar(value='10')
        self._drag = None
        self.bar.bind('<Button-1>', self._press)
        self.bar.bind('<B1-Motion>', self._move)
        self.bar.bind('<ButtonRelease-1>', lambda _e: setattr(self, '_drag', None))

        row = tk.Frame(inner, bg=C_CARD)
        row.pack(fill=tk.X, pady=(8, 0))
        self.chips = []
        for color in SPLIT_COLORS:
            chip = tk.Label(row, text='', bg=C_CARD2, fg=C_TXT1, font=('Segoe UI', 9, 'bold'), padx=10, pady=4)
            chip.pack(side=tk.LEFT, padx=(0, 8))
            chip.dot = color
            self.chips.append(chip)
        tk.Label(row, text="Train %", bg=C_CARD, fg=C_TXT3, font=('Segoe UI', 9)).pack(side=tk.LEFT, padx=(18, 4))
        self.sb_train = tk.Spinbox(row, from_=0, to=100, textvariable=self.train_var, width=5, command=self._typed)
        self.sb_train.pack(side=tk.LEFT)
        tk.Label(row, text="Valid %", bg=C_CARD, fg=C_TXT3, font=('Segoe UI', 9)).pack(side=tk.LEFT, padx=(12, 4))
        self.sb_valid = tk.Spinbox(row, from_=0, to=100, textvariable=self.valid_var, width=5, command=self._typed)
        self.sb_valid.pack(side=tk.LEFT)
        self.locks += [self.sb_train, self.sb_valid]
        self.train_var.trace_add('write', lambda *_: self._typed())
        self.valid_var.trace_add('write', lambda *_: self._typed())
        self._draw_bar()

    def _x_of(self, pct):
        return 10 + pct / 100.0 * (BAR_W - 20)

    def _pct_of(self, x):
        return round(max(0, min(100, (x - 10) / (BAR_W - 20) * 100)))

    def _draw_bar(self):
        c = self.bar
        c.delete('all')
        t, v = self.train_pct, self.valid_pct
        edges = [0, t, t + v, 100]
        for i, color in enumerate(SPLIT_COLORS):
            x1, x2 = self._x_of(edges[i]), self._x_of(edges[i + 1])
            if x2 > x1:
                c.create_rectangle(x1, 24, x2, 38, fill=color, outline='')
            mid = (x1 + x2) / 2
            if x2 - x1 > 34:
                c.create_text(mid, 11, text=f"{edges[i + 1] - edges[i]}%", fill=C_TXT1, font=('Segoe UI', 9, 'bold'))
        for x in (self._x_of(t), self._x_of(t + v)):
            c.create_oval(x - 8, 31 - 8, x + 8, 31 + 8, fill=C_CARD, outline=C_TXT1, width=2)

    def _press(self, event):
        if self.active:
            return
        t, v = self.train_pct, self.valid_pct
        first, second = abs(event.x - self._x_of(t)), abs(event.x - self._x_of(t + v))
        self._drag = 'train' if first <= second else 'valid'

    def _move(self, event):
        if not self._drag or self.active:
            return
        pct = self._pct_of(event.x)
        if self._drag == 'train':
            self.train_pct = min(pct, 100)
            self.valid_pct = min(self.valid_pct, 100 - self.train_pct)
        else:
            self.valid_pct = max(0, min(pct - self.train_pct, 100 - self.train_pct))
        self._sync_vars()
        self.refresh()

    def _sync_vars(self):
        self._setting = True
        self.train_var.set(str(self.train_pct))
        self.valid_var.set(str(self.valid_pct))
        self.test_var.set(str(100 - self.train_pct - self.valid_pct))
        self._setting = False

    def _typed(self):
        if getattr(self, '_setting', False):
            return
        self.train_pct = max(0, min(100, to_int(self.train_var.get())))
        self.valid_pct = max(0, min(100 - self.train_pct, to_int(self.valid_var.get())))
        self.test_var.set(str(100 - self.train_pct - self.valid_pct))
        self._draw_bar()
        self.refresh()

    # ------------------------------------------------------------ preprocessing
    def _build_preprocess(self):
        inner = self._section(2, "Preprocessing",
                              "Applied to every exported picture (the augmented ones too). The labels move with the "
                              "pixels, and your original pictures are never changed.")
        self.orient_var = tk.BooleanVar(value=True)
        self._check(inner, "Auto-Orient", self.orient_var, self.refresh).pack(anchor='w')
        tk.Label(inner, text="Discard the EXIF rotation tag and save the pictures upright, so every program shows them "
                             "the way you annotated them.", bg=C_CARD, fg=C_TXT3, font=('Segoe UI', 8), anchor='w',
                 wraplength=WIDTH - 90, justify=tk.LEFT).pack(fill=tk.X, padx=24)

        self.resize_var = tk.BooleanVar(value=False)
        row = tk.Frame(inner, bg=C_CARD)
        row.pack(fill=tk.X, pady=(10, 0))
        self._check(row, "Resize", self.resize_var, self._resize_toggled).pack(side=tk.LEFT)
        self.mode = ttk.Combobox(row, values=[pre.RESIZE_LABELS[m] for m in ("stretch", "fit", "letterbox")],
                                 state='disabled', width=18, font=('Segoe UI', 10))
        self.mode.set(pre.RESIZE_LABELS['stretch'])
        self.mode.pack(side=tk.LEFT, padx=(14, 8))
        self.mode.bind('<<ComboboxSelected>>', lambda _e: self.refresh())
        self.width_var, self.height_var = tk.StringVar(value='640'), tk.StringVar(value='640')
        self.size_entries = []
        for i, var in enumerate((self.width_var, self.height_var)):
            if i:
                tk.Label(row, text="x", bg=C_CARD, fg=C_TXT2).pack(side=tk.LEFT, padx=6)
            e = tk.Entry(row, textvariable=var, width=6, font=('Segoe UI', 10), bg=C_CARD2, fg=C_TXT1,
                         insertbackground=C_ACCENT, relief=tk.FLAT, highlightthickness=1,
                         highlightbackground=C_BORDER, highlightcolor=C_ACCENT, state='disabled')
            e.pack(side=tk.LEFT, ipady=2)
            var.trace_add('write', lambda *_: self.refresh())
            self.size_entries.append(e)
        self.resize_hint = tk.Label(inner, text="", bg=C_CARD, fg=C_TXT3, font=('Segoe UI', 8), anchor='w',
                                    wraplength=WIDTH - 90, justify=tk.LEFT)
        self.resize_hint.pack(fill=tk.X, padx=24)

    def _resize_toggled(self):
        state = 'readonly' if self.resize_var.get() else 'disabled'
        self.mode.config(state=state)
        for e in self.size_entries:
            e.config(state='normal' if self.resize_var.get() else 'disabled')
        self.refresh()

    def _mode_key(self):
        shown = self.mode.get()
        return next((k for k, v in pre.RESIZE_LABELS.items() if v == shown), 'stretch')

    def build_preprocess(self):
        """The Preprocess for the choices, or raises ValueError with the message to show."""
        resize = None
        if self.resize_var.get():
            w, h = to_int(self.width_var.get()), to_int(self.height_var.get())
            lo, hi = pre.SIZE_RANGE
            if not (lo <= w <= hi and lo <= h <= hi):
                raise ValueError(f"Resize needs a width and a height between {lo} and {hi} pixels.")
            resize = (self._mode_key(), w, h)
        return pre.Preprocess(self.orient_var.get(), resize)

    # ------------------------------------------------------------ augmentation
    def _build_augmentation(self):
        inner = self._section(3, "Augmentation")
        self.panel = AugmentationPanel(inner, self.workspace, training=False, on_change=self.refresh)
        self.panel.pack(fill=tk.X)
        self.panel.extra_line = self._dataset_size_text

    def _dataset_size_text(self):
        _train, _valid, _test = self.sizes()
        return f"The training split grows to {_train} pictures."

    # ------------------------------------------------------------ footer
    def _build_footer(self):
        tk.Frame(self.footer, bg=C_BORDER, height=1).pack(fill=tk.X)
        inner = tk.Frame(self.footer, bg=C_BASE)
        inner.pack(fill=tk.X, padx=18, pady=10)
        self.total_label = tk.Label(inner, text="", bg=C_BASE, fg=C_TXT1, font=('Segoe UI', 10, 'bold'), anchor='w')
        self.total_label.pack(fill=tk.X)
        self.status = tk.Label(inner, text="", bg=C_BASE, fg=C_TXT2, font=('Segoe UI', 9), anchor='w')
        self.status.pack(fill=tk.X)
        self.progress = ttk.Progressbar(inner, orient='horizontal', mode='determinate', maximum=100)
        self.progress.pack(fill=tk.X, pady=(4, 8))
        row = tk.Frame(inner, bg=C_BASE)
        row.pack(fill=tk.X)
        self.export_btn = tk.Button(row, text='Export', command=self.start, bg=C_ACCENT, fg=C_ON_ACCENT,
                                    relief=tk.FLAT, cursor='hand2', font=('Segoe UI', 10, 'bold'))
        self.export_btn.pack(side=tk.RIGHT, ipadx=28, ipady=4)
        self.close_btn = tk.Button(row, text='Close', command=self.close, bg=C_CARD2, fg=C_TXT1, relief=tk.FLAT,
                                   cursor='hand2', font=('Segoe UI', 10))
        self.close_btn.pack(side=tk.RIGHT, padx=8, ipadx=18, ipady=4)

    # ------------------------------------------------------------ numbers
    def _records(self):
        return self.gui._find_workspace_images(IMAGE_EXTS)

    def image_count(self):
        records = self._records()
        if self.fmt.get() == 'YOLO':                   # only pictures that have a label file are exported
            labels = set(os.path.splitext(f)[0] for f in os.listdir(self.labels_folder)
                         if f.endswith('.txt')) if os.path.isdir(self.labels_folder) else set()
            return sum(1 for r in records if r['base'] in labels)
        return len(records)

    def sizes(self):
        """(train, valid, test) picture counts as the export will write them, augmented copies included."""
        train, valid, test = split_counts(self._total, self.train_pct, self.valid_pct)
        try:
            cfg = self.panel.get_config() if hasattr(self, 'panel') else None
        except ValueError:
            cfg = None
        if cfg is not None and cfg.enabled:
            per = cfg.copies + (1 if cfg.keep_original else 0)
            counts = {'train': train, 'val': valid, 'test': test}
            counts = {k: n * per if k in cfg.splits else n for k, n in counts.items()}
            return counts['train'], counts['val'], counts['test']
        return train, valid, test

    def recount(self):
        """The number of pictures the chosen format exports (listing the folders: not done on every drag)."""
        self._total = self.image_count()
        self.count_label.config(text=f"{self._total} images")
        self.refresh()

    def refresh(self):
        if not hasattr(self, 'total_label') or self._busy:
            return
        self._busy = True                            # the panel's summary calls back into refresh
        try:
            self.panel.refresh_summary()
            self._show()
        finally:
            self._busy = False

    def _show(self):
        t, v, te = split_counts(self._total, self.train_pct, self.valid_pct)
        for chip, name, n, pct in zip(self.chips, ("Train", "Valid", "Test"), (t, v, te),
                                      (self.train_pct, self.valid_pct, 100 - self.train_pct - self.valid_pct)):
            chip.config(text=f"●  {name}  {n}  ·  {pct}%", fg=chip.dot)
        self._draw_bar()
        if hasattr(self, 'resize_hint'):
            self.resize_hint.config(text=pre.RESIZE_HINTS[self._mode_key()] if self.resize_var.get() else "")
        a, b, c = self.sizes()
        self.total_label.config(text=f"Export:   Train {a}   ·   Valid {b}   ·   Test {c}")

    # ------------------------------------------------------------ run
    def lock(self, locked):
        self.active = locked
        state = 'disabled' if locked else 'normal'
        self.fmt.config(state='disabled' if locked else 'readonly')
        for w in self.locks:
            w.config(state=state)
        if not locked:
            self._resize_toggled()
        else:
            self.mode.config(state='disabled')
            for e in self.size_entries:
                e.config(state='disabled')
        self.panel.set_locked(locked)
        self.export_btn.config(state=state)
        self.close_btn.config(state=state)

    def close(self):
        if self.active:
            return
        self.dialog.destroy()

    def start(self):
        if self.active:
            return
        gui, dlg = self.gui, self.dialog
        t, v = self.train_pct, self.valid_pct
        fmt = self.fmt.get()
        try:
            preprocess = self.build_preprocess()
            config = self.panel.get_config()
        except ValueError as exc:
            messagebox.showerror('Export', str(exc), parent=dlg)
            return
        augmenter = None
        if config is not None and config.enabled:
            augmenter = Augmenter(config, post=preprocess.post if preprocess.resize else None)
        seed = config.seed if config is not None else None
        if config is not None and not config.enabled:
            messagebox.showinfo('Augmentation', 'One output per image that is the original adds nothing, so no '
                                                'augmented copies are made. Raise "Outputs per training image" to '
                                                'make some.', parent=dlg)

        chosen = self.dest or None
        fallback_note = ''
        if chosen and not os.path.isdir(chosen):
            # the drive was unplugged after the folder was picked: back to the default folder
            self.missing, self.dest = chosen, ''
            self.refresh_dest()
            fallback_note = f'\n\nThe folder you chose ({chosen}) was not found, so the default folder was used.'
            chosen = None
        export_target, replace = export_module.plan_export_target(self.default_folder, chosen, self.workspace, fmt)
        if chosen:
            needed = export_module.files_size(r['source'] for r in self._records())
            problem = export_module.check_destination(chosen, needed)
            if problem:
                messagebox.showerror('Save to', problem, parent=dlg)
                return

        self.lock(True)
        self.progress['value'] = 0
        self.status.config(text='Preparing export...')
        dlg.update_idletasks()                     # show the locked, 0% state right away
        last_pct = {'value': -1}
        use_pre = preprocess if preprocess.active else None

        def progress_cb(current, total, message=None):
            def update():
                pct = int((current / total) * 100) if total else 0
                self.progress['value'] = pct
                self.status.config(text=message or f'Exporting... {pct}%')
                if pct != last_pct['value']:           # small exports finish within one event-loop turn
                    last_pct['value'] = pct
                    self.progress.update_idletasks()
            self.root.after(0, update)

        def worker():
            try:
                target = export_target
                # The default folder is replaced on every export; a folder the user chose is only added to.
                try:
                    if replace and os.path.exists(target):
                        proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
                        abs_target = os.path.abspath(target)
                        if abs_target.startswith(proj_root):
                            shutil.rmtree(abs_target)
                        else:                          # outside the project nothing is deleted
                            target = os.path.join(target, f'export_{int(time.time())}')
                    os.makedirs(target, exist_ok=True)
                except Exception:
                    os.makedirs(target, exist_ok=True)

                if seed is not None:
                    random.seed(seed)                  # makes the train/val/test shuffle repeatable too

                if fmt == 'YOLO':
                    gui._export_yolo_dataset(target, t, v, IMAGE_EXTS, progress_cb, augment=augmenter,
                                             preprocess=use_pre)
                elif 'Pascal' in fmt or 'XML' in fmt:
                    gui._export_voc_dataset(target, t, v, IMAGE_EXTS, progress_cb, augment=augmenter,
                                            preprocess=use_pre)
                elif fmt == 'COCO':
                    gui._export_coco_dataset(target, t, v, IMAGE_EXTS, progress_cb, augment=augmenter,
                                             preprocess=use_pre)
                else:
                    raise ValueError(f'Unknown format: {fmt}')

                extra = fallback_note
                if use_pre is not None:
                    extra += '\n\nPreprocessing: ' + ', '.join(use_pre.describe())
                if augmenter is not None:
                    extra += f'\n\nAugmented copies added: {augmenter.created}'
                    if augmenter.skipped:
                        extra += f' ({augmenter.skipped} skipped: no effect fired)'

                def on_success():
                    self.lock(False)
                    self.progress['value'] = 100
                    self.status.config(text=f'Export completed to: {target}')
                    messagebox.showinfo('Export Completed', f'Dataset exported to:\n{target}{extra}', parent=dlg)
                self.root.after(0, on_success)
            except Exception as e:
                err = str(e)

                def on_error():
                    self.lock(False)
                    self.progress['value'] = 0
                    self.status.config(text='Export failed')
                    messagebox.showerror('Export Failed', f'Error during export:\n{err}', parent=dlg)
                self.root.after(0, on_error)

        threading.Thread(target=worker, daemon=True).start()
