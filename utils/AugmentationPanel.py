"""The augmentation picker: a grid of tiles with a preview picture each (Flip, Brightness, Blur ...), a settings
window per tile with example pictures, and the outputs / seed controls. Used inside the Export Dataset window, and
(with `training=True`) in the Train window, where the augmentations Ultralytics already does are greyed out.

What the tiles are and how their previews are made lives in augment_catalog.py; the picture the previews use is chosen
in augment_sample.py. This file only draws.
"""
import os
import tkinter as tk
from tkinter import filedialog, messagebox

import cv2
from PIL import Image, ImageTk

from . import augment_catalog as cat
from . import augment_sample as sample
from . import augmentation as aug
from .theme import (C_BASE, C_PANEL, C_CARD, C_CARD2, C_BORDER, C_ACCENT, C_AMBER, C_GREEN, C_TXT1, C_TXT2, C_TXT3,
                    C_ON_ACCENT, C_STAGE)

THUMB = (150, 104)            # a tile's picture in the grid
TILE_W = 164
COLUMNS = 5
EXAMPLE_SIDE = (300, 190)     # the example pictures in a tile's settings window
ORIGINAL_SIDE = (400, 240)


def to_photo(bgr, box, bg=C_CARD2):
    """A BGR array as a Tk image that fits inside `box` (w, h), centred on a background of colour `bg`."""
    h, w = bgr.shape[:2]
    scale = min(box[0] / w, box[1] / h)
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    small = cv2.resize(bgr, size, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    pil = Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
    canvas = Image.new("RGB", box, bg)
    canvas.paste(pil, ((box[0] - size[0]) // 2, (box[1] - size[1]) // 2))
    return ImageTk.PhotoImage(canvas)


def _flat_button(parent, text, command, accent=False, **kw):
    return tk.Button(parent, text=text, command=command, bg=C_ACCENT if accent else C_CARD2,
                     fg=C_ON_ACCENT if accent else C_TXT1, font=('Segoe UI', 9, 'bold' if accent else 'normal'),
                     relief=tk.FLAT, cursor='hand2', borderwidth=0,
                     activebackground=C_ACCENT if accent else C_CARD2, activeforeground=C_ON_ACCENT if accent else C_TXT1,
                     **kw)


def _spin(parent, var, low, high, width=5):
    return tk.Spinbox(parent, from_=low, to=high, textvariable=var, width=width, font=('Segoe UI', 10), bg=C_CARD2,
                      fg=C_TXT1, buttonbackground=C_CARD2, relief=tk.FLAT, highlightthickness=1,
                      highlightbackground=C_BORDER, highlightcolor=C_ACCENT, insertbackground=C_ACCENT)


# ======================================================================== one tile's settings window
class TileDialog(tk.Toplevel):
    """Settings of one augmentation: the original and examples on the left (they follow the controls), the
    description and the controls on the right. Apply writes the settings into `state`; Go Back drops them."""

    def __init__(self, parent, tile, state, image, on_apply):
        super().__init__(parent)
        self.tile, self.state, self.image, self.on_apply = tile, state, image, on_apply
        # work on copies; Apply copies them back
        self.work = {op: aug.OpSetting(s.enabled, s.magnitude, s.probability, dict(s.options))
                     for op, s in state.items() if op in tile["ops"]}
        self.title(tile["title"])
        self.configure(bg=C_BASE)
        self.transient(parent.winfo_toplevel())
        self.resizable(False, False)
        self._photos = []
        self._after = None
        self.vars = {}
        self.warning = None

        head = tk.Frame(self, bg=C_PANEL, height=64)
        head.pack(fill=tk.X)
        head.pack_propagate(False)
        tk.Label(head, text=tile["title"], bg=C_PANEL, fg=C_TXT1, font=('Segoe UI', 15, 'bold')).pack(
            side=tk.LEFT, padx=26, pady=14)
        tk.Frame(self, bg=C_BORDER, height=1).pack(fill=tk.X)

        body = tk.Frame(self, bg=C_BASE)
        body.pack(fill=tk.BOTH, expand=True)
        self.stage = tk.Frame(body, bg=C_STAGE, padx=18, pady=16)
        self.stage.pack(side=tk.LEFT, fill=tk.BOTH)
        side = tk.Frame(body, bg=C_BASE, padx=24, pady=20, width=380)
        side.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        side.pack_propagate(False)
        self._controls(side)
        self.stage.configure(width=ORIGINAL_SIDE[0] + 36)

        tk.Frame(self, bg=C_BORDER, height=1).pack(fill=tk.X)
        foot = tk.Frame(self, bg=C_PANEL)
        foot.pack(fill=tk.X)
        _flat_button(foot, "Go Back", self.destroy).pack(side=tk.LEFT, padx=20, pady=12, ipadx=14, ipady=6)
        _flat_button(foot, "Apply", self._apply, accent=True).pack(side=tk.RIGHT, padx=20, pady=12, ipadx=26, ipady=6)
        self.remove_btn = _flat_button(foot, "Remove", self._remove)
        self.remove_btn.pack(side=tk.RIGHT, pady=12, ipadx=14, ipady=6)
        if not any(s.enabled for s in self.work.values()):
            self.remove_btn.config(state=tk.DISABLED)

        self._render()
        self.bind('<Escape>', lambda e: self.destroy())
        self.update_idletasks()
        top = parent.winfo_toplevel()
        x = top.winfo_rootx() + max(0, (top.winfo_width() - self.winfo_reqwidth()) // 2)
        y = max(0, top.winfo_rooty() + 40)
        self.geometry(f"+{x}+{y}")
        self.grab_set()
        self.focus_force()

    # ------------------------------------------------------------ controls
    def _controls(self, side):
        tile = self.tile
        tk.Label(side, text=tile["description"], bg=C_BASE, fg=C_TXT2, font=('Segoe UI', 10), justify=tk.LEFT,
                 anchor='w', wraplength=330).pack(fill=tk.X)
        slider = tile["slider"]
        if slider:
            op, words, unit = slider
            spec = aug.OPERATIONS[op]
            row = tk.Frame(side, bg=C_BASE)
            row.pack(fill=tk.X, pady=(20, 0))
            tk.Label(row, text=words, bg=C_BASE, fg=C_TXT2, font=('Segoe UI', 9)).pack(side=tk.LEFT)
            self.amount_badge = tk.Label(row, text="", bg=C_ACCENT, fg=C_ON_ACCENT, font=('Segoe UI', 9, 'bold'),
                                         padx=10, pady=2)
            self.amount_badge.pack(side=tk.RIGHT)
            var = tk.DoubleVar(value=min(spec["range"][1], max(spec["range"][0], self.work[op].magnitude)))
            self.vars["magnitude"] = var
            tk.Scale(side, variable=var, from_=spec["range"][0], to=spec["range"][1], resolution=1, orient='horizontal',
                     showvalue=False, command=lambda _v: self._changed(), bg=C_BASE, fg=C_TXT1, troughcolor=C_CARD2,
                     highlightthickness=0, activebackground=C_ACCENT, bd=0, length=330).pack(fill=tk.X, pady=(2, 0))
            self._unit = unit
        if tile["choices"]:
            box = tk.Frame(side, bg=C_BASE)
            box.pack(fill=tk.X, pady=(14, 0))
            for op, option, text in tile["choices"]:
                if option is None:
                    value = self.work[op].enabled or not any(s.enabled for s in self.work.values())
                else:
                    value = self.work[op].option(option)
                var = tk.BooleanVar(value=bool(value))
                self.vars[(op, option)] = var
                tk.Checkbutton(box, text=text, variable=var, command=self._changed, bg=C_BASE, fg=C_TXT1,
                               selectcolor=C_CARD, activebackground=C_BASE, activeforeground=C_ACCENT,
                               font=('Segoe UI', 10, 'bold'), bd=0, highlightthickness=0, cursor='hand2',
                               anchor='w').pack(fill=tk.X, pady=2)
        if tile.get("count"):
            op, option, text, low, high = tile["count"]
            row = tk.Frame(side, bg=C_BASE)
            row.pack(fill=tk.X, pady=(12, 0))
            tk.Label(row, text=text, bg=C_BASE, fg=C_TXT1, font=('Segoe UI', 10, 'bold')).pack(side=tk.LEFT)
            var = tk.IntVar(value=int(self.work[op].options.get(option, 3)))
            self.vars["count"] = var
            spin = _spin(row, var, low, high)
            spin.pack(side=tk.LEFT, padx=10)
            var.trace_add("write", lambda *_: self._changed())
        # how many of the outputs get it
        row = tk.Frame(side, bg=C_BASE)
        row.pack(fill=tk.X, pady=(20, 0))
        tk.Label(row, text="Applied to", bg=C_BASE, fg=C_TXT2, font=('Segoe UI', 9)).pack(side=tk.LEFT)
        self.chance_badge = tk.Label(row, text="", bg=C_CARD2, fg=C_TXT1, font=('Segoe UI', 9, 'bold'), padx=10, pady=2)
        self.chance_badge.pack(side=tk.RIGHT)
        chance = tk.DoubleVar(value=self.work[tile["ops"][0]].probability)
        self.vars["chance"] = chance
        tk.Scale(side, variable=chance, from_=5, to=100, resolution=5, orient='horizontal', showvalue=False,
                 command=lambda _v: self._changed(), bg=C_BASE, fg=C_TXT1, troughcolor=C_CARD2, highlightthickness=0,
                 activebackground=C_ACCENT, bd=0, length=330).pack(fill=tk.X, pady=(2, 0))
        tk.Label(side, text="of the augmented pictures (each picture draws its own combination).", bg=C_BASE, fg=C_TXT3,
                 font=('Segoe UI', 8), anchor='w', justify=tk.LEFT, wraplength=330).pack(fill=tk.X)
        self.warning = tk.Label(side, text="", bg=C_BASE, fg=C_AMBER, font=('Segoe UI', 9, 'bold'), anchor='w',
                                justify=tk.LEFT, wraplength=330)
        self.warning.pack(fill=tk.X, pady=(12, 0))

    def _read(self):
        """Write what the controls say into the working copies."""
        tile = self.tile
        chance = float(self.vars["chance"].get())
        for op in tile["ops"]:
            self.work[op].probability = chance
        if tile["slider"]:
            self.work[tile["slider"][0]].magnitude = float(self.vars["magnitude"].get())
        for op, option, _ in tile["choices"]:
            value = bool(self.vars[(op, option)].get())
            if option is None:
                self.work[op].enabled = value
            else:
                self.work[op].options[option] = value
        if tile.get("count"):
            op, option = tile["count"][0], tile["count"][1]
            try:
                self.work[op].options[option] = max(tile["count"][3], min(tile["count"][4], int(self.vars["count"].get())))
            except (tk.TclError, ValueError):
                pass

    def _valid(self):
        """At least one of a tile's choices must stay ticked."""
        tile = self.tile
        if not tile["choices"]:
            return True
        if tile["choices"][0][1] is None:                     # separate operations (Flip): any tick at all
            return any(self.work[op].enabled for op, _, _ in tile["choices"])
        return any(self.work[op].option(option) for op, option, _ in tile["choices"])

    def _changed(self):
        self._read()
        if self.vars.get("magnitude") is not None:
            self.amount_badge.config(text=f"{int(self.vars['magnitude'].get())}{self._unit}")
        self.chance_badge.config(text=f"{int(self.vars['chance'].get())}%")
        self.warning.config(text="" if self._valid() else "Tick at least one choice.")
        if self._after:
            self.after_cancel(self._after)
        self._after = self.after(70, self._render)            # redraw once the slider rests

    # ------------------------------------------------------------ previews
    def _render(self):
        self._after = None
        if not self.winfo_exists():
            return
        self._read()
        self.chance_badge.config(text=f"{int(self.vars['chance'].get())}%")
        if self.vars.get("magnitude") is not None:
            self.amount_badge.config(text=f"{int(self.vars['magnitude'].get())}{self._unit}")
        for child in self.stage.winfo_children():
            child.destroy()
        self._photos = []
        shown = cat.previews(self.tile, {**self.state, **self.work}, self.image)
        first = shown[0]
        self._picture(self.stage, first, ORIGINAL_SIDE).pack(pady=(0, 14))
        row = tk.Frame(self.stage, bg=C_STAGE)
        row.pack()
        for example in shown[1:]:
            self._picture(row, example, EXAMPLE_SIDE if len(shown) > 2 else ORIGINAL_SIDE).pack(side=tk.LEFT, padx=6)

    def _picture(self, parent, item, box):
        label, img = item
        cell = tk.Frame(parent, bg=C_STAGE)
        photo = to_photo(img, box, C_STAGE)
        self._photos.append(photo)
        tk.Label(cell, image=photo, bg=C_STAGE, bd=0).pack()
        tk.Label(cell, text=label, bg=C_STAGE, fg=C_TXT2, font=('Segoe UI', 9)).pack(pady=(4, 0))
        return cell

    # ------------------------------------------------------------ buttons
    def _apply(self):
        self._read()
        if not self._valid():
            self.warning.config(text="Tick at least one choice.")
            return
        for op, setting in self.work.items():
            if self.tile["choices"] and self.tile["choices"][0][1] is None:
                enabled = setting.enabled                      # Flip: each operation has its own tick
            else:
                enabled = True
            self.state[op] = aug.OpSetting(enabled, setting.magnitude, setting.probability, dict(setting.options))
        self.on_apply(self.tile)
        self.destroy()

    def _remove(self):
        for op in self.tile["ops"]:
            s = self.state[op]
            self.state[op] = aug.OpSetting(False, s.magnitude, s.probability, dict(s.options))
        self.on_apply(self.tile)
        self.destroy()


# ======================================================================== the grid
class AugmentationPanel(tk.Frame):
    """The picker. `training=True` greys out the tiles Ultralytics already does and says so."""

    def __init__(self, parent, workspace, training=False, on_change=None, bg=C_CARD, state=None, outputs=None,
                 keep_original=True, seed=None):
        super().__init__(parent, bg=bg)
        self.bg, self.workspace, self.training, self.on_change = bg, workspace, training, on_change
        self.state = state if state is not None else cat.default_state()
        self.outputs_var = tk.IntVar(value=outputs or cat.DEFAULT_OUTPUTS)
        self.keep_var = tk.BooleanVar(value=bool(keep_original))
        self.seed_var = tk.StringVar(value="" if seed is None else str(seed))
        self.extra_line = None                       # callable -> text shown under the summary (e.g. dataset size)
        self.locked = False
        self._photos = []
        self._tiles = {}
        self._build()
        self.load_sample()

    # ------------------------------------------------------------ layout
    def _build(self):
        bg = self.bg
        tk.Label(self, text="Augmentations create new training examples for your model to learn from. Click one to "
                            "set it up; the previews use the picture below.", bg=bg, fg=C_TXT3, font=('Segoe UI', 9),
                 anchor='w', justify=tk.LEFT, wraplength=840).pack(fill=tk.X)
        row = tk.Frame(self, bg=bg)
        row.pack(fill=tk.X, pady=(8, 8))
        self.choose_btn = _flat_button(row, "Choose preview picture...", self.choose_sample)
        self.choose_btn.pack(side=tk.LEFT, ipadx=10, ipady=4)
        self.reset_btn = _flat_button(row, "First picture", self.reset_sample)
        self.reset_btn.pack(side=tk.LEFT, padx=(8, 0), ipadx=10, ipady=4)
        self.sample_label = tk.Label(row, text="", bg=bg, fg=C_TXT2, font=('Segoe UI', 9), anchor='w')
        self.sample_label.pack(side=tk.LEFT, padx=12)
        self.grid_frame = tk.Frame(self, bg=bg)
        self.grid_frame.pack(fill=tk.X)
        self.note = tk.Label(self, text="", bg=bg, fg=C_AMBER, font=('Segoe UI', 9), anchor='w', justify=tk.LEFT,
                             wraplength=840)
        self.note.pack(fill=tk.X, pady=(6, 0))

        out = tk.Frame(self, bg=bg)
        out.pack(fill=tk.X, pady=(10, 0))
        tk.Label(out, text="Outputs per training image", bg=bg, fg=C_TXT1, font=('Segoe UI', 10, 'bold')).pack(side=tk.LEFT)
        self.spin = _spin(out, self.outputs_var, cat.OUTPUTS_RANGE[0], cat.OUTPUTS_RANGE[1], width=4)
        self.spin.pack(side=tk.LEFT, padx=8)
        tk.Label(out, text="x", bg=bg, fg=C_TXT2, font=('Segoe UI', 10, 'bold')).pack(side=tk.LEFT)
        self.keep_chk = tk.Checkbutton(out, text="Keep the original as one of them", variable=self.keep_var,
                                       bg=bg, fg=C_TXT1, selectcolor=C_BASE, activebackground=bg,
                                       activeforeground=C_ACCENT, font=('Segoe UI', 9), bd=0, highlightthickness=0,
                                       cursor='hand2')
        self.keep_chk.pack(side=tk.LEFT, padx=22)
        seed = tk.Frame(self, bg=bg)
        seed.pack(fill=tk.X, pady=(6, 0))
        tk.Label(seed, text="Seed (optional)", bg=bg, fg=C_TXT2, font=('Segoe UI', 9)).pack(side=tk.LEFT)
        self.seed_entry = tk.Entry(seed, textvariable=self.seed_var, width=10, font=('Segoe UI', 10), bg=C_CARD2,
                                   fg=C_TXT1, insertbackground=C_ACCENT, relief=tk.FLAT, highlightthickness=1,
                                   highlightbackground=C_BORDER, highlightcolor=C_ACCENT)
        self.seed_entry.pack(side=tk.LEFT, padx=8, ipady=2)
        tk.Label(seed, text="the same number gives the same pictures again", bg=bg, fg=C_TXT3,
                 font=('Segoe UI', 8)).pack(side=tk.LEFT)
        self.status = tk.Label(self, text="", bg=bg, fg=C_TXT2, font=('Segoe UI', 9, 'bold'), anchor='w',
                               justify=tk.LEFT, wraplength=840)
        self.status.pack(fill=tk.X, pady=(8, 0))
        self.detail = tk.Label(self, text="", bg=bg, fg=C_TXT3, font=('Segoe UI', 8), anchor='w', justify=tk.LEFT,
                               wraplength=840)
        self.detail.pack(fill=tk.X)
        for var in (self.outputs_var, self.keep_var, self.seed_var):
            var.trace_add("write", lambda *_: self.refresh_summary())

    # ------------------------------------------------------------ the preview picture
    def load_sample(self):
        path, is_default = sample.pick_sample(self.workspace)
        self.sample_path = path
        self.image = sample.load_sample(path)
        if self.image is None and path:                    # a picture that cannot be read: try the first one
            path, is_default = (sample.workspace_images(self.workspace) or [None])[0], True
            self.sample_path, self.image = path, sample.load_sample(path)
        if self.image is None:
            self.sample_label.config(text="This workspace has no pictures to preview with yet.")
        else:
            kind = "the first picture" if is_default else "your choice"
            self.sample_label.config(text=f"{os.path.basename(path)}  ({kind})")
        self.reset_btn.config(state=tk.DISABLED if is_default else tk.NORMAL)
        self.redraw()

    def choose_sample(self):
        images = sample.workspace_images(self.workspace)
        start = os.path.dirname(images[0]) if images else None
        path = filedialog.askopenfilename(parent=self, title="Choose the picture the previews use",
                                          initialdir=start, filetypes=[("Pictures", "*.jpg *.jpeg *.png *.bmp *.webp *.tiff"),
                                                                       ("All files", "*.*")])
        if not path:
            return
        if sample.load_sample(path) is None:
            messagebox.showwarning("Preview picture", "That file could not be read as a picture.", parent=self)
            return
        sample.remember_sample(self.workspace, path)
        self.load_sample()

    def reset_sample(self):
        sample.forget_sample(self.workspace)
        self.load_sample()

    # ------------------------------------------------------------ the grid
    def redraw(self):
        for child in self.grid_frame.winfo_children():
            child.destroy()
        self._photos, self._tiles = [], {}
        for index, (tile, note) in enumerate(cat.usable_tiles(self.training)):
            cell = self._tile(tile, note)
            cell.grid(row=index // COLUMNS, column=index % COLUMNS, padx=4, pady=4, sticky='n')
        self.refresh_summary()

    def _tile(self, tile, note):
        selected = cat.tile_enabled(tile, self.state)
        disabled = note is not None
        border = C_ACCENT if selected else C_BORDER
        cell = tk.Frame(self.grid_frame, bg=self.bg, width=TILE_W, highlightbackground=border,
                        highlightthickness=2 if selected else 1, cursor='arrow' if disabled else 'hand2')
        if self.image is not None:
            thumb = cat.thumbnail(tile, self.state, self.image)
            if disabled:
                thumb = (thumb * 0.45 + 90 * 0.55).astype(thumb.dtype)       # greyed out
        else:
            import numpy as np
            thumb = np.full((THUMB[1], THUMB[0], 3), 200, np.uint8)
        photo = to_photo(thumb, THUMB, self.bg)
        self._photos.append(photo)
        picture = tk.Label(cell, image=photo, bg=self.bg, bd=0)
        picture.pack(padx=4, pady=(4, 0))
        title = tile["title"] + ("  ✓" if selected else "")
        caption = tk.Label(cell, text=title, bg=self.bg, fg=C_TXT3 if disabled else (C_ACCENT if selected else C_TXT1),
                           font=('Segoe UI', 9, 'bold' if selected else 'normal'))
        caption.pack(pady=(2, 0))
        widgets = [cell, picture, caption]
        if disabled:
            badge = tk.Label(cell, text="Built into Ultralytics", bg=self.bg, fg=C_AMBER, font=('Segoe UI', 7, 'bold'))
            badge.pack(pady=(0, 4))
            widgets.append(badge)
        else:
            tk.Frame(cell, bg=self.bg, height=4).pack()
        for w in widgets:
            w.bind("<Button-1>", lambda _e, t=tile, n=note: self.open_tile(t, n))
        self._tiles[tile["key"]] = cell
        return cell

    def set_locked(self, locked):
        """Lock the controls (the tiles open nothing) while an export runs."""
        self.locked = bool(locked)
        state = tk.DISABLED if locked else tk.NORMAL
        for w in (self.keep_chk, self.seed_entry, self.spin, self.choose_btn):
            w.config(state=state)
        if locked:
            self.reset_btn.config(state=tk.DISABLED)
        else:
            self.reset_btn.config(state=tk.DISABLED if sample.pick_sample(self.workspace)[1] else tk.NORMAL)

    def open_tile(self, tile, note=None):
        if self.locked:
            return
        if note:
            self.note.config(text=f"{tile['title']}: {note}")
            return
        self.note.config(text="")
        if self.image is None:
            messagebox.showinfo("Augmentation", "Add pictures to this workspace first: the previews need one.",
                                parent=self)
            return
        TileDialog(self, tile, self.state, self.image, self._tile_applied)

    def _tile_applied(self, tile):
        self.redraw()
        if self.on_change:
            self.on_change()

    # ------------------------------------------------------------ result
    def seed(self):
        text = self.seed_var.get().strip()
        if not text:
            return None
        try:
            return int(text)
        except ValueError:
            raise ValueError("The seed must be a whole number, or empty.")

    def outputs(self):
        try:
            return max(cat.OUTPUTS_RANGE[0], min(cat.OUTPUTS_RANGE[1], int(self.outputs_var.get())))
        except (tk.TclError, ValueError):
            return cat.DEFAULT_OUTPUTS

    def picked(self):
        return cat.selected_tiles(self.state)

    def get_config(self):
        """The AugmentConfig for the picks, or None when nothing is picked. ValueError for a bad seed."""
        if not self.picked():
            return None
        return cat.to_config(self.state, self.outputs(), self.keep_var.get(), self.seed())

    def refresh_summary(self):
        names = cat.summary(self.state)
        outputs = self.outputs()
        if not names:
            self.status.config(text="No augmentation selected.", fg=C_TXT2)
            self.detail.config(text="")
        else:
            self.status.config(text=f"{len(names)} selected: " + ", ".join(names), fg=C_GREEN)
            copies = outputs - 1 if self.keep_var.get() else outputs
            if copies <= 0:
                detail = "One output that is the original adds nothing: raise the outputs or untick \"Keep the original\"."
            elif self.keep_var.get():
                detail = f"Each training image becomes {outputs} pictures: the original and {copies} augmented."
            else:
                detail = f"Each training image is replaced by {outputs} augmented pictures."
            extra = self.extra_line() if self.extra_line else ""
            self.detail.config(text=(detail + ("  " + extra if extra else "")))
        if self.on_change:
            self.on_change()


# ======================================================================== the training window's copy
class AugmentationDialog(tk.Toplevel):
    """The picker in a window of its own (the Train windows open it from a button). OK keeps the picks."""

    def __init__(self, parent, workspace, training=True, state=None, outputs=None, keep_original=True, seed=None,
                 extra_line=None):
        super().__init__(parent)
        self.title("Augmentation")
        self.configure(bg=C_BASE)
        self.transient(parent.winfo_toplevel())
        self.result = None
        head = tk.Frame(self, bg=C_PANEL, height=56)
        head.pack(fill=tk.X)
        head.pack_propagate(False)
        tk.Label(head, text="Augmentation", bg=C_PANEL, fg=C_TXT1, font=('Segoe UI', 13, 'bold')).pack(
            side=tk.LEFT, padx=22, pady=12)
        tk.Frame(self, bg=C_ACCENT, height=2).pack(fill=tk.X)
        body = tk.Frame(self, bg=C_CARD, highlightbackground=C_BORDER, highlightthickness=1)
        body.pack(fill=tk.BOTH, expand=True, padx=18, pady=14)
        self.panel = AugmentationPanel(body, workspace, training=training, state=state, outputs=outputs,
                                       keep_original=keep_original, seed=seed)
        self.panel.extra_line = extra_line
        self.panel.pack(fill=tk.BOTH, expand=True, padx=14, pady=12)
        self.panel.refresh_summary()
        foot = tk.Frame(self, bg=C_BASE)
        foot.pack(fill=tk.X, padx=18, pady=(0, 14))
        _flat_button(foot, "Cancel", self.destroy).pack(side=tk.LEFT, ipadx=16, ipady=6)
        _flat_button(foot, "Clear all", self._clear).pack(side=tk.LEFT, padx=8, ipadx=14, ipady=6)
        _flat_button(foot, "OK", self._ok, accent=True).pack(side=tk.RIGHT, ipadx=30, ipady=6)
        self.bind('<Escape>', lambda e: self.destroy())
        self.update_idletasks()
        top = parent.winfo_toplevel()
        self.geometry(f"+{max(0, top.winfo_rootx() + 30)}+{max(0, top.winfo_rooty() + 20)}")
        self.grab_set()

    def _clear(self):
        self.panel.state = cat.default_state()
        self.panel.redraw()

    def _ok(self):
        try:
            config = self.panel.get_config()
        except ValueError as exc:
            messagebox.showwarning("Augmentation", str(exc), parent=self)
            return
        self.result = {"config": config, "state": self.panel.state, "outputs": self.panel.outputs(),
                       "keep_original": bool(self.panel.keep_var.get()), "seed": self.panel.seed()}
        self.destroy()


def ask_augmentation(parent, workspace, training=True, previous=None, extra_line=None):
    """Open the picker as a window and wait. Returns its result dict ({"config", "state", "outputs",
    "keep_original", "seed"}) or None when cancelled; `previous` is an earlier result, to start from its picks."""
    previous = previous or {}
    dlg = AugmentationDialog(parent, workspace, training, previous.get("state"), previous.get("outputs"),
                             previous.get("keep_original", True), previous.get("seed"), extra_line)
    parent.wait_window(dlg)
    return dlg.result


class AugmentChoice:
    """The Train windows' Augmentation row: a button that opens the picker, and one line saying what is picked.
    `base_count()` returns how many training pictures there are before augmentation (None = unknown)."""

    def __init__(self, parent, window, workspace, base_count=None, bg=C_CARD, wrap=520):
        self.window, self.workspace, self.base_count = window, workspace, base_count
        self.result = None
        self.frame = tk.Frame(parent, bg=bg)
        self.button = _flat_button(self.frame, "Augmentation ...", self.open)
        self.button.pack(side=tk.LEFT, ipadx=10, ipady=3)
        self.label = tk.Label(self.frame, text="", bg=bg, fg=C_TXT2, font=('Segoe UI', 9), anchor='w', justify=tk.LEFT,
                              wraplength=wrap)
        self.label.pack(side=tk.LEFT, padx=12, fill=tk.X, expand=True)
        self.refresh()

    def growth(self):
        """The line the picker shows under its summary: how many training pictures the run will use."""
        base = self.base_count() if self.base_count else None
        cfg = self.config()
        if not base or cfg is None:
            return ""
        return f"Training grows from {base} to {base * (cfg.copies + (1 if cfg.keep_original else 0))} pictures."

    def open(self):
        got = ask_augmentation(self.window, self.workspace, training=True, previous=self.result, extra_line=self.growth)
        if got is not None:
            self.result = got
            self.refresh()

    def config(self):
        """The AugmentConfig that makes copies, or None."""
        cfg = self.result["config"] if self.result else None
        return cfg if cfg is not None and cfg.enabled else None

    def job_dict(self):
        """What goes into the job file (config_to_dict), or None when nothing is picked."""
        cfg = self.config()
        return aug.config_to_dict(cfg) if cfg is not None else None

    def refresh(self):
        cfg = self.config()
        if cfg is None:
            self.label.config(text="None. Ultralytics still applies its own (mosaic, flips, colour shifts) while it trains.",
                              fg=C_TXT3)
            return
        names = cat.summary(self.result["state"])
        text = "Adds copies of the training pictures: " + ", ".join(names) + "."
        grown = self.growth()
        self.label.config(text=text + ("  " + grown if grown else ""), fg=C_GREEN)
