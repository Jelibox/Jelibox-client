"""The Train / Valid / Test split control: a colour bar with two handles to drag, a chip per split with its picture
count and percentage, and two boxes to type the percentages. Used by Export Dataset and by the full YOLO training window.
"""
import tkinter as tk

from .theme import C_CARD, C_CARD2, C_ACCENT, C_AMBER, C_GREEN, C_TXT1, C_TXT3

SPLIT_COLORS = (C_ACCENT, C_AMBER, C_GREEN)
NAMES = ("Train", "Valid", "Test")


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


class SplitBar(tk.Frame):
    def __init__(self, parent, bg=C_CARD, on_change=None, train=70, valid=20, width=820):
        super().__init__(parent, bg=bg)
        self.bg, self.on_change, self.width = bg, on_change, width
        self.train_pct, self.valid_pct = train, valid
        self.total = 0
        self.locked = False
        self._drag = None
        self._setting = False
        self.train_var = tk.StringVar(value=str(train))
        self.valid_var = tk.StringVar(value=str(valid))
        self.test_var = tk.StringVar(value=str(100 - train - valid))
        self.bar = tk.Canvas(self, width=width, height=46, bg=bg, highlightthickness=0)
        self.bar.pack(anchor='w', pady=(6, 0))
        self.bar.bind('<Button-1>', self._press)
        self.bar.bind('<B1-Motion>', self._move)
        self.bar.bind('<ButtonRelease-1>', lambda _e: setattr(self, '_drag', None))
        row = tk.Frame(self, bg=bg)
        row.pack(fill=tk.X, pady=(8, 0))
        self.chips = []
        for color in SPLIT_COLORS:
            chip = tk.Label(row, text='', bg=C_CARD2, fg=C_TXT1, font=('Segoe UI', 9, 'bold'), padx=10, pady=4)
            chip.pack(side=tk.LEFT, padx=(0, 8))
            chip.dot = color
            self.chips.append(chip)
        tk.Label(row, text="Train %", bg=bg, fg=C_TXT3, font=('Segoe UI', 9)).pack(side=tk.LEFT, padx=(18, 4))
        self.sb_train = tk.Spinbox(row, from_=0, to=100, textvariable=self.train_var, width=5, command=self._typed)
        self.sb_train.pack(side=tk.LEFT)
        tk.Label(row, text="Valid %", bg=bg, fg=C_TXT3, font=('Segoe UI', 9)).pack(side=tk.LEFT, padx=(12, 4))
        self.sb_valid = tk.Spinbox(row, from_=0, to=100, textvariable=self.valid_var, width=5, command=self._typed)
        self.sb_valid.pack(side=tk.LEFT)
        self.train_var.trace_add('write', lambda *_: self._typed())
        self.valid_var.trace_add('write', lambda *_: self._typed())
        self.refresh()

    @property
    def test_pct(self):
        return 100 - self.train_pct - self.valid_pct

    def counts(self):
        return split_counts(self.total, self.train_pct, self.valid_pct)

    def set_total(self, total):
        self.total = int(total)
        self.refresh()

    def set_locked(self, locked):
        self.locked = bool(locked)
        for w in (self.sb_train, self.sb_valid):
            w.config(state='disabled' if locked else 'normal')

    # ------------------------------------------------------------ drawing
    def _x_of(self, pct):
        return 10 + pct / 100.0 * (self.width - 20)

    def _pct_of(self, x):
        return round(max(0, min(100, (x - 10) / (self.width - 20) * 100)))

    def refresh(self):
        c = self.bar
        c.delete('all')
        t, v = self.train_pct, self.valid_pct
        edges = [0, t, t + v, 100]
        for i, color in enumerate(SPLIT_COLORS):
            x1, x2 = self._x_of(edges[i]), self._x_of(edges[i + 1])
            if x2 > x1:
                c.create_rectangle(x1, 24, x2, 38, fill=color, outline='')
            if x2 - x1 > 34:
                c.create_text((x1 + x2) / 2, 11, text=f"{edges[i + 1] - edges[i]}%", fill=C_TXT1,
                              font=('Segoe UI', 9, 'bold'))
        for x in (self._x_of(t), self._x_of(t + v)):
            c.create_oval(x - 8, 23, x + 8, 39, fill=self.bg, outline=C_TXT1, width=2)
        for chip, name, n, pct in zip(self.chips, NAMES, self.counts(), (t, v, self.test_pct)):
            chip.config(text=f"●  {name}  {n}  ·  {pct}%", fg=chip.dot)

    def _changed(self):
        self.refresh()
        if self.on_change:
            self.on_change()

    # ------------------------------------------------------------ mouse + typing
    def _press(self, event):
        if self.locked:
            return
        first = abs(event.x - self._x_of(self.train_pct))
        second = abs(event.x - self._x_of(self.train_pct + self.valid_pct))
        self._drag = 'train' if first <= second else 'valid'

    def _move(self, event):
        if not self._drag or self.locked:
            return
        pct = self._pct_of(event.x)
        if self._drag == 'train':
            self.train_pct = pct
            self.valid_pct = min(self.valid_pct, 100 - self.train_pct)
        else:
            self.valid_pct = max(0, min(pct - self.train_pct, 100 - self.train_pct))
        self._sync_vars()
        self._changed()

    def _sync_vars(self):
        self._setting = True
        self.train_var.set(str(self.train_pct))
        self.valid_var.set(str(self.valid_pct))
        self.test_var.set(str(self.test_pct))
        self._setting = False

    def _typed(self):
        if self._setting:
            return
        self.train_pct = max(0, min(100, to_int(self.train_var.get())))
        self.valid_pct = max(0, min(100 - self.train_pct, to_int(self.valid_var.get())))
        self.test_var.set(str(self.test_pct))
        self._changed()
