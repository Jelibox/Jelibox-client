"""
A vertically scrolling body for the Label Assistant dialogs.

Their content grows with the number of classes (a 14-class head, an 80-class model), so on a smaller or scaled
screen (125% on a 1080p display leaves about 860 px) the bottom of a fixed-size window, Save and Cancel included,
fell off the screen. Header and footer stay put; only the body scrolls, and only when it has to.
"""
import tkinter as tk

from .theme import C_BORDER, C_BASE

SCREEN_MARGIN = 110          # room for the title bar and the taskbar


class ScrollBody:
    def __init__(self, parent, bg=C_BASE):
        self.frame = tk.Frame(parent, bg=bg)
        self.canvas = tk.Canvas(self.frame, bg=bg, highlightthickness=0, height=1)
        self.bar = tk.Scrollbar(self.frame, orient='vertical', command=self.canvas.yview, bg=C_BORDER,
                                troughcolor=bg, relief=tk.FLAT, width=10)
        self.canvas.configure(yscrollcommand=self.bar.set)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.inner = tk.Frame(self.canvas, bg=bg)
        self._window = self.canvas.create_window((0, 0), window=self.inner, anchor='nw')
        self.inner.bind('<Configure>', self._on_inner)
        self.canvas.bind('<Configure>', lambda e: self.canvas.itemconfigure(self._window, width=e.width))
        self.nested = []         # widgets with their own mouse-wheel scrolling (they keep the wheel)
        self._placed = False
        top = parent.winfo_toplevel()
        top.bind('<MouseWheel>', self._wheel, add='+')

    def pack(self, **kw):
        self.frame.pack(**kw)

    def _on_inner(self, _=None):
        self.canvas.configure(scrollregion=self.canvas.bbox('all'))
        self._sync_bar()

    def _sync_bar(self):
        if self.inner.winfo_reqheight() > self.canvas.winfo_height() + 1:
            if not self.bar.winfo_ismapped():
                self.bar.pack(side=tk.RIGHT, fill=tk.Y)
        elif self.bar.winfo_ismapped():
            self.bar.pack_forget()
            self.canvas.yview_moveto(0)

    def _wheel(self, event):
        if not self.bar.winfo_ismapped():
            return
        w = event.widget
        while w is not None:
            if w in self.nested:
                return
            w = getattr(w, 'master', None)
        self.scroll_by(event.delta)

    def scroll_by(self, delta):
        if self.bar.winfo_ismapped():
            self.canvas.yview_scroll(int(-delta / 120) or (-1 if delta > 0 else 1), 'units')

    def fit(self, win, width, parent=None):
        """Size `win` to its content, but never taller than the screen. The first call centres it over `parent`;
        later calls (the content changed) keep where the window is."""
        win.update_idletasks()
        need = self.inner.winfo_reqheight()
        self.canvas.configure(height=need)
        win.update_idletasks()
        total = win.winfo_reqheight()
        limit = win.winfo_screenheight() - SCREEN_MARGIN
        if total > limit:
            self.canvas.configure(height=max(120, need - (total - limit)))
            win.update_idletasks()
            total = win.winfo_reqheight()
        if not self._placed and parent is not None:
            x = parent.winfo_x() + parent.winfo_width() // 2 - width // 2
            y = parent.winfo_y() + parent.winfo_height() // 2 - total // 2
            self._placed = True
        else:
            x, y = win.winfo_x(), win.winfo_y()
        y = max(0, min(y, win.winfo_screenheight() - total - 40))
        win.geometry(f"{width}x{total}+{max(x, 0)}+{y}")
        win.update_idletasks()
        self._sync_bar()
