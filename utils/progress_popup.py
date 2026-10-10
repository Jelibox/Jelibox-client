"""A small centered popup with a status line and a progress bar, shared by the workspace picker and Analyze Dataset."""
import tkinter as tk

from .theme import C_BASE, C_PANEL, C_TXT1


def show_progress_popup(parent, title, bar_color):
    """Returns (popup, update_fn): update_fn(done, total, label=None) advances the bar and repaints it - call
    popup.destroy() when the work is finished. The caller works on the Tk thread (no worker thread here)."""
    popup = tk.Toplevel(parent)
    popup.title(title)
    popup.configure(bg=C_BASE)
    popup.transient(parent)
    popup.grab_set()
    popup.resizable(False, False)
    popup.protocol("WM_DELETE_WINDOW", lambda: None)         # can't be cancelled half way

    width = 420
    tk.Frame(popup, bg=C_BASE, width=width, height=1).pack(side=tk.TOP)

    status_var = tk.StringVar(value=title)
    tk.Label(popup, textvariable=status_var, bg=C_BASE, fg=C_TXT1, font=('Segoe UI', 10)).pack(pady=(20, 14))

    bar_w = width - 48
    bar_bg = tk.Frame(popup, bg=C_PANEL, width=bar_w, height=6)
    bar_bg.pack(pady=(0, 20))
    bar_bg.pack_propagate(False)
    bar_fill = tk.Frame(bar_bg, bg=bar_color, width=0, height=6)
    bar_fill.place(x=0, y=0, relheight=1)

    def update_fn(done, total, label=None):
        if label:
            status_var.set(label)
        frac = (done / total) if total else 1.0
        bar_fill.place(width=int(bar_w * max(0.0, min(1.0, frac))))
        popup.update()

    popup.update_idletasks()
    x = (popup.winfo_screenwidth() // 2) - (popup.winfo_width() // 2)
    y = (popup.winfo_screenheight() // 2) - (popup.winfo_height() // 2)
    popup.geometry(f"+{x}+{y}")
    popup.update()

    return popup, update_fn
