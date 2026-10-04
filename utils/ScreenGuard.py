"""
Screen guard for the annotation window.

Jelibox's annotation GUI is built for desktop-class landscape displays. Before
the window is shown we inspect the screen: if it does not look like a desktop
monitor we refuse to open (with a Jelibox-styled dialog explaining why),
otherwise the window is maximized.
"""
import tkinter as tk
from PIL import Image, ImageTk

from .theme import (C_BASE, C_PANEL, C_CARD, C_BORDER, C_ACCENT, C_RED, C_ON_ACCENT,
                    C_TXT1, C_TXT2, C_TXT3)

MIN_WIDTH = 1024
MIN_HEIGHT = 600
MIN_ASPECT = 1.3   # width / height - 4:3 and wider; portrait/phone-like screens fail


def check_screen(root):
    """Return (ok, reason, width, height). `reason` is English, empty if ok."""
    w, h = root.winfo_screenwidth(), root.winfo_screenheight()
    aspect = w / h if h else 0
    problems = []
    if aspect < MIN_ASPECT:
        problems.append(
            f"The screen is not in landscape desktop proportions "
            f"(aspect ratio {aspect:.2f}, at least {MIN_ASPECT:.2f} is required).")
    if w < MIN_WIDTH or h < MIN_HEIGHT:
        problems.append(
            f"The screen is too small (minimum {MIN_WIDTH}×{MIN_HEIGHT} required).")
    return (not problems), " ".join(problems), w, h


def maximize(root):
    """Fill the screen with the window (keeps the title bar and taskbar)."""
    try:
        root.state('zoomed')                  # Windows
    except tk.TclError:
        try:
            root.attributes('-zoomed', True)  # Linux
        except tk.TclError:
            root.geometry(f"{root.winfo_screenwidth()}x{root.winfo_screenheight()}+0+0")


def show_unsupported_dialog(root, reason, w, h):
    """Modal Jelibox-styled window explaining why Jelibox can't open."""
    win = tk.Toplevel(root)
    win.title("Jelibox")
    win.configure(bg=C_BASE)
    win.resizable(False, False)

    width, height = 480, 340
    x = (win.winfo_screenwidth() // 2) - (width // 2)
    y = (win.winfo_screenheight() // 2) - (height // 2)
    win.geometry(f"{width}x{height}+{max(x, 0)}+{max(y, 0)}")

    # Header bar with logo + brand (same look as the main GUI header)
    header = tk.Frame(win, bg=C_PANEL, height=54)
    header.pack(fill=tk.X)
    header.pack_propagate(False)
    try:
        logo = Image.open("assets/jelibox.png").resize((32, 32))
        win._logo = ImageTk.PhotoImage(logo)
        tk.Label(header, image=win._logo, bg=C_PANEL).pack(side=tk.LEFT, padx=(14, 8))
    except Exception:
        pass
    tk.Label(header, text="Jelibox", bg=C_PANEL, fg=C_TXT1,
             font=('Segoe UI', 13, 'bold')).pack(side=tk.LEFT)
    tk.Frame(win, bg=C_ACCENT, height=2).pack(fill=tk.X)

    body = tk.Frame(win, bg=C_BASE)
    body.pack(fill=tk.BOTH, expand=True, padx=24, pady=18)

    tk.Label(body, text="Unsupported display", bg=C_BASE, fg=C_RED,
             font=('Segoe UI', 13, 'bold')).pack(anchor='w')
    tk.Label(body, text="Jelibox can't open on this screen.", bg=C_BASE, fg=C_TXT1,
             font=('Segoe UI', 10)).pack(anchor='w', pady=(2, 10))

    card = tk.Frame(body, bg=C_CARD, highlightbackground=C_BORDER, highlightthickness=1)
    card.pack(fill=tk.X)
    tk.Label(card, text=reason, bg=C_CARD, fg=C_TXT2, font=('Segoe UI', 9),
             wraplength=width - 80, justify=tk.LEFT).pack(anchor='w', padx=12, pady=(10, 4))
    tk.Label(card, text=f"Detected screen: {w}×{h}", bg=C_CARD, fg=C_TXT3,
             font=('Segoe UI', 8)).pack(anchor='w', padx=12, pady=(0, 10))

    tk.Label(body, text="Please open Jelibox on a desktop or laptop monitor.",
             bg=C_BASE, fg=C_TXT3, font=('Segoe UI', 9)).pack(anchor='w', pady=(10, 0))

    tk.Button(win, text="Close", command=win.destroy, bg=C_ACCENT, fg=C_ON_ACCENT,
              font=('Segoe UI', 9, 'bold'), relief=tk.FLAT, cursor='hand2',
              activebackground=C_ACCENT, borderwidth=0
              ).pack(side=tk.BOTTOM, pady=(0, 18), ipadx=22, ipady=4)

    win.bind('<Return>', lambda e: win.destroy())
    win.bind('<Escape>', lambda e: win.destroy())
    win.protocol("WM_DELETE_WINDOW", win.destroy)
    win.grab_set()
    win.focus_force()
    root.wait_window(win)
