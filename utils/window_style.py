"""
Native window polish so Jelibox looks like its own app instead of "a Tk window":

  - the Jelibox icon replaces Tk's default feather (window + taskbar)
  - Windows: the taskbar groups under "Jelibox" instead of python.exe
  - Windows 10/11: the title bar takes the theme's colors (DWM caption/text/border
    color + dark-mode flag) so it blends into the header. Windows 11 recolors the
    whole bar; Windows 10 only honors the dark-mode flag.

The title bar is deliberately NOT removed (overrideredirect): that costs the
taskbar entry, minimize/maximize, window snapping and resizing, and behaves very
differently across Linux window managers and Wayland. Linux keeps its native
title bar, which follows the user's desktop theme.
"""
import ctypes
import os
import sys

from . import theme

_APP_ID = "Jelibox.App"
_ICON_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "assets"))

_DWMWA_USE_IMMERSIVE_DARK_MODE = 20
_DWMWA_BORDER_COLOR = 34
_DWMWA_CAPTION_COLOR = 35
_DWMWA_TEXT_COLOR = 36


def _colorref(hex_color):
    h = hex_color.lstrip('#')
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return (b << 16) | (g << 8) | r            # COLORREF is 0x00BBGGRR


def set_app_id():
    """Call once before the first Tk window is created (Windows only)."""
    if sys.platform == "win32":
        try:
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(_APP_ID)
        except Exception:
            pass


def _style_titlebar(widget):
    if sys.platform != "win32":
        return
    try:
        widget.update_idletasks()
        hwnd = ctypes.windll.user32.GetParent(widget.winfo_id())
        if not hwnd:
            return
        dwm = ctypes.windll.dwmapi.DwmSetWindowAttribute

        def put(attr, value):
            v = ctypes.c_int(value)
            dwm(ctypes.c_void_p(hwnd), attr, ctypes.byref(v), ctypes.sizeof(v))

        put(_DWMWA_USE_IMMERSIVE_DARK_MODE, 1 if theme.MODE == "dark" else 0)
        put(_DWMWA_CAPTION_COLOR, _colorref(theme.C_PANEL))
        put(_DWMWA_TEXT_COLOR, _colorref(theme.C_TXT1))
        put(_DWMWA_BORDER_COLOR, _colorref(theme.C_BORDER))
    except Exception:
        pass            # older Windows / missing dwmapi: keep the default title bar


def setup(root):
    """Icon + title bar styling for `root` and every window opened from it."""
    ico = os.path.join(_ICON_DIR, "jelibox.ico")
    if sys.platform == "win32" and os.path.exists(ico):
        # The .ico carries hand-tuned 16-40 px frames (bolder mark) that stay
        # recognisable in the title bar / taskbar; a downscaled photo would not.
        try:
            root.iconbitmap(ico)
            root.iconbitmap(default=ico)
        except Exception:
            pass
    else:
        try:
            from PIL import Image, ImageTk
            root._jelibox_icon = ImageTk.PhotoImage(
                Image.open(os.path.join(_ICON_DIR, "jelibox.png")).resize((64, 64)))
            root.iconphoto(True, root._jelibox_icon)    # True: default for new Toplevels too
        except Exception:
            pass

    handler = lambda e: _style_titlebar(e.widget) if e.widget.winfo_toplevel() is e.widget else None
    root.bind_class("Toplevel", "<Map>", handler, add="+")
    root.bind_class("Tk", "<Map>", handler, add="+")
    _style_titlebar(root)
