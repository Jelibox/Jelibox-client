"""
JELIBOX DESIGN SYSTEM

Warm cream neutrals do the work, one indigo accent marks what matters, red is
reserved for danger. Class colors (the boxes on the image) live in
ClassManager, not here.

Two modes, chosen by the "theme" app setting ("light" is the default) and fixed
for the lifetime of the process - change it with set_mode() and it applies the
next time Jelibox starts. Every module does `from .theme import C_*`, so the
colors must be resolved at import time.

Legacy names (C_PURPLE, C_BLUE, C_ORANGE) are kept as aliases of the accent /
warning colors so older call sites still work; new code should use the roles
below.
"""
from . import app_settings

MODE = app_settings.get("theme", "light")
if MODE not in ("light", "dark"):
    MODE = "light"

_LIGHT = dict(
    BASE='#EAE5DA',      # app background (canvas)
    PANEL='#F4F0E6',     # header, side panels, bars (surface)
    CARD='#EDE8DC',      # wells: inputs, list backgrounds, cards inside a panel
    CARD2='#DED8CB',     # secondary buttons, hover wells (surface 2)
    BORDER='#CFC8B8',    # dividers / hairlines
    ACCENT='#4A5BF0',    # indigo - the one accent
    ACCENT_TINT='#DFDEF3',
    ON_ACCENT='#FFFFFF',
    GREEN='#1F7A59',     # success (text-safe on cream)
    AMBER='#9A6412',     # caution
    RED='#C9353A',       # danger fill / text
    DANGER_BG='#F0D9D2', # tinted destructive button
    DANGER_FG='#B3262B',
    TXT1='#1F1D1A',
    TXT2='#625D55',
    TXT3='#7A746A',
    STAGE='#CFC8B8',     # backdrop behind the image canvas
)

_DARK = dict(
    BASE='#0B0C12',
    PANEL='#14161D',
    CARD='#1B1D26',
    CARD2='#2B2E3A',
    BORDER='#2A2C36',
    ACCENT='#7C8CFF',
    ACCENT_TINT='#262A4D',
    ON_ACCENT='#10131F',
    GREEN='#5BD6A8',
    AMBER='#F5B84B',
    RED='#D93A3F',
    DANGER_BG='#3A2227',
    DANGER_FG='#FF8589',
    TXT1='#F2EFE9',
    TXT2='#A1A1AA',
    TXT3='#6B6B75',
    STAGE='#07080C',
)

_P = _DARK if MODE == "dark" else _LIGHT

C_BASE = _P['BASE']
C_PANEL = _P['PANEL']
C_CARD = _P['CARD']
C_CARD2 = _P['CARD2']
C_BORDER = _P['BORDER']
C_ACCENT = _P['ACCENT']
C_ACCENT_TINT = _P['ACCENT_TINT']
C_ON_ACCENT = _P['ON_ACCENT']
C_GREEN = _P['GREEN']
C_AMBER = _P['AMBER']
C_RED = _P['RED']
C_DANGER_BG = _P['DANGER_BG']
C_DANGER_FG = _P['DANGER_FG']
C_TXT1 = _P['TXT1']
C_TXT2 = _P['TXT2']
C_TXT3 = _P['TXT3']
C_STAGE = _P['STAGE']
C_ON_RED = '#FFFFFF'

# Legacy aliases (one accent, one warning color - no more per-button hues)
C_PURPLE = C_ACCENT
C_BLUE = C_ACCENT
C_ORANGE = C_AMBER


def toggle_mode():
    """Flip the saved theme; returns the new mode (applies on next start)."""
    new = "dark" if MODE == "light" else "light"
    set_mode(new)
    return new


def set_mode(mode):
    """Persist the theme; takes effect the next time Jelibox starts."""
    if mode in ("light", "dark"):
        app_settings.set("theme", mode)
