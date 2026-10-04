import importlib
import json
import os
import tempfile
import unittest

from utils import app_settings


def luminance(hex_color):
    h = hex_color.lstrip('#')
    r, g, b = (int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))
    f = lambda c: c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b)


def contrast(a, b):
    la, lb = sorted((luminance(a), luminance(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


class AppSettingsTests(unittest.TestCase):
    def setUp(self):
        self._old = app_settings._PATH
        self.tmp = tempfile.mkdtemp()
        app_settings._PATH = os.path.join(self.tmp, "configs", "_app.json")

    def tearDown(self):
        app_settings._PATH = self._old

    def test_default_when_missing(self):
        self.assertEqual(app_settings.get("theme", "light"), "light")

    def test_roundtrip_and_keys_are_independent(self):
        app_settings.set("theme", "dark")
        app_settings.set("resume", {"folder": "x", "image": "y.jpg"})
        self.assertEqual(app_settings.get("theme"), "dark")
        self.assertEqual(app_settings.get("resume")["image"], "y.jpg")
        app_settings.set("resume", None)
        self.assertIsNone(app_settings.get("resume"))
        self.assertEqual(app_settings.get("theme"), "dark")

    def test_corrupt_file_falls_back_to_default(self):
        os.makedirs(os.path.dirname(app_settings._PATH), exist_ok=True)
        with open(app_settings._PATH, "w") as f:
            f.write("{oops")
        self.assertEqual(app_settings.get("theme", "light"), "light")


class ThemeTests(unittest.TestCase):
    """Reload utils.theme for each mode and check the palette rules."""

    def setUp(self):
        self._old = app_settings._PATH
        app_settings._PATH = os.path.join(tempfile.mkdtemp(), "_app.json")

    def tearDown(self):
        app_settings._PATH = self._old
        import utils.theme
        importlib.reload(utils.theme)

    def load(self, mode):
        if mode:
            app_settings.set("theme", mode)
        import utils.theme as t
        return importlib.reload(t)

    def test_light_is_default_and_invalid_falls_back(self):
        self.assertEqual(self.load(None).MODE, "light")
        self.assertEqual(self.load("neon").MODE, "light")

    def test_dark_mode_loads(self):
        t = self.load("dark")
        self.assertEqual(t.MODE, "dark")
        self.assertEqual(t.C_BASE, "#0B0C12")

    def test_toggle_mode_flips_and_persists(self):
        t = self.load("light")
        self.assertEqual(t.toggle_mode(), "dark")
        self.assertEqual(app_settings.get("theme"), "dark")
        self.assertEqual(self.load(None).MODE, "dark")

    def test_every_color_is_a_hex_in_both_modes(self):
        for mode in ("light", "dark"):
            t = self.load(mode)
            for name in dir(t):
                if name.startswith("C_"):
                    v = getattr(t, name)
                    self.assertRegex(v, r"^#[0-9A-Fa-f]{6}$", f"{mode}:{name}")

    def test_light_mode_has_no_pure_white_surfaces(self):
        t = self.load("light")
        for name in ("C_BASE", "C_PANEL", "C_CARD", "C_CARD2"):
            self.assertNotEqual(getattr(t, name).upper(), "#FFFFFF", name)

    def test_text_contrast_meets_wcag_aa(self):
        for mode in ("light", "dark"):
            t = self.load(mode)
            for fg, bg in (("C_TXT1", "C_BASE"), ("C_TXT1", "C_PANEL"), ("C_TXT2", "C_BASE"),
                           ("C_TXT2", "C_PANEL"), ("C_ON_ACCENT", "C_ACCENT"),
                           ("C_DANGER_FG", "C_DANGER_BG"), ("C_ON_RED", "C_RED")):
                ratio = contrast(getattr(t, fg), getattr(t, bg))
                self.assertGreaterEqual(ratio, 4.5, f"{mode}: {fg} on {bg} = {ratio:.2f}")

    def test_legacy_aliases_collapse_to_the_single_accent(self):
        t = self.load("light")
        self.assertEqual(t.C_PURPLE, t.C_ACCENT)
        self.assertEqual(t.C_BLUE, t.C_ACCENT)
        self.assertEqual(t.C_ORANGE, t.C_AMBER)


if __name__ == "__main__":
    unittest.main()
