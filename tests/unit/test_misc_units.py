import os
import tempfile
import unittest

from utils import app_settings


class FakeRoot:
    def __init__(self, w, h):
        self._w, self._h = w, h

    def winfo_screenwidth(self):
        return self._w

    def winfo_screenheight(self):
        return self._h


class ScreenGuardTests(unittest.TestCase):
    def check(self, w, h):
        from utils import ScreenGuard
        return ScreenGuard.check_screen(FakeRoot(w, h))

    def test_common_desktop_resolutions_pass(self):
        for w, h in ((1920, 1080), (2560, 1440), (3840, 2160), (1920, 1200), (1366, 768),
                     (1280, 720), (1024, 768), (5120, 1440), (3440, 1440)):
            ok, reason, *_ = self.check(w, h)
            self.assertTrue(ok, f"{w}x{h}: {reason}")

    def test_phone_and_portrait_screens_are_rejected_with_english_reason(self):
        for w, h in ((390, 844), (1080, 1920), (768, 1024), (800, 600)):
            ok, reason, *_ = self.check(w, h)
            self.assertFalse(ok, f"{w}x{h}")
            self.assertIn("screen", reason.lower())
            self.assertTrue(reason.encode('ascii', 'ignore').decode().split()[0].isalpha())

    def test_reports_detected_size(self):
        self.assertEqual(self.check(390, 844)[2:], (390, 844))


class WindowStyleTests(unittest.TestCase):
    def test_colorref_is_bgr(self):
        from utils.window_style import _colorref
        self.assertEqual(_colorref("#FF0000"), 0x0000FF)
        self.assertEqual(_colorref("#00FF00"), 0x00FF00)
        self.assertEqual(_colorref("#0000FF"), 0xFF0000)
        self.assertEqual(_colorref("#EAE5DA"), 0xDAE5EA)


class WorkspaceManagerTests(unittest.TestCase):
    def test_sanitize_workspace_name(self):
        from utils.workspace_manager import sanitize_workspace_name as s
        self.assertEqual(s("  weapon  "), "weapon")
        self.assertEqual(s("my_ws-v2"), "my_ws-v2")
        for bad in ("", "   ", "a b", "a/b", "name-3", "x.y"):
            with self.assertRaises(ValueError, msg=bad):
                s(bad)

    def test_parse_classes_input(self):
        from utils.workspace_manager import parse_classes_input as p
        self.assertEqual(p("{cat, dog, rabbit}"), ["cat", "dog", "rabbit"])
        for bad in ("", "cat, dog", "{}", "{cat, cat}", "{cat,,dog}"):
            with self.assertRaises(ValueError, msg=bad):
                p(bad)

    def test_workspace_name_for(self):
        from utils.workspace_manager import workspace_name_for as w
        self.assertEqual(w("weapon-1"), "weapon")
        self.assertEqual(w("ppeKujangv3-24"), "ppeKujangv3")
        self.assertEqual(w("my-project"), "my-project")


class ResumeAfterRestartTests(unittest.TestCase):
    """Theme restarts reopen on the image the user was on - once."""

    def setUp(self):
        self._old = app_settings._PATH
        app_settings._PATH = os.path.join(tempfile.mkdtemp(), "_app.json")
        self.ds = tempfile.mkdtemp()
        for n in ("b.png", "a.jpg", "c.jpeg", "notes.txt"):
            open(os.path.join(self.ds, n), "w").close()

    def tearDown(self):
        app_settings._PATH = self._old

    def resume(self):
        from utils import Annotator
        return Annotator._resume_image_index(self.ds)

    def test_no_marker_means_start_at_zero(self):
        self.assertIsNone(self.resume())

    def test_marker_for_this_folder_is_used_once(self):
        app_settings.set("resume", {"folder": self.ds, "image": "b.png"})
        self.assertEqual(self.resume(), 1)          # sorted: a.jpg, b.png, c.jpeg
        self.assertIsNone(self.resume())            # consumed

    def test_marker_for_another_folder_is_ignored_and_cleared(self):
        app_settings.set("resume", {"folder": os.path.join(self.ds, "other"), "image": "b.png"})
        self.assertIsNone(self.resume())
        self.assertIsNone(app_settings.get("resume"))

    def test_missing_image_is_ignored(self):
        app_settings.set("resume", {"folder": self.ds, "image": "gone.png"})
        self.assertIsNone(self.resume())

    def test_restart_exit_code_contract_with_the_picker(self):
        from utils import Annotator
        self.assertEqual(Annotator.RESTART_EXIT_CODE, 75)
        with open(os.path.join(os.path.dirname(__file__), "..", "..", "utils", "WorkspacePicker.py"), encoding="utf-8") as f:
            self.assertIn("returncode == 75", f.read())


if __name__ == "__main__":
    unittest.main()
