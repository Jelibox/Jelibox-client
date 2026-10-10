"""The Analyze Dataset window: cards, the warning strip, the resize simulation (needs matplotlib)."""
import gc
import tkinter as tk
import unittest

from utils import dataset_analysis as da

try:
    import matplotlib  # noqa: F401
    HAVE = True
except ImportError:
    HAVE = False


def analysis(objects):
    """objects: (class, width, height, image width, image height)."""
    d = da.Analysis("ws")
    d.instances = [{"name": "ws-1", "images": 3, "labelled": 2}]
    d.n_images, d.n_labelled = 3, 2
    for cls, w, h, iw, ih in objects:
        d.cls.append(cls); d.kind.append("bbox")
        d.w.append(w); d.h.append(h); d.iw.append(iw); d.ih.append(ih)
        d.image_sizes[(iw, ih)] += 1
    d.per_image = [2, 1]
    d.classes = sorted(set(d.cls))
    return d


def labels(widget):
    out = []
    for w in widget.winfo_children():
        if isinstance(w, tk.Label):
            out.append(str(w.cget("text")))
        out += labels(w)
    return out


@unittest.skipUnless(HAVE, "matplotlib not installed")
class DashboardTests(unittest.TestCase):
    def setUp(self):
        self.root = tk.Tk()
        self.addCleanup(gc.collect)             # Tk variables / images must not be collected on another thread
        self.addCleanup(self.root.destroy)

    def window(self, objects):
        from utils.DatasetAnalysisDialog import DatasetAnalysisWindow
        w = DatasetAnalysisWindow(self.root, "ws", analysis(objects))
        self.addCleanup(w.top.destroy)
        w.top.update()
        return w

    def test_no_small_objects_no_warning(self):
        w = self.window([("car", 200, 100, 640, 640), ("car", 90, 80, 640, 640)])
        self.assertFalse(w.alert.winfo_ismapped())

    def test_small_objects_show_a_warning_strip_not_a_dialog(self):
        w = self.window([("car", 200, 100, 640, 640), ("bolt", 20, 10, 640, 640)])
        self.assertTrue(w.alert.winfo_ismapped())
        text = w.alert_text.cget("text")
        self.assertIn("1 object(s)", text)
        self.assertIn("bolt", text)
        self.assertIn("32 px", text)

    def test_simulating_a_smaller_input_adds_the_warning_and_original_removes_it(self):
        w = self.window([("car", 60, 60, 1280, 1280)])                 # fine as it is
        self.assertFalse(w.alert.winfo_ismapped())
        w.size_var.set("320")
        w.simulate()
        w.top.update()
        self.assertTrue(w.alert.winfo_ismapped())                      # 60 px -> 15 px
        self.assertIn("at 320 px", w.alert_text.cget("text"))
        self.assertTrue(any("15 x 15 px" in t for t in labels(w.cards)))
        self.assertTrue(any("was 60 x 60 px" in t for t in labels(w.cards)))
        w.reset()
        w.top.update()
        self.assertFalse(w.alert.winfo_ismapped())

    def test_the_warning_strip_has_a_remove_button_that_asks_first(self):
        from unittest import mock
        w = self.window([("car", 200, 100, 640, 640), ("bolt", 20, 10, 640, 640), ("bolt", 25, 40, 640, 640)])
        self.assertTrue(w.remove_btn.winfo_ismapped())
        self.assertIn("Remove", w.remove_btn.cget("text"))
        with mock.patch("utils.DatasetAnalysisDialog.messagebox.askyesno", return_value=False) as ask, \
                mock.patch.object(da, "remove_small") as removal:
            w.remove_btn.invoke()
        removal.assert_not_called()
        title, message = ask.call_args[0]
        self.assertEqual(title, "Remove small objects")
        for must in ("MODIFY", "2 object(s)", "under 32 px", "permanently deleted", "cannot be undone",
                     "backed up", "images themselves are not touched"):
            self.assertIn(must, message)
        self.assertEqual(ask.call_args[1].get("default"), "no", "the safe answer is the default")

    def test_confirming_removes_the_objects_and_reloads_the_dashboard(self):
        from unittest import mock
        w = self.window([("car", 200, 100, 640, 640), ("bolt", 20, 10, 640, 640)])
        after = analysis([("car", 200, 100, 640, 640)])
        result = {"objects": 1, "files": 1, "emptied": 0}
        with mock.patch("utils.DatasetAnalysisDialog.messagebox.askyesno", return_value=True), \
                mock.patch("utils.DatasetAnalysisDialog.messagebox.showinfo") as info, \
                mock.patch.object(da, "remove_small", return_value=result) as removal, \
                mock.patch.object(da, "analyze_workspace", return_value=after):
            w.remove_btn.invoke()
        removal.assert_called_once()
        args = removal.call_args[0]
        self.assertEqual((args[3], args[4], args[5]), (None, da.LETTERBOX, 32))        # original sizes
        self.assertIn("Removed 1 object(s) from 1 annotation file(s)", info.call_args[0][1])
        w.top.update()
        self.assertEqual(w.data.n_objects, 1)
        self.assertFalse(w.alert.winfo_ismapped(), "nothing small is left")

    def test_in_a_simulation_the_message_and_the_removal_use_the_simulated_size(self):
        from unittest import mock
        w = self.window([("car", 60, 60, 1280, 1280)])
        w.size_var.set("320")
        w.simulate()
        w.top.update()
        with mock.patch("utils.DatasetAnalysisDialog.messagebox.askyesno", return_value=True) as ask, \
                mock.patch("utils.DatasetAnalysisDialog.messagebox.showinfo"), \
                mock.patch.object(da, "remove_small", return_value={"objects": 1, "files": 1, "emptied": 1}) as removal, \
                mock.patch.object(da, "analyze_workspace", return_value=analysis([])):
            w.remove_btn.invoke()
        self.assertIn("measured at 320 px, letterbox", ask.call_args[0][1])
        self.assertEqual(removal.call_args[0][3:5], (320, da.LETTERBOX))

    def test_a_failed_removal_is_reported_and_the_popup_is_closed(self):
        from unittest import mock
        w = self.window([("bolt", 20, 10, 640, 640)])
        with mock.patch("utils.DatasetAnalysisDialog.messagebox.askyesno", return_value=True), \
                mock.patch("utils.DatasetAnalysisDialog.messagebox.showerror") as error, \
                mock.patch.object(da, "remove_small", side_effect=OSError("disk full")):
            w.remove_btn.invoke()
        self.assertIn("disk full", error.call_args[0][1])
        self.assertEqual([c for c in w.top.winfo_children() if isinstance(c, tk.Toplevel)], [])

    def test_a_bad_input_size_is_refused(self):
        from unittest import mock
        w = self.window([("car", 60, 60, 640, 640)])
        for bad in ("abc", "5", "99999"):
            w.size_var.set(bad)
            with mock.patch("utils.DatasetAnalysisDialog.messagebox.showwarning") as warn:
                w.simulate()
            warn.assert_called_once()
            self.assertFalse(w.simulated)

    def test_every_tab_draws_and_an_empty_workspace_does_not_crash(self):
        w = self.window([("car", 200, 100, 640, 640)])
        self.assertTrue({"overview", "size", "images"} <= set(w.figures))
        empty = da.Analysis("ws")
        from utils.DatasetAnalysisDialog import DatasetAnalysisWindow
        e = DatasetAnalysisWindow(self.root, "ws", empty)
        self.addCleanup(e.top.destroy)
        e.top.update()
        e.size_var.set("640")
        e.simulate()


if __name__ == "__main__":
    unittest.main()
