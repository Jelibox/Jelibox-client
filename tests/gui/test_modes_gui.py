"""The front mode menu, the two new settings windows and the Auto-annotate all window, built for real against the
isolated temp workspace. Skipped automatically when there is no display."""
import os
import re
import time
import tkinter as tk
import unittest
from unittest import mock

from tests.helpers import isolated_workspace, WORKSPACE

ctx = isolated_workspace()

from utils import app_settings, config                           # noqa: E402
from utils import assistant_modes as am                          # noqa: E402
from utils import workspace_config as wc                         # noqa: E402

try:
    _probe = tk.Tk()
    _probe.destroy()
    HAVE_DISPLAY = True
except tk.TclError:
    HAVE_DISPLAY = False

if HAVE_DISPLAY:
    from tkinter import messagebox                               # noqa: E402

IMAGES = ["img1.png", "img2.png", "img3.png"]


def pump(root, seconds=0.3):
    end = time.time() + seconds
    while time.time() < end:
        root.update()
        time.sleep(0.01)


def pump_until(root, condition, timeout=10):
    end = time.time() + timeout
    while time.time() < end:
        root.update()
        if condition():
            return True
        time.sleep(0.02)
    return False


def walk(widget):
    yield widget
    for child in widget.winfo_children():
        yield from walk(child)


def buttons(root, text=None):
    out = [w for w in walk(root) if isinstance(w, tk.Button)]
    # a narrow annotation header shows icons only; _jb['full'] keeps the real label
    label = lambda b: getattr(b, "_jb", {}).get("full") or b.cget("text")
    return [b for b in out if text is None or re.search(re.escape(text), label(b))]


def label_texts(root):
    return " ".join(str(w.cget("text")) for w in walk(root) if isinstance(w, tk.Label))


class Recorder:
    def __init__(self):
        self.calls = []
        names = {"showinfo": None, "showwarning": None, "showerror": None, "askyesno": True}
        self.patches = [mock.patch.object(messagebox, n, side_effect=self._rec(n, rv)) for n, rv in names.items()]

    def _rec(self, name, rv):
        def f(title=None, message=None, **kw):
            self.calls.append((name, title, message))
            return rv
        return f

    def __enter__(self):
        for p in self.patches:
            p.start()
        return self

    def __exit__(self, *a):
        for p in self.patches:
            p.stop()

    def of(self, kind):
        return [c for c in self.calls if c[0] == kind]


class ModeGuiBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rec = Recorder().__enter__()
        cls.root = tk.Tk()
        cls.root.geometry("1100x700+0+0")

    @classmethod
    def tearDownClass(cls):
        cls.rec.__exit__()
        try:
            pump(cls.root, 0.2)
            cls.root.destroy()
        except tk.TclError:
            pass

    def setUp(self):
        self.rec.calls.clear()
        self._mode = app_settings.get("mode")
        a = wc.get_assistant(WORKSPACE)
        a["provider"] = wc.PROVIDER_YOLO_WORLD
        a["locate_anything"] = dict(wc.LOCATE_DEFAULTS)
        a["custom_head"] = dict(wc.HEAD_DEFAULTS)
        a["custom_model"] = {"path": "", "class_map": {}}
        wc.set_assistant(WORKSPACE, a)

    def tearDown(self):
        app_settings.set("mode", self._mode)
        for child in self.root.winfo_children():
            child.destroy()


@unittest.skipUnless(HAVE_DISPLAY, "no display available")
class FrontMenuTests(ModeGuiBase):
    def picker(self, skip_menu=False):
        from utils.WorkspacePicker import WorkspacePickerApp
        env = {"JELIBOX_SKIP_MODE_MENU": "1"} if skip_menu else {}
        with mock.patch.dict(os.environ, env, clear=False):
            if not skip_menu:
                os.environ.pop("JELIBOX_SKIP_MODE_MENU", None)
            app = WorkspacePickerApp(self.root, entry_script=os.path.join(os.getcwd(), "x.py"))
        pump(self.root, 0.3)
        return app

    def test_the_front_menu_offers_the_four_modes(self):
        app = self.picker()
        self.assertIsNotNone(app.mode_menu)
        text = label_texts(app.mode_menu)
        for title in ("YOLO-World", "LocateAnything", "Custom Model", "SAM 2 Dynamic"):
            self.assertIn(title, text)
        self.assertEqual(len(buttons(app.mode_menu)), 4)
        self.assertIn("1 / 2 / 3 / 4", text)

    def test_the_four_cards_fit_a_narrow_window(self):
        app = self.picker()
        self.root.update_idletasks()
        cards = [b.master.master for b in buttons(app.mode_menu)]
        right = max(c.winfo_rootx() - self.root.winfo_rootx() + c.winfo_width() for c in cards)
        self.assertLessEqual(right, self.root.winfo_width())

    def test_pressing_4_chooses_sam2_dynamic(self):
        app = self.picker()
        self.root.event_generate("<Key-4>")
        pump(self.root, 0.2)
        self.assertEqual(am.get_mode(), am.MODE_SAM2)
        self.assertIsNone(app.mode_menu)

    def test_choosing_a_mode_saves_it_and_shows_the_workspaces(self):
        app = self.picker()
        buttons(app.mode_menu)[1].invoke()                  # the second card: LocateAnything
        self.assertEqual(am.get_mode(), am.MODE_LOCATE)
        self.assertIsNone(app.mode_menu)
        self.assertIn("LocateAnything", app.mode_btn.cget("text"))

    def test_the_last_used_mode_is_marked(self):
        am.set_mode(am.MODE_HEAD)
        app = self.picker()
        marked = [b for b in buttons(app.mode_menu) if b.cget("text") == "Last used"]
        self.assertEqual(len(marked), 1)

    def test_number_keys_choose_a_mode(self):
        app = self.picker()
        self.assertTrue({"1", "2", "3"} <= set(self.root.bind()))
        self.root.focus_force()
        pump(self.root, 0.2)
        self.root.event_generate("<KeyPress-3>")
        pump(self.root, 0.2)
        self.assertEqual(am.get_mode(), am.MODE_HEAD)
        self.assertIsNone(app.mode_menu)

    def test_the_mode_button_reopens_the_menu(self):
        app = self.picker(skip_menu=True)
        self.assertIsNone(app.mode_menu, "a theme restart must not show the menu again")
        app.mode_btn.invoke()
        self.assertIsNotNone(app.mode_menu)


@unittest.skipUnless(HAVE_DISPLAY, "no display available")
class LocateAnythingDialogTests(ModeGuiBase):
    def open(self):
        from utils.ModeDialogs import LocateAnythingDialog
        dlg = LocateAnythingDialog(self.root)
        pump(self.root, 0.2)
        return dlg

    def test_saving_stores_categories_and_options(self):
        dlg = self.open()
        dlg.rows[0]["prompt"].set("tabby cat")
        dlg.rows[0]["cls"].set("cat")
        dlg.mode_var.set("slow")
        dlg.device_var.set("cpu")
        dlg.side_var.set(768)
        dlg.save()
        a = wc.get_assistant(WORKSPACE)
        self.assertEqual(a["provider"], wc.PROVIDER_LOCATE)
        la = a["locate_anything"]
        self.assertEqual(la["target_classes"], [{"prompt": "tabby cat", "map_to": "cat"}])
        self.assertEqual((la["generation_mode"], la["device"], la["short_side"]), ("slow", "cpu", 768))

    def test_a_category_without_a_class_is_refused(self):
        dlg = self.open()
        dlg.rows[0]["prompt"].set("tabby cat")
        dlg.save()
        self.assertTrue(self.rec.of("showwarning"))
        self.assertEqual(wc.get_assistant(WORKSPACE)["locate_anything"]["target_classes"], [])
        self.assertTrue(dlg.win.winfo_exists(), "the window stays open so the user can fix it")

    def test_rows_can_be_added_and_removed_and_saved_settings_come_back(self):
        a = wc.get_assistant(WORKSPACE)
        a["locate_anything"]["target_classes"] = [{"prompt": "cat", "map_to": "cat"}, {"prompt": "dog", "map_to": "dog"}]
        wc.set_assistant(WORKSPACE, a)
        dlg = self.open()
        self.assertEqual([r["prompt"].get() for r in dlg.rows], ["cat", "dog"])
        dlg._add_row("bird")
        dlg._remove_row(dlg.rows[0])
        self.assertEqual([r["prompt"].get() for r in dlg.rows], ["dog", "bird"])

    def test_it_tells_the_user_what_the_first_use_costs(self):
        dlg = self.open()
        text = label_texts(dlg.win)
        self.assertIn("Packages:", text)
        self.assertIn("Model:", text)
        self.assertIn("Device:", text)


@unittest.skipUnless(HAVE_DISPLAY, "no display available")
class CustomHeadDialogTests(ModeGuiBase):
    def make_head(self):
        try:
            import torch
        except Exception:
            self.skipTest("torch is not installed")
        path = os.path.join(ctx.model_dir, "heads", "head_best.pt")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({"head": {}, "names": ["cat", "walking", "other"],
                    "cfg": {"weights": "yolov9c.pt", "taps": [15, 18, 21], "imgsz": [640, 640], "nc": 3}}, path)
        return path

    def open(self):
        from utils.ModeDialogs import CustomHeadDialog
        dlg = CustomHeadDialog(self.root)
        pump(self.root, 0.2)
        return dlg

    def test_choosing_a_checkpoint_lists_its_classes_and_maps_matching_names(self):
        head = self.make_head()
        dlg = self.open()
        dlg.path_var.set(head)
        dlg._load_head(head)
        self.assertEqual(list(dlg.map_vars), ["cat", "walking", "other"])
        self.assertEqual(dlg.map_vars["cat"].get(), "cat", "same name -> mapped by itself")
        self.assertEqual(dlg.map_vars["walking"].get(), "(skip)")

    def test_saving_stores_head_detector_classes_and_mapping(self):
        head = self.make_head()
        dlg = self.open()
        dlg.path_var.set(head)
        dlg._load_head(head)
        dlg.classes_var.set("person, bottle")
        dlg.map_vars["walking"].set("dog")
        dlg.save()
        a = wc.get_assistant(WORKSPACE)
        self.assertEqual(a["provider"], wc.PROVIDER_HEAD)
        h = a["custom_head"]
        self.assertEqual(h["detector_classes"], [0, 39])
        self.assertEqual(h["class_map"], {"cat": "cat", "walking": "dog"})
        self.assertEqual(os.path.normcase(h["head_path"]), os.path.normcase(head))

    def test_missing_checkpoint_and_bad_classes_are_refused(self):
        dlg = self.open()
        dlg.save()
        self.assertIn("head_best.pt", self.rec.of("showwarning")[-1][2])
        head = self.make_head()
        dlg.path_var.set(head)
        dlg._load_head(head)
        dlg.classes_var.set("unicorn")
        dlg.save()
        self.assertIn("unicorn", self.rec.of("showwarning")[-1][2])
        self.assertEqual(wc.get_assistant(WORKSPACE)["custom_head"]["head_path"], "")

    def test_a_file_that_is_not_a_head_is_reported(self):
        try:
            import torch
        except Exception:
            self.skipTest("torch is not installed")
        junk = os.path.join(ctx.model_dir, "junk.pt")
        os.makedirs(ctx.model_dir, exist_ok=True)
        torch.save({"hello": 1}, junk)
        dlg = self.open()
        dlg.path_var.set(junk)
        dlg._load_head(junk)
        self.assertEqual(dlg.head_names, [])
        self.assertTrue(self.rec.of("showwarning"))


@unittest.skipUnless(HAVE_DISPLAY, "no display available")
class DialogScrollTests(ModeGuiBase):
    """A tall dialog (14 head classes) on a short screen (125% scale on 1080p leaves ~860 px) scrolls instead of
    pushing Save / Cancel off the screen."""
    SCREEN_H = 600

    def make_head(self, n):
        try:
            import torch
        except Exception:
            self.skipTest("torch is not installed")
        path = os.path.join(ctx.model_dir, "heads", "many.pt")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({"head": {}, "names": [f"behavior_{i}" for i in range(n)],
                    "cfg": {"weights": "yolov9c.pt", "taps": [15, 18, 21], "imgsz": [640, 640], "nc": n}}, path)
        return path

    def open_head(self, head):
        from utils.ModeDialogs import CustomHeadDialog
        a = wc.get_assistant(WORKSPACE)
        a["custom_head"]["head_path"] = head
        wc.set_assistant(WORKSPACE, a)
        with mock.patch.object(tk.Misc, "winfo_screenheight", lambda w: DialogScrollTests.SCREEN_H):
            dlg = CustomHeadDialog(self.root)
        pump(self.root, 0.3)
        return dlg

    def test_a_tall_dialog_fits_the_screen_and_keeps_its_buttons(self):
        dlg = self.open_head(self.make_head(14))
        self.assertLessEqual(dlg.win.winfo_height(), self.SCREEN_H, "never taller than the screen")
        self.assertTrue(dlg.scroll.bar.winfo_ismapped(), "scrollbar shown when the body is taller")
        footer_bottom = max(w.winfo_rooty() + w.winfo_height() for w in dlg.win.winfo_children()
                            if isinstance(w, tk.Frame) and w is not dlg.scroll.frame)
        self.assertLessEqual(footer_bottom, dlg.win.winfo_rooty() + dlg.win.winfo_height(), "Save / Cancel stay visible")
        dlg.win.destroy()

    def test_scrolling_reaches_the_last_class_and_the_wheel_works(self):
        dlg = self.open_head(self.make_head(14))
        self.assertEqual(dlg.scroll.canvas.yview()[0], 0.0)
        dlg.scroll._wheel(mock.Mock(widget=dlg.body, delta=-120))
        self.assertGreater(dlg.scroll.canvas.yview()[0], 0.0)
        dlg.scroll.canvas.yview_moveto(1.0)
        pump(self.root, 0.2)
        self.assertAlmostEqual(dlg.scroll.canvas.yview()[1], 1.0, places=2)
        dlg.win.destroy()

    def test_a_short_dialog_has_no_scrollbar(self):
        head = self.make_head(2)
        from utils.ModeDialogs import CustomHeadDialog
        a = wc.get_assistant(WORKSPACE)
        a["custom_head"]["head_path"] = head
        wc.set_assistant(WORKSPACE, a)
        dlg = CustomHeadDialog(self.root)                       # the test display is tall enough
        pump(self.root, 0.3)
        self.assertFalse(dlg.scroll.bar.winfo_ismapped())
        dlg.win.destroy()


def buttons_fit(test, dlg):
    """Finish early / Stop now / Close are all visible and inside the window (they were cut off once)."""
    pump(dlg.win, 0.3)
    bottom = dlg.win.winfo_rooty() + dlg.win.winfo_height()
    for button in (dlg.finish_btn, dlg.stop_btn, dlg.close_btn):
        test.assertTrue(button.winfo_viewable(), button.cget("text"))
        test.assertLessEqual(button.winfo_rooty() + button.winfo_height(), bottom, button.cget("text"))


FAKE_TRAINER = r'''
import json, os, sys, time
job = json.load(open(sys.argv[1], encoding="utf-8"))
def ev(event, **f):
    print("@@ " + json.dumps({"event": event, **f}), flush=True)
mode = os.environ.get("FAKE_TRAIN_MODE", "ok")
ev("status", text="Loading the detector ...")
if mode == "crash":
    raise SystemExit(3)
if mode == "error":
    ev("error", message="Layers [1, 2, 3] are not at strides 8 / 16 / 32.")
    raise SystemExit(1)
ev("start", train_images=9, val_images=3, classes=job["classes"], counts=[5, 4], epochs=job["epochs"],
   device="cpu", batch=4, imgsz=job["imgsz"], head_params_m=5.26)
for i in range(1, job["epochs"] + 1):
    if mode == "slow" and i == 2:
        time.sleep(60)
    stop_now = False
    if mode == "graceful":                      # an epoch that only the stop file ends early
        for _ in range(150):
            if os.path.exists(job["stop_file"]):
                stop_now = True
                break
            time.sleep(0.1)
    print("some plain log line", flush=True)
    with open(os.path.join(job["out_dir"], "head_best.pt"), "wb") as f:
        f.write(b"head-%d" % i)
    ev("epoch", epoch=i, epochs=job["epochs"], loss=1.0 / i, acc=0.5, f1=0.4 + i / 10, best=True,
       recall=[0.5, 0.6], support=[5, 4], seconds=0.01)
    if stop_now:
        ev("done", best_f1=0.5, best_epoch=i, epochs_run=i, stopped=True, out_dir=job["out_dir"], has_head=True,
           classes=job["classes"], last_recall=[0.5, 0.6], last_support=[5, 4])
        raise SystemExit(0)
ev("done", best_f1=0.9, best_epoch=job["epochs"], epochs_run=job["epochs"], stopped=False, out_dir=job["out_dir"],
   has_head=True, classes=job["classes"], last_recall=[0.5, 0.6], last_support=[5, 4])
'''


@unittest.skipUnless(HAVE_DISPLAY, "no display available")
class HeadTrainDialogTests(ModeGuiBase):
    """The Train button of the Custom Model mode: setup window -> progress window -> head installed."""

    def setUp(self):
        super().setUp()
        import sys
        from tests.helpers import make_images
        self.data = os.path.join(ctx.root, "headdata")
        self.images = os.path.join(self.data, "images")
        self.labels = os.path.join(self.data, "labels")
        os.makedirs(self.labels, exist_ok=True)
        for i, name in enumerate(make_images(self.images, 12)):
            with open(os.path.join(self.labels, os.path.splitext(name)[0] + ".txt"), "w") as f:
                f.write(f"{i % 2} 0.5 0.5 0.4 0.4\n")
        self.script = os.path.join(ctx.root, "fake_trainer.py")
        with open(self.script, "w") as f:
            f.write(FAKE_TRAINER)
        self.command = lambda job_path: [sys.executable, "-u", self.script, job_path]
        for old in ("head_best.pt", "head_best.prev.pt"):
            try:
                os.remove(os.path.join(ctx.model_dir, old))
            except OSError:
                pass

    @staticmethod
    def read_bytes(path):
        with open(path, "rb") as f:
            return f.read()

    def open(self, **kw):
        from utils.HeadTrainDialog import HeadTrainDialog
        dlg = HeadTrainDialog(self.root, [self.images], self.labels, ctx.model_dir, command=self.command, **kw)
        pump(self.root, 0.2)
        return dlg

    def run_to_end(self, dlg, timeout=30):
        dlg.save()
        self.assertTrue(pump_until(self.root, lambda: dlg.finished, timeout), "the run never ended")

    def test_the_defaults_are_the_recommended_ones(self):
        dlg = self.open()
        self.assertEqual((dlg.h_var.get(), dlg.w_var.get()), (384, 640))
        self.assertEqual((dlg.epochs_var.get(), dlg.batch_var.get()), (50, 16))
        self.assertEqual(dlg.weights_var.get(), "yolov8m.pt")
        self.assertEqual(dlg.taps_var.get(), "15, 18, 21")
        self.assertTrue(dlg.cw_var.get())
        self.assertEqual(dlg.lr_var.get(), "0.001")
        self.assertIn("12 labelled images: 10 to train, 2 to validate", dlg.data_info.cget("text"))
        dlg.win.destroy()

    def test_taps_follow_the_detector_and_unknown_models_are_flagged(self):
        dlg = self.open()
        dlg.weights_var.set("yolo11m.pt")
        self.assertEqual(dlg.taps_var.get(), "16, 19, 22")
        self.assertEqual(dlg.taps_hint.cget("text"), "")
        dlg.weights_var.set("my_model.pt")
        self.assertIn("not recognised", dlg.taps_hint.cget("text"))
        dlg.win.destroy()

    def test_the_validation_share_updates_the_split(self):
        dlg = self.open()
        dlg.val_var.set(50)
        self.assertIn("6 to train, 6 to validate", dlg.data_info.cget("text"))
        dlg.win.destroy()

    def test_bad_settings_are_explained_and_nothing_starts(self):
        dlg = self.open()
        dlg.h_var.set(100)
        dlg.save()
        self.assertIn("multiples of 32", self.rec.of("showwarning")[-1][2])
        dlg.h_var.set(384)
        dlg.taps_var.set("1, 2")
        dlg.save()
        self.assertIn("three layer numbers", self.rec.of("showwarning")[-1][2])
        dlg.taps_var.set("15, 18, 21")
        dlg.lr_var.set("5")
        dlg.save()
        self.assertIn("Learning rate", self.rec.of("showwarning")[-1][2])
        dlg.lr_var.set("0.001")
        dlg.weights_var.set(os.path.join(ctx.root, "nope", "model.pt"))
        dlg.save()
        self.assertIn("not found", self.rec.of("showwarning")[-1][2])
        self.assertIsNone(dlg.proc)
        dlg.win.destroy()

    def test_too_few_labelled_images_blocks_the_start(self):
        for name in os.listdir(self.labels)[:5]:
            os.remove(os.path.join(self.labels, name))
        dlg = self.open()
        self.assertIn("At least 10", dlg.classes_note.cget("text"))
        dlg.save()
        self.assertIn("Only 7", self.rec.of("showwarning")[-1][2])
        self.assertIsNone(dlg.proc)
        dlg.win.destroy()

    def test_a_finished_run_installs_the_head_and_selects_it_in_label_assistant(self):
        dlg = self.open()
        dlg.epochs_var.set(3)
        self.run_to_end(dlg)
        target = os.path.join(ctx.model_dir, "head_best.pt")
        self.assertTrue(os.path.isfile(target))
        self.assertEqual(self.read_bytes(target), b"head-3")
        h = wc.get_assistant(WORKSPACE)["custom_head"]
        self.assertTrue(h["head_path"].endswith("head_best.pt") and not os.path.isabs(h["head_path"]), h["head_path"])
        self.assertEqual(h["detector_weights"], "", "a stock name is not stored as a path")
        self.assertEqual(h["class_map"], {})
        self.assertEqual(dlg.title_label.cget("text"), "Training finished")
        self.assertEqual(float(dlg.bar.cget("value")), 3.0)
        self.assertIn("Epoch 3 / 3", dlg.stats_label.cget("text"))
        self.assertEqual(dlg.close_btn.cget("state"), "normal")
        self.assertEqual(dlg.stop_btn.cget("state"), "disabled")
        self.assertIn("some plain log line", dlg.log.get("1.0", "end"))
        with open(os.path.join(ctx.model_dir, "head_run", "job.json"), encoding="utf-8") as f:
            job = __import__("json").load(f)
        self.assertEqual((job["imgsz"], job["epochs"], job["batch"], job["weights"]), ([384, 640], 3, 16, "yolov8m.pt"))
        self.assertEqual(job["taps"], [15, 18, 21])
        self.assertEqual(job["classes"], config.class_manager.get_classes())
        dlg.win.destroy()

    def test_a_browsed_detector_file_is_remembered_for_annotating(self):
        model = os.path.join(ctx.root, "elsewhere", "my_detector.pt")
        os.makedirs(os.path.dirname(model), exist_ok=True)
        open(model, "wb").close()
        dlg = self.open()
        dlg.weights_var.set(model)
        dlg.epochs_var.set(1)
        self.run_to_end(dlg)
        self.assertEqual(wc.get_assistant(WORKSPACE)["custom_head"]["detector_weights"], model)
        dlg.win.destroy()

    def test_the_previous_head_is_kept_when_a_new_one_is_installed(self):
        os.makedirs(ctx.model_dir, exist_ok=True)
        with open(os.path.join(ctx.model_dir, "head_best.pt"), "wb") as f:
            f.write(b"old head")
        dlg = self.open()
        dlg.epochs_var.set(1)
        self.run_to_end(dlg)
        self.assertEqual(self.read_bytes(os.path.join(ctx.model_dir, "head_best.prev.pt")), b"old head")
        self.assertEqual(self.read_bytes(os.path.join(ctx.model_dir, "head_best.pt")), b"head-1")
        self.assertTrue(self.rec.of("askyesno"), "asks before replacing a head")
        dlg.win.destroy()

    def test_a_trainer_error_is_shown_and_nothing_is_installed(self):
        dlg = self.open()
        with mock.patch.dict(os.environ, {"FAKE_TRAIN_MODE": "error"}):
            self.run_to_end(dlg)
        self.assertEqual(dlg.title_label.cget("text"), "Training failed")
        self.assertIn("strides 8 / 16 / 32", dlg.status_label.cget("text"))
        self.assertFalse(os.path.exists(os.path.join(ctx.model_dir, "head_best.pt")))
        dlg.win.destroy()

    def test_a_crash_without_a_message_still_ends_cleanly(self):
        dlg = self.open()
        with mock.patch.dict(os.environ, {"FAKE_TRAIN_MODE": "crash"}):
            self.run_to_end(dlg)
        self.assertEqual(dlg.title_label.cget("text"), "Training ended unexpectedly")
        self.assertFalse(os.path.exists(os.path.join(ctx.model_dir, "head_best.pt")))
        self.assertEqual(dlg.close_btn.cget("state"), "normal")
        dlg.win.destroy()

    def test_stop_keeps_the_best_head_so_far(self):
        dlg = self.open()
        dlg.epochs_var.set(5)
        with mock.patch.dict(os.environ, {"FAKE_TRAIN_MODE": "slow"}):
            dlg.save()
            self.assertTrue(pump_until(self.root, lambda: float(dlg.bar.cget("value")) >= 1, 20), "epoch 1 never came")
            dlg._stop()
            self.assertTrue(pump_until(self.root, lambda: dlg.finished, 20))
        self.assertEqual(self.read_bytes(os.path.join(ctx.model_dir, "head_best.pt")), b"head-1")
        self.assertIn("kept", dlg.title_label.cget("text"))
        dlg.win.destroy()

    def test_the_progress_window_always_shows_its_buttons_whatever_the_text_says(self):
        dlg = self.open()
        dlg.epochs_var.set(3)
        dlg.save()
        buttons_fit(self, dlg)
        self.assertTrue(pump_until(self.root, lambda: dlg.finished, 30))
        buttons_fit(self, dlg)                                    # the long ending text is in
        dlg.win.destroy()

    def test_the_window_has_finish_early_and_stop_now_buttons(self):
        dlg = self.open()
        dlg.epochs_var.set(5)
        with mock.patch.dict(os.environ, {"FAKE_TRAIN_MODE": "slow"}):
            dlg.save()
            self.assertEqual((dlg.finish_btn.cget("text"), dlg.stop_btn.cget("text")), ("Finish early", "Stop now"))
            self.assertEqual((str(dlg.finish_btn.cget("state")), str(dlg.stop_btn.cget("state"))), ("normal", "normal"))
            self.assertTrue(pump_until(self.root, lambda: float(dlg.bar.cget("value")) >= 1, 20))
            dlg._stop()
            self.assertTrue(pump_until(self.root, lambda: dlg.finished, 20))
        self.assertEqual((str(dlg.finish_btn.cget("state")), str(dlg.stop_btn.cget("state"))), ("disabled", "disabled"))
        dlg.win.destroy()

    def test_finish_early_wraps_the_head_run_up_and_installs_the_head(self):
        dlg = self.open()
        dlg.epochs_var.set(5)
        with mock.patch.dict(os.environ, {"FAKE_TRAIN_MODE": "graceful"}):
            dlg.save()
            self.addCleanup(lambda: dlg.proc.stop())
            dlg._finish_early()
            self.assertEqual((dlg.finish_btn.cget("text"), str(dlg.finish_btn.cget("state"))), ("Finishing ...", "disabled"))
            self.assertIn("Finishing early", dlg.status_label.cget("text"))
            self.assertEqual(str(dlg.stop_btn.cget("state")), "normal", "Stop now still ends it at once")
            self.assertTrue(pump_until(self.root, lambda: dlg.finished, 30), "the run never ended")
        self.assertEqual(dlg.title_label.cget("text"), "Finished early")
        self.assertFalse(dlg.stopping)
        self.assertEqual(self.read_bytes(os.path.join(ctx.model_dir, "head_best.pt")), b"head-1")
        self.assertEqual(len(dlg.table.get_children()), 1, "the interrupted epoch is in the table")
        self.assertIn("epoch 1", dlg.status_label.cget("text"))
        self.assertFalse(os.path.exists(os.path.join(ctx.model_dir, "head_run", "finish_early.flag")), "flag cleaned up")
        self.assertTrue(wc.get_assistant(WORKSPACE)["custom_head"]["head_path"].endswith("head_best.pt"))
        dlg.win.destroy()

    def test_finish_early_twice_or_after_stop_now_does_nothing(self):
        dlg = self.open()
        dlg.epochs_var.set(5)
        with mock.patch.dict(os.environ, {"FAKE_TRAIN_MODE": "graceful"}):
            dlg.save()
            self.addCleanup(lambda: dlg.proc.stop())        # whatever happens, no trainer is left running
            dlg._finish_early()
            flag = os.path.join(ctx.model_dir, "head_run", "finish_early.flag")
            self.assertTrue(os.path.exists(flag))
            stamp = os.stat(flag).st_mtime_ns
            with mock.patch("builtins.open", side_effect=AssertionError("a second press must not touch the file")):
                dlg._finish_early()                        # the second press is ignored
            self.assertTrue(dlg.finishing)
            self.assertEqual(os.stat(flag).st_mtime_ns, stamp)
            self.assertTrue(pump_until(self.root, lambda: dlg.finished, 30))
        dlg.win.destroy()

    def test_the_job_names_the_file_the_finish_early_button_creates(self):
        dlg = self.open()
        dlg.epochs_var.set(1)
        self.run_to_end(dlg)
        with open(os.path.join(ctx.model_dir, "head_run", "job.json"), encoding="utf-8") as f:
            job = __import__("json").load(f)
        self.assertEqual(job["stop_file"], os.path.join(ctx.model_dir, "head_run", "finish_early.flag"))
        dlg.win.destroy()

    def test_closing_while_training_asks_first(self):
        dlg = self.open()
        dlg.epochs_var.set(5)
        with mock.patch.dict(os.environ, {"FAKE_TRAIN_MODE": "slow"}):
            dlg.save()
            self.assertTrue(pump_until(self.root, lambda: float(dlg.bar.cget("value")) >= 1, 20))
            with mock.patch.object(messagebox, "askyesno", return_value=False) as ask:
                dlg._close()
            self.assertTrue(ask.called)
            self.assertTrue(dlg.win.winfo_exists() and dlg.running, "answering No keeps training")
            dlg._stop()
            self.assertTrue(pump_until(self.root, lambda: dlg.finished, 20))
        dlg.win.destroy()


@unittest.skipUnless(HAVE_DISPLAY, "no display available")
class BrowseCustomModelTests(ModeGuiBase):
    """Original mode: the Browse custom model button of the Label Assistant window."""

    def setUp(self):
        super().setUp()
        self.file = os.path.join(ctx.root, "elsewhere", "finetuned_v8m.pt")
        os.makedirs(os.path.dirname(self.file), exist_ok=True)
        open(self.file, "wb").close()
        if os.path.exists(ctx.model_file):                      # these tests start without a workspace model
            os.remove(ctx.model_file)

    def open(self):
        from utils.LabelAssistantDialog import LabelAssistantDialog
        dlg = LabelAssistantDialog(self.root)
        pump(self.root, 0.2)
        return dlg

    def browse(self, dlg, picked, names=("cat", "walking", "tree")):
        from utils import LabelAssistantDialog as module
        with mock.patch.object(module.filedialog, "askopenfilename", return_value=picked), \
                mock.patch.object(module.providers, "model_class_names", return_value=list(names)):
            dlg._browse_model()
        pump(self.root, 0.1)

    def test_the_button_is_there(self):
        dlg = self.open()
        self.assertTrue(buttons(dlg.win, "Browse custom model"))

    def test_browsing_selects_the_model_and_lists_its_classes_mapped_by_name(self):
        dlg = self.open()
        self.assertEqual(dlg.custom_radio.cget("state"), "disabled", "no model yet")
        self.browse(dlg, self.file)
        self.assertEqual(dlg.provider_var.get(), wc.PROVIDER_CUSTOM)
        self.assertEqual(dlg.custom_radio.cget("state"), "normal")
        self.assertEqual(list(dlg.map_vars), ["cat", "walking", "tree"])
        self.assertEqual(dlg.map_vars["cat"].get(), "cat", "same name -> mapped by itself")
        self.assertEqual(dlg.map_vars["walking"].get(), "(skip)")
        self.assertIn(os.path.basename(self.file), dlg.custom_info.cget("text"))

    def test_saving_stores_the_path_the_mapping_and_the_provider(self):
        dlg = self.open()
        self.browse(dlg, self.file)
        dlg.map_vars["walking"].set("dog")
        dlg._save()
        a = wc.get_assistant(WORKSPACE)
        self.assertEqual(a["provider"], wc.PROVIDER_CUSTOM)
        self.assertEqual(os.path.normcase(a["custom_model"]["path"]), os.path.normcase(os.path.normpath(self.file)))
        self.assertEqual(a["custom_model"]["class_map"], {"cat": "cat", "walking": "dog"})

    def test_a_saved_model_comes_back_with_its_mapping(self):
        a = wc.get_assistant(WORKSPACE)
        a["provider"] = wc.PROVIDER_CUSTOM
        a["custom_model"] = {"path": self.file, "class_map": {"walking": "dog"}}
        wc.set_assistant(WORKSPACE, a)
        from utils import LabelAssistantDialog as module
        with mock.patch.object(module.providers, "model_class_names", return_value=["cat", "walking", "tree"]):
            dlg = self.open()
        self.assertEqual(dlg.provider_var.get(), wc.PROVIDER_CUSTOM)
        self.assertEqual(dlg.map_vars["walking"].get(), "dog")
        self.assertEqual(dlg.map_vars["cat"].get(), "cat")

    def test_cancelling_the_file_dialog_changes_nothing(self):
        dlg = self.open()
        self.browse(dlg, "")
        self.assertEqual(dlg.provider_var.get(), wc.PROVIDER_YOLO_WORLD)
        self.assertEqual(dlg.map_vars, {})

    def test_a_file_that_is_not_a_model_is_reported_and_not_selected(self):
        from utils import LabelAssistantDialog as module
        from utils.assistant_providers import AssistantError
        dlg = self.open()
        with mock.patch.object(module.filedialog, "askopenfilename", return_value=self.file), \
                mock.patch.object(module.providers, "model_class_names", side_effect=AssistantError("not a model")):
            dlg._browse_model()
        self.assertIn("not a model", self.rec.of("showwarning")[-1][2])
        self.assertEqual(dlg.model_path, "")
        self.assertEqual(dlg.provider_var.get(), wc.PROVIDER_YOLO_WORLD)

    def test_saving_without_any_mapped_class_is_refused(self):
        dlg = self.open()
        self.browse(dlg, self.file, names=("foo", "bar"))
        dlg._save()
        self.assertIn("Map at least one", self.rec.of("showwarning")[-1][2])
        self.assertEqual(wc.get_assistant(WORKSPACE)["custom_model"]["path"], "")
        self.assertTrue(dlg.win.winfo_exists())

    def test_going_back_to_the_workspace_model(self):
        dlg = self.open()
        self.browse(dlg, self.file)
        dlg._clear_model()
        self.assertEqual(dlg.model_path, "")
        self.assertEqual(dlg.provider_var.get(), wc.PROVIDER_YOLO_WORLD, "no workspace model exists")
        self.assertEqual(dlg.map_vars, {})
        dlg._save()
        self.assertEqual(wc.get_assistant(WORKSPACE)["custom_model"], {"path": "", "class_map": {}})


@unittest.skipUnless(HAVE_DISPLAY, "no display available")
class BrowseDetectorTests(ModeGuiBase):
    """Custom Model mode: the Browse custom model button next to the detector weights."""

    def setUp(self):
        super().setUp()
        self.detector = os.path.join(ctx.root, "elsewhere", "my_detector.pt")
        os.makedirs(os.path.dirname(self.detector), exist_ok=True)
        open(self.detector, "wb").close()

    def make_head(self):
        try:
            import torch
        except Exception:
            self.skipTest("torch is not installed")
        path = os.path.join(ctx.model_dir, "heads", "head_best.pt")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({"head": {}, "names": ["cat", "walking"],
                    "cfg": {"weights": "yolov9c.pt", "taps": [15, 18, 21], "imgsz": [640, 640], "nc": 2}}, path)
        return path

    def open(self):
        from utils.ModeDialogs import CustomHeadDialog
        dlg = CustomHeadDialog(self.root)
        pump(self.root, 0.2)
        return dlg

    def browse(self, dlg, picked, names=("bottle_a", "bottle_b")):
        from utils import ModeDialogs as module
        with mock.patch.object(module.filedialog, "askopenfilename", return_value=picked), \
                mock.patch.object(module.providers, "model_class_names", return_value=list(names)):
            dlg._browse_detector()
        pump(self.root, 0.1)

    def test_both_browse_buttons_exist(self):
        dlg = self.open()
        self.assertTrue(buttons(dlg.win, "Browse head"))
        self.assertTrue(buttons(dlg.win, "Browse custom model"))

    def test_the_detectors_own_class_names_are_used(self):
        dlg = self.open()
        self.assertEqual(dlg.classes_var.get(), "person")
        self.browse(dlg, self.detector)
        self.assertEqual(os.path.normcase(dlg.weights_var.get()), os.path.normcase(os.path.normpath(self.detector)))
        self.assertEqual(dlg.detector_names, ["bottle_a", "bottle_b"])
        self.assertEqual(dlg.classes_var.get(), "bottle_a", "'person' does not exist in this detector")
        self.assertIn("bottle_b", dlg.classes_hint.cget("text"))

    def test_saving_stores_the_detector_and_the_chosen_classes(self):
        head = self.make_head()
        dlg = self.open()
        dlg.path_var.set(head)
        dlg._load_head(head)
        self.browse(dlg, self.detector)
        dlg.classes_var.set("bottle_b")
        dlg.save()
        h = wc.get_assistant(WORKSPACE)["custom_head"]
        self.assertEqual(h["detector_classes"], [1])
        self.assertEqual(os.path.normcase(h["detector_weights"]), os.path.normcase(os.path.normpath(self.detector)))

    def test_cancelling_keeps_the_current_detector(self):
        dlg = self.open()
        self.browse(dlg, "")
        self.assertEqual(dlg.weights_var.get(), "")
        self.assertEqual(dlg.detector_names[0], "person")

    def test_a_missing_detector_file_is_refused_but_a_stock_name_is_fine(self):
        head = self.make_head()
        dlg = self.open()
        dlg.path_var.set(head)
        dlg._load_head(head)
        dlg.weights_var.set(os.path.join(ctx.root, "nope", "gone.pt"))
        dlg.save()
        self.assertIn("gone.pt", self.rec.of("showwarning")[-1][2])
        self.assertEqual(wc.get_assistant(WORKSPACE)["custom_head"]["head_path"], "")
        dlg.weights_var.set("yolov9c.pt")                        # ultralytics fetches stock models by name
        dlg.save()
        self.assertEqual(wc.get_assistant(WORKSPACE)["custom_head"]["detector_weights"], "yolov9c.pt")

    def test_a_saved_detector_comes_back_with_its_names(self):
        a = wc.get_assistant(WORKSPACE)
        a["custom_head"]["detector_weights"] = self.detector
        a["custom_head"]["detector_classes"] = [1]
        wc.set_assistant(WORKSPACE, a)
        from utils import ModeDialogs as module
        with mock.patch.object(module.providers, "model_class_names", return_value=["bottle_a", "bottle_b"]):
            dlg = self.open()
        self.assertEqual(dlg.classes_var.get(), "bottle_b")


@unittest.skipUnless(HAVE_DISPLAY, "no display available")
class AutoAnnotateWindowTests(ModeGuiBase):
    def fake_predict(self, bgr):
        return [{"rect": (10, 10, 90, 90), "conf": 0.9, "cls": "cat"}], False

    def test_run_blocking_returns_the_result_or_the_error(self):
        from utils.AutoAnnotateDialog import run_blocking
        result, error = run_blocking(self.root, "t", lambda status, progress, cancel: status("x") or 42)
        self.assertEqual((result, error), (42, None))

        def boom(status, progress, cancel):
            raise RuntimeError("nope")
        result, error = run_blocking(self.root, "t", boom)
        self.assertIsNone(result)
        self.assertEqual(str(error), "nope")

    def test_the_window_annotates_every_image_and_reports(self):
        from utils import inferenceObjectDetection as inf
        from utils.AutoAnnotateDialog import AutoAnnotateDialog
        from utils import file_handler
        for d in (ctx.voc, ctx.labels):
            for f in os.listdir(d):
                os.remove(os.path.join(d, f))
        events = []
        with mock.patch.object(inf, "build_predictor", return_value=self.fake_predict):
            dlg = AutoAnnotateDialog(self.root, IMAGES, before_start=lambda: events.append("before"),
                                     after_done=lambda: events.append("after"))
            pump(self.root, 0.2)
            self.assertEqual(dlg.start_btn.cget("text"), "Start")
            dlg.start_btn.invoke()
            self.assertTrue(pump_until(self.root, lambda: dlg.finished), "the batch never finished")
        self.assertEqual(events, ["before", "after"])
        self.assertIn("3 of 3 images annotated", self.rec.of("showinfo")[-1][2])
        for name in IMAGES:
            self.assertEqual(file_handler.load_annotation_local(name)[0], [[10, 10, 90, 90, "cat"]])
        self.assertEqual(dlg.start_btn.cget("text"), "Start", "can be started again")
        self.assertEqual(wc.get_assistant(WORKSPACE)["batch"]["only_unlabeled"], True)
        dlg._close()

    def test_a_class_added_after_the_window_opened_is_labelled(self):
        # config.CLASSLIST is a snapshot from when the workspace was opened; the batch must use the live classes
        from utils import inferenceObjectDetection as inf
        from utils.AutoAnnotateDialog import AutoAnnotateDialog
        from utils import file_handler
        for d in (ctx.voc, ctx.labels):
            for f in os.listdir(d):
                os.remove(os.path.join(d, f))
        config.class_manager.add_class("standing")
        self.addCleanup(config.class_manager.delete_class, "standing")
        self.assertNotIn("standing", config.CLASSLIST)
        predict = lambda bgr: ([{"rect": (10, 10, 90, 90), "conf": 0.9, "cls": "standing"}], False)
        with mock.patch.object(inf, "build_predictor", return_value=predict):
            dlg = AutoAnnotateDialog(self.root, IMAGES)
            dlg.start_btn.invoke()
            self.assertTrue(pump_until(self.root, lambda: dlg.finished))
        self.assertIn("3 of 3 images annotated", self.rec.of("showinfo")[-1][2])
        self.assertEqual(file_handler.load_annotation_local(IMAGES[0])[0], [[10, 10, 90, 90, "standing"]])
        dlg._close()

    def test_provider_problems_are_shown_not_raised(self):
        from utils import inferenceObjectDetection as inf
        from utils.AutoAnnotateDialog import AutoAnnotateDialog
        from utils.assistant_providers import AssistantError
        with mock.patch.object(inf, "build_predictor", side_effect=AssistantError("add a target class first")):
            dlg = AutoAnnotateDialog(self.root, IMAGES)
            dlg.start_btn.invoke()
            self.assertTrue(pump_until(self.root, lambda: dlg.finished))
        self.assertIn("add a target class first", self.rec.of("showwarning")[-1][2])
        self.assertEqual(dlg.close_btn.cget("state"), "normal")
        dlg._close()

    def test_labelled_images_can_be_included_or_skipped(self):
        from utils import inferenceObjectDetection as inf
        from utils.AutoAnnotateDialog import AutoAnnotateDialog
        calls = []

        def predict(bgr):
            calls.append(1)
            return [], False
        with mock.patch.object(inf, "build_predictor", return_value=predict):
            dlg = AutoAnnotateDialog(self.root, IMAGES)
            self.assertTrue(dlg.only_var.get(), "default: only images without annotations")
            dlg.only_var.set(False)
            dlg.start_btn.invoke()
            self.assertTrue(pump_until(self.root, lambda: dlg.finished))
        self.assertEqual(len(calls), 3)
        self.assertFalse(wc.get_assistant(WORKSPACE)["batch"]["only_unlabeled"])
        a = wc.get_assistant(WORKSPACE)
        a["batch"]["only_unlabeled"] = True
        wc.set_assistant(WORKSPACE, a)
        dlg._close()


@unittest.skipUnless(HAVE_DISPLAY, "no display available")
class Sam2DynamicDialogTests(ModeGuiBase):
    def setUp(self):
        super().setUp()
        from utils import sam2_dynamic as sd
        self.sd = sd
        sd.store.clear()

    def tearDown(self):
        self.sd.store.clear()
        super().tearDown()

    def open(self):
        from utils.ModeDialogs import Sam2DynamicDialog
        dlg = Sam2DynamicDialog(self.root)
        pump(self.root, 0.2)
        return dlg

    def test_the_label_assistant_button_opens_it_in_sam2_mode(self):
        from utils import LabelAssistantDialog as lad
        am.set_mode(am.MODE_SAM2)
        with mock.patch("utils.ModeDialogs.open_sam2_dynamic") as opener:
            lad.open_label_assistant(self.root)
        opener.assert_called_once()

    def test_it_shows_the_saved_settings_and_what_has_been_learned(self):
        a = wc.get_assistant(WORKSPACE)
        a["sam2_dynamic"].update(model="sam2.1_t.pt", imgsz=512, max_references=4, min_score=0.1)
        wc.set_assistant(WORKSPACE, a)
        self.sd.store.remember("x.png", [[10, 10, 90, 90, "cat"]], [])
        dlg = self.open()
        self.assertIn("sam2.1_t.pt", dlg.model_var.get())
        self.assertEqual((dlg.size_var.get(), dlg.refs_var.get(), round(dlg.score_var.get(), 2)), ("512", 4, 0.1))
        self.assertIn("1 annotated image", dlg.ref_info.cget("text"))
        dlg.win.destroy()

    def test_saving_stores_the_choices_and_selects_the_provider(self):
        dlg = self.open()
        dlg.model_var.set(next(k for k in dlg.model_names if k.startswith("sam2.1_s.pt")))
        dlg.size_var.set("768")
        dlg.refs_var.set(9)
        dlg.score_var.set(0.2)
        dlg.save()
        pump(self.root, 0.1)
        a = wc.get_assistant(WORKSPACE)
        self.assertEqual(a["sam2_dynamic"], {"model": "sam2.1_s.pt", "imgsz": 768, "max_references": 9, "min_score": 0.2})
        self.assertEqual(a["provider"], wc.PROVIDER_SAM2)

    def test_forget_all_empties_the_references(self):
        self.sd.store.remember("x.png", [[10, 10, 90, 90, "cat"]], [])
        dlg = self.open()
        dlg._forget()
        self.assertEqual(len(self.sd.store), 0)
        self.assertIn("None yet", dlg.ref_info.cget("text"))
        dlg.win.destroy()

    def test_learning_from_labeled_images_adds_each_image_once(self):
        os.makedirs(config.vocdataset_folder, exist_ok=True)
        for leftover in os.listdir(config.vocdataset_folder):               # annotations other tests left behind
            os.remove(os.path.join(config.vocdataset_folder, leftover))
        for name in IMAGES[:2]:
            with open(os.path.join(config.vocdataset_folder, os.path.splitext(name)[0] + ".xml"), "w") as f:
                f.write("<annotation><size><width>320</width><height>240</height></size><object><name>cat</name>"
                        "<bndbox><xmin>10</xmin><ymin>10</ymin><xmax>100</xmax><ymax>100</ymax></bndbox></object></annotation>")
        dlg = self.open()
        dlg._load_labeled()
        self.assertEqual(len(self.sd.store), 2)
        dlg._load_labeled()                                          # again: no duplicates
        self.assertEqual(len(self.sd.store), 2)
        self.assertIn("2 annotated image", dlg.ref_info.cget("text"))
        dlg.win.destroy()

    def test_with_no_labeled_images_it_says_so(self):
        for name in os.listdir(config.vocdataset_folder) if os.path.isdir(config.vocdataset_folder) else []:
            os.remove(os.path.join(config.vocdataset_folder, name))
        dlg = self.open()
        dlg._load_labeled()
        self.assertIn("No labeled images", self.rec.of("showinfo")[-1][2])
        dlg.win.destroy()


@unittest.skipUnless(HAVE_DISPLAY, "no display available")
class MainWindowModeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from utils.AnnotationGUI import AnnotationGUI
        cls.rec = Recorder().__enter__()
        cls.root = tk.Tk()
        cls.root.geometry("1500x850+0+0")
        cls.gui = AnnotationGUI(cls.root)
        pump(cls.root, 0.8)

    @classmethod
    def tearDownClass(cls):
        cls.rec.__exit__()
        cls.root.destroy()

    def setUp(self):
        self._mode = app_settings.get("mode")
        self.rec.calls.clear()
        config.state.bboxes, config.state.polygons = [], []
        self.gui._assist_unreviewed.clear()

    def tearDown(self):
        app_settings.set("mode", self._mode)

    def test_the_header_has_an_infer_button_instead_of_annotate_all_and_a_g_shortcut(self):
        self.assertTrue(buttons(self.root, "Infer"))
        self.assertFalse(buttons(self.root, "Annotate all"))
        self.assertIn("G", set(self.root.bind()))

    # ------------------------------------------------------------ SAM 2 Dynamic references
    def _leave_image(self, index, bboxes):
        """Annotate image `index`, then move on - what a user does."""
        config.state.current_index = index
        self.gui.load_current_image()
        config.state.bboxes = [list(b) for b in bboxes]
        self.gui.save_current()
        return os.path.join(config.input_folder, IMAGES[index])

    def test_leaving_an_image_remembers_its_annotations_once_however_often_you_return(self):
        from utils import sam2_dynamic as sd
        sd.store.clear()
        self.addCleanup(sd.store.clear)
        a = self._leave_image(0, [[10, 10, 90, 90, "cat"]])
        b = self._leave_image(1, [[20, 20, 100, 100, "dog"]])
        self._leave_image(0, [[10, 10, 90, 90, "cat"]])                  # back to image 1
        self._leave_image(1, [[20, 20, 100, 100, "dog"]])                # and to image 2 again
        self.assertEqual(len(sd.store), 2)
        self.assertEqual(sd.store.items_for(a), [("cat", "box", (10.0, 10.0, 90.0, 90.0))])
        self.assertEqual(sd.store.items_for(b), [("dog", "box", (20.0, 20.0, 100.0, 100.0))])

    def test_what_you_change_replaces_what_was_remembered(self):
        from utils import sam2_dynamic as sd
        sd.store.clear()
        self.addCleanup(sd.store.clear)
        a = self._leave_image(0, [[10, 10, 90, 90, "cat"], [100, 100, 190, 190, "dog"]])
        self.assertEqual(len(sd.store.items_for(a)), 2)
        self._leave_image(0, [[10, 10, 90, 90, "cat"]])                  # you deleted the dog
        self.assertEqual(sd.store.items_for(a), [("cat", "box", (10.0, 10.0, 90.0, 90.0))])

    def test_annotations_the_assistant_made_are_not_learned_until_you_have_touched_them(self):
        from utils import sam2_dynamic as sd
        sd.store.clear()
        self.addCleanup(sd.store.clear)
        config.state.current_index = 2
        self.gui.load_current_image()
        config.state.bboxes, config.state.polygons = [], []                # a clean image
        predict = lambda bgr: ([{"rect": (20, 20, 120, 120), "conf": 0.9, "cls": "cat"}], False)
        from utils import inferenceObjectDetection as inf
        with mock.patch.object(inf, "build_predictor", return_value=predict):
            self.gui.run_inference()
        self.assertEqual(len(config.state.bboxes), 1)
        self.gui.save_current()                                          # left without a look
        path = os.path.join(config.input_folder, IMAGES[2])
        self.assertNotIn(path, sd.store)
        self.gui.push_undo()                                             # you edit it...
        self.gui.save_current()                                          # ...and now it counts
        self.assertIn(path, sd.store)
        self.gui._assist_unreviewed.clear()

    def test_deleting_an_image_forgets_it(self):
        from utils import sam2_dynamic as sd
        sd.store.clear()
        self.addCleanup(sd.store.clear)
        path = self._leave_image(0, [[10, 10, 90, 90, "cat"]])
        self.assertIn(path, sd.store)
        with mock.patch("utils.AnnotationGUI.delete_current_image", return_value=(0, list(self.gui.images))):
            self.gui.delete_image()
        self.assertNotIn(path, sd.store)

    def test_train_in_sam2_mode_explains_that_nothing_needs_training(self):
        from utils import sam2_dynamic as sd
        am.set_mode(am.MODE_SAM2)
        with mock.patch("utils.AnnotationGUI.HeadTrainDialog") as yolo, \
                mock.patch("utils.AnnotationGUI.open_head_training") as head:
            self.gui.start_training()
        self.assertFalse(yolo.called or head.called)
        self.assertIn("needs no training", self.rec.of("showinfo")[-1][2])

    def test_infer_in_sam2_mode_runs_behind_a_progress_window_and_merges_the_result(self):
        from utils import inferenceObjectDetection as inf
        am.set_mode(am.MODE_SAM2)
        predict = lambda bgr: ([{"rect": (20, 20, 120, 120), "conf": 0.2, "cls": "cat"}], False)
        with mock.patch.object(inf, "sam2_predictor", return_value=predict), \
                mock.patch("utils.AnnotationGUI.confirm_download_if_needed", return_value=True):
            self.gui.run_inference()
        self.assertEqual(config.state.bboxes, [[20, 20, 120, 120, "cat"]])
        self.assertFalse(self.rec.of("showwarning"))

    def test_infer_in_sam2_mode_with_nothing_learned_yet_explains_what_to_do(self):
        from utils import sam2_dynamic as sd
        sd.store.clear()
        am.set_mode(am.MODE_SAM2)
        with mock.patch.object(sd, "predictor_problem", return_value=None), \
                mock.patch("utils.AnnotationGUI.confirm_download_if_needed", return_value=True):
            self.gui.run_inference()
        self.assertIn("annotated", self.rec.of("showwarning")[-1][2])
        self.assertEqual(config.state.bboxes, [])

    def test_infer_in_sam2_mode_passes_the_current_image_so_it_is_never_its_own_reference(self):
        from utils import inferenceObjectDetection as inf
        am.set_mode(am.MODE_SAM2)
        config.state.current_index = 1
        self.gui.load_current_image()
        with mock.patch.object(inf, "sam2_predictor", return_value=lambda bgr: ([], False)) as build, \
                mock.patch("utils.AnnotationGUI.confirm_download_if_needed", return_value=True):
            self.gui.run_inference()
        self.assertEqual(os.path.basename(build.call_args.kwargs["exclude"]), IMAGES[1])

    def test_the_header_names_sam2_dynamic_in_its_mode(self):
        self.assertEqual(am.title(am.MODE_SAM2), "SAM 2 Dynamic")

    def test_the_header_names_the_current_mode(self):
        self.assertIn("YOLO-WORLD", label_texts(self.root).upper())

    def test_infer_in_locate_mode_runs_behind_a_progress_window_and_merges_the_result(self):
        from utils import inferenceObjectDetection as inf
        am.set_mode(am.MODE_LOCATE)
        a = wc.get_assistant(WORKSPACE)
        a["locate_anything"]["target_classes"] = [{"prompt": "cat", "map_to": "cat"}]
        wc.set_assistant(WORKSPACE, a)
        predict = lambda bgr: ([{"rect": (20, 20, 120, 120), "conf": 0.9, "cls": "cat"}], False)
        with mock.patch.object(inf, "locate_predictor", return_value=predict), \
                mock.patch("utils.AnnotationGUI.confirm_download_if_needed", return_value=True):
            self.gui.run_inference()
        self.assertEqual(config.state.bboxes, [[20, 20, 120, 120, "cat"]])
        self.assertFalse(self.rec.of("showwarning"))

    def test_infer_in_locate_mode_shows_provider_problems_as_a_warning(self):
        am.set_mode(am.MODE_LOCATE)
        a = wc.get_assistant(WORKSPACE)
        a["locate_anything"]["target_classes"] = []
        wc.set_assistant(WORKSPACE, a)
        with mock.patch("utils.AnnotationGUI.confirm_download_if_needed", return_value=True):
            self.gui.run_inference()
        self.assertIn("target class", self.rec.of("showwarning")[-1][2])
        self.assertEqual(config.state.bboxes, [])

    def test_declining_the_download_changes_nothing_and_shows_no_error(self):
        am.set_mode(am.MODE_LOCATE)
        with mock.patch("utils.AnnotationGUI.confirm_download_if_needed", return_value=False):
            self.gui.run_inference()
        self.assertFalse(self.rec.of("showwarning"))
        self.assertEqual(config.state.bboxes, [])

    def test_train_in_custom_head_mode_opens_the_head_trainer(self):
        am.set_mode(am.MODE_HEAD)
        with mock.patch("utils.AnnotationGUI.open_head_training") as head, \
                mock.patch("utils.AnnotationGUI.HeadTrainDialog") as yolo:
            self.gui.start_training()
        head.assert_called_once()
        args = head.call_args[0]
        self.assertEqual(args[2], config.yolo_labels_folder)
        self.assertFalse(yolo.called, "no whole-YOLO training in this mode")

    def test_train_in_the_other_modes_is_the_full_yolo_training_with_the_same_window(self):
        for mode in (am.MODE_YOLO_WORLD, am.MODE_LOCATE):
            am.set_mode(mode)
            with mock.patch("utils.AnnotationGUI.open_head_training") as head, \
                    mock.patch("utils.AnnotationGUI.HeadTrainDialog") as window:
                self.gui.start_training()
            self.assertTrue(window.called, mode)
            self.assertTrue(window.call_args.kwargs["full_only"], "no head choice outside Custom Model")
            self.assertFalse(head.called, mode)

    # ------------------------------------------------------------ the plain Train button: a window, not a terminal
    def _plain_train(self, mode="ok", cfg=None, polygon=False, slow=False):
        """Press Train in YOLO-World mode: the settings window opens (the same one as Custom Model's full training),
        the test fills it from `cfg` and presses Start, with a fake trainer process. Returns that window, now showing
        the progress (and keeps the Popen mock, which must stay unused)."""
        import sys
        from utils import HeadTrainDialog as htd
        am.set_mode(am.MODE_YOLO_WORLD)
        script = os.path.join(ctx.root, "fake_yolo_trainer.py")
        with open(script, "w") as f:
            f.write(FAKE_YOLO_TRAINER)
        cfg = cfg or {"epoch": 2, "batch": 4, "imgsz": 320, "freeze": 0, "lr0": 0.001, "weight_decay": 0.0005,
                      "base_model": "yolo11n.pt"}
        created = []

        def factory(*a, **k):
            dlg = htd.HeadTrainDialog(*a, **k)
            created.append(dlg)
            return dlg

        self.addCleanup(setattr, config.state, "annotation_mode", config.state.annotation_mode)
        config.state.annotation_mode = "polygon" if polygon else "bbox"
        with mock.patch("utils.AnnotationGUI.HeadTrainDialog", factory), \
                mock.patch("utils.HeadTrainDialog.default_command",
                           lambda job_path: [sys.executable, "-u", script, job_path]), \
                mock.patch("utils.AnnotationGUI.subprocess.Popen", wraps=__import__("subprocess").Popen) as popen, \
                mock.patch("utils.workspace_counts.get_or_empty", return_value=dict(PLENTY)), \
                mock.patch.dict(os.environ, {"FAKE_TRAIN_MODE": mode}):
            self.gui.start_training()
            self.assertEqual(len(created), 1, "the Train button opened one settings window")
            self._fill_training_window(created[0], cfg)
            created[0].save()
        self.popen = popen
        return created[0]

    def _fill_training_window(self, dlg, cfg):
        """Type the settings of the old dialog's dict into the standard window."""
        from utils import augment_catalog as cat
        from utils import augmentation as aug
        dlg.epochs_var.set(cfg["epoch"])
        dlg.batch_var.set(cfg["batch"])
        dlg.sq_var.set(cfg["imgsz"])
        dlg.freeze_var.set(cfg["freeze"])
        dlg.lr_var.set(str(cfg["lr0"]))
        dlg.wd_var.set(str(cfg["weight_decay"]))
        if cfg.get("base_model"):
            dlg.base_var.set(cfg["base_model"])
        split = cfg.get("split")
        if split:
            dlg.split.train_var.set(str(round(100 - split["val_fraction"] * 100 - split["test_fraction"] * 100)))
            dlg.split.valid_var.set(str(round(split["val_fraction"] * 100)))
            dlg.shuffle_var.set(split.get("shuffle", False))
        if cfg.get("augment"):
            config_ = aug.config_from_dict(cfg["augment"])
            dlg.augment.result = {"config": config_, "state": cat.from_config(config_), "outputs": 2,
                                  "keep_original": config_.keep_original, "seed": config_.seed,
                                  "splits": config_.splits}

    def _clean_trained_model(self):
        import shutil
        for name in ("modelAssistant.pt", "modelAssistant.prev.pt"):
            try:
                os.remove(os.path.join(ctx.model_dir, name))
            except OSError:
                pass
        shutil.rmtree(os.path.join(ctx.model_dir, "yolo_run"), ignore_errors=True)

    def _wait(self, dlg, timeout=30):
        self.assertTrue(pump_until(self.root, lambda: dlg.finished, timeout), "the training never ended")

    def test_train_in_yolo_world_mode_opens_a_progress_window_instead_of_a_terminal(self):
        self._clean_trained_model()
        self.addCleanup(self._clean_trained_model)
        dlg = self._plain_train()
        commands = [c[0][0] for c in self.popen.call_args_list]               # the argument lists, not the environment
        self.assertEqual(len(commands), 1, "only the training process itself is started: " + str(commands))
        self.assertFalse(any(word in " ".join(commands[0]) for word in ("terminal", "xterm", "konsole", "training.py")),
                         commands)
        self.assertEqual(dlg.win.title(), "Train model")
        self.assertEqual(dlg.active_kind, "yolo")
        self.assertIsNone(dlg.win.grab_status(), "the annotation window stays usable while it trains")
        self.assertTrue(config.state.training_running)
        self.assertIs(config.state.training_process, dlg.proc.proc)
        self._wait(dlg)
        rows = [dlg.table.item(i)["values"] for i in dlg.table.get_children()]
        self.assertEqual([r[0] for r in rows], [1, 2], "a table row per epoch")
        self.assertEqual(dlg.title_label.cget("text"), "Training finished")
        with open(config.model_path, "rb") as f:
            self.assertEqual(f.read(), b"model-2")
        self.assertIn("G and the full-dataset run use it", dlg.status_label.cget("text"))
        self.assertFalse(config.state.training_running)
        self.assertIsNone(config.state.training_process)
        a = wc.get_assistant(WORKSPACE)
        self.assertEqual(a["provider"], wc.PROVIDER_CUSTOM)
        dlg.win.destroy()

    def test_the_settings_of_the_old_dialog_become_the_job(self):
        self._clean_trained_model()
        self.addCleanup(self._clean_trained_model)
        cfg = {"epoch": 3, "batch": 8, "imgsz": 416, "freeze": 10, "lr0": 0.002, "weight_decay": 0.001,
               "base_model": "yolov8s.pt"}
        dlg = self._plain_train(cfg=cfg)
        self._wait(dlg)
        with open(os.path.join(ctx.model_dir, "yolo_run", "job.json"), encoding="utf-8") as f:
            job = __import__("json").load(f)
        self.assertEqual((job["kind"], job["init_weights"], job["epochs"], job["batch"], job["imgsz"], job["freeze"]),
                         ("yolo", "yolov8s.pt", 3, 8, 416, 10))
        self.assertEqual((job["lr"], job["weight_decay"], job["val_fraction"]), (0.002, 0.001, 0.2))
        self.assertAlmostEqual(job["test_fraction"], 0.1, msg="the standard split is 70 / 20 / 10 in every mode")
        self.assertEqual(job["classes"], config.class_manager.get_classes())
        self.assertEqual(job["labels_folder"], config.yolo_labels_folder)
        self.assertTrue(job["images_folders"])
        dlg.win.destroy()

    def test_the_split_chosen_in_the_old_dialog_reaches_the_job(self):
        self._clean_trained_model()
        self.addCleanup(self._clean_trained_model)
        split = {"val_fraction": 0.2, "test_fraction": 0.1, "shuffle": True}
        cfg = {"epoch": 1, "batch": 4, "imgsz": 320, "freeze": 0, "lr0": 0.001, "weight_decay": 0.0005, "split": split}
        dlg = self._plain_train(cfg=cfg)
        self._wait(dlg)
        with open(os.path.join(ctx.model_dir, "yolo_run", "job.json"), encoding="utf-8") as f:
            job = __import__("json").load(f)
        self.assertEqual((job["val_fraction"], job["test_fraction"], job["shuffle"]), (0.2, 0.1, True))
        self.assertIn("split_seed", job)
        dlg.win.destroy()

    def test_the_augmentation_chosen_in_the_old_dialog_reaches_the_job(self):
        from utils import augment_catalog as cat, augmentation as aug
        self._clean_trained_model()
        self.addCleanup(self._clean_trained_model)
        state = cat.default_state()
        state["contrast"].enabled = True
        picked = aug.config_to_dict(cat.to_config(state, 2, False, seed=8))
        cfg = {"epoch": 1, "batch": 4, "imgsz": 320, "freeze": 0, "lr0": 0.001, "weight_decay": 0.0005,
               "augment": picked}
        dlg = self._plain_train(cfg=cfg)
        self._wait(dlg)
        with open(os.path.join(ctx.model_dir, "yolo_run", "job.json"), encoding="utf-8") as f:
            job = __import__("json").load(f)
        self.assertEqual(job["augment"], picked)
        dlg.win.destroy()

    def test_an_existing_model_is_trained_on_and_a_polygon_workspace_defaults_to_a_seg_model(self):
        self._clean_trained_model()
        self.addCleanup(self._clean_trained_model)
        os.makedirs(ctx.model_dir, exist_ok=True)
        with open(config.model_path, "wb") as f:
            f.write(b"old model")
        dlg = self._plain_train()
        self._wait(dlg)
        with open(os.path.join(ctx.model_dir, "yolo_run", "job.json"), encoding="utf-8") as f:
            self.assertEqual(__import__("json").load(f)["init_weights"], config.model_path)
        with open(os.path.join(ctx.model_dir, "modelAssistant.prev.pt"), "rb") as f:
            self.assertEqual(f.read(), b"old model", "the model that was there is kept")
        dlg.win.destroy()
        self._clean_trained_model()
        cfg = {"epoch": 1, "batch": 4, "imgsz": 320, "freeze": 0, "lr0": 0.001, "weight_decay": 0.0005}   # no base chosen
        dlg = self._plain_train(cfg=cfg, polygon=True)
        self._wait(dlg)
        with open(os.path.join(ctx.model_dir, "yolo_run", "job.json"), encoding="utf-8") as f:
            self.assertEqual(__import__("json").load(f)["init_weights"], "yolo11s-seg.pt")
        dlg.win.destroy()

    def test_a_second_train_while_one_runs_is_refused_and_stop_frees_the_button(self):
        self._clean_trained_model()
        self.addCleanup(self._clean_trained_model)
        cfg = {"epoch": 5, "batch": 4, "imgsz": 320, "freeze": 0, "lr0": 0.001, "weight_decay": 0.0005,
               "base_model": "yolo11n.pt"}
        dlg = self._plain_train(mode="slow", cfg=cfg)
        self.assertTrue(pump_until(self.root, lambda: float(dlg.bar.cget("value")) >= 1, 20), "epoch 1 never came")
        self.rec.calls.clear()
        with mock.patch("utils.AnnotationGUI.HeadTrainDialog") as again:
            self.gui.start_training()
        self.assertFalse(again.called)
        self.assertIn("already running", self.rec.of("showwarning")[-1][2])
        dlg._stop()
        self._wait(dlg, 20)
        self.assertFalse(config.state.training_running)
        self.assertIsNone(config.state.training_process)
        with open(config.model_path, "rb") as f:
            self.assertEqual(f.read(), b"model-1", "the best model so far is kept")
        dlg.win.destroy()

    def test_leaving_while_a_training_runs_asks_and_keeps_the_best_model(self):
        self._clean_trained_model()
        self.addCleanup(self._clean_trained_model)
        cfg = {"epoch": 5, "batch": 4, "imgsz": 320, "freeze": 0, "lr0": 0.001, "weight_decay": 0.0005,
               "base_model": "yolo11n.pt"}
        dlg = self._plain_train(mode="slow", cfg=cfg)
        self.assertTrue(pump_until(self.root, lambda: float(dlg.bar.cget("value")) >= 1, 20), "epoch 1 never came")
        proc = dlg.proc.proc
        with mock.patch("utils.AnnotationGUI.messagebox.askyesno", return_value=False) as ask, \
                mock.patch.object(self.gui.root, "quit") as quit_, mock.patch.object(self.gui.root, "destroy") as gone:
            self.gui.close_main_gui()
        self.assertIn("still running", ask.call_args[0][1])
        self.assertFalse(quit_.called or gone.called, "No: the window stays and the training goes on")
        self.assertIsNone(proc.poll())
        with mock.patch("utils.AnnotationGUI.messagebox.askyesno", return_value=True), \
                mock.patch.object(self.gui.root, "quit") as quit_, mock.patch.object(self.gui.root, "destroy") as gone:
            self.gui.close_main_gui()
        self.assertTrue(quit_.called and gone.called, "Yes: the window closes")
        self.assertIsNotNone(proc.poll(), "and the training process is gone with it")
        self.assertIsNone(config.state.training_process)
        with open(config.model_path, "rb") as f:
            self.assertEqual(f.read(), b"model-1", "the best model so far is installed, not lost")
        dlg.win.destroy()

    def test_leaving_without_a_training_just_closes(self):
        config.state.training_process = None
        with mock.patch("utils.AnnotationGUI.messagebox.askyesno") as ask, \
                mock.patch.object(self.gui.root, "quit") as quit_, mock.patch.object(self.gui.root, "destroy"):
            self.gui.close_main_gui()
        self.assertFalse(ask.called)
        self.assertTrue(quit_.called)

    def test_a_training_running_in_another_window_blocks_a_new_one(self):
        from utils import process_guard
        self._clean_trained_model()
        self.addCleanup(self._clean_trained_model)
        process_guard.claim(config.BASE_DIR, "another-workspace", os.getpid())      # a live worker, as far as it knows
        self.addCleanup(process_guard.release, config.BASE_DIR, os.getpid())
        self.rec.calls.clear()
        cfg = {"epoch": 1, "batch": 4, "imgsz": 320, "freeze": 0, "lr0": 0.001, "weight_decay": 0.0005,
               "base_model": "yolo11n.pt"}
        with mock.patch("utils.HeadTrainDialog.messagebox.showwarning") as warn, \
                mock.patch("utils.HeadTrainDialog.TrainProcess") as started:
            try:
                self._plain_train(cfg=cfg)
            except AssertionError:
                pass                                           # the helper expects a window: there is none
        self.assertFalse(started.called, "no second process on the same GPU")
        self.assertIn("another-workspace", warn.call_args[0][1])
        self.assertFalse(config.state.training_running)

    def test_finish_early_works_in_the_window_of_the_plain_train_button_too(self):
        self._clean_trained_model()
        self.addCleanup(self._clean_trained_model)
        cfg = {"epoch": 5, "batch": 4, "imgsz": 320, "freeze": 0, "lr0": 0.001, "weight_decay": 0.0005,
               "base_model": "yolo11n.pt"}
        dlg = self._plain_train(mode="graceful", cfg=cfg)
        dlg._finish_early()
        self._wait(dlg, 30)
        self.assertEqual(dlg.title_label.cget("text"), "Finished early")
        with open(config.model_path, "rb") as f:
            self.assertEqual(f.read(), b"model-1")
        self.assertIn("G and the full-dataset run use it", dlg.status_label.cget("text"))
        self.assertFalse(config.state.training_running)
        self.assertIsNone(config.state.training_process)
        dlg.win.destroy()

    def test_a_failed_training_is_explained_in_the_window_and_frees_the_button(self):
        self._clean_trained_model()
        self.addCleanup(self._clean_trained_model)
        dlg = self._plain_train(mode="error")
        self._wait(dlg)
        self.assertEqual(dlg.title_label.cget("text"), "Training failed")
        self.assertIn("CUDA out of memory", dlg.status_label.cget("text"))
        self.assertFalse(os.path.exists(config.model_path))
        self.assertFalse(config.state.training_running)
        self.assertIsNone(config.state.training_process)
        dlg.win.destroy()



PLENTY = {"images": 60, "labelled": 40, "background": 0, "boxes": [20, 20], "instances": {}, "counted": 1}   # enough to train
FAKE_YOLO_TRAINER = r"""
import json, os, sys, time
job = json.load(open(sys.argv[1], encoding="utf-8"))
def ev(event, **f):
    print("@@ " + json.dumps({"event": event, **f}), flush=True)
mode = os.environ.get("FAKE_TRAIN_MODE", "ok")
ev("status", text="Loading the model ...")
if mode == "crash":
    raise SystemExit(3)
if mode == "error":
    ev("error", message="CUDA out of memory")
    raise SystemExit(1)
ev("start", kind="yolo", train_images=9, val_images=3, classes=job["classes"], counts=[5, 4], epochs=job["epochs"],
   device="cpu", batch=job["batch"], imgsz=[job["imgsz"], job["imgsz"]], model=os.path.basename(job["init_weights"]),
   freeze=job["freeze"])
weights = os.path.join(job["out_dir"], "train_run", "weights")
os.makedirs(weights, exist_ok=True)
for i in range(1, job["epochs"] + 1):
    if mode == "slow" and i == 2:
        time.sleep(60)
    stop_now = False
    if mode == "graceful":                      # an epoch that only the stop file ends early
        for _ in range(150):
            if os.path.exists(job["stop_file"]):
                stop_now = True
                break
            time.sleep(0.1)
    print("some plain log line", flush=True)
    with open(os.path.join(weights, "best.pt"), "wb") as f:
        f.write(b"model-%d" % i)
    ev("epoch", epoch=i, epochs=job["epochs"], box_loss=1.0 / i, cls_loss=0.5 / i, dfl_loss=0.9 / i, precision=0.5,
       recall=0.4, map50=0.1 * i, map5095=0.05 * i, fitness=0.1 * i, best=(i != 2), seconds=0.01)
    if stop_now:
        ev("done", best_epoch=i, best_fitness=0.1 * i, epochs_run=i, stopped=True, out_dir=job["out_dir"],
           weights=os.path.join(weights, "best.pt"), has_model=True, classes=job["classes"], map50=0.1 * i,
           map5095=0.05 * i)
        raise SystemExit(0)
ev("done", best_epoch=job["epochs"], best_fitness=0.3, epochs_run=job["epochs"], stopped=False, out_dir=job["out_dir"],
   weights=os.path.join(weights, "best.pt"), has_model=True, classes=job["classes"], map50=0.1 * job["epochs"],
   map5095=0.05 * job["epochs"])
"""


@unittest.skipUnless(HAVE_DISPLAY, "no display available")
class FullYoloTrainingTests(ModeGuiBase):
    """The Train window's second choice: train every layer of a YOLO model, with a table row per epoch."""

    def setUp(self):
        super().setUp()
        import shutil
        import sys
        from tests.helpers import make_images
        self.data = os.path.join(ctx.root, "fulldata")
        self.images = os.path.join(self.data, "images")
        self.labels = os.path.join(self.data, "labels")
        os.makedirs(self.labels, exist_ok=True)
        for i, name in enumerate(make_images(self.images, 12)):
            with open(os.path.join(self.labels, os.path.splitext(name)[0] + ".txt"), "w") as f:
                f.write(f"{i % 2} 0.5 0.5 0.4 0.4\n")
        self.script = os.path.join(ctx.root, "fake_yolo_trainer.py")
        with open(self.script, "w") as f:
            f.write(FAKE_YOLO_TRAINER)
        self.command = lambda job_path: [sys.executable, "-u", self.script, job_path]
        os.makedirs(ctx.model_dir, exist_ok=True)
        self.model_file = os.path.join(ctx.model_dir, "modelAssistant.pt")
        self.prev_file = os.path.join(ctx.model_dir, "modelAssistant.prev.pt")
        for old in (self.model_file, self.prev_file):
            try:
                os.remove(old)
            except OSError:
                pass
        shutil.rmtree(os.path.join(ctx.model_dir, "yolo_run"), ignore_errors=True)

    @staticmethod
    def read_bytes(path):
        with open(path, "rb") as f:
            return f.read()

    def open(self, full=True, **kw):
        from utils.HeadTrainDialog import HeadTrainDialog
        done = []
        dlg = HeadTrainDialog(self.root, [self.images], self.labels, ctx.model_dir, command=self.command,
                              model_path=self.model_file, on_done=done.append, **kw)
        dlg.done_calls = done
        if full:
            dlg.kind_var.set("yolo")
            dlg._apply_kind()
        pump(self.root, 0.2)
        return dlg

    def run_to_end(self, dlg, timeout=30):
        dlg.save()
        self.assertTrue(pump_until(self.root, lambda: dlg.finished, timeout), "the run never ended")

    def job(self):
        with open(os.path.join(ctx.model_dir, "yolo_run", "job.json"), encoding="utf-8") as f:
            return __import__("json").load(f)

    # ---------------------------------------------------------------- the setup window
    def test_the_window_offers_both_kinds_and_starts_on_the_head(self):
        dlg = self.open(full=False)
        self.assertEqual(dlg.kind_var.get(), "head")
        self.assertEqual(dlg.det_title.cget("text"), "DETECTOR")
        self.assertTrue(dlg.taps_row.winfo_ismapped())
        self.assertFalse(dlg.continue_chk.winfo_ismapped())
        radios = [w.cget("text") for w in walk(dlg.win) if isinstance(w, tk.Radiobutton)]
        self.assertEqual(radios, ["Head only (fast)", "Full YOLO training (every layer)"])
        dlg.win.destroy()

    # ---------------------------------------------------------------- augmentation
    def _picked(self, *tiles_ops, outputs=3, keep=True):
        """What ask_augmentation returns when the user picks these operations and presses OK."""
        from utils import augment_catalog as cat
        state = cat.default_state()
        for name in tiles_ops:
            state[name].enabled = True
        return {"config": cat.to_config(state, outputs, keep, seed=4), "state": state, "outputs": outputs,
                "keep_original": keep, "seed": 4}

    def test_the_augmentation_row_belongs_to_full_training_only(self):
        dlg = self.open(full=False)
        self.assertFalse(dlg.aug_frame.winfo_ismapped())
        dlg.kind_var.set("yolo")
        dlg._apply_kind()
        pump(self.root, 0.2)
        self.assertTrue(dlg.aug_frame.winfo_ismapped())
        self.assertIn("Ultralytics still applies its own", dlg.augment.label.cget("text"))
        dlg.kind_var.set("head")
        dlg._apply_kind()
        pump(self.root, 0.2)
        self.assertFalse(dlg.aug_frame.winfo_ismapped())
        dlg.win.destroy()

    def test_the_augmentation_picked_in_the_window_goes_into_the_job(self):
        dlg = self.open()
        with mock.patch("utils.AugmentationPanel.ask_augmentation", return_value=self._picked("blur", "noise")) as ask:
            dlg.augment.button.invoke()
        self.assertTrue(ask.call_args.kwargs["training"], "the picker greys out what Ultralytics already does")
        text = dlg.augment.label.cget("text")
        self.assertIn("Blur", text)
        self.assertIn("Noise", text)
        self.assertIn("grows from 9 to 27", text, "9 training pictures (70% of 12), each with 2 copies besides itself")
        self.run_to_end(dlg)
        aug = self.job()["augment"]
        self.assertEqual((aug["copies"], aug["keep_original"], aug["splits"], aug["seed"]), (2, True, ["train"], 4))
        self.assertTrue(aug["ops"]["blur"]["enabled"] and aug["ops"]["noise"]["enabled"])
        self.assertFalse(aug["ops"]["flip_h"]["enabled"])
        dlg.win.destroy()

    def test_the_preparing_phase_shows_what_is_going_on_with_a_bar(self):
        dlg = self.open()
        dlg.save()
        dlg._handle('@@ {"event": "prepare", "done": 20, "total": 40, "text": "Step 3/3  Preparing the pictures"}')
        self.assertEqual(dlg.title_label.cget("text"), "Preparing the data ...")
        self.assertIn("Step 3/3", dlg.status_label.cget("text"))
        self.assertEqual((float(dlg.bar.cget("maximum")), float(dlg.bar.cget("value"))), (40.0, 20.0))
        self.assertIn("20 of 40 pictures", dlg.stats_label.cget("text"))
        pump_until(self.root, lambda: dlg.finished, 30)
        self.assertEqual(dlg.title_label.cget("text"), "Training finished", "back to training once it starts and ends")
        dlg.win.destroy()

    def test_the_full_training_has_the_split_bar_and_the_head_keeps_its_box(self):
        dlg = self.open()
        self.assertTrue(dlg.split_box.winfo_ismapped())
        self.assertFalse(dlg.val_row.winfo_ismapped())
        self.assertEqual((dlg.split.train_pct, dlg.split.valid_pct, dlg.split.test_pct), (70, 20, 10))
        self.assertIn("9 to train, 2 to validate, 1 to test", dlg.data_info.cget("text"))
        dlg.split.train_var.set("50")
        dlg.split.valid_var.set("30")
        pump(self.root, 0.1)
        self.assertIn("6 to train, 4 to validate, 2 to test", dlg.data_info.cget("text"))
        dlg.kind_var.set("head")
        dlg._apply_kind()
        pump(self.root, 0.2)
        self.assertFalse(dlg.split_box.winfo_ismapped())
        self.assertTrue(dlg.val_row.winfo_ismapped())
        self.assertNotIn("test", dlg.data_info.cget("text"))
        dlg.win.destroy()

    def test_the_split_chosen_in_the_bar_goes_into_the_job(self):
        dlg = self.open()
        dlg.split.train_var.set("60")
        dlg.split.valid_var.set("30")
        dlg.shuffle_var.set(True)
        self.run_to_end(dlg)
        job = self.job()
        self.assertEqual((job["val_fraction"], round(job["test_fraction"], 2), job["shuffle"]), (0.3, 0.1, True))
        self.assertIn("split_seed", job)
        dlg.win.destroy()

    def test_a_split_without_validation_pictures_is_refused(self):
        dlg = self.open()
        dlg.split.valid_var.set("0")
        with mock.patch("utils.HeadTrainDialog.messagebox.showwarning") as warn:
            dlg.save()
        self.assertIn("validation", warn.call_args[0][1])
        self.assertIsNone(dlg.proc)
        dlg.win.destroy()

    def _add_null_pictures(self, how_many=4):
        """Pictures saved with nothing in them: an empty label file next to the picture (plus 2 never annotated)."""
        import cv2
        import numpy as np

        def remove():
            for folder in (self.images, self.labels):
                for f in os.listdir(folder):
                    if f.startswith("null_"):
                        os.remove(os.path.join(folder, f))
        self.addCleanup(remove)                                # the folder is shared with the other tests
        for i in range(how_many + 2):
            name = f"null_{i:02d}"
            cv2.imwrite(os.path.join(self.images, name + ".png"), np.full((40, 60, 3), 90, np.uint8))
            if i < how_many:                                   # reviewed, nothing in it
                open(os.path.join(self.labels, name + ".txt"), "w").close()
        return how_many

    def test_pictures_saved_with_nothing_in_them_are_offered_as_background(self):
        nulls = self._add_null_pictures(4)
        dlg = self.open()
        self.assertEqual(len(dlg.pairs), 12, "the pictures with boxes")
        self.assertEqual(len(dlg.background), nulls, "the 2 never annotated ones are not background")
        self.assertTrue(dlg.bg_var.get())
        self.assertIn("12 labelled images + 4 with nothing in them (background)", dlg.data_info.cget("text"))
        self.assertEqual(dlg.split.total, 16)
        text = " ".join(str(w.cget("text")) for w in walk(dlg.win) if isinstance(w, tk.Checkbutton))
        self.assertIn("4 pictures with nothing in them as background", text)
        dlg.bg_var.set(False)
        self.assertEqual(dlg.split.total, 12)
        self.assertNotIn("background", dlg.data_info.cget("text"))
        dlg.bg_var.set(True)
        self.run_to_end(dlg)
        self.assertTrue(self.job()["background"])
        dlg.win.destroy()

    def test_background_can_be_switched_off_and_a_workspace_without_it_has_no_box(self):
        self._add_null_pictures(3)
        dlg = self.open()
        dlg.bg_var.set(False)
        self.run_to_end(dlg)
        self.assertNotIn("background", self.job())
        dlg.win.destroy()
        for name in os.listdir(self.labels):
            if name.startswith("null_"):
                os.remove(os.path.join(self.labels, name))
        dlg = self.open()
        self.assertEqual(len(dlg.background), 0)
        self.assertFalse(dlg.bg_check.winfo_ismapped(), "no such pictures: no box")
        dlg.win.destroy()

    # ---------------------------------------------------------------- the same window in every mode
    def _full_only_window(self, **kw):
        from utils.HeadTrainDialog import HeadTrainDialog
        started = []
        dlg = HeadTrainDialog(self.root, [self.images], self.labels, ctx.model_dir, command=self.command,
                              model_path=self.model_file, on_done=lambda _i: None, full_only=True,
                              on_start=started.append, **kw)
        dlg.started = started
        pump(self.root, 0.2)
        return dlg

    def test_outside_custom_model_the_window_is_the_full_training_without_the_head_choice(self):
        dlg = self._full_only_window()
        self.assertEqual(dlg.kind_var.get(), "yolo")
        self.assertEqual([w for w in walk(dlg.win) if isinstance(w, tk.Radiobutton)], [], "no head / full choice")
        self.assertEqual(dlg.det_title.cget("text"), "BASE MODEL")
        self.assertTrue(dlg.split_box.winfo_ismapped())
        self.assertTrue(dlg.aug_frame.winfo_ismapped())
        self.assertFalse(dlg.taps_row.winfo_ismapped())
        for part in (dlg.sq_var, dlg.epochs_var, dlg.batch_var, dlg.freeze_var, dlg.lr_var, dlg.wd_var, dlg.base_var):
            self.assertIsNotNone(part.get())
        self.assertEqual(dlg.wd_var.get(), "0.0005")
        dlg.win.destroy()

    def test_the_window_lets_go_of_the_annotation_window_once_training_starts(self):
        dlg = self._full_only_window()
        self.assertEqual(dlg.win.grab_status(), "local", "modal while it asks")
        dlg.epochs_var.set(1)
        self.run_to_end(dlg)
        self.assertEqual(len(dlg.started), 1)
        self.assertIs(dlg.started[0], dlg, "on_start is told which window started")
        self.assertIsNone(dlg.win.grab_status(), "hours of training must not lock the annotation window")
        dlg.win.destroy()

    def test_weight_decay_is_checked_and_goes_into_the_job(self):
        dlg = self._full_only_window()
        dlg.wd_var.set("0.5")
        with mock.patch("utils.HeadTrainDialog.messagebox.showwarning") as warn:
            dlg.save()
        self.assertIn("Weight decay", warn.call_args[0][1])
        self.assertIsNone(dlg.proc)
        dlg.wd_var.set("abc")
        with mock.patch("utils.HeadTrainDialog.messagebox.showwarning") as warn:
            dlg.save()
        self.assertIn("Weight decay", warn.call_args[0][1])
        dlg.wd_var.set("0.001")
        dlg.epochs_var.set(1)
        self.run_to_end(dlg)
        self.assertEqual(self.job()["weight_decay"], 0.001)
        dlg.win.destroy()

    def test_an_existing_model_is_continued_by_default_and_the_choice_is_still_there(self):
        self.assertFalse(os.path.isfile(self.model_file))
        dlg = self._full_only_window()
        self.assertFalse(dlg.continue_var.get(), "nothing to continue from yet")
        dlg.win.destroy()
        with open(self.model_file, "wb") as f:
            f.write(b"old model")
        dlg = self._full_only_window()
        self.assertTrue(dlg.continue_var.get(), "keeps what it learned, as the plain Train button always did")
        dlg.epochs_var.set(1)
        self.run_to_end(dlg)
        self.assertEqual(self.job()["init_weights"], self.model_file)
        dlg.win.destroy()

    # ---------------------------------------------------------------- counting only on request
    UNCOUNTED = {"images": 0, "instances": {}, "labelled": 0, "background": 0, "boxes": [], "counted": None}

    def test_nothing_is_counted_to_open_the_window(self):
        from utils import workspace_counts as wcounts
        with mock.patch.object(wcounts, "recount", side_effect=AssertionError("counted")), \
                mock.patch.object(wcounts, "compute", side_effect=AssertionError("counted")), \
                mock.patch("utils.head_data.scan_labels", side_effect=AssertionError("scanned")):
            dlg = self._full_only_window(counts=dict(self.UNCOUNTED))
        self.assertIn("not counted yet", dlg.data_info.cget("text"))
        self.assertIn("Press Refresh", dlg.counted_label.cget("text"))
        self.assertEqual(dlg.split.total, 0)
        dlg.win.destroy()

    def test_training_can_start_without_a_count_because_the_run_counts_for_itself(self):
        dlg = self._full_only_window(counts=dict(self.UNCOUNTED))
        dlg.epochs_var.set(1)
        self.run_to_end(dlg)                                    # no "at least 10 images" refusal from a number of 0
        self.assertEqual(len(dlg.started), 1)
        dlg.win.destroy()

    def test_refresh_counts_again_and_fills_the_window(self):
        stats = {"images": 40, "instances": {}, "labelled": 25, "background": 5, "boxes": [15, 10],
                 "counted": 1700000000.0}
        dlg = self._full_only_window(counts=dict(self.UNCOUNTED))
        self.assertFalse(dlg.bg_check.winfo_ismapped(), "no pictures with nothing in them known yet")
        with mock.patch("utils.workspace_counts.recount", return_value=stats) as counted:
            dlg.refresh_btn.invoke()
            self.assertEqual(str(dlg.refresh_btn.cget("state")), "disabled", "busy while it counts")
            self.assertTrue(pump_until(self.root, lambda: not dlg._counting, 10))
        self.assertEqual(counted.call_count, 1)
        self.assertEqual(str(dlg.refresh_btn.cget("state")), "normal")
        self.assertIn("25 labelled images + 5 with nothing in them", dlg.data_info.cget("text"))
        self.assertEqual(dlg.split.total, 30)
        self.assertTrue(dlg.bg_check.winfo_ismapped(), "the background box appears with the new pictures")
        self.assertTrue(dlg.bg_var.get())
        self.assertIn("Counted", dlg.counted_label.cget("text"))
        dlg.win.destroy()

    def test_a_refresh_that_fails_says_so_and_keeps_the_old_numbers(self):
        dlg = self._full_only_window(counts={**self.UNCOUNTED, "labelled": 12, "counted": 1700000000.0})
        with mock.patch("utils.workspace_counts.recount", side_effect=OSError("disk gone")):
            dlg.refresh_btn.invoke()
            self.assertTrue(pump_until(self.root, lambda: not dlg._counting, 10))
        self.assertIn("Could not count: disk gone", dlg.counted_label.cget("text"))
        self.assertEqual(dlg.n_labelled, 12)
        self.assertEqual(str(dlg.refresh_btn.cget("state")), "normal")
        dlg.win.destroy()

    def test_no_augmentation_means_no_augment_key_in_the_job(self):
        dlg = self.open()
        self.run_to_end(dlg)
        self.assertNotIn("augment", self.job())
        dlg.win.destroy()

    def test_cancelling_the_picker_keeps_the_earlier_picks(self):
        dlg = self.open()
        with mock.patch("utils.AugmentationPanel.ask_augmentation", return_value=self._picked("blur")):
            dlg.augment.button.invoke()
        with mock.patch("utils.AugmentationPanel.ask_augmentation", return_value=None) as ask:
            dlg.augment.button.invoke()
        self.assertIsNotNone(ask.call_args.kwargs["previous"], "the picker starts from the earlier picks")
        self.assertIn("Blur", dlg.augment.label.cget("text"))
        dlg.win.destroy()

    def test_choosing_full_training_swaps_the_fields_and_choosing_head_swaps_them_back(self):
        dlg = self.open(full=False)
        dlg.kind_var.set("yolo")
        dlg._apply_kind()
        pump(self.root, 0.2)
        self.assertEqual(dlg.det_title.cget("text"), "BASE MODEL")
        self.assertTrue(dlg.intro.cget("text").startswith("Trains a whole YOLO model"))
        self.assertEqual(dlg.base_var.get(), "yolo11s.pt")
        self.assertEqual(str(dlg.weights_box.cget("textvariable")), str(dlg.base_var))
        self.assertIn("yolo11n.pt", dlg.weights_box.cget("values"))
        self.assertFalse(dlg.taps_row.winfo_ismapped(), "taps belong to the head")
        self.assertTrue(dlg.continue_chk.winfo_ismapped())
        self.assertEqual(str(dlg.continue_chk.cget("state")), "disabled", "no trained model here yet")
        self.assertTrue(all(w.winfo_ismapped() for w in dlg._yolo_cells))
        self.assertFalse(any(w.winfo_ismapped() for w in dlg._head_cells))
        self.assertEqual((dlg.sq_var.get(), dlg.freeze_var.get(), dlg.epochs_var.get(), dlg.batch_var.get()),
                         (640, 0, 50, 16))
        dlg.kind_var.set("head")
        dlg._apply_kind()
        pump(self.root, 0.2)
        self.assertEqual(dlg.det_title.cget("text"), "DETECTOR")
        self.assertTrue(dlg.taps_row.winfo_ismapped())
        self.assertTrue(all(w.winfo_ismapped() for w in dlg._head_cells))
        self.assertFalse(any(w.winfo_ismapped() for w in dlg._yolo_cells))
        dlg.win.destroy()

    def test_a_polygon_workspace_gets_the_segmentation_models(self):
        dlg = self.open(task="seg")
        self.assertEqual(dlg.base_var.get(), "yolo11s-seg.pt")
        self.assertTrue(all("-seg" in v for v in dlg.weights_box.cget("values")))
        self.assertIn("polygon", dlg.det_note.cget("text"))
        dlg.win.destroy()

    def test_continue_is_offered_once_a_model_has_been_trained(self):
        with open(self.model_file, "wb") as f:
            f.write(b"old model")
        dlg = self.open()
        self.assertEqual(str(dlg.continue_chk.cget("state")), "normal")
        dlg.win.destroy()

    def test_bad_settings_are_explained_and_nothing_starts(self):
        dlg = self.open()
        dlg.sq_var.set(100)
        dlg.save()
        self.assertIn("multiple of 32", self.rec.of("showwarning")[-1][2])
        dlg.sq_var.set(640)
        dlg.freeze_var.set(200)
        dlg.save()
        self.assertIn("Freeze", self.rec.of("showwarning")[-1][2])
        dlg.freeze_var.set(0)
        dlg.lr_var.set("5")
        dlg.save()
        self.assertIn("Learning rate", self.rec.of("showwarning")[-1][2])
        dlg.lr_var.set("0.001")
        dlg.base_var.set(os.path.join(ctx.root, "nope", "base.pt"))
        dlg.save()
        self.assertIn("not found", self.rec.of("showwarning")[-1][2])
        self.assertIsNone(dlg.proc)
        dlg.win.destroy()

    # ---------------------------------------------------------------- a run
    def test_a_run_shows_a_row_per_epoch_and_installs_the_model_as_my_trained_model(self):
        dlg = self.open()
        dlg.epochs_var.set(3)
        self.run_to_end(dlg)
        rows = [dlg.table.item(i) for i in dlg.table.get_children()]
        self.assertEqual(len(rows), 3, "one table row per epoch")
        self.assertEqual([r["values"][0] for r in rows], [1, 2, 3])
        self.assertEqual([c[1] for c in __import__("utils.HeadTrainDialog", fromlist=["x"]).YOLO_COLUMNS][:4],
                         ["Epoch", "Box", "Cls", "DFL"])
        self.assertEqual([dlg.table.heading(c, "text") for c in dlg.table["columns"]],
                         ["Epoch", "Box", "Cls", "DFL", "Precision", "Recall", "mAP50", "mAP50-95", "Time"])
        self.assertEqual(rows[0]["values"][1:4], ["1.000", "0.500", "0.900"])
        self.assertEqual(rows[2]["values"][6:8], ["0.300", "0.150"])
        self.assertIn("best", rows[0]["tags"])
        self.assertNotIn("best", rows[1]["tags"])
        self.assertIn("Epoch 3 / 3", dlg.stats_label.cget("text"))
        self.assertIn("mAP50", dlg.stats_label.cget("text"))
        self.assertIn("some plain log line", dlg.log.get("1.0", "end"))
        self.assertEqual(self.read_bytes(self.model_file), b"model-3")
        a = wc.get_assistant(WORKSPACE)
        self.assertEqual((a["provider"], a["custom_model"]["path"]), (wc.PROVIDER_CUSTOM, ""))
        self.assertEqual(dlg.title_label.cget("text"), "Training finished")
        self.assertIn("My trained model", dlg.status_label.cget("text"))
        self.assertIn("YOLO-World", dlg.status_label.cget("text"))
        self.assertEqual(float(dlg.bar.cget("value")), 3.0)
        self.assertEqual((dlg.close_btn.cget("state"), dlg.stop_btn.cget("state")), ("normal", "disabled"))
        self.assertEqual(dlg.done_calls, [self.model_file])
        dlg.win.destroy()

    def test_the_job_asks_for_every_layer_from_the_chosen_base_model(self):
        dlg = self.open()
        dlg.epochs_var.set(1)
        dlg.base_var.set("yolo11m.pt")
        dlg.sq_var.set(512)
        self.run_to_end(dlg)
        job = self.job()
        self.assertEqual((job["kind"], job["init_weights"], job["imgsz"], job["epochs"], job["batch"], job["freeze"]),
                         ("yolo", "yolo11m.pt", 512, 1, 16, 0))
        self.assertEqual(job["classes"], config.class_manager.get_classes())
        self.assertEqual(job["val_fraction"], 0.2)
        self.assertTrue(job["scratch_dir"].endswith("dataset"))
        self.assertTrue(job["out_dir"].endswith("yolo_run"))
        dlg.win.destroy()

    def test_freeze_is_passed_on_when_the_user_asks_for_it(self):
        dlg = self.open()
        dlg.epochs_var.set(1)
        dlg.freeze_var.set(10)
        self.run_to_end(dlg)
        self.assertEqual(self.job()["freeze"], 10)
        dlg.win.destroy()

    def test_a_new_model_keeps_the_previous_one_after_asking(self):
        with open(self.model_file, "wb") as f:
            f.write(b"old model")
        dlg = self.open()
        dlg.continue_var.set(False)                              # on by default when a model is here
        dlg.epochs_var.set(1)
        self.run_to_end(dlg)
        self.assertTrue(self.rec.of("askyesno"), "asks before replacing a trained model")
        self.assertEqual(self.read_bytes(self.prev_file), b"old model")
        self.assertEqual(self.read_bytes(self.model_file), b"model-1")
        dlg.win.destroy()

    def test_declining_the_replacement_starts_nothing(self):
        with open(self.model_file, "wb") as f:
            f.write(b"old model")
        dlg = self.open()
        dlg.continue_var.set(False)
        with mock.patch("utils.HeadTrainDialog.messagebox.askyesno", return_value=False):
            dlg.save()
        self.assertIsNone(dlg.proc)
        self.assertEqual(self.read_bytes(self.model_file), b"old model")
        dlg.win.destroy()

    def test_continue_trains_on_from_the_model_already_here(self):
        with open(self.model_file, "wb") as f:
            f.write(b"old model")
        dlg = self.open()
        dlg.continue_var.set(True)
        dlg.epochs_var.set(1)
        before = len(self.rec.of("askyesno"))
        self.run_to_end(dlg)
        self.assertEqual(self.job()["init_weights"], self.model_file)
        self.assertEqual(len(self.rec.of("askyesno")), before, "continuing replaces nothing, so nothing to ask")
        dlg.win.destroy()

    def test_a_trainer_error_is_shown_and_nothing_is_installed(self):
        dlg = self.open()
        with mock.patch.dict(os.environ, {"FAKE_TRAIN_MODE": "error"}):
            self.run_to_end(dlg)
        self.assertEqual(dlg.title_label.cget("text"), "Training failed")
        self.assertIn("CUDA out of memory", dlg.status_label.cget("text"))
        self.assertFalse(os.path.exists(self.model_file))
        self.assertEqual(wc.get_assistant(WORKSPACE)["provider"], wc.PROVIDER_YOLO_WORLD, "nothing was selected")
        dlg.win.destroy()

    def test_a_crash_without_a_message_still_ends_cleanly(self):
        dlg = self.open()
        with mock.patch.dict(os.environ, {"FAKE_TRAIN_MODE": "crash"}):
            self.run_to_end(dlg)
        self.assertEqual(dlg.title_label.cget("text"), "Training ended unexpectedly")
        self.assertIn("yolo_run/train.log", dlg.status_label.cget("text"))
        self.assertFalse(os.path.exists(self.model_file))
        dlg.win.destroy()

    def test_stop_keeps_the_best_model_so_far(self):
        dlg = self.open()
        dlg.epochs_var.set(5)
        with mock.patch.dict(os.environ, {"FAKE_TRAIN_MODE": "slow"}):
            dlg.save()
            self.assertTrue(pump_until(self.root, lambda: float(dlg.bar.cget("value")) >= 1, 20), "epoch 1 never came")
            dlg._stop()
            self.assertTrue(pump_until(self.root, lambda: dlg.finished, 20))
        self.assertEqual(self.read_bytes(self.model_file), b"model-1")
        self.assertIn("best model so far was kept", dlg.title_label.cget("text"))
        self.assertEqual(wc.get_assistant(WORKSPACE)["provider"], wc.PROVIDER_CUSTOM)
        dlg.win.destroy()

    def test_finish_early_wraps_the_full_training_up_and_installs_the_model(self):
        dlg = self.open()
        dlg.epochs_var.set(5)
        with mock.patch.dict(os.environ, {"FAKE_TRAIN_MODE": "graceful"}):
            dlg.save()
            self.addCleanup(lambda: dlg.proc.stop())
            dlg._finish_early()
            self.assertEqual((dlg.finish_btn.cget("text"), str(dlg.finish_btn.cget("state"))), ("Finishing ...", "disabled"))
            self.assertTrue(pump_until(self.root, lambda: dlg.finished, 30), "the run never ended")
        self.assertEqual(dlg.title_label.cget("text"), "Finished early")
        self.assertIn("then evaluated", dlg.status_label.cget("text"))
        self.assertIn("My trained model", dlg.status_label.cget("text"))
        self.assertEqual(self.read_bytes(self.model_file), b"model-1")
        self.assertEqual(len(dlg.table.get_children()), 1)
        self.assertEqual(float(dlg.bar.cget("value")), 1.0)
        self.assertFalse(os.path.exists(os.path.join(ctx.model_dir, "yolo_run", "finish_early.flag")))
        self.assertEqual(wc.get_assistant(WORKSPACE)["provider"], wc.PROVIDER_CUSTOM)
        self.assertEqual(dlg.done_calls, [self.model_file])
        dlg.win.destroy()

    def test_stop_now_after_finish_early_still_ends_it_at_once(self):
        dlg = self.open()
        dlg.epochs_var.set(5)
        with mock.patch.dict(os.environ, {"FAKE_TRAIN_MODE": "slow"}):
            dlg.save()
            self.assertTrue(pump_until(self.root, lambda: float(dlg.bar.cget("value")) >= 1, 20))
            dlg._finish_early()                            # the slow fake ignores the file
            dlg._stop()
            self.assertTrue(pump_until(self.root, lambda: dlg.finished, 20))
        self.assertIn("kept", dlg.title_label.cget("text"))
        self.assertNotEqual(dlg.title_label.cget("text"), "Finished early")
        dlg.win.destroy()

    def test_the_job_names_the_file_the_finish_early_button_creates(self):
        dlg = self.open()
        dlg.epochs_var.set(1)
        self.run_to_end(dlg)
        self.assertEqual(self.job()["stop_file"], os.path.join(ctx.model_dir, "yolo_run", "finish_early.flag"))
        dlg.win.destroy()

    def test_the_progress_window_always_shows_its_buttons_whatever_the_text_says(self):
        dlg = self.open()
        dlg.epochs_var.set(3)
        dlg.save()
        buttons_fit(self, dlg)
        self.assertTrue(pump_until(self.root, lambda: dlg.finished, 30))
        buttons_fit(self, dlg)                                    # the long ending text is in, the table has rows
        self.assertEqual(len(dlg.table.get_children()), 3)
        dlg.win.destroy()

    def test_the_head_run_still_has_a_table_too(self):
        import sys
        self.script_head = os.path.join(ctx.root, "fake_trainer.py")
        with open(self.script_head, "w") as f:
            f.write(FAKE_TRAINER)
        self.command = lambda job_path: [sys.executable, "-u", self.script_head, job_path]
        dlg = self.open(full=False)
        dlg.epochs_var.set(2)
        self.run_to_end(dlg)
        self.assertEqual(len(dlg.table.get_children()), 2)
        self.assertEqual([dlg.table.heading(c, "text") for c in dlg.table["columns"]],
                         ["Epoch", "Loss", "Accuracy", "Macro-F1", "Time"])
        dlg.win.destroy()

    # ---------------------------------------------------------------- the pieces without a window
    def test_the_default_command_runs_the_trainer_the_job_asks_for(self):
        import json
        from utils import HeadTrainDialog as htd
        folder = os.path.join(ctx.root, "jobs")
        os.makedirs(folder, exist_ok=True)
        head, full, junk = (os.path.join(folder, n) for n in ("head.json", "full.json", "junk.json"))
        with open(head, "w") as f:
            json.dump({"classes": []}, f)
        with open(full, "w") as f:
            json.dump({"kind": "yolo"}, f)
        with open(junk, "w") as f:
            f.write("not json")
        self.assertEqual(htd.default_command(head)[-2:], ["utils.head_train", head])
        self.assertEqual(htd.default_command(full)[-2:], ["utils.yolo_train_job", full])
        self.assertEqual(htd.default_command(junk)[-2], "utils.head_train")
        self.assertEqual(htd.default_command(os.path.join(folder, "missing.json"))[-2], "utils.head_train")

    def test_install_yolo_copies_the_best_model_and_keeps_the_old_one(self):
        from utils import HeadTrainDialog as htd
        best = os.path.join(ctx.root, "best_test.pt")
        with open(best, "wb") as f:
            f.write(b"new")
        target = os.path.join(ctx.root, "installed", "modelAssistant.pt")
        self.assertIsNone(htd.install_yolo(os.path.join(ctx.root, "missing.pt"), target))
        self.assertIsNone(htd.install_yolo(None, target))
        self.assertEqual(htd.install_yolo(best, target), target)
        self.assertEqual(self.read_bytes(target), b"new")
        with open(best, "wb") as f:
            f.write(b"newer")
        htd.install_yolo(best, target)
        self.assertEqual(self.read_bytes(os.path.join(ctx.root, "installed", "modelAssistant.prev.pt")), b"new")
        self.assertEqual(self.read_bytes(target), b"newer")

    def test_the_start_button_of_the_annotation_window_passes_the_model_path_and_the_task(self):
        # wired in AnnotationGUI._train_head: the window needs both to offer full training
        with open(os.path.join(os.path.dirname(__file__), "..", "..", "utils", "AnnotationGUI.py"), encoding="utf-8") as f:
            text = f.read()
        self.assertIn('model_path=model_path, task="seg" if state.annotation_mode == "polygon" else "detect"', text)


if __name__ == "__main__":
    unittest.main()
