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
    return [b for b in out if text is None or re.search(re.escape(text), b.cget("text"))]


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

    def test_the_front_menu_offers_the_three_modes(self):
        app = self.picker()
        self.assertIsNotNone(app.mode_menu)
        text = label_texts(app.mode_menu)
        for title in ("YOLO-World", "LocateAnything", "Custom head"):
            self.assertIn(title, text)
        self.assertEqual(len(buttons(app.mode_menu)), 3)

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
    print("some plain log line", flush=True)
    with open(os.path.join(job["out_dir"], "head_best.pt"), "wb") as f:
        f.write(b"head-%d" % i)
    ev("epoch", epoch=i, epochs=job["epochs"], loss=1.0 / i, acc=0.5, f1=0.4 + i / 10, best=True,
       recall=[0.5, 0.6], support=[5, 4], seconds=0.01)
ev("done", best_f1=0.9, best_epoch=job["epochs"], epochs_run=job["epochs"], stopped=False, out_dir=job["out_dir"],
   has_head=True, classes=job["classes"], last_recall=[0.5, 0.6], last_support=[5, 4])
'''


@unittest.skipUnless(HAVE_DISPLAY, "no display available")
class HeadTrainDialogTests(ModeGuiBase):
    """The Train button of the Custom head mode: setup window -> progress window -> head installed."""

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
    """Custom head mode: the Browse custom model button next to the detector weights."""

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

    def tearDown(self):
        app_settings.set("mode", self._mode)

    def test_the_header_has_an_annotate_all_button_and_a_shortcut(self):
        self.assertTrue(buttons(self.root, "Annotate all"))
        self.assertIn("G", set(self.root.bind()))

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
        with mock.patch("utils.AnnotationGUI.open_head_training") as head,                 mock.patch("utils.AnnotationGUI.TrainingConfigDialog") as yolo:
            self.gui.start_training()
        head.assert_called_once()
        args = head.call_args[0]
        self.assertEqual(args[2], config.yolo_labels_folder)
        self.assertFalse(yolo.called, "no whole-YOLO training in this mode")

    def test_train_in_the_other_modes_is_still_the_yolo_training(self):
        am.set_mode(am.MODE_YOLO_WORLD)
        dialog = mock.Mock()
        dialog.show.return_value = None                      # the user cancels
        with mock.patch("utils.AnnotationGUI.open_head_training") as head,                 mock.patch("utils.AnnotationGUI.TrainingConfigDialog", return_value=dialog) as yolo:
            self.gui.start_training()
        self.assertTrue(yolo.called)
        self.assertFalse(head.called)


if __name__ == "__main__":
    unittest.main()
