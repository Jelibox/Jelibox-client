"""Builds the real windows against the isolated temp workspace and pokes them.
Skipped automatically when there is no display (e.g. a headless CI box)."""
import os
import re
import time
import tkinter as tk
from tkinter import ttk
import unittest
from unittest import mock

from tests.helpers import isolated_workspace, WORKSPACE

ctx = isolated_workspace()

from utils import app_settings, config                          # noqa: E402
from utils import theme                                          # noqa: E402
from utils import workspace_config as wc                         # noqa: E402

try:
    _probe = tk.Tk()
    _probe.destroy()
    HAVE_DISPLAY = True
except tk.TclError:
    HAVE_DISPLAY = False

if HAVE_DISPLAY:
    from tkinter import messagebox                               # noqa: E402
    from utils.AnnotationGUI import AnnotationGUI                # noqa: E402


def pump(root, seconds=0.3):
    end = time.time() + seconds
    while time.time() < end:
        root.update()
        time.sleep(0.01)


def walk(widget):
    yield widget
    for child in widget.winfo_children():
        yield from walk(child)


def buttons(root):
    return [w for w in walk(root) if isinstance(w, tk.Button)]


def find_button(root, text):
    for b in buttons(root):
        if re.search(rf"(?<![A-Za-z]){re.escape(text)}(?![A-Za-z])", b.cget("text")):
            return b
    raise AssertionError(f"no button '{text}' among {[b.cget('text') for b in buttons(root)]}")


class Dialogs:
    """Replaces the blocking message boxes with recorders."""

    def __init__(self):
        self.calls = []
        names = {"showinfo": None, "showwarning": None, "showerror": None, "askyesno": True, "askokcancel": True}
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

    def last(self, kind):
        return [c for c in self.calls if c[0] == kind][-1]


@unittest.skipUnless(HAVE_DISPLAY, "no display available")
class MainWindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dialogs = Dialogs()
        cls.dialogs.__enter__()
        cls.root = tk.Tk()
        cls.root.geometry("1500x850+0+0")
        cls.gui = AnnotationGUI(cls.root)
        pump(cls.root, 1.0)

    @classmethod
    def tearDownClass(cls):
        cls.dialogs.__exit__()
        try:
            cls.root.destroy()
        except tk.TclError:
            pass

    def setUp(self):
        config.state.bboxes, config.state.polygons = [], []
        self.dialogs.calls.clear()

    # ------------------------------------------------------------------ layout
    def test_window_title_and_branding(self):
        self.assertTrue(self.root.title().startswith("Jelibox"))
        texts = " ".join(str(w.cget("text")) for w in walk(self.root) if isinstance(w, tk.Label) and w.cget("text"))
        self.assertIn("Jelibox", texts)
        self.assertNotIn("Boxify", texts.replace("BOXIFY", "Boxify"))

    def test_every_header_button_exists(self):
        for label in ("Prev", "Next", "Repeat", "Infer", "Delete", "Stream", "Mode", "ZeroFill: OFF",
                      "Label Assistant", "Train", "Export Model", "Export Dataset"):
            find_button(self.root, label)

    def test_quick_actions_sit_by_navigation_and_config_buttons_on_the_right(self):
        x = lambda t: find_button(self.root, t).winfo_rootx()
        self.assertLess(x("Prev"), x("Next"))
        self.assertLess(x("Next"), x("Repeat"))
        self.assertLess(x("Repeat"), x("ZeroFill: OFF"))
        self.assertLess(x("ZeroFill: OFF"), x("Label Assistant"))
        self.assertLess(x("Label Assistant"), x("Export Dataset"))

    def test_button_hierarchy_colors(self):
        bg = lambda t: find_button(self.root, t).cget("bg")
        self.assertEqual(bg("Infer"), theme.C_ACCENT)                  # the one primary action
        self.assertEqual(bg("Delete"), theme.C_DANGER_BG)              # destructive
        for plain in ("Repeat", "Stream", "Mode", "Label Assistant", "Train", "Export Model", "Prev", "Next"):
            self.assertEqual(bg(plain), theme.C_CARD2, plain)

    def test_keyboard_shortcuts_are_bound(self):
        bound = set(self.root.bind())
        for seq in ("b", "m", "g", "t", "e", "r", "s", "1", "9",
                    "<Key-Delete>", "<Key-Escape>", "<Key-Return>", "<Key-Left>", "<Key-Right>"):
            self.assertIn(seq, bound)

    # ---------------------------------------------------------------- behaviours
    def test_label_text_button_toggles_without_any_popup(self):
        btn = self.gui.text_btn
        self.assertEqual(btn.cget("text"), "Hide label text")
        btn.invoke()
        self.assertEqual(btn.cget("text"), "Show label text")
        self.assertFalse(config.state.show_bbox_text)
        btn.invoke()
        self.assertEqual(btn.cget("text"), "Hide label text")
        self.assertTrue(config.state.show_bbox_text)
        self.assertEqual(self.dialogs.calls, [], "label text toggle must not show any notification")

    def test_zerofill_button_toggles_state_and_look(self):
        btn = self.gui.mask_btn
        btn.invoke()
        self.assertIn("ON", btn.cget("text"))
        self.assertEqual(btn.cget("bg"), theme.C_ACCENT)
        btn.invoke()
        self.assertIn("OFF", btn.cget("text"))
        self.assertEqual(btn.cget("bg"), theme.C_CARD2)

    def test_mode_button_switches_bbox_and_polygon(self):
        self.assertEqual(config.state.annotation_mode, "bbox")
        find_button(self.root, "Mode").invoke()
        self.assertEqual(config.state.annotation_mode, "polygon")
        self.assertIn("POLYGON", self.gui.statusbar_mode.cget("text"))
        find_button(self.root, "Mode").invoke()
        self.assertEqual(config.state.annotation_mode, "bbox")
        self.assertIn("BBOX", self.gui.statusbar_mode.cget("text"))

    def test_force_indicator_lives_in_the_status_bar(self):
        self.gui.toggle_force_new_bbox()
        self.assertEqual(self.gui.force_label.cget("text"), "Force: ON")
        self.gui.toggle_force_new_bbox()
        self.assertEqual(self.gui.force_label.cget("text"), "Force: OFF")
        self.assertFalse(hasattr(self.gui, "mode_label"), "the redundant top-right mode badge is gone")
        self.assertFalse(hasattr(self.gui, "text_label"), "the redundant 'Text: ON' badge is gone")

    def test_class_list_uses_the_calm_class_palette(self):
        lb = self.gui.class_listbox
        self.assertEqual(lb.size(), 2)
        self.assertEqual(lb.itemcget(0, "background").lower(), "#e5566d")
        self.assertEqual(lb.itemcget(1, "background").lower(), "#e0a030")

    def test_image_list_is_populated(self):
        self.assertEqual(self.gui.image_listbox.size(), 3)

    def test_navigation_wraps_around_and_keys_work(self):
        config.state.current_index = 0
        self.gui.prev_image()
        self.assertEqual(config.state.current_index, 2)
        self.gui.next_image()
        self.assertEqual(config.state.current_index, 0)
        self.root.focus_force()
        pump(self.root, 0.1)
        self.root.event_generate("<KeyPress>", keysym="d")
        pump(self.root, 0.4)
        self.assertEqual(config.state.current_index, 1)
        self.root.event_generate("<KeyPress>", keysym="a")
        pump(self.root, 0.4)
        self.assertEqual(config.state.current_index, 0)

    # ------------------------------------------------------------ undo / redo
    def _fresh_image(self, index=0):
        config.state.current_index = index
        self.gui.load_current_image()
        self.gui.update_display()

    def test_undo_redo_buttons_and_shortcuts_exist(self):
        find_button(self.root, "Undo")
        find_button(self.root, "Redo")
        bound = set(self.root.bind())
        for seq in ("<Control-Key-z>", "<Control-Key-y>"):
            self.assertIn(seq, bound)

    def test_undo_and_redo_bboxes_polygons_and_edits(self):
        self._fresh_image()
        g, s = self.gui, config.state
        g.push_undo()
        s.bboxes.append([10, 10, 80, 80, "cat"])
        g.push_undo()
        s.polygons.append([[(5, 5), (50, 5), (50, 50)], "cat"])
        g.undo()
        self.assertEqual((len(s.bboxes), len(s.polygons)), (1, 0))
        g.undo()
        self.assertEqual(len(s.bboxes), 0)
        g.undo()                                            # nothing left: harmless
        g.redo()
        g.redo()
        self.assertEqual((len(s.bboxes), len(s.polygons)), (1, 1))
        g.undo()
        g.push_undo()                                       # a new edit forks history
        self.assertEqual(g.redo_stack, [])

    def test_undo_takes_back_the_last_point_of_a_half_drawn_polygon(self):
        self._fresh_image()
        s = config.state
        s.polygon_points_preview = [(1, 1), (30, 1), (30, 30)]
        self.gui.undo()
        self.assertEqual(s.polygon_points_preview, [(1, 1), (30, 1)])
        s.polygon_points_preview = []

    def test_changing_image_clears_the_record(self):
        self._fresh_image(0)
        self.gui.push_undo()
        config.state.bboxes.append([10, 10, 80, 80, "cat"])
        self.gui.next_image()
        self.assertEqual((self.gui.undo_stack, self.gui.redo_stack), ([], []))
        self.gui.undo()
        self.assertEqual(config.state.bboxes, [])           # nothing leaked into image 2
        self._fresh_image(0)

    def test_zerofill_is_pending_until_the_image_changes(self):
        import cv2
        self._fresh_image(0)
        g = self.gui
        path = os.path.join(ctx.instance_dir, "img1.png")
        before = cv2.imread(path).copy()
        g.mask_mode = True
        g.mask_polygon_points = [(20, 20), (120, 20), (120, 120), (20, 120)]
        g.finish_mask_polygon()
        g.mask_mode = False
        self.assertEqual(len(g.masked_regions), 1)
        self.assertTrue((cv2.imread(path) == before).all(), "file must not change yet")
        self.assertEqual(g._base_array()[60, 60].tolist(), [0, 0, 0], "but it shows on screen")
        g.undo()
        self.assertEqual(g.masked_regions, [])
        self.assertTrue((cv2.imread(path) == before).all())
        g.redo()
        self.assertEqual(len(g.masked_regions), 1)
        g.next_image()                                      # commit happens here
        self.assertEqual(g.masked_regions, [])
        self.assertEqual(cv2.imread(path)[60, 60].tolist(), [0, 0, 0])
        self.assertNotEqual(cv2.imread(path)[200, 300].tolist(), [0, 0, 0])
        self._fresh_image(0)

    # ------------------------------------------------- export with augmentation
    def _augmenter(self, splits=("train",), copies=1):
        from utils.augmentation import AugmentConfig, Augmenter, OpSetting
        return Augmenter(AugmentConfig(
            ops={"brightness": OpSetting(True, 25, 100), "blur": OpSetting(True, 3, 100)},
            copies=copies, splits=splits, seed=11))

    def _label_the_test_images(self):
        import utils.AnnotationGUI as agui
        os.makedirs(agui.yolo_labels_folder, exist_ok=True)
        os.makedirs(agui.vocdataset_folder, exist_ok=True)
        for name in ("img1", "img2", "img3"):
            with open(os.path.join(agui.yolo_labels_folder, name + ".txt"), "w") as f:
                f.write("0 0.5 0.5 0.4 0.4\n")
            with open(os.path.join(agui.vocdataset_folder, name + ".xml"), "w") as f:
                f.write(f"<annotation><filename>{name}.png</filename><path>/somewhere/{name}.png</path>"
                        "<size><width>320</width><height>240</height><depth>3</depth></size>"
                        "<object><name>cat</name><bndbox><xmin>10</xmin><ymin>10</ymin>"
                        "<xmax>100</xmax><ymax>100</ymax></bndbox></object></annotation>")

    def _target(self):
        import tempfile
        return tempfile.mkdtemp(prefix="jelibox_export_")

    def test_yolo_export_adds_augmented_copies_with_identical_labels(self):
        self._label_the_test_images()
        target = self._target()
        aug = self._augmenter()
        self.gui._export_yolo_dataset(target, 100, 0, (".png",), augment=aug)
        imgs = sorted(os.listdir(os.path.join(target, "train", "images")))
        lbls = sorted(os.listdir(os.path.join(target, "train", "labels")))
        self.assertEqual(imgs, ["img1.png", "img1_aug1.png", "img2.png", "img2_aug1.png", "img3.png", "img3_aug1.png"])
        self.assertEqual(lbls, [n.replace(".png", ".txt") for n in imgs])
        def read(n):
            with open(os.path.join(target, "train", "labels", n)) as f:
                return f.read()
        self.assertEqual(read("img1.txt"), read("img1_aug1.txt"))
        self.assertEqual(aug.created, 3)

    def test_yolo_export_without_augmentation_is_unchanged(self):
        self._label_the_test_images()
        target = self._target()
        self.gui._export_yolo_dataset(target, 100, 0, (".png",))
        self.assertEqual(len(os.listdir(os.path.join(target, "train", "images"))), 3)

    def test_valid_and_test_are_only_augmented_when_asked(self):
        self._label_the_test_images()
        names = lambda t, s: os.listdir(os.path.join(t, s, "images"))
        only_train = self._target()
        self.gui._export_yolo_dataset(only_train, 34, 33, (".png",), augment=self._augmenter())
        self.assertEqual((len(names(only_train, "train")), len(names(only_train, "val")), len(names(only_train, "test"))),
                         (2, 1, 1))
        everything = self._target()
        self.gui._export_yolo_dataset(everything, 34, 33, (".png",),
                                      augment=self._augmenter(splits=("train", "val", "test")))
        self.assertEqual((len(names(everything, "train")), len(names(everything, "val")), len(names(everything, "test"))),
                         (2, 2, 2))                        # 3 images split 1/1/1, each gets one copy

    def test_voc_export_copies_xml_and_points_it_at_the_new_image(self):
        import xml.etree.ElementTree as ET
        self._label_the_test_images()
        target = self._target()
        self.gui._export_voc_dataset(target, 100, 0, (".png",), augment=self._augmenter())
        files = sorted(os.listdir(os.path.join(target, "train")))
        self.assertIn("img1_aug1.png", files)
        self.assertIn("img1_aug1.xml", files)
        root = ET.parse(os.path.join(target, "train", "img1_aug1.xml")).getroot()
        self.assertEqual(root.findtext("filename"), "img1_aug1.png")
        self.assertEqual(root.findtext("path"), "img1_aug1.png")
        self.assertEqual(root.findtext("object/bndbox/xmax"), "100")

    def test_coco_export_duplicates_annotations_under_new_image_ids(self):
        import json
        self._label_the_test_images()
        target = self._target()
        self.gui._export_coco_dataset(target, 100, 0, (".png",), augment=self._augmenter())
        with open(os.path.join(target, "annotations", "instances_train.json"), encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(len(data["images"]), 6)
        self.assertEqual(len(data["annotations"]), 6)
        ids = [i["id"] for i in data["images"]]
        self.assertEqual(len(set(ids)), 6)
        self.assertEqual({a["image_id"] for a in data["annotations"]}, set(ids))
        self.assertEqual(len({a["id"] for a in data["annotations"]}), 6)
        by_name = {i["file_name"]: i["id"] for i in data["images"]}
        boxes = {a["image_id"]: a["bbox"] for a in data["annotations"]}
        self.assertEqual(boxes[by_name["img1.png"]], boxes[by_name["img1_aug1.png"]])
        for i in data["images"]:
            self.assertTrue(os.path.exists(os.path.join(target, "images", "train", i["file_name"])))

    def test_export_dialog_augmentation_panel(self):
        opened = []
        real = tk.Toplevel
        with mock.patch.object(tk, "Toplevel", side_effect=lambda *a, **k: opened.append(real(*a, **k)) or opened[-1]):
            self.gui.show_export_dataset_dialog()
        dlg = opened[0]
        try:
            pump(self.root, 0.2)
            shown = lambda: " ".join(str(w.cget("text")) for w in walk(dlg)
                                     if isinstance(w, (tk.Checkbutton, tk.Label)) and w.winfo_ismapped())
            labels = shown()
            self.assertIn("Augment exported images", labels)
            self.assertNotIn("Brightness", labels)             # panel starts collapsed
            toggle = next(w for w in walk(dlg) if isinstance(w, tk.Checkbutton)
                          and "Augment exported images" in str(w.cget("text")))
            toggle.invoke()
            pump(self.root, 0.2)
            labels = shown()
            for op in ("Brightness", "Contrast", "Saturation", "Hue", "Blur", "Noise", "Grayscale", "JPEG compression", "Effect", "Value", "Chance"):
                self.assertIn(op, labels)
            self.assertLess(dlg.winfo_reqheight(), 800)

            tabs = next(w for w in walk(dlg) if isinstance(w, ttk.Notebook))
            self.assertEqual(len(tabs.tabs()), 2)
            self.assertIn("(2)", tabs.tab(0, "text"))           # brightness + blur are ticked by default
            tabs.select(1)
            pump(self.root, 0.2)
            labels = shown()
            for op in ("Flip horizontal", "Flip vertical", "Rotation"):
                self.assertIn(op, labels)
            rotation_box = next(w for w in walk(dlg) if isinstance(w, tk.Checkbutton)
                                and w.cget("text") == "Rotation")
            rotation_box.invoke()
            self.assertIn("(1)", tabs.tab(1, "text"))
        finally:
            dlg.destroy()

    def test_yolo_export_flips_and_rotates_labels_with_the_pictures(self):
        from utils.augmentation import AugmentConfig, Augmenter, OpSetting
        import cv2
        self._label_the_test_images()
        flipper = Augmenter(AugmentConfig(ops={"flip_h": OpSetting(True, 0, 100)},
                                          copies=1, splits=("train",), seed=1))
        target = self._target()
        self.gui._export_yolo_dataset(target, 100, 0, (".png",), augment=flipper)
        def label(n):
            with open(os.path.join(target, "train", "labels", n), encoding="utf-8") as f:
                return f.read().split()
        self.assertEqual(label("img1.txt"), "0 0.5 0.5 0.4 0.4".split())
        self.assertEqual(label("img1_aug1.txt"), "0 0.500000 0.500000 0.400000 0.400000".split())

        rotator = Augmenter(AugmentConfig(ops={"rotate": OpSetting(True, 90, 100)},
                                          copies=1, splits=("train",), seed=2))
        target = self._target()
        self.gui._export_yolo_dataset(target, 100, 0, (".png",), augment=rotator)
        for name in ("img1", "img2", "img3"):
            img = cv2.imread(os.path.join(target, "train", "images", name + "_aug1.png"))
            self.assertGreaterEqual(img.shape[0], 240)            # canvas never shrinks below the original
            parts = label(name + "_aug1.txt")
            self.assertEqual(len(parts), 5)
            self.assertTrue(all(0.0 <= float(v) <= 1.0 for v in parts[1:]))

    def test_voc_and_coco_exports_report_the_rotated_image_size(self):
        import json
        import xml.etree.ElementTree as ET
        import cv2
        from utils.augmentation import AugmentConfig, Augmenter, OpSetting
        self._label_the_test_images()
        make = lambda: Augmenter(AugmentConfig(ops={"rotate": OpSetting(True, 45, 100)},
                                               copies=1, splits=("train",), seed=4))
        target = self._target()
        self.gui._export_voc_dataset(target, 100, 0, (".png",), augment=make())
        img = cv2.imread(os.path.join(target, "train", "img1_aug1.png"))
        root = ET.parse(os.path.join(target, "train", "img1_aug1.xml")).getroot()
        self.assertEqual((int(root.findtext("size/width")), int(root.findtext("size/height"))),
                         (img.shape[1], img.shape[0]))
        xmax = int(root.findtext("object/bndbox/xmax"))
        self.assertTrue(0 < xmax <= img.shape[1])

        target = self._target()
        self.gui._export_coco_dataset(target, 100, 0, (".png",), augment=make())
        with open(os.path.join(target, "annotations", "instances_train.json"), encoding="utf-8") as f:
            data = json.load(f)
        for entry in data["images"]:
            img = cv2.imread(os.path.join(target, "images", "train", entry["file_name"]))
            self.assertEqual((entry["width"], entry["height"]), (img.shape[1], img.shape[0]))
        by_id = {i["id"]: i for i in data["images"]}
        for ann in data["annotations"]:
            x, y, w, h = ann["bbox"]
            self.assertLessEqual(x + w, by_id[ann["image_id"]]["width"] + 1e-6)
            self.assertLessEqual(y + h, by_id[ann["image_id"]]["height"] + 1e-6)
        self.assertEqual(len(data["annotations"]), 6)

    def test_theme_button_requests_a_restart_and_resumes_on_the_same_image(self):
        before = theme.MODE
        config.state.current_index = 2                       # img3.png, untouched
        xml3 = os.path.join(ctx.voc, "img3.xml")
        if os.path.exists(xml3):
            os.remove(xml3)
        with mock.patch.object(self.gui, "close_main_gui") as closer:
            self.gui.toggle_theme()
        closer.assert_called_once()
        self.assertTrue(getattr(self.root, "_jelibox_restart", False))
        self.assertEqual(app_settings.get("theme"), "dark" if before == "light" else "light")
        resume = app_settings.get("resume")
        self.assertEqual(resume["image"], "img3.png")
        self.assertFalse(os.path.exists(xml3), "an untouched image must not be marked as annotated by a restart")
        # restore
        app_settings.set("theme", before)
        app_settings.set("resume", None)
        self.root._jelibox_restart = False
        config.state.current_index = 0

    def test_theme_restart_saves_real_work(self):
        config.state.current_index = 1
        self.gui.load_current_image()
        config.state.bboxes = [[10, 10, 80, 80, "cat"]]
        with mock.patch.object(self.gui, "close_main_gui"):
            self.gui.toggle_theme()
        self.assertTrue(os.path.exists(os.path.join(ctx.voc, "img2.xml")))
        app_settings.set("theme", theme.MODE)
        app_settings.set("resume", None)
        self.root._jelibox_restart = False
        config.state.current_index = 0


@unittest.skipUnless(HAVE_DISPLAY, "no display available")
class LabelAssistantDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dialogs = Dialogs()
        cls.dialogs.__enter__()
        cls.root = tk.Tk()
        cls.root.geometry("1200x800+0+0")

    @classmethod
    def tearDownClass(cls):
        cls.dialogs.__exit__()
        cls.root.destroy()

    def setUp(self):
        self.dialogs.calls.clear()
        if os.path.exists(ctx.model_file):
            os.remove(ctx.model_file)
        a = wc.get_assistant(WORKSPACE)
        a.update(provider=wc.PROVIDER_YOLO_WORLD, confidence=0.3)
        a["yolo_world"].update(model="yolov8s-world", target_classes=[])
        wc.set_assistant(WORKSPACE, a)

    def open(self):
        from utils.LabelAssistantDialog import LabelAssistantDialog
        dlg = LabelAssistantDialog(self.root)
        pump(self.root, 0.2)
        self.addCleanup(lambda: dlg.win.winfo_exists() and dlg.win.destroy())
        return dlg

    def test_defaults_to_yolo_world_and_disables_trained_model_when_missing(self):
        dlg = self.open()
        self.assertEqual(dlg.provider_var.get(), wc.PROVIDER_YOLO_WORLD)
        radios = [w for w in walk(dlg.win) if isinstance(w, tk.Radiobutton)]
        trained = [r for r in radios if "trained" in r.cget("text")][0]
        self.assertEqual(str(trained.cget("state")), "disabled")
        shown = " ".join(str(w.cget("text")) for w in walk(dlg.win) if isinstance(w, tk.Label))
        self.assertIn("Your model was not found in this workspace.", shown)

    def test_trained_model_option_unlocks_when_the_model_exists(self):
        os.makedirs(ctx.model_dir, exist_ok=True)
        open(ctx.model_file, "wb").close()
        dlg = self.open()
        radios = [w for w in walk(dlg.win) if isinstance(w, tk.Radiobutton)]
        trained = [r for r in radios if "trained" in r.cget("text")][0]
        self.assertEqual(str(trained.cget("state")), "normal")
        trained.invoke()
        self.assertEqual(dlg.provider_var.get(), wc.PROVIDER_CUSTOM)
        self.assertEqual(str(dlg.model_combo.cget("state")), "disabled", "YOLO-World controls dim when not selected")
        dlg._save()
        self.assertEqual(wc.get_assistant(WORKSPACE)["provider"], wc.PROVIDER_CUSTOM)

    def test_dropdown_offers_the_eight_models(self):
        dlg = self.open()
        self.assertEqual(list(dlg.model_combo.cget("values")), wc.YOLO_WORLD_MODELS)

    def test_prompt_entry_rejects_digits_and_symbols_but_allows_spaces(self):
        dlg = self.open()
        entry = dlg.rows[0]["entry"]
        entry.insert(0, "white horse")
        entry.insert(tk.END, "2")
        entry.insert(tk.END, "-")
        entry.insert(tk.END, "!")
        self.assertEqual(entry.get(), "white horse")

    def test_add_and_remove_target_rows(self):
        dlg = self.open()
        n = len(dlg.rows)
        dlg.add_btn.invoke()
        self.assertEqual(len(dlg.rows), n + 1)
        dlg.rows[-1]["x"].invoke()
        self.assertEqual(len(dlg.rows), n)

    def test_class_dropdown_lists_workspace_classes(self):
        dlg = self.open()
        self.assertEqual(list(dlg.rows[0]["combo"].cget("values")), ["cat", "dog"])

    def test_saving_targets_writes_them_to_the_config(self):
        dlg = self.open()
        dlg.rows[0]["prompt"].set("tabby cat")
        dlg.rows[0]["cls"].set("cat")
        dlg.add_btn.invoke()
        dlg.rows[1]["prompt"].set("puppy")
        dlg.rows[1]["cls"].set("dog")
        dlg.conf_var.set(0.45)
        dlg.model_var.set("yolov8x-worldv2")
        dlg._save()
        a = wc.get_assistant(WORKSPACE)
        self.assertEqual(a["yolo_world"]["target_classes"],
                         [{"prompt": "tabby cat", "map_to": "cat"}, {"prompt": "puppy", "map_to": "dog"}])
        self.assertEqual(a["yolo_world"]["model"], "yolov8x-worldv2")
        self.assertAlmostEqual(a["confidence"], 0.45)

    def test_incomplete_row_blocks_saving(self):
        dlg = self.open()
        dlg.rows[0]["prompt"].set("tabby cat")           # no class chosen
        dlg._save()
        self.assertEqual(self.dialogs.last("showwarning")[0], "showwarning")
        self.assertTrue(dlg.win.winfo_exists(), "dialog stays open")
        self.assertEqual(wc.get_assistant(WORKSPACE)["yolo_world"]["target_classes"], [])

    def test_no_target_class_warns_in_english_but_still_saves(self):
        dlg = self.open()
        dlg._save()
        title, msg = self.dialogs.last("showwarning")[1:]
        self.assertIn("can't work without a target class", msg)
        self.assertIn("keep annotating", msg)
        self.assertFalse(dlg.win.winfo_exists())


@unittest.skipUnless(HAVE_DISPLAY, "no display available")
class OtherWindowTests(unittest.TestCase):
    def setUp(self):
        self.dialogs = Dialogs()
        self.dialogs.__enter__()
        self.root = tk.Tk()
        self.root.geometry("1100x700+0+0")

    def tearDown(self):
        self.dialogs.__exit__()
        self.root.destroy()

    def test_workspace_picker_lists_workspaces_and_has_theme_toggle(self):
        from utils.WorkspacePicker import WorkspacePickerApp
        WorkspacePickerApp(self.root, entry_script=os.path.join(os.getcwd(), "x.py"))
        pump(self.root, 0.5)
        texts = " ".join(str(w.cget("text")) for w in walk(self.root) if isinstance(w, (tk.Label, tk.Button)))
        self.assertIn(WORKSPACE, texts)
        self.assertIn("Jelibox", texts)
        self.assertTrue(self.root.title().startswith("Jelibox"))
        find_button(self.root, "Dark" if theme.MODE == "light" else "Light")

    def test_open_folder_button_sits_after_import_dataset_and_opens_the_install_folder(self):
        import sys
        from utils.WorkspacePicker import WorkspacePickerApp
        app = WorkspacePickerApp(self.root, entry_script=os.path.join(os.getcwd(), "x.py"))
        pump(self.root, 0.3)
        self.assertLess(find_button(self.root, "Import Dataset").winfo_rootx(),
                        find_button(self.root, "Open Folder").winfo_rootx())
        target = "os.startfile" if sys.platform == "win32" else "subprocess.Popen"
        with mock.patch(target, create=True) as opener:
            find_button(self.root, "Open Folder").invoke()
        opener.assert_called_once()
        shown = opener.call_args[0][0]
        shown = os.path.abspath(shown if isinstance(shown, str) else shown[-1])
        self.assertEqual(shown, os.path.abspath(ctx.root))
        self.assertEqual(self.dialogs.calls, [])
        with mock.patch(target, create=True, side_effect=OSError("no file manager")):
            find_button(self.root, "Open Folder").invoke()
        self.assertEqual(len(self.dialogs.calls), 1)

    def _picker(self, flag):
        from utils.WorkspacePicker import WorkspacePickerApp
        env = {"JELIBOX_COLLAB": "1"} if flag else {}
        with mock.patch.dict(os.environ, env):
            if not flag:
                os.environ.pop("JELIBOX_COLLAB", None)
            app = WorkspacePickerApp(self.root, entry_script=os.path.join(os.getcwd(), "x.py"))
        pump(self.root, 0.3)
        return app

    def test_server_gear_is_hidden_unless_the_feature_is_on(self):
        app_settings.set("collab_enabled", None)
        self.assertIsNone(self._picker(False).server_btn)

    def test_server_gear_opens_the_settings_dialog(self):
        app = self._picker(True)
        self.assertIsNotNone(app.server_btn)
        app.server_btn.invoke()
        pump(self.root, 0.3)
        dialogs = [w for w in self.root.winfo_children() if isinstance(w, tk.Toplevel)]
        self.assertEqual(len(dialogs), 1)
        self.assertEqual(dialogs[0].title(), "Server settings")

    def test_server_dialog_adds_edits_and_removes_a_server(self):
        from utils.ServerSettingsDialog import ServerSettingsDialog
        from utils.collab import servers, identity
        for e in servers.list_servers():
            servers.remove_server(e["id"])
        dlg = ServerSettingsDialog(self.root)
        pump(self.root, 0.2)

        texts = lambda: " ".join(str(w.cget("text")) for w in walk(dlg.win) if isinstance(w, (tk.Label, tk.Button)))
        self.assertIn(identity.get_device_id(), texts())
        self.assertIn("No servers yet", texts())

        dlg.url_var.set("not a valid address")
        dlg.save_btn.invoke()
        self.assertIn("✗", dlg.status.cget("text"))
        self.assertEqual(servers.list_servers(), [])

        dlg.name_var.set("Lab")
        dlg.url_var.set("192.168.1.10")
        dlg.save_btn.invoke()
        pump(self.root, 0.2)
        self.assertEqual([e["url"] for e in servers.list_servers()], ["http://192.168.1.10:8420"])
        self.assertIn("Not connected", texts())

        entry = servers.list_servers()[0]
        dlg._edit(entry)
        dlg.url_var.set("192.168.1.11:9000")
        dlg.save_btn.invoke()
        self.assertEqual(servers.list_servers()[0]["url"], "http://192.168.1.11:9000")

        servers.store_access_key(entry["id"], "jbk_TESTKEY00000000")
        dlg.refresh()
        self.assertIn("Connected", texts())
        self.assertNotIn("jbk_TESTKEY", texts())                           # a key is never shown

        dlg._remove(entry)                                                 # the stubbed confirmation answers yes
        self.assertEqual(servers.list_servers(), [])
        self.assertIsNone(servers.get_access_key(entry["id"]))
        dlg.win.destroy()

    def test_unsupported_display_dialog_is_english_and_closes(self):
        from utils import ScreenGuard
        self.root.withdraw()
        self.root.after(400, lambda: [w.destroy() for w in self.root.winfo_children() if isinstance(w, tk.Toplevel)])
        ok, reason, w, h = ScreenGuard.check_screen(type("R", (), {"winfo_screenwidth": lambda s: 390,
                                                                   "winfo_screenheight": lambda s: 844})())
        self.assertFalse(ok)
        ScreenGuard.show_unsupported_dialog(self.root, reason, w, h)      # returns once the dialog is closed

    def test_window_style_setup_does_not_break_a_window(self):
        from utils import window_style
        window_style.set_app_id()
        window_style.setup(self.root)
        pump(self.root, 0.3)
        self.assertTrue(self.root.winfo_exists())


if __name__ == "__main__":
    unittest.main()
