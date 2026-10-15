"""The augmentation picker: tiles with previews, a settings window per tile, the preview picture and the result."""
import gc
import os
import time
import tkinter as tk
import unittest
from unittest import mock

from tests.helpers import isolated_workspace, WORKSPACE

ctx = isolated_workspace()

from utils import augment_catalog as cat                          # noqa: E402
from utils import augment_sample as sample                        # noqa: E402
from utils import workspace_config as wc                          # noqa: E402

try:
    _probe = tk.Tk()
    _probe.destroy()
    HAVE_DISPLAY = True
except tk.TclError:
    HAVE_DISPLAY = False


def pump(root, seconds=0.2):
    end = time.time() + seconds
    while time.time() < end:
        root.update()
        time.sleep(0.01)


def walk(widget):
    yield widget
    for child in widget.winfo_children():
        yield from walk(child)


def texts(widget):
    out = []
    for w in walk(widget):
        if isinstance(w, (tk.Label, tk.Button, tk.Checkbutton)):
            out.append(str(w.cget("text")))
    return "\n".join(out)


@unittest.skipUnless(HAVE_DISPLAY, "no display available")
class PanelTests(unittest.TestCase):
    training = False

    def setUp(self):
        wc.set_preview_image(WORKSPACE, "")
        self.addCleanup(wc.set_preview_image, WORKSPACE, "")
        self.root = tk.Tk()
        self.root.geometry("1000x800+0+0")
        self.addCleanup(gc.collect)
        self.addCleanup(self.root.destroy)
        from utils.AugmentationPanel import AugmentationPanel
        self.changes = []
        self.panel = AugmentationPanel(self.root, WORKSPACE, training=self.training, on_change=lambda: self.changes.append(1))
        self.panel.pack(fill=tk.BOTH, expand=True)
        pump(self.root)

    def tile_dialog(self, key):
        self.panel.open_tile(cat.TILE_BY_KEY[key])
        pump(self.root)
        dialogs = [w for w in walk(self.root) if isinstance(w, tk.Toplevel) and w is not self.root]
        self.assertEqual(len(dialogs), 1, "the tile's settings window opened")
        self.addCleanup(lambda d=dialogs[0]: d.winfo_exists() and d.destroy())
        return dialogs[0]

    def button(self, dialog, text):
        return next(w for w in walk(dialog) if isinstance(w, tk.Button) and w.cget("text") == text)

    def checks(self, dialog):
        return {w.cget("text"): w for w in walk(dialog) if isinstance(w, tk.Checkbutton)}


class GridTests(PanelTests):
    def test_there_is_a_tile_with_a_picture_for_every_augmentation(self):
        self.assertEqual(len(self.panel._tiles), len(cat.TILES))
        self.assertEqual(len(self.panel._photos), len(cat.TILES))
        text = texts(self.panel)
        for title in ("Flip", "90° Rotate", "Rotation", "Shear", "Grayscale", "Hue", "Saturation", "Brightness",
                      "Exposure", "Blur", "Motion Blur", "Noise", "Cutout"):
            self.assertIn(title, text)
        self.assertNotIn("Built into Ultralytics", text, "that note belongs to the training window")

    def test_the_first_picture_is_the_default_preview_and_there_is_nothing_to_reset(self):
        self.assertTrue(self.panel.sample_path.endswith("img1.png"))
        self.assertIn("img1.png  (the first picture)", texts(self.panel))
        self.assertEqual(str(self.panel.reset_btn.cget("state")), "disabled")

    def test_nothing_is_selected_at_first(self):
        self.assertEqual(self.panel.status.cget("text"), "No augmentation selected.")
        self.assertIsNone(self.panel.get_config())

    def test_outputs_default_to_3_and_keep_the_original(self):
        self.assertEqual((self.panel.outputs(), self.panel.keep_var.get()), (3, True))

    def test_a_choice_is_remembered_and_the_first_picture_comes_back_on_request(self):
        chosen = os.path.join(ctx.instance_dir, "img3.png")
        with mock.patch("utils.AugmentationPanel.filedialog.askopenfilename", return_value=chosen):
            self.panel.choose_sample()
        self.assertEqual(self.panel.sample_path, chosen)
        self.assertIn("img3.png  (your choice)", texts(self.panel))
        self.assertEqual(sample.pick_sample(WORKSPACE), (chosen, False))
        self.assertEqual(str(self.panel.reset_btn.cget("state")), "normal")
        self.panel.reset_sample()
        self.assertTrue(self.panel.sample_path.endswith("img1.png"))
        self.assertEqual(wc.get_preview_image(WORKSPACE), "")

    def test_a_new_panel_uses_the_saved_choice_and_falls_back_when_the_file_is_gone(self):
        from utils.AugmentationPanel import AugmentationPanel
        chosen = os.path.join(ctx.instance_dir, "img2.png")
        sample.remember_sample(WORKSPACE, chosen)
        again = AugmentationPanel(self.root, WORKSPACE)
        self.assertEqual(again.sample_path, chosen)
        sample.remember_sample(WORKSPACE, os.path.join(ctx.instance_dir, "gone.png"))
        third = AugmentationPanel(self.root, WORKSPACE)
        self.assertTrue(third.sample_path.endswith("img1.png"))
        self.assertEqual(str(third.reset_btn.cget("state")), "disabled")

    def test_a_file_that_is_not_a_picture_is_refused(self):
        junk = os.path.join(ctx.root, "junk.png")
        with open(junk, "wb") as f:
            f.write(b"not a picture")
        with mock.patch("utils.AugmentationPanel.filedialog.askopenfilename", return_value=junk), \
                mock.patch("utils.AugmentationPanel.messagebox.showwarning") as warn:
            self.panel.choose_sample()
        warn.assert_called_once()
        self.assertTrue(self.panel.sample_path.endswith("img1.png"))
        self.assertEqual(wc.get_preview_image(WORKSPACE), "")

    def test_cancelling_the_file_chooser_changes_nothing(self):
        with mock.patch("utils.AugmentationPanel.filedialog.askopenfilename", return_value=""):
            self.panel.choose_sample()
        self.assertTrue(self.panel.sample_path.endswith("img1.png"))


class TileDialogTests(PanelTests):
    def test_apply_picks_the_tile_with_the_slider_value(self):
        dialog = self.tile_dialog("brightness")
        scale = next(w for w in walk(dialog) if isinstance(w, tk.Scale) and w.cget("to") == 100.0)
        scale.set(30)
        pump(self.root, 0.2)
        self.button(dialog, "Apply").invoke()
        pump(self.root)
        self.assertTrue(self.panel.state["brightness"].enabled)
        self.assertEqual(self.panel.state["brightness"].magnitude, 30)
        self.assertIn("1 selected: Brightness ±30%", self.panel.status.cget("text"))
        self.assertIn("Brightness  ✓", texts(self.panel))
        self.assertTrue(self.changes)
        config = self.panel.get_config()
        self.assertEqual((config.copies, config.keep_original), (2, True))
        self.assertTrue(config.ops["brightness"].enabled)

    def test_the_window_shows_the_original_and_both_examples_and_they_follow_the_slider(self):
        dialog = self.tile_dialog("brightness")
        labels = [str(w.cget("text")) for w in walk(dialog) if isinstance(w, tk.Label)]
        for must in ("original", "-25%", "25%", "Between -N% and +N%"):
            self.assertIn(must, labels)
        scale = next(w for w in walk(dialog) if isinstance(w, tk.Scale) and w.cget("to") == 100.0)
        scale.set(40)
        pump(self.root, 0.3)
        labels = [str(w.cget("text")) for w in walk(dialog) if isinstance(w, tk.Label)]
        self.assertIn("-40%", labels)
        self.assertIn("40%", labels)

    def test_go_back_throws_the_changes_away(self):
        dialog = self.tile_dialog("blur")
        scale = next(w for w in walk(dialog) if isinstance(w, tk.Scale) and w.cget("to") == 25.0)
        scale.set(9)
        self.button(dialog, "Go Back").invoke()
        pump(self.root)
        self.assertFalse(self.panel.state["blur"].enabled)
        self.assertEqual(self.panel.state["blur"].magnitude, cat.default_state()["blur"].magnitude)
        self.assertEqual(self.panel.status.cget("text"), "No augmentation selected.")

    def test_brightness_can_be_limited_to_darkening(self):
        dialog = self.tile_dialog("brightness")
        self.checks(dialog)["Brighten"].invoke()
        self.button(dialog, "Apply").invoke()
        pump(self.root)
        self.assertEqual(self.panel.state["brightness"].options, {"brighten": False, "darken": True})

    def test_a_tile_cannot_be_applied_with_every_choice_unticked(self):
        dialog = self.tile_dialog("brightness")
        for box in self.checks(dialog).values():
            box.invoke()
        pump(self.root)
        self.assertIn("Tick at least one choice.", texts(dialog))
        self.button(dialog, "Apply").invoke()
        pump(self.root)
        self.assertTrue(dialog.winfo_exists(), "still open")
        self.assertFalse(self.panel.state["brightness"].enabled)

    def test_flip_has_a_tick_for_each_direction(self):
        dialog = self.tile_dialog("flip")
        boxes = self.checks(dialog)
        self.assertEqual(set(boxes), {"Horizontal (left-right)", "Vertical (top-bottom)"})
        boxes["Vertical (top-bottom)"].invoke()
        self.button(dialog, "Apply").invoke()
        pump(self.root)
        self.assertTrue(self.panel.state["flip_h"].enabled)
        self.assertFalse(self.panel.state["flip_v"].enabled)
        self.assertIn("1 selected: Flip", self.panel.status.cget("text"))

    def test_90_degree_rotate_offers_the_three_turns(self):
        dialog = self.tile_dialog("rot90")
        boxes = self.checks(dialog)
        self.assertEqual(set(boxes), {"Clockwise", "Counter-clockwise", "Upside down"})
        boxes["Upside down"].invoke()
        self.button(dialog, "Apply").invoke()
        pump(self.root)
        self.assertEqual(self.panel.state["rot90"].options, {"cw": True, "ccw": True, "upside_down": False})
        labels = [str(w.cget("text")) for w in walk(self.root) if isinstance(w, tk.Label)]
        self.assertIn("90° Rotate  ✓", labels)

    def test_cutout_has_a_number_of_boxes(self):
        dialog = self.tile_dialog("cutout")
        spin = next(w for w in walk(dialog) if isinstance(w, tk.Spinbox))
        dialog.setvar(str(spin.cget("textvariable")), "6")
        self.button(dialog, "Apply").invoke()
        pump(self.root)
        self.assertEqual(self.panel.state["cutout"].options["count"], 6)

    def test_the_chance_slider_sets_the_probability(self):
        dialog = self.tile_dialog("noise")
        scale = next(w for w in walk(dialog) if isinstance(w, tk.Scale) and w.cget("to") == 100.0)
        scale.set(80)
        self.button(dialog, "Apply").invoke()
        pump(self.root)
        self.assertEqual(self.panel.state["noise"].probability, 80)

    def test_remove_switches_a_picked_tile_off_again(self):
        dialog = self.tile_dialog("grayscale")
        self.button(dialog, "Apply").invoke()
        pump(self.root)
        self.assertTrue(self.panel.state["grayscale"].enabled)
        dialog = self.tile_dialog("grayscale")
        self.button(dialog, "Remove").invoke()
        pump(self.root)
        self.assertFalse(self.panel.state["grayscale"].enabled)
        self.assertEqual(self.panel.status.cget("text"), "No augmentation selected.")

    def test_remove_is_greyed_out_for_a_tile_that_is_not_picked(self):
        dialog = self.tile_dialog("hue")
        self.assertEqual(str(self.button(dialog, "Remove").cget("state")), "disabled")

    def test_every_tile_opens(self):
        for tile in cat.TILES:
            dialog = self.tile_dialog(tile["key"])
            self.assertIn(tile["description"][:30], texts(dialog))
            dialog.destroy()
            pump(self.root, 0.05)


class OutputsTests(PanelTests):
    def pick(self, key):
        dialog = self.tile_dialog(key)
        self.button(dialog, "Apply").invoke()
        pump(self.root)

    def test_the_summary_follows_outputs_and_keep_original(self):
        self.pick("blur")
        self.assertIn("Each training image becomes 3 pictures: the original and 2 augmented.", self.panel.detail.cget("text"))
        self.panel.keep_var.set(False)
        self.assertIn("is replaced by 3 augmented pictures", self.panel.detail.cget("text"))
        self.panel.outputs_var.set(1)
        self.panel.keep_var.set(True)
        self.assertIn("adds nothing", self.panel.detail.cget("text"))
        self.assertEqual(self.panel.get_config().copies, 0)

    def test_the_config_has_the_number_of_copies_and_the_seed(self):
        self.pick("blur")
        self.panel.outputs_var.set(5)
        self.panel.seed_var.set("42")
        config = self.panel.get_config()
        self.assertEqual((config.copies, config.seed, config.splits), (4, 42, ("train",)))

    def test_a_seed_that_is_not_a_number_is_refused(self):
        self.pick("blur")
        self.panel.seed_var.set("abc")
        with self.assertRaisesRegex(ValueError, "seed"):
            self.panel.get_config()

    def test_the_extra_line_is_shown_next_to_the_summary(self):
        self.panel.extra_line = lambda: "The training part grows from 20 to 60 pictures."
        self.pick("blur")
        self.assertIn("The training part grows from 20 to 60 pictures.", self.panel.detail.cget("text"))


class TrainingModeTests(PanelTests):
    training = True

    def test_the_tiles_ultralytics_already_does_are_marked_and_cannot_be_opened(self):
        badges = [w for w in walk(self.panel) if isinstance(w, tk.Label) and w.cget("text") == "Built into Ultralytics"]
        self.assertEqual(len(badges), 6)
        with mock.patch("utils.AugmentationPanel.TileDialog") as dialog:
            self.panel.open_tile(cat.TILE_BY_KEY["flip"], cat.TILE_BY_KEY["flip"]["ultralytics"])
        dialog.assert_not_called()
        self.assertIn("fliplr", self.panel.note.cget("text"))
        self.assertIn("twice", self.panel.note.cget("text"))
        self.assertFalse(self.panel.state["flip_h"].enabled)

    def test_the_other_tiles_work_as_usual(self):
        dialog = self.tile_dialog("exposure")
        self.button(dialog, "Apply").invoke()
        pump(self.root)
        self.assertTrue(self.panel.state["exposure"].enabled)
        self.assertEqual(self.panel.note.cget("text"), "")


@unittest.skipUnless(HAVE_DISPLAY, "no display available")
class DialogTests(unittest.TestCase):
    def setUp(self):
        wc.set_preview_image(WORKSPACE, "")
        self.root = tk.Tk()
        self.root.geometry("900x700+0+0")
        self.addCleanup(gc.collect)
        self.addCleanup(self.root.destroy)

    def open(self, **kw):
        from utils.AugmentationPanel import AugmentationDialog
        dlg = AugmentationDialog(self.root, WORKSPACE, **kw)
        pump(self.root)
        return dlg

    def button(self, dlg, text):
        return next(w for w in walk(dlg) if isinstance(w, tk.Button) and w.cget("text") == text)

    def test_ok_returns_the_picks_and_cancel_returns_nothing(self):
        dlg = self.open()
        dlg.panel.state["noise"].enabled = True
        dlg.panel.redraw()
        self.button(dlg, "OK").invoke()
        self.assertTrue(dlg.result["config"].ops["noise"].enabled)
        self.assertEqual((dlg.result["outputs"], dlg.result["keep_original"], dlg.result["seed"]), (3, True, None))
        dlg = self.open()
        self.button(dlg, "Cancel").invoke()
        self.assertIsNone(dlg.result)

    def test_ok_with_nothing_picked_gives_no_config(self):
        dlg = self.open()
        self.button(dlg, "OK").invoke()
        self.assertIsNone(dlg.result["config"])

    def test_a_bad_seed_keeps_the_window_open(self):
        dlg = self.open()
        dlg.panel.state["noise"].enabled = True
        dlg.panel.seed_var.set("x")
        with mock.patch("utils.AugmentationPanel.messagebox.showwarning") as warn:
            self.button(dlg, "OK").invoke()
        warn.assert_called_once()
        self.assertTrue(dlg.winfo_exists())
        dlg.destroy()

    def test_clear_all_unpicks_everything(self):
        dlg = self.open()
        dlg.panel.state["noise"].enabled = True
        dlg.panel.redraw()
        self.button(dlg, "Clear all").invoke()
        self.assertEqual(dlg.panel.status.cget("text"), "No augmentation selected.")
        dlg.destroy()

    def test_an_earlier_result_is_the_starting_point(self):
        state = cat.default_state()
        state["blur"].enabled = True
        dlg = self.open(state=state, outputs=7, keep_original=False, seed=5)
        self.assertEqual((dlg.panel.outputs(), dlg.panel.keep_var.get(), dlg.panel.seed_var.get()), (7, False, "5"))
        self.assertIn("1 selected: Blur", dlg.panel.status.cget("text"))
        dlg.destroy()

    def test_training_mode_is_the_default_for_this_window(self):
        dlg = self.open()
        self.assertTrue(dlg.panel.training)
        dlg.destroy()


if __name__ == "__main__":
    unittest.main()
