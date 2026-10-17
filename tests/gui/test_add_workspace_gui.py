"""Add Workspace is also the way to import a dataset: it scans the folder, shows what it found (and what would be
skipped) before anything is copied, then imports."""
import os
import tempfile
import tkinter as tk
import unittest
from unittest import mock

from tests.gui.test_gui_smoke import Dialogs, pump, walk, find_button, buttons, label_of, HAVE_DISPLAY
from tests.helpers import make_images, WORKSPACE, isolated_workspace

ctx = isolated_workspace()

from utils import theme                                          # noqa: E402
from utils import workspace_config as wc                         # noqa: E402
from utils import workspace_manager as wm                        # noqa: E402

SIZE = (200, 100)


def voc_folder(objects_by_image):
    """A temp folder of images with a Pascal VOC XML each; objects_by_image: [[class, ...], ...]."""
    src = tempfile.mkdtemp()
    names = make_images(src, count=len(objects_by_image), size=SIZE)
    for name, classes in zip(names, objects_by_image):
        rows = "".join(f"<object><name>{c}</name><bndbox><xmin>{10 * i + 5}</xmin><ymin>5</ymin><xmax>{10 * i + 15}</xmax>"
                       f"<ymax>30</ymax></bndbox></object>" for i, c in enumerate(classes))
        with open(os.path.join(src, name.replace(".png", ".xml")), "w") as f:
            f.write(f"<annotation><filename>{name}</filename><size><width>{SIZE[0]}</width><height>{SIZE[1]}</height>"
                    f"<depth>3</depth></size>{rows}</annotation>")
    return src


@unittest.skipUnless(HAVE_DISPLAY, "no display available")
class AddWorkspaceBase(unittest.TestCase):
    """The window, its helpers and a clean-up; the tests are in the classes below."""
    def setUp(self):
        self.dialogs = Dialogs()
        self.dialogs.__enter__()
        self.root = tk.Tk()
        self.root.geometry("1100x700+0+0")
        from utils.WorkspacePicker import WorkspacePickerApp
        self.app = WorkspacePickerApp(self.root, entry_script=os.path.join(os.getcwd(), "x.py"))
        pump(self.root, 0.3)
        self.created = []

    def tearDown(self):
        for name in self.created:
            wm.delete_workspace(name)
        for inst in self.new_instances():
            wm.delete_instance(inst)
        self.dialogs.__exit__()
        self.root.destroy()

    def new_instances(self):
        return [i for i in wm.list_workspaces().get(WORKSPACE, []) if i != "testws-1"]

    def dialog(self):
        return [w for w in self.root.winfo_children() if isinstance(w, tk.Toplevel)][-1]

    def entries(self):
        return [w for w in walk(self.dialog()) if isinstance(w, tk.Entry)]

    def texts(self):
        return "\n".join(str(w.cget("text")) for w in walk(self.dialog()) if isinstance(w, tk.Label))

    def type_into(self, entry, text):
        entry.delete(0, tk.END)
        entry.insert(0, text)

    def browse(self, folder):
        with mock.patch("utils.WorkspacePicker.filedialog.askdirectory", return_value=folder):
            find_button(self.dialog(), "Browse").invoke()
        pump(self.root, 0.3)

@unittest.skipUnless(HAVE_DISPLAY, "no display available")
class AddWorkspaceImportTests(AddWorkspaceBase):
    def test_there_is_no_import_dataset_button_any_more(self):
        self.assertFalse([b for b in buttons(self.root) if "Import Dataset" in label_of(b)])

    def test_browsing_shows_what_was_found_and_suggests_the_classes(self):
        src = voc_folder([["cat", "dog"], ["cat"]])
        self.app._open_add_workspace_dialog()
        pump(self.root, 0.2)
        name, classes, prefix = self.entries()
        self.type_into(name, "guinew")
        self.browse(src)
        text = self.texts()
        self.assertIn("2 image(s)", text)
        self.assertIn("Pascal VOC", text)
        self.assertIn("2 class(es): cat, dog", text)
        self.assertEqual(classes.get(), "{cat, dog}")
        self.assertNotIn("SKIPPED", text)

    def test_create_imports_the_annotations_with_prefix_and_classes(self):
        src = voc_folder([["cat", "dog"], ["cat"]])
        self.app._open_add_workspace_dialog()
        name, classes, prefix = self.entries()
        self.type_into(name, "guinew")
        self.created.append("guinew")
        self.browse(src)
        self.type_into(classes, "{dog, cat}")                  # the user's order wins
        self.type_into(prefix, "g-")
        find_button(self.dialog(), "Create Workspace").invoke()
        pump(self.root, 0.5)
        self.assertEqual(wc.get_classes("guinew"), ["dog", "cat"])
        folder = os.path.join(ctx.datasets, "guinew-1")
        self.assertEqual(sorted(os.listdir(folder)), ["g-1.png", "g-2.png"])
        self.assertEqual(self.dialogs.last("showinfo")[1], "Workspace Ready")
        self.assertIn("2 image(s)", self.dialogs.last("showinfo")[2])

    def test_warns_before_importing_when_classes_are_not_in_the_workspace(self):
        src = voc_folder([["cat", "gloves"], ["gloves"]])
        self.app._open_add_workspace_dialog()
        name, classes, prefix = self.entries()
        self.type_into(name, WORKSPACE)                         # an existing workspace: cat, dog
        self.browse(src)
        text = self.texts()
        self.assertIn("SKIPPED", text)
        self.assertIn("gloves (2)", text)
        warning = [w for w in walk(self.dialog()) if isinstance(w, tk.Label) and "SKIPPED" in str(w.cget("text"))][0]
        self.assertEqual(warning.cget("fg"), theme.C_AMBER)
        self.assertEqual(self.dialogs.calls, [], "no dialog box: the warning is part of the window")

    def test_the_warning_follows_the_name_typed_after_the_folder(self):
        src = voc_folder([["cat", "gloves"]])
        self.app._open_add_workspace_dialog()
        name = self.entries()[0]
        self.type_into(name, "brandnew")
        self.browse(src)
        self.assertNotIn("SKIPPED", self.texts())
        self.type_into(name, WORKSPACE)
        self.assertIn("SKIPPED", self.texts())
        self.type_into(name, "brandnew")
        self.assertNotIn("SKIPPED", self.texts())

    def test_adding_to_an_existing_workspace_skips_the_unknown_class_and_reports_it(self):
        src = voc_folder([["cat", "gloves"]])
        self.app._open_add_workspace_dialog(existing_workspace=WORKSPACE)
        self.browse(src)
        self.assertIn("SKIPPED", self.texts())
        (prefix,) = self.entries()
        self.type_into(prefix, "gui-")
        find_button(self.dialog(), "Add Instance").invoke()
        pump(self.root, 0.5)
        kind, title, message = self.dialogs.last("showwarning")
        self.assertEqual(title, "Workspace Ready")
        self.assertIn("1 object(s) skipped", message)
        self.assertIn("gloves", message)
        self.assertEqual(wc.get_classes(WORKSPACE), ["cat", "dog"])
        self.assertEqual(len(self.new_instances()), 1)

    def test_a_folder_without_images_is_refused_in_the_window(self):
        self.app._open_add_workspace_dialog()
        self.type_into(self.entries()[0], "guiempty")
        self.browse(tempfile.mkdtemp())
        self.assertIn("No images or videos found", self.texts())
        find_button(self.dialog(), "Create Workspace").invoke()
        pump(self.root, 0.2)
        self.assertNotIn("guiempty", wm.list_workspaces())




def video_folder(with_images=False, frames=25):
    """A temp folder with one video (and, optionally, two images)."""
    import cv2
    import numpy as np
    src = tempfile.mkdtemp()
    if with_images:
        make_images(src, count=2, size=SIZE)
    writer = cv2.VideoWriter(os.path.join(src, "clip.avi"), cv2.VideoWriter_fourcc(*"MJPG"), 25, (64, 48))
    for k in range(frames):
        writer.write(np.full((48, 64, 3), 10 + 8 * k, np.uint8))
    writer.release()
    return src


@unittest.skipUnless(HAVE_DISPLAY, "no display available")
class SkipFramesTests(AddWorkspaceBase):
    """Video files in the chosen folder: a Skip frames box appears, default 10, and the frames become pictures."""

    def box_label(self):
        return [w for w in walk(self.dialog()) if isinstance(w, tk.Label)
                and str(w.cget("text")).startswith("SKIP FRAMES")][0]

    def spin(self):
        return [w for w in walk(self.dialog()) if isinstance(w, tk.Spinbox)][0]

    def set_skip(self, value):
        self.dialog().setvar(str(self.spin().cget("textvariable")), value)
        pump(self.root, 0.1)

    def estimate(self):
        return [str(w.cget("text")) for w in walk(self.dialog()) if isinstance(w, tk.Label)
                and str(w.cget("text")).startswith(("→", "✗ Skip"))]

    def test_the_box_is_hidden_until_a_folder_with_videos_is_chosen(self):
        self.app._open_add_workspace_dialog()
        pump(self.root, 0.2)
        self.assertFalse(self.box_label().winfo_ismapped())
        self.type_into(self.entries()[0], "guivid")
        self.browse(voc_folder([["cat"]]))                                # images only
        self.assertFalse(self.box_label().winfo_ismapped())
        self.browse(video_folder())
        self.assertTrue(self.box_label().winfo_ismapped())
        self.browse(voc_folder([["cat"]]))                                # back to a folder without videos
        self.assertFalse(self.box_label().winfo_ismapped())

    def test_the_default_is_10_and_the_description_explains_it_in_plain_english(self):
        self.app._open_add_workspace_dialog()
        self.type_into(self.entries()[0], "guivid")
        self.browse(video_folder())
        self.assertEqual(self.spin().get(), "10")
        text = self.texts()
        for must in ("SKIP FRAMES (VIDEO FILES)", "Video files were found", "skips the next N - 1 frames",
                     "10 means every 10th frame is kept", "30 fps video gives 3 pictures per second",
                     "Use 1 to keep every frame", "without annotations"):
            self.assertIn(must, text)
        self.assertIn("1 video(s) found and no images", text)

    def test_the_estimate_follows_the_number(self):
        self.app._open_add_workspace_dialog()
        self.type_into(self.entries()[0], "guivid")
        self.browse(video_folder(frames=25))
        self.assertEqual(self.estimate(), ["→ about 3 picture(s) from 1 video(s)"])
        self.set_skip("5")
        self.assertEqual(self.estimate(), ["→ about 5 picture(s) from 1 video(s)"])
        self.set_skip("1")
        self.assertEqual(self.estimate(), ["→ about 25 picture(s) from 1 video(s)"])
        self.set_skip("0")
        self.assertIn("✗ Skip frames must be a whole number", self.estimate()[0])

    def test_create_saves_the_frames_next_to_the_images_and_says_what_it_did(self):
        self.app._open_add_workspace_dialog()
        name = self.entries()[0]
        self.type_into(name, "guivid")
        self.created.append("guivid")
        self.browse(video_folder(with_images=True))
        self.set_skip("10")
        find_button(self.dialog(), "Create Workspace").invoke()
        pump(self.root, 0.5)
        folder = os.path.join(ctx.datasets, "guivid-1")
        self.assertEqual(sorted(os.listdir(folder)),
                         ["clip_f000000.jpg", "clip_f000010.jpg", "clip_f000020.jpg", "img1.png", "img2.png"])
        title, message = self.dialogs.last("showinfo")[1:]
        self.assertEqual(title, "Workspace Ready")
        self.assertIn("created with 5 image(s)", message)
        self.assertIn("Video frames: 3 picture(s) saved from 1 video(s), 1 frame out of every 10.", message)

    def test_a_different_skip_value_reaches_the_import(self):
        self.app._open_add_workspace_dialog()
        self.type_into(self.entries()[0], "guivid")
        self.created.append("guivid")
        self.browse(video_folder(frames=25))
        self.set_skip("5")
        find_button(self.dialog(), "Create Workspace").invoke()
        pump(self.root, 0.5)
        self.assertEqual(len(os.listdir(os.path.join(ctx.datasets, "guivid-1"))), 5)
        self.assertIn("1 frame out of every 5", self.dialogs.last("showinfo")[2])

    def test_a_bad_skip_value_blocks_the_import_and_says_why_in_the_window(self):
        self.app._open_add_workspace_dialog()
        self.type_into(self.entries()[0], "guivid")
        self.browse(video_folder())
        for bad in ("0", "abc", "", "99999"):
            self.set_skip(bad)
            find_button(self.dialog(), "Create Workspace").invoke()
            pump(self.root, 0.2)
            self.assertIn("✗ Skip frames must be a whole number", self.estimate()[0], bad)
        self.assertNotIn("guivid", wm.list_workspaces())
        self.assertEqual(self.dialogs.calls, [])

    def test_the_add_instance_window_has_the_box_too(self):
        self.app._open_add_workspace_dialog(existing_workspace=WORKSPACE)
        pump(self.root, 0.2)
        self.assertFalse(self.box_label().winfo_ismapped())
        self.browse(video_folder())
        self.assertTrue(self.box_label().winfo_ismapped())
        self.assertEqual(self.spin().get(), "10")

    def test_the_buttons_stay_visible_when_the_box_makes_the_window_taller(self):
        self.app._open_add_workspace_dialog()
        self.type_into(self.entries()[0], "guivid")
        self.browse(video_folder())
        pump(self.root, 0.3)
        dlg = self.dialog()
        bottom = dlg.winfo_rooty() + dlg.winfo_height()
        for text in ("Create Workspace", "Cancel"):
            button = find_button(dlg, text)
            self.assertTrue(button.winfo_viewable(), text)
            self.assertLessEqual(button.winfo_rooty() + button.winfo_height(), bottom, text)
        self.assertLessEqual(dlg.winfo_height(), dlg.winfo_screenheight())


if __name__ == "__main__":
    unittest.main()


class RefreshRecountsTests(AddWorkspaceBase):
    """The Refresh button counts every workspace again; the list and the Train windows read the stored numbers."""

    def test_refresh_recounts_every_workspace_and_forgets_the_ones_that_are_gone(self):
        from utils import workspace_counts as counts
        counts.recount_all()
        ghost = counts._read()
        ghost["vanished"] = {"counted": 1, "images": 9, "labelled": 9, "background": 0, "boxes": [], "instances": {}}
        counts._write(ghost)
        self.assertIn("vanished", counts._read())
        find_button(self.root, "Refresh").invoke()
        pump(self.root, 0.3)
        stored = counts._read()
        self.assertNotIn("vanished", stored, "a workspace that no longer exists is dropped")
        self.assertIn(WORKSPACE, stored)
        self.assertEqual(stored[WORKSPACE]["images"], wm.count_images(WORKSPACE + "-1"))

    def test_the_list_shows_the_stored_picture_count_and_does_not_list_the_folder_again(self):
        from utils import workspace_counts as counts
        counts.recount_all()
        self.app.selected = WORKSPACE
        with mock.patch.object(wm, "count_images", side_effect=AssertionError("listed a folder")):
            self.app.refresh_workspaces()                       # expanding a row draws its folders
            self.app._add_instance_row(WORKSPACE + "-1")
        pump(self.root, 0.2)

