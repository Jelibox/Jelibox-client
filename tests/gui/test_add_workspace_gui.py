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
class AddWorkspaceImportTests(unittest.TestCase):
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
        self.assertIn("No images found", self.texts())
        find_button(self.dialog(), "Create Workspace").invoke()
        pump(self.root, 0.2)
        self.assertNotIn("guiempty", wm.list_workspaces())


if __name__ == "__main__":
    unittest.main()
