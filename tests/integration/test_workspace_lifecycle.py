import os
import shutil
import tempfile
import unittest

from tests.helpers import isolated_workspace, make_images, WORKSPACE

ctx = isolated_workspace()

from utils import workspace_config as wc                        # noqa: E402
from utils import workspace_manager as wm                       # noqa: E402
from utils import dataset_import as di                          # noqa: E402


class WorkspaceLifecycleTests(unittest.TestCase):
    def test_the_test_workspace_is_listed_with_its_instance(self):
        ws = wm.list_workspaces()
        self.assertIn(WORKSPACE, ws)
        self.assertIn("testws-1", ws[WORKSPACE])

    def test_existing_classes_come_from_the_json_config(self):
        self.assertEqual(wm.existing_classes_for_workspace(WORKSPACE), ["cat", "dog"])
        self.assertIsNone(wm.existing_classes_for_workspace("never_created"))

    def test_create_instance_for_a_new_workspace_stores_classes_in_json(self):
        src = tempfile.mkdtemp()
        make_images(src, count=2)
        name, count = wm.create_workspace_instance("fresh", src, classes=["apple", "pear"])
        try:
            self.assertEqual((name, count), ("fresh-1", 2))
            self.assertTrue(os.path.isdir(os.path.join(ctx.datasets, "fresh-1")))
            self.assertEqual(wc.get_classes("fresh"), ["apple", "pear"])
            self.assertTrue(os.path.exists(os.path.join(ctx.configs, "fresh.json")))
            # a second instance reuses the workspace classes and never overwrites them
            name2, _ = wm.create_workspace_instance("fresh", src, classes=["other"])
            self.assertEqual(name2, "fresh-2")
            self.assertEqual(wc.get_classes("fresh"), ["apple", "pear"])
        finally:
            wm.delete_workspace("fresh")
        self.assertFalse(wc.exists("fresh"), "deleting a workspace removes its config")
        self.assertFalse(os.path.exists(os.path.join(ctx.datasets, "fresh-1")))

    def test_folders_inside_datasets_root_are_refused(self):
        with self.assertRaises(ValueError):
            wm.create_workspace_instance("loop", ctx.instance_dir)

    def test_empty_folder_is_refused(self):
        with self.assertRaises(ValueError):
            wm.create_workspace_instance("empty", tempfile.mkdtemp())


class YoloImportTests(unittest.TestCase):
    def test_yolo_dataset_is_converted_to_voc_and_native_labels(self):
        src = tempfile.mkdtemp()
        names = make_images(src, count=2, size=(200, 100))
        with open(os.path.join(src, "classes.txt"), "w") as f:
            f.write("apple\npear\n")
        with open(os.path.join(src, names[0].replace(".png", ".txt")), "w") as f:
            f.write("1 0.5 0.5 0.5 0.5\n")
        try:
            summary = di.import_dataset(src, "imported")
            self.assertEqual(summary["format"], "YOLO")
            self.assertEqual(summary["image_count"], 2)
            self.assertEqual(summary["annotated_count"], 1)
            self.assertEqual(summary["classes"], ["apple", "pear"])
            self.assertEqual(wc.get_classes("imported"), ["apple", "pear"])
            xml = os.path.join(ctx.root, "vocdataset", "imported", names[0].replace(".png", ".xml"))
            self.assertTrue(os.path.exists(xml))
            with open(xml) as f:
                content = f.read()
            self.assertIn("<name>pear</name>", content)
            self.assertIn("<xmin>50</xmin>", content)      # 0.5 - 0.25 of width 200
        finally:
            wm.delete_workspace("imported")

    def test_folder_without_images_is_refused(self):
        with self.assertRaises(ValueError):
            di.import_dataset(tempfile.mkdtemp(), "noimages")


if __name__ == "__main__":
    unittest.main()
