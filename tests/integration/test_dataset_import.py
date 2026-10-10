"""Add Workspace imports a whole folder tree (images + VOC / YOLO / COCO): prefix, class order, and what happens to
objects of classes an existing workspace does not have."""
import json
import os
import tempfile
import unittest

from tests.helpers import isolated_workspace, make_images

ctx = isolated_workspace()

from utils import workspace_config as wc                        # noqa: E402
from utils import workspace_manager as wm                       # noqa: E402
from utils import dataset_import as di                          # noqa: E402

SIZE = (200, 100)


def voc_xml(path, name, objects):
    rows = "".join(f"<object><name>{c}</name><bndbox><xmin>{a}</xmin><ymin>{b}</ymin><xmax>{x}</xmax><ymax>{y}</ymax>"
                   f"</bndbox></object>" for c, (a, b, x, y) in objects)
    with open(path, "w") as f:
        f.write(f"<annotation><filename>{name}</filename><size><width>{SIZE[0]}</width><height>{SIZE[1]}</height>"
                f"<depth>3</depth></size>{rows}</annotation>")


class Base(unittest.TestCase):
    def setUp(self):
        self.src = tempfile.mkdtemp()
        self.made = []

    def workspace(self, name, classes=None):
        self.made.append(name)
        if classes:
            wc.set_classes(name, classes)
        return name

    def tearDown(self):
        for name in self.made:
            try:
                wm.delete_workspace(name)
            except Exception:
                pass
            if wc.exists(name):
                os.remove(os.path.join(ctx.configs, name + ".json"))

    def voc(self, name):
        with open(os.path.join(ctx.root, "vocdataset", name_ws(self), name + ".xml")) as f:
            return f.read()

    def label(self, ws, name):
        with open(os.path.join(ctx.root, "YOLOdataset", ws, "labels", name + ".txt")) as f:
            return f.read().split("\n")


def name_ws(test):
    return test.made[-1]


class NewWorkspaceTests(Base):
    def test_typed_classes_come_first_then_the_ones_found_in_the_annotations(self):
        names = make_images(self.src, count=1, size=SIZE)
        voc_xml(os.path.join(self.src, "img1.xml"), names[0], [("pear", (10, 10, 50, 50)), ("apple", (60, 10, 90, 50))])
        ws = self.workspace("newws")
        result = di.import_dataset(self.src, ws, classes=["apple", "kiwi"])
        self.assertEqual(result["classes"], ["apple", "kiwi", "pear"])
        self.assertEqual(wc.get_classes(ws), ["apple", "kiwi", "pear"])
        self.assertEqual(result["skipped"], {})
        self.assertEqual(sorted(l.split()[0] for l in self.label(ws, "img1")), ["0", "2"])

    def test_images_only_with_typed_classes_keeps_them(self):
        make_images(self.src, count=2, size=SIZE)
        ws = self.workspace("plain")
        result = di.import_dataset(self.src, ws, classes=["a", "b"])
        self.assertEqual((result["format"], result["image_count"]), (None, 2))
        self.assertEqual(wc.get_classes(ws), ["a", "b"])

    def test_every_subfolder_is_searched_and_the_prefix_renames_images_and_annotations(self):
        sub = os.path.join(self.src, "train", "images")
        names = make_images(sub, count=2, size=SIZE)
        voc_xml(os.path.join(self.src, "train", "x.xml"), names[0], [])      # not matching any image
        voc_xml(os.path.join(sub, "img1.xml"), names[0], [("cat", (10, 10, 50, 50))])
        ws = self.workspace("tree")
        result = di.import_dataset(self.src, ws, prefix="p-")
        self.assertEqual(result["image_count"], 2)
        folder = os.path.join(ctx.datasets, result["instance_name"])
        self.assertEqual(sorted(os.listdir(folder)), ["p-1.png", "p-2.png"])
        self.assertIn("<filename>p-1.png</filename>", self.voc("p-1"))


class ExistingWorkspaceTests(Base):
    def test_voc_objects_of_unknown_classes_are_skipped_and_counted(self):
        names = make_images(self.src, count=1, size=SIZE)
        voc_xml(os.path.join(self.src, "img1.xml"), names[0],
                [("cat", (10, 10, 50, 50)), ("gloves", (60, 10, 90, 50)), ("gloves", (100, 10, 150, 50))])
        ws = self.workspace("locked", ["dog", "cat"])
        result = di.import_dataset(self.src, ws)
        self.assertEqual(result["skipped"], {"gloves": 2})
        self.assertEqual(wc.get_classes(ws), ["dog", "cat"], "the workspace's classes are never changed")
        xml = self.voc("img1")
        self.assertIn("<name>cat</name>", xml)
        self.assertNotIn("gloves", xml)
        self.assertEqual([l.split()[0] for l in self.label(ws, "img1")], ["1"])        # cat = index 1 in the workspace

    def test_yolo_labels_follow_the_workspace_order_not_the_source_order(self):
        names = make_images(self.src, count=1, size=SIZE)
        with open(os.path.join(self.src, "classes.txt"), "w") as f:
            f.write("cat\ndog\ngloves\n")                      # the source has cat = 0, dog = 1
        with open(os.path.join(self.src, "img1.txt"), "w") as f:
            f.write("0 0.5 0.5 0.2 0.2\n1 0.3 0.3 0.1 0.1\n2 0.8 0.8 0.1 0.1\n")
        ws = self.workspace("yolo_locked", ["dog", "cat"])     # the workspace has dog = 0, cat = 1
        result = di.import_dataset(self.src, ws)
        self.assertEqual(result["skipped"], {"gloves": 1})
        self.assertEqual([l.split()[0] for l in self.label(ws, "img1")], ["1", "0"])
        xml = self.voc("img1")
        self.assertNotIn("gloves", xml)
        self.assertEqual(wc.get_classes(ws), ["dog", "cat"])

    def test_coco_objects_of_unknown_classes_are_skipped(self):
        names = make_images(self.src, count=1, size=SIZE)
        coco = {"images": [{"id": 1, "file_name": names[0], "width": SIZE[0], "height": SIZE[1]}],
                "categories": [{"id": 1, "name": "cat"}, {"id": 2, "name": "gloves"}],
                "annotations": [{"image_id": 1, "category_id": 1, "bbox": [10, 10, 40, 40]},
                                {"image_id": 1, "category_id": 2, "bbox": [60, 10, 30, 30]}]}
        with open(os.path.join(self.src, "_annotations.coco.json"), "w") as f:
            json.dump(coco, f)
        ws = self.workspace("coco_locked", ["cat"])
        result = di.import_dataset(self.src, ws)
        self.assertEqual(result["skipped"], {"gloves": 1})
        self.assertEqual(len(self.label(ws, "img1")), 1)

    def test_a_workspace_without_classes_yet_takes_them_from_the_annotations(self):
        names = make_images(self.src, count=1, size=SIZE)
        voc_xml(os.path.join(self.src, "img1.xml"), names[0], [("cat", (10, 10, 50, 50))])
        ws = self.workspace("empty_cfg")
        wc.set_classes(ws, [])
        result = di.import_dataset(self.src, ws)
        self.assertEqual(result["skipped"], {})
        self.assertEqual(wc.get_classes(ws), ["cat"])


class PreviewTests(Base):
    def test_scan_classes_counts_objects_per_class_for_every_format(self):
        names = make_images(self.src, count=2, size=SIZE)
        voc_xml(os.path.join(self.src, "img1.xml"), names[0], [("cat", (1, 1, 9, 9)), ("cat", (1, 1, 9, 9)),
                                                              ("dog", (1, 1, 9, 9))])
        scan = di.scan_dataset_folder(self.src)
        self.assertEqual(di.scan_classes(scan, "voc"), {"cat": 2, "dog": 1})
        self.assertEqual(di.scan_classes(scan, None), {})

    def test_scan_classes_for_yolo_uses_the_class_file_names(self):
        make_images(self.src, count=1, size=SIZE)
        with open(os.path.join(self.src, "classes.txt"), "w") as f:
            f.write("cat\ndog\n")
        with open(os.path.join(self.src, "img1.txt"), "w") as f:
            f.write("1 0.5 0.5 0.2 0.2\n1 0.5 0.5 0.2 0.2\n0 0.5 0.5 0.2 0.2\n")
        scan = di.scan_dataset_folder(self.src)
        self.assertEqual(di.scan_classes(scan, "yolo"), {"cat": 1, "dog": 2})

    def test_same_file_name_in_two_subfolders_is_counted(self):
        make_images(os.path.join(self.src, "train"), count=1, size=SIZE)
        make_images(os.path.join(self.src, "valid"), count=1, size=SIZE)
        scan = di.scan_dataset_folder(self.src)
        self.assertEqual((len(scan["images"]), scan["duplicates"]), (1, 1))

    def test_a_folder_inside_datasets_root_is_refused(self):
        with self.assertRaises(ValueError):
            di.import_dataset(ctx.instance_dir, "loopws")


if __name__ == "__main__":
    unittest.main()
