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


def make_video(path, frames=25, size=(64, 48), fps=25):
    """A small MJPG .avi whose frame k has the grey level 10 + 8 * k, so a saved picture says which frame it is."""
    import cv2
    import numpy as np
    os.makedirs(os.path.dirname(path), exist_ok=True)
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
    assert writer.isOpened(), "this OpenCV cannot write MJPG"
    for k in range(frames):
        writer.write(np.full((size[1], size[0], 3), 10 + 8 * k, np.uint8))
    writer.release()
    return path


def grey_level(path):
    import cv2
    import numpy as np
    return float(cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR).mean())


class VideoTests(Base):
    def folder_of(self, instance):
        return os.path.join(ctx.datasets, instance)

    def test_videos_are_found_by_extension_and_are_not_images(self):
        for name in ("a.AVI", "b.mp4", "c.MOV", "d.mkv", "e.webm", "notes.txt"):
            open(os.path.join(self.src, name), "wb").close()
        make_images(self.src, count=1, size=SIZE)
        scan = di.scan_dataset_folder(self.src)
        self.assertEqual([os.path.basename(v) for v in scan["videos"]], ["a.AVI", "b.mp4", "c.MOV", "d.mkv", "e.webm"])
        self.assertEqual(len(scan["images"]), 1)

    def test_skip_frames_numbers(self):
        self.assertEqual((di.sanitize_skip_frames("10"), di.sanitize_skip_frames(" 3 "), di.sanitize_skip_frames(1)),
                         (10, 3, 1))
        for bad in ("0", "-1", "abc", "", None, "2.5", "20000"):
            with self.assertRaisesRegex(ValueError, "whole number from 1 to 10000"):
                di.sanitize_skip_frames(bad)
        self.assertEqual(di.DEFAULT_SKIP_FRAMES, 10)

    def test_how_many_pictures_a_video_gives(self):
        self.assertEqual([di.frames_to_save(n, 10) for n in (0, 1, 10, 11, 25, 100)], [0, 1, 1, 2, 3, 10])
        self.assertEqual(di.frames_to_save(25, 1), 25)
        self.assertEqual(di.frames_to_save(-1, 10), 0)

    def test_probing_reads_the_length_from_the_header_and_survives_a_broken_file(self):
        good = make_video(os.path.join(self.src, "clip.avi"), frames=25)
        bad = os.path.join(self.src, "broken.mp4")
        with open(bad, "wb") as f:
            f.write(b"this is not a video")
        found = {os.path.basename(p["path"]): p for p in di.probe_videos([good, bad])}
        self.assertEqual(found["clip.avi"]["frames"], 25)
        self.assertEqual(found["broken.mp4"]["frames"], 0)

    def test_every_tenth_frame_by_default_named_after_the_video_and_the_frame(self):
        make_video(os.path.join(self.src, "clip.avi"), frames=25)
        ws = self.workspace("vid1")
        result = di.import_dataset(self.src, ws)
        folder = self.folder_of(result["instance_name"])
        names = sorted(os.listdir(folder))
        self.assertEqual(names, ["clip_f000000.jpg", "clip_f000010.jpg", "clip_f000020.jpg"])
        self.assertEqual((result["frames"], result["videos"], result["image_count"], result["skip_frames"]), (3, 1, 3, 10))
        self.assertEqual(result["video_errors"], [])
        self.assertEqual((result["format"], result["annotated_count"]), (None, 0))
        levels = [grey_level(os.path.join(folder, n)) for n in names]
        for got, frame in zip(levels, (0, 10, 20)):                        # they really are frames 0, 10 and 20
            self.assertAlmostEqual(got, 10 + 8 * frame, delta=6)

    def test_other_skip_values(self):
        make_video(os.path.join(self.src, "clip.avi"), frames=25)
        ws = self.workspace("vid2")
        every = di.import_dataset(self.src, ws, skip_frames=1)
        self.assertEqual(every["frames"], 25)
        seven = di.import_dataset(self.src, ws, skip_frames=7)
        self.assertEqual(seven["frames"], 4)                                # frames 0, 7, 14, 21
        self.assertEqual(len(os.listdir(self.folder_of(seven["instance_name"]))), 4)

    def test_images_and_a_video_in_one_folder_and_only_the_images_have_annotations(self):
        names = make_images(self.src, count=2, size=SIZE)
        voc_xml(os.path.join(self.src, "img1.xml"), names[0], [("cat", (10, 10, 50, 50))])
        make_video(os.path.join(self.src, "sub", "clip.avi"), frames=25)
        ws = self.workspace("vid3")
        result = di.import_dataset(self.src, ws, classes=["cat"])
        folder = self.folder_of(result["instance_name"])
        self.assertEqual(sorted(os.listdir(folder)),
                         ["clip_f000000.jpg", "clip_f000010.jpg", "clip_f000020.jpg", "img1.png", "img2.png"])
        self.assertEqual((result["image_count"], result["frames"], result["annotated_count"]), (5, 3, 1))
        voc_files = os.listdir(os.path.join(ctx.root, "vocdataset", ws))
        self.assertEqual(voc_files, ["img1.xml"], "a picture cut from a video starts without annotations")

    def test_a_prefix_numbers_the_frames_after_the_images(self):
        make_images(self.src, count=2, size=SIZE)
        make_video(os.path.join(self.src, "clip.avi"), frames=25)
        ws = self.workspace("vid4")
        result = di.import_dataset(self.src, ws, prefix="p-")
        self.assertEqual(sorted(os.listdir(self.folder_of(result["instance_name"]))),
                         ["p-1.png", "p-2.png", "p-3.jpg", "p-4.jpg", "p-5.jpg"])

    def test_two_videos_with_the_same_name_do_not_overwrite_each_other(self):
        make_video(os.path.join(self.src, "a", "clip.avi"), frames=25)
        make_video(os.path.join(self.src, "b", "clip.avi"), frames=25)
        ws = self.workspace("vid5")
        result = di.import_dataset(self.src, ws)
        names = sorted(os.listdir(self.folder_of(result["instance_name"])))
        self.assertEqual(len(names), 6)
        self.assertEqual(len(set(names)), 6)
        self.assertEqual((result["frames"], result["videos"]), (6, 2))

    def test_an_unreadable_video_is_reported_and_the_rest_still_imports(self):
        make_video(os.path.join(self.src, "clip.avi"), frames=25)
        with open(os.path.join(self.src, "broken.mp4"), "wb") as f:
            f.write(b"this is not a video")
        ws = self.workspace("vid6")
        result = di.import_dataset(self.src, ws)
        self.assertEqual((result["frames"], result["videos"]), (3, 1))
        self.assertEqual(len(result["video_errors"]), 1)
        self.assertIn("broken.mp4", result["video_errors"][0])

    def test_only_unreadable_videos_is_an_error_and_leaves_no_empty_dataset_behind(self):
        with open(os.path.join(self.src, "broken.mp4"), "wb") as f:
            f.write(b"this is not a video")
        ws = self.workspace("vid7")
        with self.assertRaisesRegex(ValueError, "No pictures could be read"):
            di.import_dataset(self.src, ws)
        self.assertFalse(os.path.isdir(self.folder_of("vid7-1")))

    def test_a_bad_skip_value_is_refused_before_anything_is_copied(self):
        make_video(os.path.join(self.src, "clip.avi"), frames=25)
        ws = self.workspace("vid8")
        with self.assertRaisesRegex(ValueError, "Skip frames"):
            di.import_dataset(self.src, ws, skip_frames=0)
        self.assertFalse(os.path.isdir(self.folder_of("vid8-1")))

    def test_progress_names_the_video_it_is_working_on(self):
        make_video(os.path.join(self.src, "clip.avi"), frames=25)
        ws = self.workspace("vid9")
        seen = []
        di.import_dataset(self.src, ws, progress_cb=lambda done, total, label=None: seen.append((done, total, label)))
        labels = [l for _, _, l in seen if l]
        self.assertTrue(any("Extracting frames from clip.avi" in l for l in labels), labels)
        self.assertTrue(all(0 <= d <= t for d, t, _ in seen), seen)

    def test_a_folder_without_images_or_videos_says_so(self):
        with self.assertRaisesRegex(ValueError, "No images or videos"):
            di.import_dataset(tempfile.mkdtemp(), "novid")


if __name__ == "__main__":
    unittest.main()
