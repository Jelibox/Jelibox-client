"""The picture counts of each workspace are written down (configs/_stats.json) and only recounted on request."""
import json
import os
import shutil
import unittest
from unittest import mock

from tests.helpers import isolated_workspace, make_images

ctx = isolated_workspace()

from utils import workspace_config as wc                    # noqa: E402
from utils import workspace_counts as counts                # noqa: E402
from utils import workspace_manager as wm                   # noqa: E402

A, B = "countsa", "countsb"


def write_label(ws, name, text):
    folder = os.path.join(wm.BASE_DIR, "YOLOdataset", ws, "labels")
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, name + ".txt"), "w") as f:
        f.write(text)


class CountsTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(self.cleanup)
        self.cleanup()
        make_images(os.path.join(ctx.datasets, A + "-1"), count=5)               # img1..img5
        make_images(os.path.join(ctx.datasets, A + "-2"), count=2)               # img1, img2 again: same names
        make_images(os.path.join(ctx.datasets, B + "-1"), count=3)
        wc.set_classes(A, ["cat", "dog"])
        wc.set_classes(B, ["bird"])
        write_label(A, "img1", "0 0.5 0.5 0.2 0.2" + chr(10) + "1 0.3 0.3 0.1 0.1")
        write_label(A, "img2", "1 0.5 0.5 0.2 0.2")
        write_label(A, "img3", "")                                               # saved with nothing in it
        write_label(B, "img1", "0 0.5 0.5 0.2 0.2")

    def cleanup(self):
        for ws in (A, B):
            for root in (ctx.datasets,):
                for name in os.listdir(root) if os.path.isdir(root) else []:
                    if name.startswith(ws + "-"):
                        shutil.rmtree(os.path.join(root, name), ignore_errors=True)
            shutil.rmtree(os.path.join(wm.BASE_DIR, "YOLOdataset", ws), ignore_errors=True)
        try:
            os.remove(os.path.join(ctx.configs, counts.FILE))
        except OSError:
            pass

    def test_a_workspace_is_counted_in_pictures_labelled_background_and_boxes(self):
        stats = counts.compute(A)
        self.assertEqual(stats["instances"], {A + "-1": 5, A + "-2": 2})
        self.assertEqual((stats["images"], stats["labelled"], stats["background"]), (7, 2, 1))
        self.assertEqual(stats["boxes"], [1, 2], "one cat, two dogs")

    def test_the_numbers_are_stored_and_read_without_looking_at_the_folders(self):
        counts.recount(A)
        with mock.patch.object(counts.head_data, "scan_labels", side_effect=AssertionError("looked at the labels")), \
                mock.patch.object(counts.wm, "count_images", side_effect=AssertionError("listed a folder")):
            self.assertEqual(counts.get(A)["labelled"], 2)
            self.assertEqual(counts.get_or_empty(A)["background"], 1)
            self.assertEqual(counts.instance_images(A, A + "-1"), 5)
        self.assertTrue(os.path.isfile(os.path.join(ctx.configs, counts.FILE)))

    def test_nothing_is_counted_by_itself_when_nothing_is_stored(self):
        self.assertIsNone(counts.get(A))
        with mock.patch.object(counts.head_data, "scan_labels", side_effect=AssertionError("counted")), \
                mock.patch.object(counts.wm, "count_images", side_effect=AssertionError("counted")):
            empty = counts.get_or_empty(A)
            self.assertIsNone(counts.instance_images(A, A + "-1"))
        self.assertEqual((empty["labelled"], empty["background"], empty["counted"]), (0, 0, None))
        self.assertIsNone(counts.get(A), "and nothing was written")
        self.assertFalse(os.path.exists(os.path.join(ctx.configs, counts.FILE)))

    def test_the_stored_numbers_do_not_follow_the_disk_until_a_recount(self):
        counts.recount(A)
        write_label(A, "img4", "0 0.5 0.5 0.2 0.2")
        self.assertEqual(counts.get(A)["labelled"], 2, "not looked at again by itself")
        self.assertEqual(counts.recount(A)["labelled"], 3)

    def test_recounting_everything_forgets_the_workspaces_that_are_gone(self):
        counts.recount_all()
        self.assertIn(A, json.load(open(os.path.join(ctx.configs, counts.FILE)))["workspaces"])
        shutil.rmtree(os.path.join(ctx.datasets, B + "-1"))
        seen = []
        fresh = counts.recount_all(lambda done, total, text: seen.append((done, total)))
        self.assertNotIn(B, fresh)
        self.assertNotIn(B, json.load(open(os.path.join(ctx.configs, counts.FILE)))["workspaces"])
        self.assertIn(A, fresh)
        self.assertEqual(seen[-1][0], seen[-1][1], "the progress ends complete")

    def test_the_count_per_folder_is_known_after_a_recount_and_not_before(self):
        self.assertIsNone(counts.instance_images(B, B + "-1"))
        counts.recount(B)
        with mock.patch.object(counts.wm, "count_images", side_effect=AssertionError("listed again")):
            self.assertEqual(counts.instance_images(B, B + "-1"), 3)

    def test_a_damaged_file_is_just_recounted(self):
        with open(os.path.join(ctx.configs, counts.FILE), "w") as f:
            f.write("{ not json")
        self.assertIsNone(counts.get(A))
        self.assertEqual(counts.get_or_empty(A)["counted"], None)
        self.assertEqual(counts.recount(A)["labelled"], 2)


if __name__ == "__main__":
    unittest.main()
