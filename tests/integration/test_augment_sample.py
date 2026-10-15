"""The picture the augmentation previews are made from: the first picture by default, a chosen one kept in the
workspace's config, and the first one again when the chosen file is gone."""
import os
import shutil
import tempfile
import unittest

import numpy as np

from tests.helpers import isolated_workspace, make_images

ctx = isolated_workspace()

from utils import augment_sample as sample                       # noqa: E402
from utils import workspace_config as wc                         # noqa: E402
from utils import workspace_manager as wm                        # noqa: E402

WS = "sampleaug"


class SampleTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(self.cleanup)
        self.first = os.path.join(ctx.datasets, WS + "-1")
        self.second = os.path.join(ctx.datasets, WS + "-2")
        make_images(self.first, count=3, size=(120, 80))           # img1.png img2.png img3.png
        make_images(self.second, count=2, size=(90, 60))
        wc.set_classes(WS, ["a"])

    def cleanup(self):
        for folder in (self.first, self.second):
            shutil.rmtree(folder, ignore_errors=True)
        if wc.exists(WS):
            os.remove(os.path.join(ctx.configs, WS + ".json"))

    def test_the_images_come_in_the_order_the_exporter_uses(self):
        names = [os.path.relpath(p, ctx.datasets).replace("\\", "/") for p in sample.workspace_images(WS)]
        self.assertEqual(names, [f"{WS}-1/img1.png", f"{WS}-1/img2.png", f"{WS}-1/img3.png",
                                 f"{WS}-2/img1.png", f"{WS}-2/img2.png"])

    def test_other_workspaces_and_non_images_are_not_included(self):
        open(os.path.join(self.first, "notes.txt"), "w").close()
        shutil.copytree(self.first, os.path.join(ctx.datasets, WS + "x-1"))
        self.addCleanup(shutil.rmtree, os.path.join(ctx.datasets, WS + "x-1"), True)
        self.assertEqual(len(sample.workspace_images(WS)), 5)

    def test_by_default_the_first_picture_is_used(self):
        path, is_default = sample.pick_sample(WS)
        self.assertTrue(is_default)
        self.assertEqual(path, os.path.join(self.first, "img1.png"))

    def test_a_workspace_without_pictures_has_no_sample(self):
        self.assertEqual(sample.pick_sample("nothing-here"), (None, True))

    def test_a_chosen_picture_is_remembered_in_the_workspace_config(self):
        chosen = os.path.join(self.second, "img2.png")
        sample.remember_sample(WS, chosen)
        self.assertEqual(sample.pick_sample(WS), (chosen, False))
        stored = wc.get_preview_image(WS)
        self.assertFalse(os.path.isabs(stored), "inside the Jelibox folder it is stored relative: " + stored)
        self.assertNotIn("\\", stored)
        with open(os.path.join(ctx.configs, WS + ".json"), encoding="utf-8") as f:
            self.assertIn("preview", f.read())

    def test_a_picture_from_anywhere_is_stored_with_its_full_path(self):
        elsewhere = os.path.join(tempfile.mkdtemp(), "mine.png")
        make_images(os.path.dirname(elsewhere), count=1)
        os.replace(os.path.join(os.path.dirname(elsewhere), "img1.png"), elsewhere)
        sample.remember_sample(WS, elsewhere)
        self.assertEqual(sample.pick_sample(WS), (elsewhere, False))

    def test_when_the_chosen_file_is_gone_the_first_picture_is_back(self):
        chosen = os.path.join(self.second, "img2.png")
        sample.remember_sample(WS, chosen)
        os.remove(chosen)
        self.assertEqual(sample.pick_sample(WS), (os.path.join(self.first, "img1.png"), True))

    def test_something_that_is_not_a_picture_is_ignored(self):
        text = os.path.join(self.first, "readme.txt")
        open(text, "w").close()
        sample.remember_sample(WS, text)
        self.assertTrue(sample.pick_sample(WS)[1])

    def test_forgetting_goes_back_to_the_first_picture(self):
        sample.remember_sample(WS, os.path.join(self.second, "img2.png"))
        sample.forget_sample(WS)
        self.assertEqual(wc.get_preview_image(WS), "")
        self.assertTrue(sample.pick_sample(WS)[1])

    def test_the_choice_survives_other_changes_to_the_config(self):
        sample.remember_sample(WS, os.path.join(self.second, "img2.png"))
        wc.set_classes(WS, ["a", "b"])
        a = wc.get_assistant(WS)
        wc.set_assistant(WS, a)
        self.assertFalse(sample.pick_sample(WS)[1])

    def test_a_malformed_preview_entry_in_the_config_is_ignored(self):
        import json
        path = os.path.join(ctx.configs, WS + ".json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"classes": ["a"], "preview": {"image": 5}}, f)
        self.assertEqual(wc.get_preview_image(WS), "")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"classes": ["a"], "preview": "nope"}, f)
        self.assertEqual(wc.get_preview_image(WS), "")

    def test_the_sample_is_loaded_small_and_a_bad_file_gives_none(self):
        big = os.path.join(self.first, "big.png")
        from PIL import Image
        Image.new("RGB", (1000, 500), (10, 20, 30)).save(big)
        img = sample.load_sample(big)
        self.assertEqual(max(img.shape[:2]), sample.PREVIEW_SIDE)
        self.assertEqual(img.shape[:2], (180, 360))
        small = sample.load_sample(os.path.join(self.first, "img1.png"))
        self.assertEqual(small.shape[:2], (80, 120), "a small picture is not enlarged")
        junk = os.path.join(self.first, "junk.png")
        with open(junk, "wb") as f:
            f.write(b"not an image")
        self.assertIsNone(sample.load_sample(junk))
        self.assertIsNone(sample.load_sample(None))
        self.assertIsNone(sample.load_sample(os.path.join(self.first, "missing.png")))


if __name__ == "__main__":
    unittest.main()
