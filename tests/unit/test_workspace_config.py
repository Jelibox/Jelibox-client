import json
import os
import tempfile
import unittest

from utils import workspace_config as wc


class WorkspaceConfigTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self._old = wc.CONFIGS_ROOT
        wc.CONFIGS_ROOT = self.tmp

    def tearDown(self):
        wc.CONFIGS_ROOT = self._old

    def test_missing_config_is_none(self):
        self.assertIsNone(wc.load("nope"))
        self.assertIsNone(wc.get_classes("nope"))

    def test_legacy_txt_is_migrated_in_order_and_removed(self):
        with open(os.path.join(self.tmp, "w.txt"), "w", encoding="utf-8") as f:
            f.write("handgun\nrifle\nknife\n")
        self.assertEqual(wc.get_classes("w"), ["handgun", "rifle", "knife"])
        self.assertTrue(os.path.exists(os.path.join(self.tmp, "w.json")))
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "w.txt")))
        # second read comes from JSON, same order
        self.assertEqual(wc.get_classes("w"), ["handgun", "rifle", "knife"])

    def test_empty_legacy_txt_gives_empty_list_not_none(self):
        open(os.path.join(self.tmp, "e.txt"), "w").close()
        self.assertEqual(wc.get_classes("e"), [])

    def test_defaults_are_filled_in(self):
        wc.set_classes("w", ["a"])
        a = wc.get_assistant("w")
        self.assertEqual(a["provider"], wc.PROVIDER_YOLO_WORLD)
        self.assertEqual(a["confidence"], 0.3)
        self.assertEqual(a["yolo_world"]["model"], "yolov8s-world")
        self.assertEqual(a["yolo_world"]["target_classes"], [])

    def test_garbage_in_file_is_normalized(self):
        with open(os.path.join(self.tmp, "g.json"), "w") as f:
            json.dump({"classes": ["a", 3, None],
                       "label_assistant": {"provider": "nonsense", "confidence": 7,
                                           "yolo_world": {"model": "bogus",
                                                          "target_classes": [{"prompt": "x"}, "junk", {"prompt": 5}]}}}, f)
        d = wc.load("g")
        self.assertEqual(d["classes"], ["a"])
        a = d["label_assistant"]
        self.assertEqual(a["provider"], wc.PROVIDER_YOLO_WORLD)
        self.assertEqual(a["confidence"], 0.3)
        self.assertEqual(a["yolo_world"]["model"], "yolov8s-world")
        self.assertEqual(a["yolo_world"]["target_classes"], [{"prompt": "x", "map_to": ""}])

    def test_corrupt_json_is_treated_as_missing(self):
        with open(os.path.join(self.tmp, "c.json"), "w") as f:
            f.write("{not json")
        self.assertIsNone(wc.load("c"))

    def test_all_dropdown_models_are_valid_choices(self):
        self.assertEqual(len(wc.YOLO_WORLD_MODELS), 8)
        for m in wc.YOLO_WORLD_MODELS:
            wc.set_classes("m", ["a"])
            a = wc.get_assistant("m")
            a["yolo_world"]["model"] = m
            wc.set_assistant("m", a)
            self.assertEqual(wc.get_assistant("m")["yolo_world"]["model"], m)

    def test_set_classes_keeps_assistant_settings(self):
        wc.set_classes("w", ["horse"])
        a = wc.get_assistant("w")
        a["confidence"] = 0.55
        a["yolo_world"]["target_classes"] = [{"prompt": "white horse", "map_to": "horse"}]
        wc.set_assistant("w", a)
        wc.set_classes("w", ["horse", "rider"])
        self.assertEqual(wc.get_assistant("w")["confidence"], 0.55)
        self.assertEqual(len(wc.get_assistant("w")["yolo_world"]["target_classes"]), 1)

    def test_deleting_a_class_removes_its_yolo_world_targets(self):
        wc.set_classes("w", ["horse", "rider"])
        a = wc.get_assistant("w")
        a["yolo_world"]["target_classes"] = [{"prompt": "white horse", "map_to": "horse"},
                                             {"prompt": "jockey", "map_to": "rider"}]
        wc.set_assistant("w", a)
        removed = wc.set_classes("w", ["horse"])
        self.assertEqual(removed, ["jockey"])
        self.assertEqual(wc.get_assistant("w")["yolo_world"]["target_classes"],
                         [{"prompt": "white horse", "map_to": "horse"}])

    def test_delete_removes_json_and_legacy(self):
        wc.set_classes("w", ["a"])
        open(os.path.join(self.tmp, "w.txt"), "w").close()
        wc.delete("w")
        self.assertFalse(wc.exists("w"))

    def test_prompt_text_rules(self):
        ok = ["white horse", "Horse", "a", "kuda putih", ""]
        bad = ["white-horse", "horse2", "horse!", "a_b", "x" * 41, "😀"]
        for t in ok:
            self.assertTrue(wc.is_valid_prompt_text(t), t)
        for t in bad:
            self.assertFalse(wc.is_valid_prompt_text(t), t)

    def test_validate_targets(self):
        self.assertEqual(wc.validate_targets([("", ""), (" ", "")]), ([], None))
        t, e = wc.validate_targets([("  white   horse ", "horse")])
        self.assertEqual((t, e), ([{"prompt": "white horse", "map_to": "horse"}], None))
        self.assertIsNotNone(wc.validate_targets([("", "horse")])[1])        # class without prompt
        self.assertIn("Choose", wc.validate_targets([("horse", "")])[1])     # prompt without class
        self.assertIn("more than once",
                      wc.validate_targets([("Horse", "horse"), ("horse", "horse")])[1])


if __name__ == "__main__":
    unittest.main()
