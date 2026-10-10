"""The three modes: the front-menu setting, the per-workspace config schema for LocateAnything and custom heads,
and the detector class helpers."""
import os
import shutil
import tempfile
import unittest

from utils import app_settings
from utils import assistant_modes as am
from utils import workspace_config as wc
from utils import detector_classes as dc


class ModeSettingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self._old = app_settings._PATH
        app_settings._PATH = os.path.join(self.tmp, "configs", "_app.json")

    def tearDown(self):
        app_settings._PATH = self._old
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_default_is_the_original_jelibox_mode(self):
        self.assertEqual(am.get_mode(), am.MODE_YOLO_WORLD)

    def test_choice_is_remembered_and_validated(self):
        am.set_mode(am.MODE_LOCATE)
        self.assertEqual(am.get_mode(), am.MODE_LOCATE)
        with self.assertRaises(ValueError):
            am.set_mode("nonsense")
        app_settings.set("mode", "garbage")                      # a hand-edited file must not break the app
        self.assertEqual(am.get_mode(), am.MODE_YOLO_WORLD)

    def test_there_are_exactly_four_modes_with_titles(self):
        ids = [m[0] for m in am.MODES]
        self.assertEqual(ids, [am.MODE_YOLO_WORLD, am.MODE_LOCATE, am.MODE_HEAD, am.MODE_SAM2])
        for mode in ids:
            self.assertTrue(am.title(mode))
        self.assertEqual(am.title("unknown"), am.title(am.DEFAULT_MODE))

    def test_provider_follows_the_mode(self):
        self.assertEqual(am.provider_for(am.MODE_LOCATE, wc.PROVIDER_YOLO_WORLD), wc.PROVIDER_LOCATE)
        self.assertEqual(am.provider_for(am.MODE_HEAD, wc.PROVIDER_CUSTOM), wc.PROVIDER_HEAD)
        self.assertEqual(am.provider_for(am.MODE_SAM2, wc.PROVIDER_CUSTOM), wc.PROVIDER_SAM2)
        # the first mode keeps the old choice between YOLO-World and "my trained model"
        self.assertEqual(am.provider_for(am.MODE_YOLO_WORLD, wc.PROVIDER_CUSTOM), wc.PROVIDER_CUSTOM)
        self.assertEqual(am.provider_for(am.MODE_YOLO_WORLD, wc.PROVIDER_LOCATE), wc.PROVIDER_YOLO_WORLD)


class ConfigSchemaTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self._old = wc.CONFIGS_ROOT
        wc.CONFIGS_ROOT = self.tmp

    def tearDown(self):
        wc.CONFIGS_ROOT = self._old
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_old_configs_get_the_new_blocks_with_defaults(self):
        data = wc._normalize({"classes": ["a"], "label_assistant": {"provider": "yolo_world", "confidence": 0.4}})
        a = data["label_assistant"]
        self.assertEqual(a["confidence"], 0.4)
        self.assertEqual(a["locate_anything"], wc.LOCATE_DEFAULTS)
        self.assertEqual(a["custom_head"], wc.HEAD_DEFAULTS)
        self.assertTrue(a["batch"]["only_unlabeled"])

    def test_new_providers_are_accepted_and_unknown_ones_are_not(self):
        for p in (wc.PROVIDER_LOCATE, wc.PROVIDER_HEAD):
            self.assertEqual(wc._normalize({"label_assistant": {"provider": p}})["label_assistant"]["provider"], p)
        self.assertEqual(wc._normalize({"label_assistant": {"provider": "x"}})["label_assistant"]["provider"],
                         wc.PROVIDER_YOLO_WORLD)

    def test_locate_settings_are_validated(self):
        raw = {"label_assistant": {"locate_anything": {
            "generation_mode": "turbo", "device": "tpu", "short_side": 99999, "max_new_tokens": "many",
            "passes": 3, "temperature": 5, "target_classes": [{"prompt": "cat", "map_to": "c"}, "junk", {"x": 1}]}}}
        la = wc._normalize(raw)["label_assistant"]["locate_anything"]
        self.assertEqual(la["generation_mode"], "hybrid")
        self.assertEqual(la["device"], "auto")
        self.assertEqual(la["short_side"], wc.LOCATE_DEFAULTS["short_side"])
        self.assertEqual(la["max_new_tokens"], wc.LOCATE_DEFAULTS["max_new_tokens"])
        self.assertEqual(la["passes"], 3)
        self.assertEqual(la["temperature"], wc.LOCATE_DEFAULTS["temperature"])
        self.assertEqual(la["target_classes"], [{"prompt": "cat", "map_to": "c"}])

    def test_head_settings_are_validated(self):
        raw = {"label_assistant": {"custom_head": {
            "head_path": "  models/x/head_best.pt ", "detector_classes": [39, 0, 0, -1, "a", True],
            "iou": 7, "min_head_conf": 0.25, "class_map": {"sleeping": "rider", "x": 1, 2: "y"}}}}
        h = wc._normalize(raw)["label_assistant"]["custom_head"]
        self.assertEqual(h["head_path"], "models/x/head_best.pt")
        self.assertEqual(h["detector_classes"], [0, 39])
        self.assertEqual(h["iou"], wc.HEAD_DEFAULTS["iou"])
        self.assertEqual(h["min_head_conf"], 0.25)
        self.assertEqual(h["class_map"], {"sleeping": "rider"})

    def test_empty_detector_classes_fall_back_to_person(self):
        h = wc._normalize({"label_assistant": {"custom_head": {"detector_classes": []}}})["label_assistant"]["custom_head"]
        self.assertEqual(h["detector_classes"], [0])

    def test_round_trip_through_the_file(self):
        a = wc.get_assistant("ws")
        a["provider"] = wc.PROVIDER_LOCATE
        a["locate_anything"]["target_classes"] = [{"prompt": "person", "map_to": "rider"}]
        a["custom_head"]["class_map"] = {"sleeping": "rider"}
        wc.set_assistant("ws", a)
        back = wc.get_assistant("ws")
        self.assertEqual(back["provider"], wc.PROVIDER_LOCATE)
        self.assertEqual(back["locate_anything"]["target_classes"], [{"prompt": "person", "map_to": "rider"}])
        self.assertEqual(back["custom_head"]["class_map"], {"sleeping": "rider"})

    def test_removing_a_class_prunes_every_mode(self):
        wc.set_classes("ws", ["horse", "rider"])
        a = wc.get_assistant("ws")
        a["yolo_world"]["target_classes"] = [{"prompt": "h", "map_to": "horse"}]
        a["locate_anything"]["target_classes"] = [{"prompt": "r", "map_to": "rider"}, {"prompt": "h2", "map_to": "horse"}]
        a["custom_head"]["class_map"] = {"sleeping": "rider", "awake": "horse"}
        wc.set_assistant("ws", a)
        removed = wc.set_classes("ws", ["horse"])
        self.assertEqual(sorted(removed), ["r"])
        a = wc.get_assistant("ws")
        self.assertEqual([t["map_to"] for t in a["locate_anything"]["target_classes"]], ["horse"])
        self.assertEqual(a["custom_head"]["class_map"], {"awake": "horse"})
        self.assertEqual(len(a["yolo_world"]["target_classes"]), 1)


class DetectorClassTests(unittest.TestCase):
    def test_names_and_numbers(self):
        self.assertEqual(dc.parse_class_ids("person"), [0])
        self.assertEqual(dc.parse_class_ids("Bottle, cup; 0, 39"), [0, 39, 41])
        self.assertEqual(dc.COCO_NAMES[0], "person")
        self.assertEqual(dc.COCO_NAMES[39], "bottle")
        self.assertEqual(len(dc.COCO_NAMES), 80)

    def test_unknown_names_and_empty_input_are_refused_with_a_message(self):
        with self.assertRaises(ValueError) as e:
            dc.parse_class_ids("person, unicorn")
        self.assertIn("unicorn", str(e.exception))
        with self.assertRaises(ValueError):
            dc.parse_class_ids("  ")

    def test_custom_name_lists(self):
        self.assertEqual(dc.parse_class_ids("b", names=["a", "b"]), [1])
        self.assertEqual(dc.format_class_ids([1, 7], names=["a", "b"]), "b, 7")

    def test_format_round_trip(self):
        self.assertEqual(dc.format_class_ids([0, 39]), "person, bottle")
        self.assertEqual(dc.parse_class_ids(dc.format_class_ids([0, 39])), [0, 39])


if __name__ == "__main__":
    unittest.main()
