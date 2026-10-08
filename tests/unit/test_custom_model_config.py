"""Settings of a model chosen with "Browse custom model" in the original mode."""
import shutil
import tempfile
import unittest

from utils import workspace_config as wc


class CustomModelConfigTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self._old = wc.CONFIGS_ROOT
        wc.CONFIGS_ROOT = self.tmp

    def tearDown(self):
        wc.CONFIGS_ROOT = self._old
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_old_configs_get_an_empty_custom_model_block(self):
        a = wc._normalize({"label_assistant": {"provider": "custom_model"}})["label_assistant"]
        self.assertEqual(a["custom_model"], {"path": "", "class_map": {}})

    def test_path_and_mapping_are_validated(self):
        raw = {"label_assistant": {"custom_model": {"path": "  C:/m/best.pt ", "class_map": {"a": "b", "c": 1, 2: "d"}}}}
        cm = wc._normalize(raw)["label_assistant"]["custom_model"]
        self.assertEqual(cm, {"path": "C:/m/best.pt", "class_map": {"a": "b"}})
        junk = wc._normalize({"label_assistant": {"custom_model": "nope"}})["label_assistant"]["custom_model"]
        self.assertEqual(junk, {"path": "", "class_map": {}})

    def test_removing_a_class_prunes_the_model_mapping_too(self):
        wc.set_classes("ws", ["horse", "rider"])
        a = wc.get_assistant("ws")
        a["custom_model"] = {"path": "m.pt", "class_map": {"h": "horse", "r": "rider"}}
        wc.set_assistant("ws", a)
        wc.set_classes("ws", ["horse"])
        self.assertEqual(wc.get_assistant("ws")["custom_model"], {"path": "m.pt", "class_map": {"h": "horse"}})

    def test_round_trip_through_the_file(self):
        a = wc.get_assistant("ws")
        a["custom_model"] = {"path": "D:/models/best.pt", "class_map": {"x": "y"}}
        wc.set_assistant("ws", a)
        self.assertEqual(wc.get_assistant("ws")["custom_model"], {"path": "D:/models/best.pt", "class_map": {"x": "y"}})


if __name__ == "__main__":
    unittest.main()
