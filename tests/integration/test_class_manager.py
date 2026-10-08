import os
import unittest

from tests.helpers import isolated_workspace

ctx = isolated_workspace()

from utils import workspace_config as wc                      # noqa: E402
from utils.class_manager import ClassManager                  # noqa: E402


class ClassManagerTests(unittest.TestCase):
    def setUp(self):
        self.ws = f"cm{self._testMethodName[-12:]}".replace("_", "")
        self.cm = ClassManager(self.ws)

    def tearDown(self):
        wc.delete(self.ws)

    def test_add_class_validation(self):
        self.assertTrue(self.cm.add_class("cat")[0])
        for bad, why in (("two words", "space"), ("cat!", "symbol"), ("", "empty"),
                         ("cat", "duplicate"), ("x" * 17, "too long")):
            ok, msg = self.cm.add_class(bad)
            self.assertFalse(ok, why)
            self.assertTrue(msg)
        self.assertEqual(self.cm.get_classes(), ["cat"])

    def test_classes_persist_in_the_json_config_in_order(self):
        for c in ("b", "a", "c"):
            self.cm.add_class(c)
        self.assertEqual(wc.get_classes(self.ws), ["b", "a", "c"])
        self.assertEqual(ClassManager(self.ws).get_classes(), ["b", "a", "c"])

    def test_class_colors_use_the_calm_palette_then_generate(self):
        for i in range(10):
            self.cm.add_class(f"c{i}")
        colors = self.cm.get_colors()
        self.assertEqual(len(colors), 10)
        for i, hexc in enumerate(ClassManager.CLASS_PALETTE):
            r, g, b = (int(hexc[k:k + 2], 16) for k in (1, 3, 5))
            self.assertEqual(colors[i], (b, g, r), f"class {i} should be BGR of {hexc}")
        self.assertEqual(len(set(colors)), 10, "every class needs a distinct color")

    def test_delete_class_reindexes_yolo_labels_and_drops_xml_objects(self):
        for c in ("a", "b", "c"):
            self.cm.add_class(c)
        os.makedirs(self.cm.labels_folder, exist_ok=True)
        label = os.path.join(self.cm.labels_folder, "x.txt")
        with open(label, "w") as f:
            f.write("0 0.1 0.1 0.2 0.2\n1 0.5 0.5 0.2 0.2\n2 0.8 0.8 0.1 0.1\n")
        xml = os.path.join(self.cm.voc_dataset, "x.xml")
        with open(xml, "w") as f:
            f.write("<annotation><object><name>a</name></object><object><name>b</name></object></annotation>")

        ok, msg = self.cm.delete_class("b")
        self.assertTrue(ok, msg)
        self.assertEqual(self.cm.get_classes(), ["a", "c"])
        with open(label) as f:
            idx = [line.split()[0] for line in f.read().splitlines()]
        self.assertEqual(idx, ["0", "1"], "c moved from index 2 to 1, b's line is gone")
        with open(xml) as f:
            content = f.read()
        self.assertNotIn("<name>b</name>", content)
        self.assertIn("<name>a</name>", content)

    def test_delete_class_removes_its_yolo_world_targets_and_says_so(self):
        for c in ("horse", "rider"):
            self.cm.add_class(c)
        a = wc.get_assistant(self.ws)
        a["yolo_world"]["target_classes"] = [{"prompt": "white horse", "map_to": "horse"},
                                             {"prompt": "jockey", "map_to": "rider"}]
        wc.set_assistant(self.ws, a)
        ok, msg = self.cm.delete_class("rider")
        self.assertTrue(ok)
        self.assertIn("jockey", msg)
        self.assertEqual(wc.get_assistant(self.ws)["yolo_world"]["target_classes"],
                         [{"prompt": "white horse", "map_to": "horse"}])

    def test_delete_unknown_class(self):
        self.assertFalse(self.cm.delete_class("ghost")[0])


if __name__ == "__main__":
    unittest.main()
