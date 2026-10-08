"""Original mode, "Browse custom model": a model from anywhere (e.g. a fine-tuned YOLOv8m), classes mapped by NAME.
The model is faked, so nothing is downloaded and no GPU is needed."""
import os
import sys
import types
import unittest
from unittest import mock

from tests.helpers import isolated_workspace, WORKSPACE

ctx = isolated_workspace()

from utils import config                                        # noqa: E402
from utils import app_settings, assistant_modes as am            # noqa: E402
from utils import assistant_providers as ap                      # noqa: E402
from utils import inferenceObjectDetection as inf                # noqa: E402
from utils import workspace_config as wc                         # noqa: E402

IMAGES = ["img1.png", "img2.png", "img3.png"]


class Vec(list):
    def tolist(self):
        return list(self)

    def item(self):
        return self[0]


class Box:
    def __init__(self, xyxy, conf, cls):
        self.xyxy, self.conf, self.cls = [Vec(xyxy)], [Vec([conf])], [Vec([cls])]


class Result:
    def __init__(self, boxes):
        self.boxes, self.masks = boxes, None


class FakeYolo:
    """Knows its class names and returns canned boxes."""

    def __init__(self, names, boxes):
        self.names, self._boxes, self.calls = names, boxes, []

    def predict(self, img, conf=None, iou=None, verbose=None):
        self.calls.append(conf)
        return [Result(self._boxes)]


class BrowsedModelTests(unittest.TestCase):
    def setUp(self):
        config.state.bboxes, config.state.polygons = [], []
        self._mode = app_settings.get("mode")
        am.set_mode(am.MODE_YOLO_WORLD)
        self.file = os.path.join(ctx.root, "elsewhere", "finetuned_v8m.pt")
        os.makedirs(os.path.dirname(self.file), exist_ok=True)
        open(self.file, "wb").close()
        self._yolo = inf.YOLO
        if os.path.exists(ctx.model_file):
            os.remove(ctx.model_file)

    def tearDown(self):
        inf.YOLO = self._yolo
        a = wc.get_assistant(WORKSPACE)
        a["provider"], a["confidence"] = wc.PROVIDER_YOLO_WORLD, 0.3
        a["custom_model"] = {"path": "", "class_map": {}}
        wc.set_assistant(WORKSPACE, a)
        app_settings.set("mode", self._mode)
        if os.path.exists(ctx.model_file):
            os.remove(ctx.model_file)

    def use(self, path, class_map, names, boxes, conf=0.3):
        a = wc.get_assistant(WORKSPACE)
        a["provider"], a["confidence"] = wc.PROVIDER_CUSTOM, conf
        a["custom_model"] = {"path": path, "class_map": class_map}
        wc.set_assistant(WORKSPACE, a)
        fake = FakeYolo(names, boxes)
        self.opened = []
        inf.YOLO = lambda p: (self.opened.append(p), fake)[1]
        return fake

    def test_a_browsed_model_is_available_without_the_workspace_model(self):
        self.assertFalse(inf.custom_model_available())
        self.use(self.file, {}, {0: "cat"}, [])
        self.assertTrue(inf.custom_model_available())
        self.assertEqual(inf.custom_model_file(), self.file)

    def test_a_missing_browsed_file_is_not_available_and_says_so(self):
        self.use(os.path.join(ctx.root, "gone.pt"), {"x": "cat"}, {0: "x"}, [])
        self.assertFalse(inf.custom_model_available())
        ok, msg = inf.inference_current(IMAGES, 0)
        self.assertFalse(ok)
        self.assertIn("gone.pt", msg)

    def test_classes_are_mapped_by_name_not_by_position(self):
        # the model's order is the reverse of the workspace's (cat, dog): index mapping would swap the labels
        self.use(self.file, {}, {0: "dog", 1: "cat", 2: "bird"},
                 [Box((10, 10, 60, 60), 0.9, 0), Box((100, 100, 160, 160), 0.8, 1), Box((200, 20, 250, 70), 0.7, 2)])
        ok, msg = inf.inference_current(IMAGES, 0)
        self.assertTrue(ok, msg)
        self.assertEqual(self.opened, [self.file])
        self.assertEqual(sorted((b[4], b[0]) for b in config.state.bboxes), [("cat", 100), ("dog", 10)])
        self.assertEqual(len(config.state.bboxes), 2, "bird is not a workspace class: skipped")

    def test_explicit_mapping_wins_and_unmapped_classes_are_skipped(self):
        self.use(self.file, {"pup": "dog", "kitty": "cat", "ghost": "nope"},
                 {0: "pup", 1: "kitty", 2: "ghost", 3: "tree"},
                 [Box((10, 10, 60, 60), 0.9, 0), Box((100, 100, 160, 160), 0.8, 1), Box((1, 1, 9, 9), 0.9, 2),
                  Box((200, 20, 250, 70), 0.7, 3)])
        ok, msg = inf.inference_current(IMAGES, 0)
        self.assertTrue(ok, msg)
        self.assertEqual(sorted(b[4] for b in config.state.bboxes), ["cat", "dog"])

    def test_no_mapped_class_is_an_error_with_a_hint(self):
        self.use(self.file, {}, {0: "foo", 1: "bar"}, [])
        ok, msg = inf.inference_current(IMAGES, 0)
        self.assertFalse(ok)
        self.assertIn("foo", msg)
        self.assertIn("map", msg)

    def test_the_confidence_setting_is_used(self):
        fake = self.use(self.file, {"a": "cat"}, {0: "a"}, [], conf=0.61)
        inf.inference_current(IMAGES, 0)
        self.assertEqual(fake.calls, [0.61])

    def test_the_workspaces_own_model_still_maps_by_position(self):
        a = wc.get_assistant(WORKSPACE)
        a["provider"] = wc.PROVIDER_CUSTOM
        wc.set_assistant(WORKSPACE, a)
        os.makedirs(ctx.model_dir, exist_ok=True)
        open(ctx.model_file, "wb").close()
        fake = FakeYolo({0: "whatever", 1: "names"}, [Box((10, 10, 60, 60), 0.9, 1)])
        inf.YOLO = lambda p: fake
        ok, msg = inf.inference_current(IMAGES, 0)
        self.assertTrue(ok, msg)
        self.assertEqual([b[4] for b in config.state.bboxes], ["dog"], "index 1 = the workspace's second class")

    def test_it_also_works_for_auto_annotate_all(self):
        from utils import file_handler as fh
        from utils.auto_annotate import run_batch
        for d in (ctx.voc, ctx.labels):
            for f in os.listdir(d):
                os.remove(os.path.join(d, f))
        self.use(self.file, {"kitty": "cat"}, {0: "kitty", 1: "tree"},
                 [Box((10, 10, 90, 90), 0.9, 0), Box((100, 100, 150, 150), 0.9, 1)])
        predict = inf.build_predictor()
        res = run_batch(
            IMAGES, read_image=lambda n: inf.read_image(os.path.join(config.input_folder, n)), predict=predict,
            load_annotations=fh.load_annotation_local,
            save_annotations=lambda n, s, b, p: fh.save_annotations(n, s, b, p, config.CLASSLIST),
            classes=config.CLASSLIST)
        self.assertEqual((res.annotated, res.added), (3, 3))
        self.assertEqual(fh.load_annotation_local("img1.png")[0], [[10, 10, 90, 90, "cat"]])

    def test_model_class_names_reads_names_and_fills_gaps(self):
        fake_mod = types.ModuleType("ultralytics")
        fake_mod.YOLO = lambda path: types.SimpleNamespace(names={0: "a", 2: "c"})
        with mock.patch.dict(sys.modules, {"ultralytics": fake_mod}):
            self.assertEqual(ap.model_class_names(self.file), ["a", "", "c"])
        with self.assertRaises(ap.AssistantError):
            ap.model_class_names(os.path.join(ctx.root, "nope.pt"))
        broken = types.ModuleType("ultralytics")

        def boom(path):
            raise RuntimeError("not a model")
        broken.YOLO = boom
        with mock.patch.dict(sys.modules, {"ultralytics": broken}):
            with self.assertRaises(ap.AssistantError) as e:
                ap.model_class_names(self.file)
        self.assertIn("not a model", str(e.exception))


if __name__ == "__main__":
    unittest.main()
