"""End-to-end Label Assistant (the G key): config -> model -> merge into state.
The model itself is faked, so no weights are downloaded and no GPU is needed."""
import os
import unittest

from tests.helpers import isolated_workspace, WORKSPACE

ctx = isolated_workspace()

from utils import config                                        # noqa: E402
from utils import inferenceObjectDetection as inf               # noqa: E402
from utils import workspace_config as wc                        # noqa: E402

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


class FakeModel:
    def __init__(self, boxes):
        self._boxes, self.calls = boxes, []

    def predict(self, img, conf=None, iou=None, verbose=None):
        self.calls.append({"conf": conf, "iou": iou})
        return [Result(self._boxes)]


def set_assistant(provider=wc.PROVIDER_YOLO_WORLD, conf=0.3, targets=()):
    a = wc.get_assistant(WORKSPACE)
    a["provider"], a["confidence"] = provider, conf
    a["yolo_world"]["target_classes"] = list(targets)
    wc.set_assistant(WORKSPACE, a)


class LabelAssistantFlowTests(unittest.TestCase):
    def setUp(self):
        config.state.bboxes = []
        config.state.polygons = []
        self._orig_loader = inf._load_world_model
        self._orig_yolo = inf.YOLO
        if os.path.exists(ctx.model_file):
            os.remove(ctx.model_file)

    def tearDown(self):
        inf._load_world_model = self._orig_loader
        inf.YOLO = self._orig_yolo
        if os.path.exists(ctx.model_file):
            os.remove(ctx.model_file)

    def use_world_model(self, boxes):
        fake = FakeModel(boxes)
        self.loaded_prompts = []

        def loader(name, prompts):
            self.loaded_prompts.append((name, list(prompts)))
            return fake
        inf._load_world_model = loader
        return fake

    # ---------------------------------------------------------------- YOLO-World
    def test_without_target_classes_it_refuses_politely(self):
        set_assistant(targets=[])
        ok, msg = inf.inference_current(IMAGES, 0)
        self.assertFalse(ok)
        self.assertIn("target class", msg)
        self.assertEqual(config.state.bboxes, [])

    def test_prompts_are_mapped_to_workspace_classes(self):
        set_assistant(targets=[{"prompt": "tabby cat", "map_to": "cat"},
                               {"prompt": "puppy", "map_to": "dog"},
                               {"prompt": "kitten", "map_to": "cat"}])
        self.use_world_model([Box((10, 10, 60, 60), 0.9, 0),      # tabby cat
                              Box((100, 100, 160, 160), 0.8, 1),  # puppy
                              Box((200, 20, 250, 70), 0.7, 2)])   # kitten -> cat too
        ok, msg = inf.inference_current(IMAGES, 0)
        self.assertTrue(ok, msg)
        self.assertEqual(sorted(b[4] for b in config.state.bboxes), ["cat", "cat", "dog"])
        self.assertEqual(self.loaded_prompts[0][1], ["tabby cat", "puppy", "kitten"])

    def test_chosen_model_and_confidence_are_used(self):
        a = wc.get_assistant(WORKSPACE)
        a["yolo_world"]["model"] = "yolov8m-worldv2"
        wc.set_assistant(WORKSPACE, a)
        set_assistant(conf=0.55, targets=[{"prompt": "cat", "map_to": "cat"}])
        fake = self.use_world_model([])
        inf.inference_current(IMAGES, 0)
        self.assertEqual(self.loaded_prompts[0][0], "yolov8m-worldv2")
        self.assertEqual(fake.calls[0]["conf"], 0.55)

    def test_overlapping_predictions_do_not_duplicate_existing_boxes(self):
        set_assistant(targets=[{"prompt": "cat", "map_to": "cat"}])
        config.state.bboxes = [[10, 10, 100, 100, "dog"]]          # manual, different class
        self.use_world_model([Box((12, 12, 98, 98), 0.99, 0),      # same object -> dropped
                              Box((200, 150, 300, 230), 0.6, 0)])  # new object -> added
        ok, msg = inf.inference_current(IMAGES, 0)
        self.assertTrue(ok)
        self.assertEqual(len(config.state.bboxes), 2)
        self.assertEqual(config.state.bboxes[0], [10, 10, 100, 100, "dog"], "existing box is untouched")
        self.assertIn("skipped", msg)

    def test_running_twice_does_not_stack_duplicates(self):
        set_assistant(targets=[{"prompt": "cat", "map_to": "cat"}])
        self.use_world_model([Box((10, 10, 60, 60), 0.9, 0)])
        inf.inference_current(IMAGES, 0)
        inf.inference_current(IMAGES, 0)
        self.assertEqual(len(config.state.bboxes), 1)

    def test_load_failure_is_reported_not_raised(self):
        set_assistant(targets=[{"prompt": "cat", "map_to": "cat"}])

        def boom(name, prompts):
            raise RuntimeError("no internet")
        inf._load_world_model = boom
        ok, msg = inf.inference_current(IMAGES, 0)
        self.assertFalse(ok)
        self.assertIn("no internet", msg)

    # ------------------------------------------------------------ trained model
    def test_trained_model_missing_is_reported(self):
        set_assistant(provider=wc.PROVIDER_CUSTOM)
        ok, msg = inf.inference_current(IMAGES, 0)
        self.assertFalse(ok)
        self.assertIn("not found", msg)
        self.assertFalse(inf.custom_model_available())

    def test_trained_model_uses_workspace_class_names(self):
        set_assistant(provider=wc.PROVIDER_CUSTOM, conf=0.4)
        os.makedirs(ctx.model_dir, exist_ok=True)
        open(ctx.model_file, "wb").close()
        self.assertTrue(inf.custom_model_available())
        fake = FakeModel([Box((10, 10, 60, 60), 0.9, 1)])           # class index 1 = "dog"
        inf.YOLO = lambda path: fake
        ok, msg = inf.inference_current(IMAGES, 0)
        self.assertTrue(ok, msg)
        self.assertEqual([b[4] for b in config.state.bboxes], ["dog"])
        self.assertEqual(fake.calls[0]["conf"], 0.4)

    def test_unreadable_image_is_reported(self):
        set_assistant(targets=[{"prompt": "cat", "map_to": "cat"}])
        ok, msg = inf.inference_current(["missing.png"], 0)
        self.assertFalse(ok)
        self.assertIn("missing.png", msg)


class YoloWorldModelCacheTests(unittest.TestCase):
    def test_same_prompts_reuse_the_loaded_model(self):
        made = []

        class Stub:
            def __init__(self, path):
                made.append(path)
                self.classes = None

            def set_classes(self, c):
                self.classes = c

        orig = inf.YOLOWorld
        inf.YOLOWorld = Stub
        inf._world_cache.clear()
        try:
            m1 = inf._load_world_model("yolov8s-world", ["a", "b"])
            m2 = inf._load_world_model("yolov8s-world", ["a", "b"])
            self.assertIs(m1, m2)
            self.assertEqual(len(made), 1)
            inf._load_world_model("yolov8s-world", ["a", "c"])
            self.assertEqual(len(made), 2)
            self.assertTrue(made[0].endswith(os.path.join("_yolo_world", "yolov8s-world.pt")))
        finally:
            inf.YOLOWorld = orig
            inf._world_cache.clear()


if __name__ == "__main__":
    unittest.main()
