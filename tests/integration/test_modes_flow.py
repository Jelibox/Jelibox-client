"""The three modes end to end against the isolated workspace: mode -> provider -> predictions -> merge / files.
The models themselves are faked, so nothing is downloaded and no GPU is needed."""
import os
import unittest
from unittest import mock

from tests.helpers import isolated_workspace, WORKSPACE

ctx = isolated_workspace()

from utils import config                                        # noqa: E402
from utils import app_settings, assistant_modes as am            # noqa: E402
from utils import file_handler as fh                             # noqa: E402
from utils import inferenceObjectDetection as inf                # noqa: E402
from utils import workspace_config as wc                         # noqa: E402
from utils.assistant_providers import AssistantError             # noqa: E402
from utils.auto_annotate import run_batch                        # noqa: E402

IMAGES = ["img1.png", "img2.png", "img3.png"]


def fake_predictor(preds=None, is_polygon=False):
    calls = []

    def predict(bgr):
        calls.append(bgr.shape)
        return list(preds if preds is not None else [{"rect": (10, 10, 100, 100), "conf": 0.9, "cls": "cat"}]), is_polygon
    predict.calls = calls
    return predict


def set_assistant(**changes):
    a = wc.get_assistant(WORKSPACE)
    for k, v in changes.items():
        a[k] = v
    wc.set_assistant(WORKSPACE, a)


class ModeFlowBase(unittest.TestCase):
    def setUp(self):
        config.state.bboxes, config.state.polygons = [], []
        self._mode = app_settings.get("mode")
        for d in (ctx.voc, ctx.labels):
            if os.path.isdir(d):
                for f in os.listdir(d):
                    os.remove(os.path.join(d, f))
        a = wc.get_assistant(WORKSPACE)
        a["provider"] = wc.PROVIDER_YOLO_WORLD
        a["locate_anything"]["target_classes"] = [{"prompt": "tabby cat", "map_to": "cat"}]
        wc.set_assistant(WORKSPACE, a)

    def tearDown(self):
        app_settings.set("mode", self._mode)


class SaveAnnotationsTests(ModeFlowBase):
    def test_voc_and_yolo_files_are_written_without_touching_the_gui_state(self):
        config.state.bboxes = [[1, 1, 2, 2, "dog"]]
        fh.save_annotations("img1.png", (240, 320, 3), [[10, 20, 110, 120, "cat"]], [], config.CLASSLIST)
        self.assertEqual(config.state.bboxes, [[1, 1, 2, 2, "dog"]])
        boxes, polys = fh.load_annotation_local("img1.png")
        self.assertEqual((boxes, polys), ([[10, 20, 110, 120, "cat"]], []))
        with open(os.path.join(ctx.labels, "img1.txt")) as f:
            cls, cx, cy, w, h = f.read().split()
        self.assertEqual(cls, str(config.CLASSLIST.index("cat")))
        self.assertAlmostEqual(float(cx), 60 / 320, places=4)
        self.assertAlmostEqual(float(h), 100 / 240, places=4)


class LocateModeTests(ModeFlowBase):
    def test_g_uses_locate_anything_in_that_mode_and_maps_categories_to_classes(self):
        am.set_mode(am.MODE_LOCATE)
        self.assertEqual(inf.current_provider(), wc.PROVIDER_LOCATE)
        self.assertTrue(inf.is_heavy())
        with mock.patch.object(inf, "locate_predictor", return_value=fake_predictor()) as make:
            ok, msg = inf.inference_current(IMAGES, 0)
        self.assertTrue(ok, msg)
        self.assertEqual(config.state.bboxes, [[10, 10, 100, 100, "cat"]])
        settings = make.call_args[0][0]
        self.assertEqual(settings["target_classes"], [{"prompt": "tabby cat", "map_to": "cat"}])

    def test_without_a_category_it_refuses_politely(self):
        am.set_mode(am.MODE_LOCATE)
        a = wc.get_assistant(WORKSPACE)
        a["locate_anything"]["target_classes"] = []
        wc.set_assistant(WORKSPACE, a)
        ok, msg = inf.inference_current(IMAGES, 0)
        self.assertFalse(ok)
        self.assertIn("target class", msg)

    def test_yolo_world_mode_is_not_heavy(self):
        am.set_mode(am.MODE_YOLO_WORLD)
        self.assertFalse(inf.is_heavy())
        self.assertEqual(inf.current_provider(), wc.PROVIDER_YOLO_WORLD)


class CustomHeadModeTests(ModeFlowBase):
    def test_g_uses_the_head_model_and_passes_the_detector_confidence(self):
        am.set_mode(am.MODE_HEAD)
        set_assistant(confidence=0.42)
        pred = fake_predictor([{"rect": (5, 5, 60, 90), "conf": 0.8, "cls": "dog"}])
        with mock.patch.object(inf, "head_predictor", return_value=pred) as make:
            ok, msg = inf.inference_current(IMAGES, 0)
        self.assertTrue(ok, msg)
        self.assertEqual(config.state.bboxes, [[5, 5, 60, 90, "dog"]])
        args = make.call_args[0]
        self.assertEqual(args[1], 0.42)                       # detector confidence
        self.assertEqual(args[2], config.CLASSLIST)           # workspace classes for the mapping

    def test_conf_override_wins_for_one_call_without_being_saved(self):
        am.set_mode(am.MODE_HEAD)
        set_assistant(confidence=0.3)
        with mock.patch.object(inf, "head_predictor", return_value=fake_predictor()) as make:
            inf.inference_current(IMAGES, 0, conf=0.77)
        self.assertEqual(make.call_args[0][1], 0.77)
        self.assertEqual(wc.get_assistant(WORKSPACE)["confidence"], 0.3)

    def test_provider_errors_come_back_as_a_message(self):
        am.set_mode(am.MODE_HEAD)
        ok, msg = inf.inference_current(IMAGES, 0)           # no head checkpoint configured
        self.assertFalse(ok)
        self.assertIn("head_best.pt", msg)


class BatchOverTheWorkspaceTests(ModeFlowBase):
    def run_all(self, predictor, **kw):
        return run_batch(
            IMAGES, read_image=lambda n: inf.read_image(os.path.join(config.input_folder, n)), predict=predictor,
            load_annotations=fh.load_annotation_local,
            save_annotations=lambda n, s, b, p: fh.save_annotations(n, s, b, p, config.CLASSLIST),
            classes=config.CLASSLIST, **kw)

    def test_every_image_in_the_folder_gets_labels_on_disk(self):
        pred = fake_predictor()
        res = self.run_all(pred)
        self.assertEqual((res.annotated, res.added, len(pred.calls)), (3, 3, 3))
        for name in IMAGES:
            base = os.path.splitext(name)[0]
            self.assertTrue(os.path.exists(os.path.join(ctx.voc, base + ".xml")), name)
            self.assertTrue(os.path.exists(os.path.join(ctx.labels, base + ".txt")), name)
        self.assertEqual(fh.load_annotation_local("img2.png")[0], [[10, 10, 100, 100, "cat"]])

    def test_second_run_skips_what_is_already_labelled(self):
        self.run_all(fake_predictor())
        pred = fake_predictor()
        res = self.run_all(pred, only_unlabeled=True)
        self.assertEqual((res.skipped_labeled, len(pred.calls)), (3, 0))
        res = self.run_all(fake_predictor(), only_unlabeled=False)
        self.assertEqual(res.added, 0, "the same boxes overlap what is there")

    def test_images_are_read_even_when_the_path_is_not_ascii(self):
        from PIL import Image
        odd = os.path.join(config.input_folder, "ünï.png")
        Image.new("RGB", (40, 30)).save(odd)
        try:
            self.assertEqual(inf.read_image(odd).shape[:2], (30, 40))
            self.assertIsNone(inf.read_image(os.path.join(config.input_folder, "missing.png")))
        finally:
            os.remove(odd)


if __name__ == "__main__":
    unittest.main()
