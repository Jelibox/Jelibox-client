"""SAM 2 Dynamic: the reference store (one record per image - nothing is ever counted twice), the prompts built
from it, the memory the engine assembles, and how predicted masks become annotations.

Nothing here loads SAM 2: the engine runs against a stand-in predictor that only counts what it is asked to
encode. (A real run on a small model is done by hand; see CLAUDE.md.)"""
import os
import subprocess
import sys
import tempfile
import unittest
from collections import OrderedDict
from unittest import mock

import numpy as np

from tests.helpers import isolated_workspace

isolated_workspace()

from utils import assistant_modes as am                           # noqa: E402
from utils import assistant_providers as ap                       # noqa: E402
from utils import sam2_dynamic as sd                              # noqa: E402
from utils import workspace_config as wc                          # noqa: E402

CLASSES = ["helmet", "body", "shoes"]


def write_png(path, value=100, size=(120, 160)):
    import cv2
    cv2.imwrite(path, np.full((size[0], size[1], 3), value, np.uint8))
    return path


class ReferenceStoreTests(unittest.TestCase):
    def setUp(self):
        self.store = sd.ReferenceStore()

    def test_the_same_image_is_stored_once_however_often_you_come_back_to_it(self):
        # image 1, then 2, back to 1, back to 2 - the case that must never double a reference
        boxes_1 = [[10, 10, 90, 90, "helmet"]]
        boxes_2 = [[20, 20, 100, 100, "body"]]
        for name, boxes in (("1.png", boxes_1), ("2.png", boxes_2), ("1.png", boxes_1), ("2.png", boxes_2)):
            self.store.remember(name, boxes, [])
        self.assertEqual(len(self.store), 2)
        self.assertEqual(self.store.items_for("1.png"), [("helmet", (10.0, 10.0, 90.0, 90.0))])

    def test_leaving_an_image_again_replaces_its_annotations_instead_of_adding_to_them(self):
        self.store.remember("1.png", [[10, 10, 90, 90, "helmet"], [10, 100, 90, 190, "body"]], [])
        self.store.remember("1.png", [[12, 12, 92, 92, "helmet"]], [])             # you deleted the body
        self.assertEqual(self.store.items_for("1.png"), [("helmet", (12.0, 12.0, 92.0, 92.0))])

    def test_an_image_left_without_annotations_is_dropped(self):
        self.store.remember("1.png", [[10, 10, 90, 90, "helmet"]], [])
        self.assertEqual(self.store.remember("1.png", [], []), 0)
        self.assertEqual(len(self.store), 0)

    def test_polygons_count_through_their_bounding_box(self):
        self.store.remember("1.png", [], [[[(10, 10), (50, 10), (50, 40), (10, 40)], "shoes"]])
        self.assertEqual(self.store.items_for("1.png"), [("shoes", (10.0, 10.0, 50.0, 40.0))])

    def test_specks_are_not_references(self):
        self.assertEqual(self.store.remember("1.png", [[10, 10, 14, 14, "helmet"]], []), 0)
        self.assertEqual(len(self.store), 0)

    def test_the_newest_images_are_chosen_and_the_one_being_annotated_never_is(self):
        for n in range(1, 6):
            self.store.remember(f"{n}.png", [[10, 10, 90, 90, "helmet"]], [])
        self.store.remember("2.png", [[10, 10, 90, 90, "helmet"]], [])             # visited again: now the newest
        picked = [p for p, _ in self.store.select(3, CLASSES)]
        self.assertEqual(picked, ["4.png", "5.png", "2.png"])
        picked = [p for p, _ in self.store.select(3, CLASSES, exclude="2.png")]
        self.assertEqual(picked, ["3.png", "4.png", "5.png"])

    def test_annotations_of_classes_that_no_longer_exist_are_ignored(self):
        self.store.remember("1.png", [[10, 10, 90, 90, "helmet"], [10, 10, 90, 90, "gone"]], [])
        self.assertEqual(self.store.select(5, CLASSES), [("1.png", [("helmet", (10.0, 10.0, 90.0, 90.0))])])
        self.assertEqual(self.store.select(5, ["gone2"]), [])

    def test_forget_and_clear(self):
        self.store.remember("1.png", [[10, 10, 90, 90, "helmet"]], [])
        self.store.remember("2.png", [[10, 10, 90, 90, "helmet"]], [])
        self.store.forget("1.png")
        self.assertNotIn("1.png", self.store)
        self.store.clear()
        self.assertEqual(len(self.store), 0)

    def test_loading_labeled_images_from_disk_keeps_one_record_each_and_the_newest_limit(self):
        tmp = tempfile.mkdtemp()
        voc = os.path.join(tmp, "voc")
        os.makedirs(voc)
        records = []
        for n in range(4):
            img = write_png(os.path.join(tmp, f"i{n}.png"))
            records.append({"source": img, "base": f"i{n}"})
            with open(os.path.join(voc, f"i{n}.xml"), "w") as f:
                f.write("<annotation><size><width>160</width><height>120</height></size><object><name>helmet</name>"
                        "<bndbox><xmin>10</xmin><ymin>10</ymin><xmax>90</xmax><ymax>90</ymax></bndbox></object></annotation>")
            os.utime(os.path.join(voc, f"i{n}.xml"), (1000 + n, 1000 + n))
        self.assertEqual(self.store.load_labeled(records, voc, limit=2), 2)
        self.assertEqual([os.path.basename(p) for p, _ in self.store.select(9, CLASSES)], ["i2.png", "i3.png"])
        self.store.load_labeled(records, voc, limit=2)                              # again: still no duplicates
        self.assertEqual(len(self.store), 2)


class PromptTests(unittest.TestCase):
    def test_one_box_per_class_the_largest_and_ids_follow_the_class_list(self):
        items = [("body", (0, 0, 10, 10)), ("body", (0, 0, 50, 50)), ("helmet", (5, 5, 25, 25)), ("alien", (0, 0, 9, 9))]
        boxes, ids = sd.reference_prompts(items, CLASSES)
        self.assertEqual(ids, [0, 1])
        self.assertEqual(boxes, [[5, 5, 25, 25], [0, 0, 50, 50]])

    def test_nothing_to_prompt_with(self):
        self.assertEqual(sd.reference_prompts([("alien", (0, 0, 9, 9))], CLASSES), ([], []))


class MaskToAnnotationTests(unittest.TestCase):
    def blob(self, shape, *rects):
        mask = np.zeros(shape, bool)
        for x1, y1, x2, y2 in rects:
            mask[y1:y2, x1:x2] = True
        return mask

    def test_every_separate_piece_of_a_mask_becomes_its_own_annotation(self):
        mask = self.blob((200, 300), (10, 10, 60, 80), (150, 100, 230, 190))
        preds = sd.masks_to_predictions([mask], [0.2], [1], CLASSES, 0.05, polygon=False)
        self.assertEqual(sorted(p["rect"] for p in preds), [(10, 10, 60, 80), (150, 100, 230, 190)])
        self.assertTrue(all(p["cls"] == "body" and p["conf"] == 0.2 and "poly" not in p for p in preds))

    def test_polygon_output_has_an_outline(self):
        preds = sd.masks_to_predictions([self.blob((200, 300), (10, 10, 60, 80))], [0.3], [0], CLASSES, 0.0, polygon=True)
        self.assertEqual(len(preds), 1)
        self.assertGreaterEqual(len(preds[0]["poly"]), 3)
        self.assertEqual(preds[0]["cls"], "helmet")

    def test_low_scores_empty_masks_and_specks_are_dropped(self):
        masks = [self.blob((200, 300), (10, 10, 60, 80)), self.blob((200, 300)), self.blob((200, 300), (5, 5, 8, 8))]
        preds = sd.masks_to_predictions(masks, [0.01, 0.9, 0.9], [0, 1, 2], CLASSES, 0.05, polygon=False)
        self.assertEqual(preds, [])

    def test_a_slot_that_is_not_a_class_is_ignored(self):
        preds = sd.masks_to_predictions([self.blob((200, 300), (10, 10, 60, 80))], [0.5], [7], CLASSES, 0.0, False)
        self.assertEqual(preds, [])


# ---------------------------------------------------------------------------------------------- the engine
class FakeResults:
    """What predictor(source=image) returns: boxes.cls are the output rows that survived the score filter."""
    def __init__(self, rows, masks, scores):
        self.boxes = mock.Mock()
        self.boxes.cls.tolist.return_value = rows
        self.boxes.conf.tolist.return_value = scores
        self.boxes.__len__ = lambda s: len(rows)
        self.masks = mock.Mock()
        self.masks.data.cpu.return_value.numpy.return_value = np.array(masks)


class FakePredictor:
    """Counts what it is asked to encode; memory entries are plain dicts."""
    def __init__(self):
        self.memory_bank = []
        self.obj_idx_set = set()
        self.encoded = []               # image values, one per update_memory call
        self.next_results = None

    def __call__(self, source=None, bboxes=None, obj_ids=None, update_memory=False):
        if update_memory:
            self.encoded.append(int(source[0, 0, 0]))
            self.memory_bank.append({"image": int(source[0, 0, 0]), "ids": list(obj_ids)})
            self.obj_idx_set.update(obj_ids)
            return []
        return [self.next_results]


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.engine = sd.Sam2Engine()
        self.fake = FakePredictor()

        def fake_load(engine, weights, imgsz, slots, status):
            engine.predictor, engine.key = self.fake, (weights, imgsz, slots)
        patcher = mock.patch.object(sd.Sam2Engine, "_load", fake_load)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.images = {n: write_png(os.path.join(self.tmp, f"{n}.png"), value=n * 20) for n in (1, 2, 3)}

    def refs(self, *numbers, boxes=None):
        boxes = boxes or [("helmet", (10, 10, 90, 90)), ("body", (10, 100, 90, 190))]
        return [(self.images[n], boxes) for n in numbers]

    def build(self, *numbers, **kw):
        return self.engine.build("w.pt", 1024, CLASSES, self.refs(*numbers, **kw), 0.05, False)

    def test_the_memory_has_exactly_one_entry_per_reference_image(self):
        self.build(1, 2)
        self.assertEqual([e["image"] for e in self.fake.memory_bank], [20, 40])
        self.assertEqual(self.fake.obj_idx_set, {0, 1})

    def test_building_again_does_not_encode_or_add_anything_twice(self):
        self.build(1, 2)
        self.build(1, 2)
        self.build(2, 1)                                                           # order does not matter either
        self.assertEqual(self.fake.encoded, [20, 40])
        self.assertEqual(len(self.fake.memory_bank), 2)

    def test_back_and_forth_between_two_images_keeps_two_entries(self):
        for numbers in ((1, 2), (1, 2), (1, 2)):
            self.build(*numbers)
        self.assertEqual(len(self.fake.memory_bank), 2)
        self.assertEqual(len({e["image"] for e in self.fake.memory_bank}), 2)

    def test_editing_one_image_re_encodes_only_that_one_and_replaces_its_entry(self):
        self.build(1, 2)
        self.engine.build("w.pt", 1024, CLASSES,
                          [(self.images[1], [("helmet", (12, 12, 95, 95))]), (self.images[2], self.refs(2)[0][1])], 0.05, False)
        self.assertEqual(self.fake.encoded, [20, 40, 20])                          # image 1 again, image 2 not
        self.assertEqual(len(self.fake.memory_bank), 2)
        self.assertEqual(self.fake.obj_idx_set, {0, 1})                            # image 2 still teaches 'body'

    def test_an_image_that_is_no_longer_a_reference_leaves_the_memory(self):
        self.build(1, 2, 3)
        self.build(2, 3)
        self.assertEqual(sorted(e["image"] for e in self.fake.memory_bank), [40, 60])

    def test_unreadable_references_are_skipped_and_none_readable_is_an_error(self):
        refs = self.refs(1) + [(os.path.join(self.tmp, "missing.png"), self.refs(1)[0][1])]
        self.engine.build("w.pt", 1024, CLASSES, refs, 0.05, False)
        self.assertEqual(len(self.fake.memory_bank), 1)
        with self.assertRaises(RuntimeError):
            self.engine.build("w.pt", 1024, CLASSES, [(os.path.join(self.tmp, "missing.png"), self.refs(1)[0][1])], 0.05, False)

    def test_predictions_use_the_slot_each_output_row_belongs_to(self):
        predict = self.build(1, 2)                                                 # slots {0, 1}
        mask = np.zeros((100, 100), bool)
        mask[10:60, 10:60] = True
        self.fake.next_results = FakeResults(rows=[1], masks=[mask], scores=[0.3])  # only row 1 survived: slot 1 = body
        preds, is_polygon = predict(np.zeros((100, 100, 3), np.uint8))
        self.assertEqual([(p["cls"], p["rect"]) for p in preds], [("body", (10, 10, 60, 60))])
        self.assertFalse(is_polygon)

    def test_a_query_with_no_surviving_object_gives_no_annotations(self):
        predict = self.build(1)
        self.fake.next_results = FakeResults(rows=[], masks=np.zeros((0, 10, 10), bool), scores=[])
        self.assertEqual(predict(np.zeros((10, 10, 3), np.uint8)), ([], False))

    def test_release_forgets_everything(self):
        self.build(1)
        self.engine.release()
        self.assertFalse(self.engine.loaded)
        self.assertEqual(len(self.engine.entries), 0)


class ProviderTests(unittest.TestCase):
    def setUp(self):
        sd.store.clear()
        self.addCleanup(sd.store.clear)
        self.settings = dict(wc.SAM2_DEFAULTS)

    def test_with_nothing_learned_yet_the_message_says_what_to_do(self):
        with mock.patch.object(sd, "predictor_problem", return_value=None):
            with self.assertRaises(ap.AssistantError) as ctx:
                ap.sam2_predictor(self.settings, CLASSES, tempfile.mkdtemp())
        self.assertIn("annotated", str(ctx.exception))
        self.assertIn("Infer", str(ctx.exception))

    def test_a_workspace_without_classes_is_explained(self):
        with mock.patch.object(sd, "predictor_problem", return_value=None):
            with self.assertRaises(ap.AssistantError) as ctx:
                ap.sam2_predictor(self.settings, [], tempfile.mkdtemp())
        self.assertIn("classes", str(ctx.exception))

    def test_an_old_ultralytics_is_reported_clearly(self):
        with mock.patch.object(sd, "predictor_problem", return_value="SAM 2 Dynamic needs a recent ultralytics"):
            with self.assertRaises(ap.AssistantError) as ctx:
                ap.sam2_predictor(self.settings, CLASSES, tempfile.mkdtemp())
        self.assertIn("recent ultralytics", str(ctx.exception))

    def test_the_image_being_annotated_is_not_its_own_reference(self):
        sd.store.remember("a.png", [[10, 10, 90, 90, "helmet"]], [])
        sd.store.remember("b.png", [[10, 10, 90, 90, "helmet"]], [])
        seen = {}

        def fake_build(weights, imgsz, classes, references, min_score, polygon, status=None):
            seen["refs"] = [p for p, _ in references]
            seen["args"] = (os.path.basename(weights), imgsz, min_score, polygon)
            return lambda bgr: ([], polygon)

        with mock.patch.object(sd, "predictor_problem", return_value=None), \
                mock.patch.object(sd.engine, "build", side_effect=fake_build):
            ap.sam2_predictor(self.settings, CLASSES, tempfile.mkdtemp(), polygon=True, exclude="b.png")
        self.assertEqual(seen["refs"], ["a.png"])
        self.assertEqual(seen["args"], ("sam2.1_b.pt", 1024, 0.05, True))

    def test_a_failure_while_loading_is_wrapped_and_frees_the_model(self):
        sd.store.remember("a.png", [[10, 10, 90, 90, "helmet"]], [])
        with mock.patch.object(sd, "predictor_problem", return_value=None), \
                mock.patch.object(sd.engine, "build", side_effect=RuntimeError("download failed")), \
                mock.patch.object(sd.engine, "release") as release:
            with self.assertRaises(ap.AssistantError) as ctx:
                ap.sam2_predictor(self.settings, CLASSES, tempfile.mkdtemp())
        self.assertIn("download failed", str(ctx.exception))
        release.assert_called_once()


class ConfigAndModeTests(unittest.TestCase):
    def test_defaults_and_validation(self):
        self.assertEqual(wc._normalize({})["label_assistant"]["sam2_dynamic"], wc.SAM2_DEFAULTS)
        raw = {"label_assistant": {"sam2_dynamic": {"model": "evil.pt", "imgsz": 999, "max_references": 99, "min_score": -1}}}
        self.assertEqual(wc._normalize(raw)["label_assistant"]["sam2_dynamic"], wc.SAM2_DEFAULTS)
        good = {"label_assistant": {"sam2_dynamic": {"model": "sam2.1_t.pt", "imgsz": 512, "max_references": 3, "min_score": 0.2}}}
        self.assertEqual(wc._normalize(good)["label_assistant"]["sam2_dynamic"],
                         {"model": "sam2.1_t.pt", "imgsz": 512, "max_references": 3, "min_score": 0.2})
        self.assertEqual(wc._normalize({"label_assistant": {"provider": "sam2_dynamic"}})["label_assistant"]["provider"],
                         wc.PROVIDER_SAM2)

    def test_it_is_the_fourth_mode_and_has_its_own_provider(self):
        self.assertEqual([m[0] for m in am.MODES][-1], am.MODE_SAM2)
        self.assertEqual(am.title(am.MODE_SAM2), "SAM 2 Dynamic")
        self.assertEqual(am.provider_for(am.MODE_SAM2, wc.PROVIDER_YOLO_WORLD), wc.PROVIDER_SAM2)

    def test_importing_the_module_does_not_load_torch(self):
        code = "import sys; from utils import sam2_dynamic; print([m for m in ('torch', 'ultralytics') if m in sys.modules])"
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                             cwd=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        self.assertEqual(out.stdout.strip(), "[]", out.stderr)

    def test_model_catalogue_is_consistent(self):
        self.assertEqual(sd.MODELS, wc.SAM2_MODELS)
        self.assertEqual(sd.IMAGE_SIZES, wc.SAM2_IMAGE_SIZES)
        self.assertEqual(set(sd.MODEL_INFO), set(sd.MODELS))
        self.assertTrue(sd.weights_path("/x", "sam2.1_t.pt").replace("\\", "/").endswith("models/_sam2/sam2.1_t.pt"))


if __name__ == "__main__":
    unittest.main()
