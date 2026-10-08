"""Training a custom head: pairing labels with images, the position-based split, and the trainer itself.
The data tests need nothing heavy. The trainer tests need torch + ultralytics (skipped without them) and swap the real
detector for a tiny fake one, so no model is downloaded."""
import contextlib
import io
import json
import os
import tempfile
import unittest
from unittest import mock

import numpy as np

from utils import head_data

try:
    import cv2
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from utils import custom_head as ch
    from utils import head_train as ht
    HAVE = True
except Exception:                                  # torch / ultralytics missing
    HAVE = False


def write_dataset(root, count=12, rows=None, size=(96, 64), polygon_for=()):
    """root/images/img_01.png ... + root/labels/img_01.txt; returns (images dir, labels dir)."""
    images, labels = os.path.join(root, "images"), os.path.join(root, "labels")
    os.makedirs(images)
    os.makedirs(labels)
    for i in range(1, count + 1):
        name = f"img_{i:02d}"
        cv2_ok = HAVE
        img = np.full((size[1], size[0], 3), 40 + i * 5, np.uint8)
        if cv2_ok:
            cv2.imwrite(os.path.join(images, name + ".png"), img)
        else:                                                            # data tests work without cv2 too
            open(os.path.join(images, name + ".png"), "wb").close()
        text = rows(i) if callable(rows) else (rows or "0 0.5 0.5 0.4 0.6\n1 0.25 0.5 0.2 0.3\n")
        if i in polygon_for:
            text = "1 0.1 0.1 0.5 0.1 0.5 0.7 0.1 0.7\n"
        with open(os.path.join(labels, name + ".txt"), "w") as f:
            f.write(text)
    return images, labels


class DataTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_pairs_are_the_labelled_images_in_natural_order(self):
        images, labels = write_dataset(self.tmp.name, 12)
        os.remove(os.path.join(labels, "img_03.txt"))                     # no label file -> not trainable
        open(os.path.join(labels, "img_04.txt"), "w").close()             # empty label -> not trainable
        with open(os.path.join(labels, "orphan.txt"), "w") as f:          # no image of that name
            f.write("0 0.5 0.5 0.1 0.1\n")
        pairs = head_data.labelled_pairs([images], labels)
        names = [os.path.basename(p[0]) for p in pairs]
        self.assertEqual(len(pairs), 10)
        self.assertEqual(names[:3], ["img_01.png", "img_02.png", "img_05.png"])
        self.assertNotIn("orphan", " ".join(names))

    def test_natural_order_puts_2_before_10(self):
        folder = os.path.join(self.tmp.name, "i")
        labels = os.path.join(self.tmp.name, "l")
        os.makedirs(folder), os.makedirs(labels)
        for n in (10, 2, 1):
            open(os.path.join(folder, f"scene{n}.png"), "wb").close()
            with open(os.path.join(labels, f"scene{n}.txt"), "w") as f:
                f.write("0 0.5 0.5 0.1 0.1\n")
        pairs = head_data.labelled_pairs([folder], labels)
        self.assertEqual([os.path.basename(p[0]) for p in pairs], ["scene1.png", "scene2.png", "scene10.png"])

    def test_the_split_is_one_block_at_the_end_not_random(self):
        pairs = [(f"i{i}", f"l{i}") for i in range(10)]
        train, val = head_data.split_contiguous(pairs, 0.2)
        self.assertEqual((len(train), len(val)), (8, 2))
        self.assertEqual(train, pairs[:8])
        self.assertEqual(val, pairs[8:])

    def test_both_sides_always_get_an_image(self):
        pairs = [("a", "a"), ("b", "b")]
        train, val = head_data.split_contiguous(pairs, 0.01)
        self.assertEqual((len(train), len(val)), (1, 1))
        train, val = head_data.split_contiguous(pairs, 0.99)
        self.assertEqual((len(train), len(val)), (1, 1))
        with self.assertRaises(ValueError):
            head_data.split_contiguous(pairs[:1])

    def test_a_segmentation_row_becomes_its_bounding_box(self):
        path = os.path.join(self.tmp.name, "p.txt")
        with open(path, "w") as f:
            f.write("2 0.1 0.2 0.5 0.2 0.5 0.8 0.1 0.8\n0 0.5 0.5 0.2 0.2\nbad line\n3 0.5 0.5\n")
        rows = head_data.read_label_rows(path)
        self.assertEqual(len(rows), 2)
        cls, cx, cy, w, h = rows[0]
        self.assertEqual(cls, 2)
        self.assertAlmostEqual(cx, 0.3)
        self.assertAlmostEqual(cy, 0.5)
        self.assertAlmostEqual(w, 0.4)
        self.assertAlmostEqual(h, 0.6)
        self.assertEqual(head_data.read_label_rows(os.path.join(self.tmp.name, "missing.txt")), [])

    def test_class_counts(self):
        images, labels = write_dataset(self.tmp.name, 4)
        pairs = head_data.labelled_pairs([images], labels)
        self.assertEqual(head_data.class_counts(pairs, 3), [4, 4, 0])

    def test_taps_follow_the_detector_family(self):
        self.assertEqual(head_data.taps_for("yolov8m.pt"), ((15, 18, 21), True))
        self.assertEqual(head_data.taps_for(r"C:\models\yolov9c.pt"), ((15, 18, 21), True))
        self.assertEqual(head_data.taps_for("yolo11m.pt"), ((16, 19, 22), True))
        self.assertEqual(head_data.taps_for("yolo26m.pt"), ((16, 19, 22), True))
        taps, known = head_data.taps_for("my_finetuned.pt")
        self.assertFalse(known)
        self.assertEqual(taps, head_data.DEFAULT_TAPS)


if HAVE:
    class FakeBackbone(nn.Module):
        """Stands in for the frozen detector: three feature maps at strides 8 / 16 / 32, nothing to download."""

        def __init__(self, strides_ok=True):
            super().__init__()
            self.channels = [3, 3, 3]
            self.w = nn.Parameter(torch.ones(1), requires_grad=False)
            self._ok = strides_ok

        def strides_ok(self):
            return self._ok

        def fingerprint(self):
            return float(self.w.sum())

        def forward(self, x):
            return None, [F.avg_pool2d(x, s) for s in (8, 16, 32)]

    class FakeModel(nn.Module):
        strides_ok = True

        def __init__(self, weights, num_classes, taps=(15, 18, 21)):
            super().__init__()
            self.backbone = FakeBackbone(FakeModel.strides_ok)
            self.head = ch.BehaviorHead([3, 3, 3], num_classes)

        def forward(self, imgs, bidx, boxes):
            _, feats = self.backbone(imgs)
            return self.head(feats, bidx, boxes, imgs.shape[-2:])


@unittest.skipUnless(HAVE, "torch and ultralytics are required")
class TrainerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        FakeModel.strides_ok = True
        patch = mock.patch.object(ht, "BehaviorModel", FakeModel)
        patch.start()
        self.addCleanup(patch.stop)

    def job(self, **over):
        images, labels = write_dataset(os.path.join(self.tmp.name, "data"), over.pop("count", 12), over.pop("rows", None))
        job = dict(images_folders=[images], labels_folder=labels, classes=["a", "b", "unused"], weights="yolov8m.pt",
                   detectors_dir=None, taps=[15, 18, 21], imgsz=[64, 96], epochs=2, batch=4, lr=0.001, workers=0,
                   class_weights=True, hflip=True, val_fraction=0.25, out_dir=os.path.join(self.tmp.name, "out"))
        job.update(over)
        return job

    def run_job(self, job):
        events = []
        summary = ht.train(job, emit=lambda event, **f: events.append({"event": event, **f}))
        return summary, events

    def test_training_writes_a_head_that_the_annotation_side_can_read(self):
        summary, events = self.run_job(self.job())
        kinds = [e["event"] for e in events]
        self.assertEqual(kinds[0], "status")
        self.assertEqual(kinds[1], "start")
        self.assertEqual(kinds.count("epoch"), 2)
        self.assertEqual(kinds[-1], "done")
        start = events[1]
        self.assertEqual((start["train_images"], start["val_images"]), (9, 3))
        self.assertEqual(start["counts"], [9, 9, 0])
        out = summary["out_dir"]
        self.assertTrue(summary["has_head"])
        ck = ch.read_checkpoint(os.path.join(out, "head_best.pt"))             # weights_only load, as at annotation time
        self.assertEqual(ck["names"], ["a", "b", "unused"])
        self.assertEqual(ck["cfg"], dict(weights="yolov8m.pt", taps=[15, 18, 21], imgsz=[64, 96], nc=3))
        self.assertTrue(os.path.isfile(os.path.join(out, "head_last.pt")))

    def test_losses_are_finite_even_with_a_class_that_has_no_examples(self):
        _, events = self.run_job(self.job(classes=["a", "b", "never", "unused"]))
        for e in (e for e in events if e["event"] == "epoch"):
            self.assertTrue(np.isfinite(e["loss"]), e)

    def test_the_stock_weights_name_is_stored_not_a_machine_path(self):
        summary, _ = self.run_job(self.job(weights=os.path.join(self.tmp.name, "yolov8m.pt")))
        ck = ch.read_checkpoint(os.path.join(summary["out_dir"], "head_best.pt"))
        self.assertEqual(ck["cfg"]["weights"], "yolov8m.pt")

    def test_too_few_labelled_images_is_refused_with_a_clear_message(self):
        with self.assertRaisesRegex(ValueError, "at least 10"):
            self.run_job(self.job(count=5))

    def test_the_image_size_must_be_multiples_of_32(self):
        with self.assertRaisesRegex(ValueError, "multiples of 32"):
            self.run_job(self.job(imgsz=[100, 96]))

    def test_wrong_taps_are_caught_before_training(self):
        FakeModel.strides_ok = False
        with self.assertRaisesRegex(ValueError, "strides 8 / 16 / 32"):
            self.run_job(self.job())

    def test_labels_that_use_no_workspace_class_are_refused(self):
        with self.assertRaisesRegex(ValueError, "none of the workspace classes"):
            self.run_job(self.job(classes=["x"], rows="7 0.5 0.5 0.4 0.6\n"))

    def test_a_batch_bigger_than_the_data_still_trains(self):
        summary, events = self.run_job(self.job(batch=64, epochs=1))
        self.assertEqual(events[1]["batch"], 9)
        self.assertEqual(summary["epochs_run"], 1)

    def test_stopping_keeps_what_was_trained_so_far(self):
        stop_after = {"n": 0}

        def should_stop():
            stop_after["n"] += 1
            return stop_after["n"] > 1                      # lets the first epoch run, stops before the second

        events = []
        summary = ht.train(self.job(epochs=5), emit=lambda event, **f: events.append({"event": event, **f}),
                           should_stop=should_stop)
        self.assertTrue(summary["stopped"])
        self.assertEqual(summary["epochs_run"], 1)
        self.assertTrue(summary["has_head"])

    def test_the_detector_changing_is_an_error(self):
        class Moving(FakeBackbone):
            calls = 0

            def fingerprint(self):
                Moving.calls += 1
                return float(Moving.calls)

        class MovingModel(FakeModel):
            def __init__(self, *a, **k):
                super().__init__(*a, **k)
                self.backbone = Moving()
        with mock.patch.object(ht, "BehaviorModel", MovingModel), self.assertRaisesRegex(RuntimeError, "changed"):
            self.run_job(self.job())

    def test_main_reports_a_failure_as_an_error_event_and_a_non_zero_exit(self):
        job = self.job(count=3)
        path = os.path.join(self.tmp.name, "job.json")
        with open(path, "w") as f:
            json.dump(job, f)
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = ht.main([path])
        self.assertEqual(code, 1)
        events = [json.loads(l[len(ht.PREFIX):]) for l in out.getvalue().splitlines() if l.startswith(ht.PREFIX)]
        self.assertEqual(events[-1]["event"], "error")
        self.assertIn("at least 10", events[-1]["message"])

    def test_the_dataset_reads_boxes_in_the_letterboxed_frame(self):
        images, labels = write_dataset(os.path.join(self.tmp.name, "d2"), 1, rows="1 0.5 0.5 0.5 0.5\n", size=(200, 100))
        pairs = head_data.labelled_pairs([images], labels)
        img, boxes, cls = ht.HeadDataset(pairs, (64, 96), 3, train=False)[0]
        self.assertEqual(tuple(img.shape), (3, 64, 96))
        self.assertEqual(cls.tolist(), [1])
        x1, y1, x2, y2 = boxes[0].tolist()                  # 200x100 -> 96x48, 8 px of padding above and below
        self.assertAlmostEqual((x1 + x2) / 2, 48, delta=1)
        self.assertAlmostEqual((y1 + y2) / 2, 32, delta=1)
        self.assertAlmostEqual(x2 - x1, 48, delta=1)

    def test_a_segmentation_label_trains_like_a_box(self):
        images, labels = write_dataset(os.path.join(self.tmp.name, "d3"), 1, polygon_for=(1,))
        pairs = head_data.labelled_pairs([images], labels)
        _, boxes, cls = ht.HeadDataset(pairs, (64, 96), 3, train=False)[0]
        self.assertEqual((len(boxes), cls.tolist()), (1, [1]))


if __name__ == "__main__":
    unittest.main()
