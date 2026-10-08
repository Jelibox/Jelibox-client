"""Auto-annotate all images: the batch runner and its worker thread, with fake models and fake storage."""
import threading
import time
import unittest

import numpy as np

from utils.auto_annotate import BatchResult, BatchWorker, run_batch


def img(h=100, w=200):
    return np.zeros((h, w, 3), np.uint8)


class Store:
    """In-memory annotations: name -> (bboxes, polygons)."""

    def __init__(self, existing=None):
        self.data = dict(existing or {})
        self.saved = []

    def load(self, name):
        b, p = self.data.get(name, ([], []))
        return list(b), list(p)

    def save(self, name, shape, bboxes, polygons):
        self.data[name] = (list(bboxes), list(polygons))
        self.saved.append((name, shape))


def pred(rect, cls="cat", conf=0.9):
    return {"rect": rect, "conf": conf, "cls": cls}


def run(images, store, predict, **kw):
    return run_batch(images, read_image=lambda n: img(), predict=predict, load_annotations=store.load,
                     save_annotations=store.save, classes=["cat", "dog"], **kw)


class RunBatchTests(unittest.TestCase):
    def test_every_image_is_annotated_and_saved(self):
        store = Store()
        res = run(["a.png", "b.png"], store, lambda im: ([pred((1, 1, 50, 50))], False))
        self.assertEqual((res.total, res.processed, res.annotated, res.added), (2, 2, 2, 2))
        self.assertEqual(store.data["a.png"][0], [[1, 1, 50, 50, "cat"]])
        self.assertEqual(store.saved[0][1], (100, 200, 3))
        self.assertIn("2 of 2 images annotated", res.summary())

    def test_labelled_images_are_skipped_when_asked(self):
        store = Store({"a.png": ([[0, 0, 10, 10, "dog"]], [])})
        seen = []
        res = run(["a.png", "b.png"], store, lambda im: (seen.append(1) or [pred((1, 1, 50, 50))], False),
                  only_unlabeled=True)
        self.assertEqual((res.skipped_labeled, res.processed, len(seen)), (1, 1, 1))
        self.assertEqual(store.data["a.png"][0], [[0, 0, 10, 10, "dog"]], "skipped image is untouched")

    def test_new_boxes_merge_into_existing_ones(self):
        store = Store({"a.png": ([[10, 10, 100, 100, "dog"]], [])})
        preds = [pred((12, 12, 98, 98)), pred((120, 10, 180, 90))]       # first overlaps the existing dog
        res = run(["a.png"], store, lambda im: (preds, False), only_unlabeled=False)
        self.assertEqual(res.added, 1)
        boxes = store.data["a.png"][0]
        self.assertEqual(boxes[0], [10, 10, 100, 100, "dog"])
        self.assertEqual(boxes[1], [120, 10, 180, 90, "cat"])

    def test_running_twice_does_not_stack_duplicates(self):
        store = Store()
        predict = lambda im: ([pred((1, 1, 50, 50))], False)
        run(["a.png"], store, predict, only_unlabeled=False)
        res = run(["a.png"], store, predict, only_unlabeled=False)
        self.assertEqual((res.added, res.no_detection), (0, 1))
        self.assertEqual(len(store.data["a.png"][0]), 1)

    def test_images_without_detections_are_not_written(self):
        store = Store()
        res = run(["a.png"], store, lambda im: ([], False))
        self.assertEqual((res.annotated, res.no_detection, store.saved), (0, 1, []))

    def test_unknown_classes_are_dropped(self):
        store = Store()
        res = run(["a.png"], store, lambda im: ([pred((1, 1, 50, 50), "bird")], False))
        self.assertEqual((res.added, store.saved), (0, []))

    def test_polygons_are_merged_into_polygons(self):
        store = Store()
        poly = [(0, 0), (40, 0), (40, 40)]
        res = run(["a.png"], store, lambda im: ([{"rect": (0, 0, 40, 40), "conf": 0.8, "cls": "dog", "poly": poly}], True))
        self.assertEqual(res.added, 1)
        self.assertEqual(store.data["a.png"][1], [[poly, "dog"]])
        self.assertEqual(store.data["a.png"][0], [])

    def test_one_bad_image_does_not_stop_the_run(self):
        store = Store()

        def predict(im):
            if predict.n == 0:
                predict.n += 1
                raise RuntimeError("boom")
            return [pred((1, 1, 50, 50))], False
        predict.n = 0
        res = run(["a.png", "b.png"], store, predict)
        self.assertEqual([n for n, _ in res.failed], ["a.png"])
        self.assertEqual(res.annotated, 1)
        self.assertIn("1 failed", res.summary())

    def test_unreadable_images_are_reported(self):
        store = Store()
        res = run_batch(["a.png"], read_image=lambda n: None, predict=lambda im: ([], False),
                        load_annotations=store.load, save_annotations=store.save, classes=["cat"])
        self.assertEqual(res.failed[0][0], "a.png")
        self.assertIn("read", res.failed[0][1])

    def test_cancel_stops_before_the_next_image(self):
        store = Store()
        calls = []

        def predict(im):
            calls.append(1)
            return [pred((1, 1, 50, 50))], False
        res = run(["a.png", "b.png", "c.png"], store, predict, should_cancel=lambda: len(calls) >= 1)
        self.assertTrue(res.cancelled)
        self.assertEqual(len(calls), 1)
        self.assertIn("stopped", res.summary())

    def test_progress_is_reported_before_and_after_each_image(self):
        events = []
        run(["a.png", "b.png"], Store(), lambda im: ([], False),
            progress=lambda done, total, name, r: events.append((done, total, name)))
        self.assertEqual(events, [(0, 2, ""), (1, 2, "a.png"), (2, 2, "b.png")])

    def test_empty_folder(self):
        res = run([], Store(), lambda im: ([], False))
        self.assertEqual((res.total, res.annotated), (0, 0))


class BatchWorkerTests(unittest.TestCase):
    def wait(self, worker, timeout=5):
        end = time.time() + timeout
        while time.time() < end:
            if worker.snapshot()[0]:
                return worker.snapshot()
            time.sleep(0.01)
        self.fail("worker did not finish")

    def test_result_and_progress_come_back_through_snapshot(self):
        def work(status, progress, cancel):
            status("loading")
            progress(3, 10, "x.png")
            return "ok"
        w = BatchWorker(work)
        w.start()
        done, status, progress, result, error = self.wait(w)
        self.assertEqual((status, progress, result, error), ("loading", (3, 10, "x.png"), "ok", None))

    def test_errors_are_captured_not_raised(self):
        def work(status, progress, cancel):
            raise ValueError("bad")
        w = BatchWorker(work)
        w.start()
        done, _s, _p, result, error = self.wait(w)
        self.assertIsNone(result)
        self.assertEqual(str(error), "bad")

    def test_cancel_flag_reaches_the_job(self):
        started = threading.Event()

        def work(status, progress, cancel):
            started.set()
            while not cancel.is_set():
                time.sleep(0.01)
            return "stopped"
        w = BatchWorker(work)
        w.start()
        started.wait(2)
        w.cancel()
        self.assertEqual(self.wait(w)[3], "stopped")


if __name__ == "__main__":
    unittest.main()
