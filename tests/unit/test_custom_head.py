"""Custom dual-head models: geometry, the RoI head, checkpoint validation and detector output handling.
Needs torch + ultralytics (skipped without them). The test that runs a real detector downloads a 5 MB model and is
only run with JELIBOX_TEST_DOWNLOAD=1."""
import os
import tempfile
import unittest

import numpy as np

try:
    import torch
    from utils import custom_head as ch
    HAVE = True
except Exception:                                  # torch / ultralytics missing
    HAVE = False


@unittest.skipUnless(HAVE, "torch and ultralytics are required")
class GeometryTests(unittest.TestCase):
    def test_letterbox_pads_to_the_target_and_keeps_the_aspect(self):
        img = np.full((100, 200, 3), 7, np.uint8)
        out, r, left, top = ch.letterbox(img, (640, 640))
        self.assertEqual(out.shape, (640, 640, 3))
        self.assertAlmostEqual(r, 3.2)
        self.assertEqual((left, top), (0, 160))
        self.assertEqual(out[0, 0, 0], 114, "padding colour")
        self.assertEqual(out[320, 320, 0], 7)

    def test_unletterbox_inverts_letterbox_and_clamps(self):
        _, r, left, top = ch.letterbox(np.zeros((100, 200, 3), np.uint8), (640, 640))
        # a box covering the picture inside the letterbox maps back to the full original image
        self.assertEqual(ch.unletterbox((0, 160, 640, 480), r, left, top, 200, 100), (0, 0, 200, 100))
        # outside the picture is clamped
        self.assertEqual(ch.unletterbox((-50, 100, 700, 700), r, left, top, 200, 100), (0, 0, 200, 100))
        x1, y1, x2, y2 = ch.unletterbox((320, 320, 480, 400), r, left, top, 200, 100)
        self.assertEqual((x1, y1, x2, y2), (100, 50, 150, 75))

    def test_scale_boxes_grows_around_the_centre_and_stays_inside(self):
        b = torch.tensor([[100.0, 100.0, 200.0, 200.0]])
        out = ch.scale_boxes(b, 2.0, (300, 300))
        self.assertEqual(out.tolist(), [[50.0, 50.0, 250.0, 250.0]])
        self.assertEqual(ch.scale_boxes(b, 10.0, (300, 300)).tolist(), [[0.0, 0.0, 300.0, 300.0]])


@unittest.skipUnless(HAVE, "torch and ultralytics are required")
class HeadTests(unittest.TestCase):
    def test_head_gives_one_row_of_logits_per_box_in_any_order_of_sizes(self):
        torch.manual_seed(0)
        chs = [64, 128, 256]
        feats = [torch.randn(2, c, 640 // s, 640 // s) for c, s in zip(chs, (8, 16, 32))]
        head = ch.BehaviorHead(chs, 4).eval()
        boxes = torch.tensor([[10.0, 10.0, 40.0, 40.0],      # small -> P3
                              [100.0, 100.0, 400.0, 500.0],  # big   -> P5
                              [50.0, 60.0, 250.0, 260.0]])   # mid   -> P4
        bidx = torch.tensor([0, 1, 1])
        with torch.no_grad():
            out = head(feats, bidx, boxes, (640, 640))
            reordered = head(feats, bidx[[2, 0, 1]], boxes[[2, 0, 1]], (640, 640))
        self.assertEqual(tuple(out.shape), (3, 4))
        torch.testing.assert_close(reordered, out[[2, 0, 1]], rtol=1e-4, atol=1e-4)   # box order must not matter

    def test_end_to_end_detector_output_is_filtered_by_confidence_and_class(self):
        fake = ch.HeadDetector.__new__(ch.HeadDetector)         # only _detections is used
        out = torch.tensor([[[10, 10, 50, 50, 0.9, 0], [60, 60, 90, 90, 0.2, 0], [5, 5, 30, 30, 0.8, 39]]], dtype=torch.float32)
        kept = fake._detections((out, {}), 0.3, 0.6, [0])
        self.assertEqual(kept.shape[0], 1)
        kept = fake._detections((out, {}), 0.3, 0.6, [0, 39])
        self.assertEqual(kept.shape[0], 2)
        kept = fake._detections((out, {}), 0.3, 0.6, None)
        self.assertEqual(kept.shape[0], 2)


@unittest.skipUnless(HAVE, "torch and ultralytics are required")
class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def save(self, obj, name="h.pt"):
        path = os.path.join(self.tmp, name)
        torch.save(obj, path)
        return path

    def good(self):
        return {"head": {"w": torch.zeros(1)}, "names": ["a", "b"],
                "cfg": {"weights": "yolo11n.pt", "taps": [16, 19, 22], "imgsz": [640, 640], "nc": 2}}

    def test_a_valid_checkpoint_is_read(self):
        ck = ch.read_checkpoint(self.save(self.good()))
        self.assertEqual(ck["names"], ["a", "b"])
        self.assertEqual(ck["cfg"]["taps"], [16, 19, 22])

    def test_missing_keys_and_missing_files_are_explained(self):
        with self.assertRaises(ValueError) as e:
            ch.read_checkpoint(self.save({"head": {}}))
        self.assertIn("cfg", str(e.exception))
        with self.assertRaises(FileNotFoundError):
            ch.read_checkpoint(os.path.join(self.tmp, "nope.pt"))

    def test_checkpoints_that_would_run_code_are_refused(self):
        import pickle

        class Evil:
            def __reduce__(self):
                return (os.getcwd, ())
        path = os.path.join(self.tmp, "evil.pt")
        with open(path, "wb") as f:
            pickle.dump({"head": Evil(), "names": [], "cfg": {}}, f)
        with self.assertRaises(ValueError):
            ch.read_checkpoint(path)

    def test_ultralytics_version_gate(self):
        import ultralytics
        old = ultralytics.__version__
        try:
            ultralytics.__version__ = "8.4.67"
            self.assertIn("8.4.68", ch.ultralytics_problem())
            ultralytics.__version__ = "8.4.68"
            self.assertIsNone(ch.ultralytics_problem())
            ultralytics.__version__ = "8.5.0"
            self.assertIsNone(ch.ultralytics_problem())
        finally:
            ultralytics.__version__ = old


@unittest.skipUnless(HAVE and os.environ.get("JELIBOX_TEST_DOWNLOAD") == "1", "set JELIBOX_TEST_DOWNLOAD=1 (downloads yolo11n.pt)")
class RealDetectorTests(unittest.TestCase):
    """A real yolo11n with a randomly initialised head: the answers are meaningless, the plumbing is checked."""

    def test_person_boxes_are_found_filtered_and_labelled(self):
        import cv2
        import ultralytics
        work = tempfile.mkdtemp()
        taps = (16, 19, 22)
        det_dir = os.path.join(work, "detectors")
        os.makedirs(det_dir)
        frozen = ch.FrozenYOLO(os.path.join(det_dir, "yolo11n.pt"), taps)
        head = ch.BehaviorHead(frozen.channels, 3)
        path = os.path.join(work, "head.pt")
        torch.save({"head": head.state_dict(), "names": ["a", "b", "c"],
                    "cfg": {"weights": "yolo11n.pt", "taps": list(taps), "imgsz": [640, 640], "nc": 3}}, path)
        det = ch.HeadDetector(path, detectors_dir=det_dir, device="cpu")
        frame = cv2.imread(os.path.join(os.path.dirname(ultralytics.__file__), "assets", "bus.jpg"))
        h, w = frame.shape[:2]
        persons = det.predict(frame, classes=[0])
        self.assertGreaterEqual(len(persons), 3)
        self.assertEqual(len(det.predict(frame, classes=[5])), 1)           # the bus
        self.assertEqual(det.predict(frame, classes=[39]), [])               # no bottles
        for p in persons:
            x1, y1, x2, y2 = p["rect"]
            self.assertTrue(0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h)
            self.assertIn(p["head"], ["a", "b", "c"])


if __name__ == "__main__":
    unittest.main()
