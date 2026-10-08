import unittest

from utils.annotation_merge import iou, merge, poly_rect


def pred(rect, conf, cls="c"):
    return {"rect": rect, "conf": conf, "cls": cls}


class IouTests(unittest.TestCase):
    def test_identical(self):
        self.assertAlmostEqual(iou((0, 0, 10, 10), (0, 0, 10, 10)), 1.0)

    def test_disjoint_and_touching(self):
        self.assertEqual(iou((0, 0, 10, 10), (20, 20, 30, 30)), 0.0)
        self.assertEqual(iou((0, 0, 10, 10), (10, 0, 20, 10)), 0.0)

    def test_half_overlap(self):
        # intersection 50, union 150
        self.assertAlmostEqual(iou((0, 0, 10, 10), (5, 0, 15, 10)), 50 / 150)

    def test_degenerate_box_does_not_crash(self):
        self.assertEqual(iou((5, 5, 5, 5), (0, 0, 10, 10)), 0.0)

    def test_poly_rect(self):
        self.assertEqual(poly_rect([(3, 9), (10, 2), (7, 7)]), (3, 2, 10, 9))


class MergeTests(unittest.TestCase):
    def test_existing_annotation_wins(self):
        kept = merge([pred((12, 12, 98, 98), 0.99)], [(10, 10, 100, 100)])
        self.assertEqual(kept, [])

    def test_non_overlapping_prediction_is_added(self):
        kept = merge([pred((200, 200, 300, 300), 0.5)], [(10, 10, 100, 100)])
        self.assertEqual(len(kept), 1)

    def test_between_predictions_highest_confidence_wins(self):
        a, b = pred((200, 200, 300, 300), 0.4, "low"), pred((205, 205, 300, 300), 0.9, "high")
        kept = merge([a, b], [])
        self.assertEqual([k["cls"] for k in kept], ["high"])

    def test_threshold_boundary(self):
        # IoU is exactly 1/3 here: kept at threshold 0.5, dropped at 0.3
        p = pred((5, 0, 15, 10), 0.8)
        self.assertEqual(len(merge([p], [(0, 0, 10, 10)], threshold=0.5)), 1)
        self.assertEqual(len(merge([p], [(0, 0, 10, 10)], threshold=0.3)), 0)

    def test_empty_inputs(self):
        self.assertEqual(merge([], [(0, 0, 1, 1)]), [])
        self.assertEqual(len(merge([pred((0, 0, 5, 5), 0.1)], [])), 1)

    def test_rerunning_is_idempotent(self):
        first = merge([pred((0, 0, 50, 50), 0.9), pred((100, 100, 150, 150), 0.8)], [])
        existing = [k["rect"] for k in first]
        again = merge([pred((0, 0, 50, 50), 0.9), pred((100, 100, 150, 150), 0.8)], existing)
        self.assertEqual(again, [])


if __name__ == "__main__":
    unittest.main()
