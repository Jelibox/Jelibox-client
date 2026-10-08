import os
import unittest
import xml.etree.ElementTree as ET

from tests.helpers import isolated_workspace

ctx = isolated_workspace()

from utils import file_handler as fh                           # noqa: E402


class VocYoloTests(unittest.TestCase):
    SHAPE = (240, 320, 3)           # h, w, c

    def test_voc_xml_contains_size_and_both_object_kinds(self):
        ann = fh.build_voc_xml("a.png", self.SHAPE, [[10, 20, 110, 120, "cat"]],
                               [[[(5, 5), (50, 5), (50, 50)], "dog"]])
        self.assertEqual(ann.findtext("size/width"), "320")
        self.assertEqual(ann.findtext("size/height"), "240")
        types = [o.findtext("type") for o in ann.findall("object")]
        self.assertEqual(types, ["bbox", "polygon"])
        self.assertEqual(ann.findtext("object/bndbox/xmin"), "10")

    def test_negative_coordinates_are_clamped(self):
        ann = fh.build_voc_xml("a.png", self.SHAPE, [[-5, -5, 20, 20, "cat"]], [])
        self.assertEqual(ann.findtext("object/bndbox/xmin"), "0")

    def test_yolo_detection_lines(self):
        lines = fh._yolo_lines_from_annotations([[0, 0, 160, 120, "cat"]], [], ["cat", "dog"], 320, 240)
        self.assertEqual(lines, ["0 0.250000 0.250000 0.500000 0.500000"])

    def test_yolo_uses_segmentation_format_when_any_polygon_exists(self):
        lines = fh._yolo_lines_from_annotations([[0, 0, 320, 240, "cat"]],
                                                [[[(0, 0), (320, 0), (160, 240)], "dog"]],
                                                ["cat", "dog"], 320, 240)
        self.assertEqual(len(lines), 2)
        self.assertTrue(lines[0].startswith("0 0.000000 0.000000 1.000000 0.000000"))   # bbox -> 4-pt polygon
        self.assertTrue(lines[1].startswith("1 "))

    def test_yolo_skips_unknown_classes(self):
        self.assertEqual(fh._yolo_lines_from_annotations([[0, 0, 10, 10, "ghost"]], [], ["cat"], 100, 100), [])

    def test_voc_roundtrip_through_load_annotation_local(self):
        ann = fh.build_voc_xml("img1.png", self.SHAPE, [[10, 20, 110, 120, "cat"]],
                               [[[(5, 5), (50, 5), (50, 50)], "dog"]])
        os.makedirs(ctx.voc, exist_ok=True)
        with open(os.path.join(ctx.voc, "img1.xml"), "w", encoding="utf-8") as f:
            f.write(fh.prettify_xml(ann))
        boxes, polys = fh.load_annotation_local("img1.png")
        self.assertEqual(boxes, [[10, 20, 110, 120, "cat"]])
        self.assertEqual(len(polys), 1)
        self.assertEqual(polys[0][1], "dog")
        self.assertEqual(len(polys[0][0]), 3)

    def test_loading_an_image_without_xml_gives_empty(self):
        self.assertEqual(fh.load_annotation_local("no_such_image.png"), ([], []))

    def test_sync_yolo_labels_backfills_missing_txt(self):
        ann = fh.build_voc_xml("img2.png", self.SHAPE, [[0, 0, 160, 120, "cat"]], [])
        os.makedirs(ctx.voc, exist_ok=True)
        with open(os.path.join(ctx.voc, "img2.xml"), "w", encoding="utf-8") as f:
            f.write(fh.prettify_xml(ann))
        txt = os.path.join(ctx.labels, "img2.txt")
        if os.path.exists(txt):
            os.remove(txt)
        fh.sync_yolo_labels(["cat", "dog"])
        self.assertTrue(os.path.exists(txt))
        with open(txt) as f:
            self.assertEqual(f.read().strip(), "0 0.250000 0.250000 0.500000 0.500000")


if __name__ == "__main__":
    unittest.main()
