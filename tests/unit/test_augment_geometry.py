import os
import tempfile
import unittest
import xml.etree.ElementTree as ET

import cv2
import numpy as np

from utils import augmentation as aug
from utils.augment_geometry import GeoTransform, coco_annotation, voc_xml_write, yolo_label_text

W, H = 80, 50


def rect_image(x1=20, y1=10, x2=50, y2=30):
    """Black canvas with a white rectangle - easy to find again after a warp."""
    img = np.zeros((H, W, 3), np.uint8)
    img[y1:y2, x1:x2] = 255
    return img


def white_box(img):
    ys, xs = np.where(img[..., 0] > 127)
    return xs.min(), ys.min(), xs.max() + 1, ys.max() + 1


class PixelsAndLabelsAgreeTests(unittest.TestCase):
    """The whole point: whatever happens to the pixels happens to the labels."""

    def check(self, flip_h, flip_v, angle):
        tf = GeoTransform(W, H, flip_h, flip_v, angle)
        out = tf.warp(rect_image())
        self.assertEqual(out.shape[:2], (tf.dst_h, tf.dst_w))
        want = tf.box(20, 10, 50, 30)                              # where the label says the box is
        got = white_box(out)                                       # where the pixels actually are
        for a, b in zip(want, got):
            self.assertAlmostEqual(a, b, delta=1.6, msg=f"{flip_h, flip_v, angle}: {want} vs {got}")

        # polygon: its centre must sit on the centre of mass of the white pixels
        poly = tf.points([(20, 10), (50, 10), (50, 30), (20, 30)])
        cx = sum(p[0] for p in poly) / 4
        cy = sum(p[1] for p in poly) / 4
        ys, xs = np.where(out[..., 0] > 127)
        self.assertAlmostEqual(cx, xs.mean() + 0.5, delta=0.8)
        self.assertAlmostEqual(cy, ys.mean() + 0.5, delta=0.8)

    def test_flips(self):
        for fh, fv in ((True, False), (False, True), (True, True)):
            self.check(fh, fv, 0)

    def test_rotations_with_and_without_flips(self):
        for angle in (-45, -20, -7.5, 3, 15, 30, 89, 90, -90, 150):
            for fh, fv in ((False, False), (True, False), (False, True)):
                self.check(fh, fv, angle)


class TransformShapeTests(unittest.TestCase):
    def test_flip_matches_opencv_and_keeps_the_size(self):
        img = np.random.default_rng(0).integers(0, 255, (H, W, 3), dtype=np.uint8)
        self.assertTrue(np.array_equal(GeoTransform(W, H, True, False, 0).warp(img), cv2.flip(img, 1)))
        self.assertTrue(np.array_equal(GeoTransform(W, H, False, True, 0).warp(img), cv2.flip(img, 0)))
        self.assertTrue(np.array_equal(GeoTransform(W, H, True, True, 0).warp(img), cv2.flip(img, -1)))

    def test_flip_box_is_exact(self):
        tf = GeoTransform(W, H, flip_h=True)
        self.assertEqual(tf.box(10, 5, 30, 20), (W - 30, 5, W - 10, 20))
        tf = GeoTransform(W, H, flip_v=True)
        self.assertEqual(tf.box(10, 5, 30, 20), (10, H - 20, 30, H - 5))

    def test_ninety_degrees_swaps_the_size_and_matches_rot90(self):
        img = np.random.default_rng(1).integers(0, 255, (H, W, 3), dtype=np.uint8)
        tf = GeoTransform(W, H, angle=90)
        self.assertEqual((tf.dst_w, tf.dst_h), (H, W))
        self.assertTrue(np.array_equal(tf.warp(img), np.rot90(img, 1)))

    def test_rotation_keeps_every_pixel_of_the_original(self):
        full = np.full((H, W, 3), 255, np.uint8)
        for angle in (12, 33, -47):
            out = GeoTransform(W, H, angle=angle).warp(full)
            self.assertAlmostEqual((out[..., 0] > 127).sum() / (W * H), 1.0, delta=0.03)

    def test_double_flip_is_identity(self):
        a = GeoTransform(W, H, flip_h=True)
        pts = [(5.5, 7.25), (60.0, 33.0)]
        back = GeoTransform(a.dst_w, a.dst_h, flip_h=True).points(a.points(pts))
        for p, q in zip(pts, back):
            self.assertAlmostEqual(p[0], q[0])
            self.assertAlmostEqual(p[1], q[1])

    def test_points_never_leave_the_canvas(self):
        tf = GeoTransform(W, H, angle=33)
        for x, y in tf.points([(0, 0), (W, 0), (W, H), (0, H)]):
            self.assertTrue(0 <= x <= tf.dst_w and 0 <= y <= tf.dst_h)


class LabelFormatTests(unittest.TestCase):
    def test_yolo_detection_line_on_a_flip(self):
        tf = GeoTransform(W, H, flip_h=True)
        out = yolo_label_text("2 0.250000 0.400000 0.200000 0.300000", tf)
        cls, cx, cy, w, h = out.split()
        self.assertEqual(cls, "2")
        self.assertAlmostEqual(float(cx), 0.75, places=5)
        self.assertAlmostEqual(float(cy), 0.4, places=5)
        self.assertAlmostEqual(float(w), 0.2, places=5)
        self.assertAlmostEqual(float(h), 0.3, places=5)

    def test_yolo_segmentation_line_rotates_exactly(self):
        tf = GeoTransform(W, H, angle=90)
        line = "0 0.25 0.2 0.625 0.2 0.625 0.6 0.25 0.6"           # the white rectangle, normalised
        vals = [float(v) for v in yolo_label_text(line, tf).split()[1:]]
        pts = [(vals[i] * tf.dst_w, vals[i + 1] * tf.dst_h) for i in range(0, 8, 2)]
        self.assertEqual(len(pts), 4)
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        want = tf.box(20, 10, 50, 30)
        self.assertAlmostEqual(min(xs), want[0], places=3)
        self.assertAlmostEqual(max(ys), want[3], places=3)

    def test_yolo_empty_and_junk_lines(self):
        tf = GeoTransform(W, H, flip_h=True)
        self.assertEqual(yolo_label_text("", tf), "")
        self.assertEqual(yolo_label_text("1 0.5\n", tf), "")

    def test_yolo_values_stay_inside_zero_one(self):
        tf = GeoTransform(W, H, angle=37)
        text = yolo_label_text("0 0.5 0.5 1.0 1.0\n1 0 0 1 0 1 1 0 1", tf)
        for line in text.splitlines():
            for v in line.split()[1:]:
                self.assertTrue(0.0 <= float(v) <= 1.0, line)

    def write_xml(self, body):
        d = tempfile.mkdtemp()
        path = os.path.join(d, "a.xml")
        head = f"<size><width>{W}</width><height>{H}</height><depth>3</depth></size>"
        with open(path, "w") as f:
            f.write(f"<annotation><filename>a.png</filename><path>/x/a.png</path>{head}{body}</annotation>")
        return path, os.path.join(d, "b.xml")

    def test_voc_flip_moves_the_box_and_renames(self):
        src, dst = self.write_xml("<object><name>cat</name><type>bbox</type><bndbox><xmin>10</xmin>"
                                  "<ymin>5</ymin><xmax>30</xmax><ymax>20</ymax></bndbox></object>")
        voc_xml_write(src, dst, "a_aug1.png", GeoTransform(W, H, flip_h=True))
        root = ET.parse(dst).getroot()
        self.assertEqual(root.findtext("filename"), "a_aug1.png")
        self.assertEqual(root.findtext("path"), "a_aug1.png")
        bnd = root.find("object/bndbox")
        self.assertEqual([bnd.findtext(k) for k in ("xmin", "ymin", "xmax", "ymax")],
                         [str(W - 30), "5", str(W - 10), "20"])

    def test_voc_rotation_updates_size_and_polygon_points(self):
        src, dst = self.write_xml("<object><name>cat</name><type>polygon</type><polygon>"
                                  "<point><x>20</x><y>10</y></point><point><x>50</x><y>10</y></point>"
                                  "<point><x>50</x><y>30</y></point></polygon></object>")
        tf = GeoTransform(W, H, angle=90)
        voc_xml_write(src, dst, "b.png", tf)
        root = ET.parse(dst).getroot()
        self.assertEqual((root.findtext("size/width"), root.findtext("size/height")), (str(H), str(W)))
        got = [(int(p.findtext("x")), int(p.findtext("y"))) for p in root.findall("object/polygon/point")]
        want = tf.points([(20, 10), (50, 10), (50, 30)])
        self.assertEqual(got, [(round(x), round(y)) for x, y in want])

    def test_voc_roboflow_style_polygon_is_also_moved(self):
        src, dst = self.write_xml("<object><name>cat</name><polygon><x1>20</x1><y1>10</y1>"
                                  "<x2>50</x2><y2>10</y2><x3>50</x3><y3>30</y3></polygon></object>")
        voc_xml_write(src, dst, "b.png", GeoTransform(W, H, flip_v=True))
        poly = ET.parse(dst).getroot().find("object/polygon")
        self.assertEqual((poly.findtext("y1"), poly.findtext("y3")), (str(H - 10), str(H - 30)))

    def test_voc_without_geometry_only_renames(self):
        src, dst = self.write_xml("<object><name>cat</name><bndbox><xmin>1</xmin><ymin>2</ymin>"
                                  "<xmax>3</xmax><ymax>4</ymax></bndbox></object>")
        voc_xml_write(src, dst, "z.png", None)
        root = ET.parse(dst).getroot()
        self.assertEqual(root.findtext("filename"), "z.png")
        self.assertEqual(root.findtext("object/bndbox/xmax"), "3")

    def test_coco_box_only_and_segmentation(self):
        tf = GeoTransform(W, H, flip_h=True)
        box = coco_annotation({"bbox": [10, 5, 20, 15], "area": 300.0, "segmentation": [], "category_id": 1}, tf)
        self.assertEqual(box["bbox"], [W - 30, 5, 20, 15])
        self.assertEqual(box["area"], 300.0)
        self.assertEqual(box["category_id"], 1)
        seg = coco_annotation({"bbox": [20, 10, 30, 20], "area": 600.0,
                               "segmentation": [[20, 10, 50, 10, 50, 30, 20, 30]]}, GeoTransform(W, H, angle=90))
        self.assertAlmostEqual(seg["area"], 600.0, places=3)         # rotation keeps the area
        self.assertAlmostEqual(seg["bbox"][2], 20.0, places=3)       # 30x20 turned on its side
        self.assertAlmostEqual(seg["bbox"][3], 30.0, places=3)

    def test_coco_does_not_mutate_the_original(self):
        ann = {"bbox": [10, 5, 20, 15], "area": 300.0, "segmentation": []}
        coco_annotation(ann, GeoTransform(W, H, flip_h=True))
        self.assertEqual(ann["bbox"], [10, 5, 20, 15])


class AugmenterGeometryTests(unittest.TestCase):
    def setUp(self):
        d = tempfile.mkdtemp()
        self.path = os.path.join(d, "r.png")
        aug.write_image(self.path, rect_image())

    def make(self, **ops):
        config = aug.AugmentConfig(ops={n: aug.OpSetting(True, m, 100) for n, m in ops.items()},
                                   copies=1, splits=("train",), seed=3)
        return aug.Augmenter(config)

    def test_photometric_only_has_no_transform(self):
        (_, _, tf), = list(self.make(brightness=25).copies(self.path, "train"))
        self.assertIsNone(tf)

    def test_geometric_copy_carries_a_transform_that_matches_its_pixels(self):
        for ops in ({"flip_h": 0}, {"flip_v": 0}, {"rotate": 30}, {"flip_h": 0, "rotate": 25}):
            (_, img, tf), = list(self.make(**ops).copies(self.path, "train"))
            self.assertIsNotNone(tf)
            self.assertEqual(img.shape[:2], (tf.dst_h, tf.dst_w))
            for a, b in zip(tf.box(20, 10, 50, 30), white_box(img)):
                self.assertAlmostEqual(a, b, delta=1.6)

    def test_rotation_angle_stays_within_the_limit(self):
        for seed in range(25):
            config = aug.AugmentConfig(ops={"rotate": aug.OpSetting(True, 10, 100)}, seed=seed)
            _, _, tf = aug.augment_image(rect_image(), config, np.random.default_rng(seed))
            self.assertLessEqual(abs(tf.angle), 10)

    def test_geometry_and_colour_combine(self):
        (_, img, tf), = list(self.make(grayscale=0, flip_h=0).copies(self.path, "train"))
        self.assertIsNotNone(tf)
        self.assertTrue((img[..., 0] == img[..., 2]).all())

    def test_same_seed_same_geometry(self):
        a = [(tf.angle, tf.flip_h) for _, _, tf in self.make(rotate=40, flip_h=0).copies(self.path, "train")]
        b = [(tf.angle, tf.flip_h) for _, _, tf in self.make(rotate=40, flip_h=0).copies(self.path, "train")]
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
