"""The newer augmentation operations (exposure, motion blur, cutout, 90 degree turn, shear), their options, the
Resize transform of the export's Preprocessing, and the helpers the settings window and the training job use."""
import unittest

import numpy as np

from utils import augmentation as aug
from utils import augment_geometry as geo
from utils.augmentation import AugmentConfig, Augmenter, OpSetting, augment_image


def photo(h=48, w=64):
    rng = np.random.default_rng(1)
    base = rng.integers(40, 200, (h, w, 3), dtype=np.uint8)
    return base


def square_image(h=60, w=80, box=(10, 5, 30, 25)):
    """Black picture with one white rectangle: where the rectangle ends up says where a label has to go."""
    img = np.zeros((h, w, 3), np.uint8)
    x1, y1, x2, y2 = box
    img[y1:y2, x1:x2] = 255
    return img


def white_box(img):
    ys, xs = np.nonzero(img[..., 0] > 128)
    return xs.min(), ys.min(), xs.max() + 1, ys.max() + 1


def one(name, magnitude=20, probability=100, **options):
    return AugmentConfig(ops={name: OpSetting(True, magnitude, probability, options)})


class DirectionTests(unittest.TestCase):
    def test_brightness_can_be_limited_to_darkening_or_brightening(self):
        img = photo()
        for seed in range(15):
            dark, _, _ = augment_image(img, one("brightness", 30, brighten=False), np.random.default_rng(seed))
            bright, _, _ = augment_image(img, one("brightness", 30, darken=False), np.random.default_rng(seed))
            self.assertLessEqual(dark.astype(int).sum(), img.astype(int).sum())
            self.assertGreaterEqual(bright.astype(int).sum(), img.astype(int).sum())

    def test_turning_both_directions_off_means_both_not_nothing(self):
        out, applied, _ = augment_image(photo(), one("brightness", 30, brighten=False, darken=False),
                                        np.random.default_rng(2))
        self.assertEqual(applied, ["brightness"])
        self.assertFalse(np.array_equal(out, photo()))

    def test_exposure_is_gamma_so_the_extremes_stay_put(self):
        img = np.zeros((10, 10, 3), np.uint8)
        img[:5] = 255
        for seed in range(10):
            out, _, _ = augment_image(img, one("exposure", 40), np.random.default_rng(seed))
            self.assertTrue((out[:5] == 255).all() and (out[5:] == 0).all(), "pure black and white are unchanged")
        mid = np.full((10, 10, 3), 100, np.uint8)
        lifted, _, _ = augment_image(mid, one("exposure", 40, darken=False), np.random.default_rng(1))
        deepened, _, _ = augment_image(mid, one("exposure", 40, brighten=False), np.random.default_rng(1))
        self.assertGreater(lifted.mean(), 100)
        self.assertLess(deepened.mean(), 100)


class PixelOnlyTests(unittest.TestCase):
    def test_cutout_paints_black_boxes_and_leaves_the_labels_alone(self):
        img = np.full((100, 100, 3), 200, np.uint8)
        out, applied, tf = augment_image(img, one("cutout", 20, count=3), np.random.default_rng(4))
        self.assertEqual(applied, ["cutout"])
        self.assertIsNone(tf, "no geometry changed, so the annotations are copied as they are")
        black = (out == 0).all(axis=2)
        self.assertGreater(black.sum(), 0)
        self.assertLessEqual(black.sum(), 3 * 20 * 20)                     # three boxes, each at most 20 % of a side
        self.assertEqual(out.shape, img.shape)

    def test_the_number_of_boxes_is_an_option(self):
        img = np.full((100, 100, 3), 200, np.uint8)
        few, _, _ = augment_image(img, one("cutout", 10, count=1), np.random.default_rng(6))
        many, _, _ = augment_image(img, one("cutout", 10, count=8), np.random.default_rng(6))
        self.assertLess((few == 0).all(axis=2).sum(), (many == 0).all(axis=2).sum())

    def test_motion_blur_smears_an_edge_but_keeps_the_average_brightness(self):
        edge = np.zeros((60, 60, 3), np.uint8)
        edge[:, 30:] = 255
        for seed in range(5):
            out, _, tf = augment_image(edge, one("motion_blur", 15), np.random.default_rng(seed))
            self.assertIsNone(tf)
            self.assertAlmostEqual(float(out.mean()), float(edge.mean()), delta=6)
            self.assertGreater(len(np.unique(out[30])), 2, "the hard edge became a gradient")


class GeometryTests(unittest.TestCase):
    def check_label_follows(self, config, seed=1, expect_size=None):
        img = square_image()
        out, applied, tf = augment_image(img, config, np.random.default_rng(seed))
        self.assertIsNotNone(tf)
        if expect_size:
            self.assertEqual((tf.dst_w, tf.dst_h), expect_size)
        self.assertEqual(out.shape[:2], (tf.dst_h, tf.dst_w))
        moved = tf.box(10, 5, 30, 25)
        seen = white_box(out)
        for a, b in zip(moved, seen):
            self.assertAlmostEqual(a, b, delta=2.5, msg=(moved, seen))
        return tf

    def test_a_clockwise_turn_swaps_the_canvas_and_the_label_follows(self):
        self.check_label_follows(one("rot90", 0, cw=True, ccw=False, upside_down=False), expect_size=(60, 80))

    def test_every_allowed_turn_is_a_multiple_of_90_and_only_the_allowed_ones(self):
        seen = set()
        for seed in range(40):
            _, _, tf = augment_image(square_image(), one("rot90", 0, cw=False, ccw=True, upside_down=True),
                                     np.random.default_rng(seed))
            seen.add(round(tf.angle))
        self.assertEqual(seen, {90, 180})

    def test_the_upside_down_turn_keeps_the_canvas_size(self):
        self.check_label_follows(one("rot90", 0, cw=False, ccw=False, upside_down=True), expect_size=None)
        _, _, tf = augment_image(square_image(), one("rot90", 0, cw=False, ccw=False, upside_down=True),
                                 np.random.default_rng(0))
        self.assertIn(round(tf.angle), (-90, 90, 180), "no choice left ticked means all of them")

    def test_shear_grows_the_canvas_and_the_label_follows(self):
        tf = self.check_label_follows(one("shear", 20, horizontal=True, vertical=False))
        self.assertGreater(tf.dst_w, 80)
        self.assertEqual(tf.dst_h, 60)
        self.assertTrue(tf.is_sheared)

    def test_shear_can_be_vertical_only(self):
        tf = self.check_label_follows(one("shear", 20, horizontal=False, vertical=True))
        self.assertEqual(tf.shear_x, 0.0)
        self.assertEqual(tf.dst_w, 80)
        self.assertGreater(tf.dst_h, 60)

    def test_rotation_and_a_turn_add_up(self):
        config = AugmentConfig(ops={"rotate": OpSetting(True, 0.0001, 100), "rot90": OpSetting(True, 0, 100, {
            "cw": False, "ccw": True, "upside_down": False})})
        _, _, tf = augment_image(square_image(), config, np.random.default_rng(0))
        self.assertAlmostEqual(tf.angle, 90.0, delta=0.01)

    def test_a_plain_flip_is_still_exact_and_has_no_shear(self):
        tf = geo.GeoTransform(80, 60, flip_h=True)
        self.assertFalse(tf.is_sheared or tf.is_rotated)
        self.assertEqual(tf.box(10, 5, 30, 25), (50.0, 5.0, 70.0, 25.0))


class ResizeTests(unittest.TestCase):
    def test_stretch_goes_to_exactly_the_size_and_scales_each_side(self):
        tf = geo.ResizeTransform(200, 100, "stretch", 64, 64)
        self.assertEqual((tf.dst_w, tf.dst_h), (64, 64))
        self.assertEqual(tf.box(0, 0, 200, 100), (0.0, 0.0, 64.0, 64.0))
        self.assertEqual(tf.warp(np.zeros((100, 200, 3), np.uint8)).shape, (64, 64, 3))

    def test_fit_keeps_the_shape(self):
        tf = geo.ResizeTransform(200, 100, "fit", 64, 64)
        self.assertEqual((tf.dst_w, tf.dst_h), (64, 32))
        self.assertEqual(tf.box(0, 0, 200, 100), (0.0, 0.0, 64.0, 32.0))
        self.assertEqual(tf.warp(np.zeros((100, 200, 3), np.uint8)).shape, (32, 64, 3))

    def test_letterbox_fills_the_rest_with_black_and_moves_the_labels_by_the_bars(self):
        tf = geo.ResizeTransform(200, 100, "letterbox", 64, 64)
        self.assertEqual((tf.dst_w, tf.dst_h), (64, 64))
        self.assertEqual(tf.box(0, 0, 200, 100), (0.0, 16.0, 64.0, 48.0))
        out = tf.warp(np.full((100, 200, 3), 255, np.uint8))
        self.assertEqual(out.shape, (64, 64, 3))
        self.assertTrue((out[:16] == 0).all() and (out[48:] == 0).all() and (out[16:48] == 255).all())

    def test_a_tall_picture_gets_side_bars_and_the_label_follows_the_pixels(self):
        img = square_image(h=100, w=50, box=(10, 20, 30, 60))
        tf = geo.ResizeTransform(50, 100, "letterbox", 64, 64)
        out = tf.warp(img)
        for a, b in zip(tf.box(10, 20, 30, 60), white_box(out)):
            self.assertAlmostEqual(a, b, delta=2.0)

    def test_yolo_labels_change_with_the_bars_but_not_with_a_plain_stretch(self):
        text = "0 0.500000 0.500000 0.500000 0.500000"
        stretch = geo.yolo_label_text(text, geo.ResizeTransform(200, 100, "stretch", 64, 64))
        self.assertEqual(stretch, "0 0.500000 0.500000 0.500000 0.500000")
        bars = geo.yolo_label_text(text, geo.ResizeTransform(200, 100, "letterbox", 64, 64))
        self.assertEqual(bars, "0 0.500000 0.500000 0.500000 0.250000")        # half as tall: the bars took the rest

    def test_a_chain_is_one_matrix_for_the_labels_and_both_steps_for_the_pixels(self):
        first = geo.GeoTransform(80, 60, flip_h=True)
        second = geo.ResizeTransform(80, 60, "stretch", 40, 30)
        chain = geo.ChainTransform(first, second)
        self.assertEqual((chain.src_w, chain.src_h, chain.dst_w, chain.dst_h), (80, 60, 40, 30))
        a = chain.box(10, 5, 30, 25)
        b = second.box(*first.box(10, 5, 30, 25))
        for x, y in zip(a, b):
            self.assertAlmostEqual(x, y, places=6)
        self.assertEqual(chain.warp(square_image()).shape, (30, 40, 3))

    def test_unknown_mode_is_refused(self):
        with self.assertRaises(ValueError):
            geo.ResizeTransform(10, 10, "zoom", 8, 8)


class ConfigHelpersTests(unittest.TestCase):
    def test_a_config_survives_a_trip_through_json(self):
        import json
        config = AugmentConfig(ops={"brightness": OpSetting(True, 30, 60, {"darken": False}),
                                    "cutout": OpSetting(False, 10, 50, {"count": 5})},
                               copies=4, splits=("train", "val"), seed=7, keep_original=False)
        again = aug.config_from_dict(json.loads(json.dumps(aug.config_to_dict(config))))
        self.assertEqual(again, config)

    def test_unknown_operations_and_missing_fields_are_tolerated(self):
        config = aug.config_from_dict({"ops": {"nonsense": {"enabled": True}, "blur": {"enabled": True}}})
        self.assertEqual(list(config.ops), ["blur"])
        self.assertEqual((config.copies, config.splits, config.seed, config.keep_original), (1, ("train",), None, True))

    def test_keeps_original_only_matters_for_the_augmented_splits(self):
        config = one("blur", 5)
        config.keep_original = False
        augmenter = Augmenter(config)
        self.assertFalse(augmenter.keeps_original("train"))
        self.assertTrue(augmenter.keeps_original("val"))
        self.assertTrue(augmenter.keeps_original("test"))
        self.assertTrue(Augmenter(AugmentConfig()).keeps_original("train"), "nothing to replace them with")

    def test_the_post_step_is_applied_to_every_copy(self):
        calls = []

        def post(img, tf, src_hw):
            calls.append((img.shape[:2], src_hw))
            second = geo.ResizeTransform(img.shape[1], img.shape[0], "stretch", 32, 32)
            return second.warp(img), second if tf is None else geo.ChainTransform(tf, second)

        augmenter = Augmenter(one("flip_h", 0), post=post)
        path = None
        import os
        import tempfile
        path = os.path.join(tempfile.mkdtemp(), "a.png")
        aug.write_image(path, photo(48, 64))
        (suffix, out, tf), = list(augmenter.copies(path, "train"))
        self.assertEqual(out.shape, (32, 32, 3))
        self.assertEqual((tf.dst_w, tf.dst_h), (32, 32))
        self.assertEqual(calls, [((48, 64), (48, 64))])


class PreviewTests(unittest.TestCase):
    def test_a_preview_shows_the_operation_at_its_limit_up_and_down(self):
        mid = np.full((20, 20, 3), 100, np.uint8)
        setting = OpSetting(True, 30, 100, {})
        up = aug.preview_operation(mid, "brightness", setting, high=True)
        down = aug.preview_operation(mid, "brightness", setting, high=False)
        self.assertAlmostEqual(float(up.mean()), 130.0, delta=1)
        self.assertAlmostEqual(float(down.mean()), 70.0, delta=1)

    def test_previews_of_geometry_come_with_the_new_canvas(self):
        out = aug.preview_operation(square_image(), "rotate", OpSetting(True, 30, 100, {}), high=True)
        self.assertGreater(out.shape[0], 60)
        self.assertGreater(out.shape[1], 80)

    def test_every_operation_can_be_previewed(self):
        for name, spec in aug.OPERATIONS.items():
            setting = OpSetting(True, spec["default"] or 1, 100, dict(spec.get("options") or {}))
            for high in (True, False):
                out = aug.preview_operation(photo(), name, setting, high)
                self.assertEqual(out.dtype, np.uint8, name)

    def test_fixed_rng_is_deterministic_and_always_says_yes(self):
        rng = aug.FixedRng(True)
        self.assertEqual((rng.uniform(-5, 7), rng.random(), rng.choice([1, 2, 3])), (7, 0.0, 3))
        rng = aug.FixedRng(False)
        self.assertEqual((rng.uniform(-5, 7), rng.choice([1, 2, 3])), (-5, 1))


if __name__ == "__main__":
    unittest.main()
