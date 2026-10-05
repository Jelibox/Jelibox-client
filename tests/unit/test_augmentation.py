import os
import tempfile
import unittest

import numpy as np

from utils import augmentation as aug
from utils.augmentation import AugmentConfig, Augmenter, OpSetting, augment_image


def photo(h=48, w=64):
    """A deterministic, colourful, non-flat test image (BGR uint8)."""
    rng = np.random.default_rng(1)
    base = rng.integers(40, 215, (h, w, 3), dtype=np.uint8)
    base[:, : w // 2, 2] = 220                       # strong red half so hue/saturation matter
    return base


def cfg(**ops):
    """cfg(brightness=(25, 100)) -> only brightness, magnitude 25, 100% chance."""
    return AugmentConfig(ops={n: OpSetting(True, m, p) for n, (m, p) in ops.items()})


class OperationTests(unittest.TestCase):
    def setUp(self):
        self.img = photo()
        self.rng = np.random.default_rng(7)

    def test_every_operation_keeps_shape_and_dtype(self):
        for name in aug.OPERATIONS:
            mag = aug.OPERATIONS[name]["default"]
            out, applied, _ = augment_image(self.img, cfg(**{name: (mag, 100)}), self.rng)
            self.assertEqual(applied, [name])
            if name != "rotate":                      # rotation grows the canvas on purpose
                self.assertEqual(out.shape, self.img.shape, name)
            self.assertEqual(out.dtype, np.uint8, name)

    def test_every_operation_actually_changes_the_pixels(self):
        for name in aug.OPERATIONS:
            mag = aug.OPERATIONS[name]["default"] or 1
            changed = False
            for seed in range(10):                    # a random draw can land near zero
                out, _, _ = augment_image(self.img, cfg(**{name: (mag, 100)}), np.random.default_rng(seed))
                changed = changed or not np.array_equal(out, self.img)
            self.assertTrue(changed, name)

    def test_brightness_stays_within_its_limit(self):
        for seed in range(30):
            out, _, _ = augment_image(self.img, cfg(brightness=(10, 100)), np.random.default_rng(seed))
            ratio = out.astype(float).sum() / self.img.astype(float).sum()
            self.assertGreaterEqual(ratio, 0.89)
            self.assertLessEqual(ratio, 1.11)

    def test_blur_kernel_is_odd_and_not_above_the_limit(self):
        edge = np.zeros((40, 40, 3), np.uint8)
        edge[:, 20:] = 255
        for seed in range(20):
            out, _, _ = augment_image(edge, cfg(blur=(3, 100)), np.random.default_rng(seed))
            # a 3x3 blur can only smear the edge by one pixel on each side
            self.assertTrue((out[:, :18] == 0).all() and (out[:, 22:] == 255).all())

    def test_grayscale_removes_colour(self):
        out, _, _ = augment_image(self.img, cfg(grayscale=(0, 100)), self.rng)
        self.assertTrue((out[..., 0] == out[..., 1]).all() and (out[..., 1] == out[..., 2]).all())

    def test_values_never_overflow(self):
        white = np.full((8, 8, 3), 255, np.uint8)
        out, _, _ = augment_image(white, cfg(brightness=(100, 100), contrast=(100, 100), noise=(50, 100)), self.rng)
        self.assertEqual(out.dtype, np.uint8)


class ProbabilityTests(unittest.TestCase):
    def test_each_operation_rolls_its_own_chance(self):
        img = photo()
        config = cfg(brightness=(25, 50), blur=(3, 50))
        rng = np.random.default_rng(3)
        counts = {"brightness": 0, "blur": 0, "both": 0}
        runs = 600
        for _ in range(runs):
            _, applied, _ = augment_image(img, config, rng)
            counts["brightness"] += "brightness" in applied
            counts["blur"] += "blur" in applied
            counts["both"] += len(applied) == 2
        # Conditioned on "at least one fired" (a copy that nothing touched is re-drawn):
        # P(a) = .5/.75, P(both) = .25/.75
        self.assertAlmostEqual(counts["brightness"] / runs, 2 / 3, delta=0.07)
        self.assertAlmostEqual(counts["blur"] / runs, 2 / 3, delta=0.07)
        self.assertAlmostEqual(counts["both"] / runs, 1 / 3, delta=0.07)

    def test_a_copy_is_never_an_untouched_duplicate(self):
        img = photo()
        rng = np.random.default_rng(0)
        for _ in range(100):
            out, applied, _ = augment_image(img, cfg(brightness=(25, 5)), rng)
            if out is not None:
                self.assertTrue(applied)

    def test_nothing_enabled_means_no_copy(self):
        config = AugmentConfig(ops={"blur": OpSetting(False, 3, 100)})
        self.assertFalse(config.enabled)
        self.assertEqual(augment_image(photo(), config, np.random.default_rng(0)), (None, [], None))

    def test_disabled_and_zero_chance_ops_are_ignored(self):
        config = AugmentConfig(ops={"blur": OpSetting(True, 3, 0), "noise": OpSetting(False, 10, 100)})
        self.assertEqual(config.active_ops(), {})


class SeedTests(unittest.TestCase):
    def run_once(self, seed):
        config = cfg(brightness=(25, 50), blur=(3, 50), noise=(10, 50))
        config.seed = seed
        config.copies = 3
        d = tempfile.mkdtemp()
        path = os.path.join(d, "a.png")
        aug.write_image(path, photo())
        return [img for _, img, _ in Augmenter(config).copies(path, "train")]

    def test_same_seed_gives_identical_copies(self):
        a, b = self.run_once(42), self.run_once(42)
        self.assertEqual(len(a), len(b))
        for x, y in zip(a, b):
            self.assertTrue(np.array_equal(x, y))

    def test_different_seed_gives_different_copies(self):
        a, b = self.run_once(1), self.run_once(2)
        self.assertFalse(all(np.array_equal(x, y) for x, y in zip(a, b)))


class AugmenterTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.path = os.path.join(self.dir, "cat.png")
        aug.write_image(self.path, photo())

    def make(self, splits=("train",), copies=2):
        config = cfg(brightness=(25, 100))
        config.splits, config.copies, config.seed = splits, copies, 5
        return Augmenter(config)

    def test_makes_the_requested_number_of_named_copies(self):
        a = self.make(copies=3)
        copies = list(a.copies(self.path, "train"))
        self.assertEqual([c[0] for c in copies], ["_aug1", "_aug2", "_aug3"])
        self.assertEqual(a.created, 3)

    def test_only_the_selected_splits_are_augmented(self):
        a = self.make(splits=("train",))
        self.assertEqual(list(a.copies(self.path, "val")), [])
        self.assertEqual(list(a.copies(self.path, "valid")), [])
        self.assertEqual(list(a.copies(self.path, "test")), [])
        a = self.make(splits=("train", "val"))
        self.assertTrue(a.wants("valid") and a.wants("val") and not a.wants("test"))

    def test_unreadable_image_yields_nothing(self):
        self.assertEqual(list(self.make().copies(os.path.join(self.dir, "missing.png"), "train")), [])

    def test_non_ascii_paths_round_trip(self):
        path = os.path.join(self.dir, "kucing_üñí_猫.png")
        aug.write_image(path, photo())
        self.assertTrue(np.array_equal(aug.read_image(path), photo()))


if __name__ == "__main__":
    unittest.main()
