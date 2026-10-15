"""The augmentation picker's catalog: its tiles, previews, summary and the config built from the picks."""
import unittest

import numpy as np

from utils import augment_catalog as cat
from utils import augmentation as aug
from utils.augmentation import OpSetting


def photo(h=48, w=64):
    return np.random.default_rng(2).integers(30, 220, (h, w, 3), dtype=np.uint8)


class TileTests(unittest.TestCase):
    def test_every_tile_controls_real_operations_and_every_operation_has_a_tile(self):
        covered = set()
        for tile in cat.TILES:
            for op in tile["ops"]:
                self.assertIn(op, aug.OPERATIONS, tile["key"])
                covered.add(op)
        self.assertEqual(covered, set(aug.OPERATIONS))

    def test_tile_keys_are_unique_and_every_tile_explains_itself(self):
        keys = [t["key"] for t in cat.TILES]
        self.assertEqual(len(keys), len(set(keys)))
        for tile in cat.TILES:
            self.assertGreater(len(tile["description"]), 30, tile["key"])
            self.assertTrue(tile["previews"], tile["key"])

    def test_the_roboflow_style_tiles_are_there_in_order(self):
        titles = [t["title"] for t in cat.TILES]
        for must in ("Flip", "90° Rotate", "Rotation", "Shear", "Grayscale", "Hue", "Saturation", "Brightness",
                     "Exposure", "Blur", "Motion Blur", "Noise", "Cutout"):
            self.assertIn(must, titles)
        self.assertLess(titles.index("Flip"), titles.index("Brightness"))
        self.assertNotIn("Crop", titles, "crop was left out on purpose")

    def test_sliders_and_choices_refer_to_known_operations_and_options(self):
        for tile in cat.TILES:
            if tile["slider"]:
                self.assertIn(tile["slider"][0], tile["ops"])
                self.assertNotEqual(aug.OPERATIONS[tile["slider"][0]]["range"], (0, 0), tile["key"])
            for op, option, _ in tile["choices"]:
                self.assertIn(op, tile["ops"])
                if option:
                    self.assertIn(option, aug.OPERATIONS[op]["options"], (tile["key"], option))

    def test_the_tiles_ultralytics_already_does_are_the_ones_it_really_has(self):
        done = {t["key"] for t in cat.TILES if t["ultralytics"]}
        self.assertEqual(done, {"flip", "rotate", "shear", "hue", "saturation", "brightness"})
        for tile in cat.TILES:
            if tile["ultralytics"]:
                self.assertIn("Ultralytics", tile["ultralytics"])

    def test_usable_tiles_greys_out_only_when_training(self):
        export = cat.usable_tiles(training=False)
        self.assertTrue(all(note is None for _, note in export))
        training = {t["key"]: note for t, note in cat.usable_tiles(training=True)}
        self.assertIn("fliplr", training["flip"])
        self.assertIn("hsv_h", training["hue"])
        self.assertIn("hsv_v", training["brightness"])
        self.assertIsNone(training["exposure"])
        self.assertIsNone(training["cutout"])


class StateTests(unittest.TestCase):
    def test_the_default_state_switches_nothing_on_and_uses_each_operation_default(self):
        state = cat.default_state()
        self.assertEqual(set(state), set(aug.OPERATIONS))
        self.assertFalse(any(s.enabled for s in state.values()))
        self.assertEqual(state["brightness"].magnitude, aug.OPERATIONS["brightness"]["default"])
        self.assertEqual(state["rot90"].options, {"cw": True, "ccw": True, "upside_down": True})
        self.assertEqual(cat.selected_tiles(state), [])

    def test_a_tile_counts_as_picked_when_any_of_its_operations_is_on(self):
        state = cat.default_state()
        state["flip_v"].enabled = True
        self.assertEqual([t["key"] for t in cat.selected_tiles(state)], ["flip"])

    def test_the_summary_names_the_picks_with_their_strength(self):
        state = cat.default_state()
        state["flip_h"].enabled = True
        state["brightness"].enabled, state["brightness"].magnitude = True, 30
        state["blur"].enabled, state["blur"].magnitude = True, 5
        self.assertEqual(cat.summary(state), ["Flip", "Brightness ±30%", "Blur up to 5px"])

    def test_outputs_and_the_original_become_the_number_of_copies(self):
        state = cat.default_state()
        state["flip_h"].enabled = True
        self.assertEqual(cat.to_config(state, 3, True).copies, 2)
        self.assertEqual(cat.to_config(state, 3, False).copies, 3)
        self.assertEqual(cat.to_config(state, 1, True).copies, 0)
        self.assertFalse(cat.to_config(state, 1, True).enabled, "one output that is the original adds nothing")
        self.assertEqual(cat.to_config(state, 99, False).copies, cat.OUTPUTS_RANGE[1])
        self.assertEqual(cat.to_config(state, 0, False).copies, cat.OUTPUTS_RANGE[0])

    def test_the_config_carries_seed_splits_and_keep_original(self):
        state = cat.default_state()
        state["noise"].enabled = True
        config = cat.to_config(state, 4, False, seed=9, splits=("train", "val"))
        self.assertEqual((config.seed, config.splits, config.keep_original), (9, ("train", "val"), False))
        self.assertTrue(config.ops["noise"].enabled)
        self.assertIsNot(config.ops["noise"], state["noise"], "the config is a copy")

    def test_from_config_is_the_inverse_of_to_config(self):
        state = cat.default_state()
        state["exposure"] = OpSetting(True, 40, 80, {"brighten": False, "darken": True})
        again = cat.from_config(cat.to_config(state, 3, True))
        self.assertEqual(again["exposure"], state["exposure"])
        self.assertEqual(set(again), set(state))


class PreviewTests(unittest.TestCase):
    def test_a_tile_previews_with_the_original_first(self):
        state = cat.default_state()
        for tile in cat.TILES:
            shown = cat.previews(tile, state, photo())
            self.assertEqual(shown[0][0], "original", tile["key"])
            self.assertEqual(len(shown), 1 + len(tile["previews"]), tile["key"])
            for label, img in shown:
                self.assertEqual(img.dtype, np.uint8, (tile["key"], label))

    def test_the_labels_carry_the_current_slider_value(self):
        state = cat.default_state()
        state["brightness"].magnitude = 30
        labels = [l for l, _ in cat.previews(cat.TILE_BY_KEY["brightness"], state, photo())]
        self.assertEqual(labels, ["original", "-30%", "30%"])
        state["rotate"].magnitude = 20
        labels = [l for l, _ in cat.previews(cat.TILE_BY_KEY["rotate"], state, photo())]
        self.assertEqual(labels, ["original", "-20°", "+20°"])

    def test_previews_follow_the_options(self):
        state = cat.default_state()
        img = np.full((40, 40, 3), 100, np.uint8)
        state["brightness"].magnitude = 30
        state["brightness"].options = {"brighten": False, "darken": True}
        (_, _), (_, low), (_, high) = cat.previews(cat.TILE_BY_KEY["brightness"], state, img)
        self.assertAlmostEqual(float(low.mean()), 70.0, delta=1)
        self.assertAlmostEqual(float(high.mean()), 100.0, delta=1, msg="brightening is switched off")

    def test_the_turn_previews_show_one_direction_each(self):
        state = cat.default_state()
        img = np.zeros((40, 60, 3), np.uint8)
        img[:10, :20] = 255                                          # white patch in the top-left corner
        shown = dict(cat.previews(cat.TILE_BY_KEY["rot90"], state, img))
        self.assertEqual(shown["clockwise"].shape[:2], (60, 40))
        self.assertTrue(shown["clockwise"][:20, 30:].min() == 255, "top-left goes to top-right when turned clockwise")
        self.assertTrue(shown["counter-clockwise"][40:, :10].min() == 255, "and to bottom-left the other way")
        self.assertTrue(shown["upside down"][30:, 40:].min() == 255)

    def test_the_thumbnail_is_the_strongest_example(self):
        state = cat.default_state()
        state["brightness"].magnitude = 40
        img = np.full((40, 40, 3), 100, np.uint8)
        self.assertAlmostEqual(float(cat.thumbnail(cat.TILE_BY_KEY["brightness"], state, img).mean()), 140.0, delta=1)


if __name__ == "__main__":
    unittest.main()
