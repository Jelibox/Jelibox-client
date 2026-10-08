"""LocateAnything prompt building, output parsing, pass merging and label -> workspace class mapping (pure logic)."""
import unittest

from utils import locate_parsing as lp


class PromptTests(unittest.TestCase):
    def test_categories_are_joined_with_the_models_separator(self):
        self.assertEqual(lp.build_prompt(["person", " car "]),
                         "Locate all the instances that matches the following description: person</c>car.")

    def test_empty_list_falls_back_to_objects(self):
        self.assertIn("objects", lp.build_prompt([]))
        self.assertIn("objects", lp.build_prompt(["", "  "]))

    def test_chinese_categories_pass_through(self):
        self.assertIn("人</c>车", lp.build_prompt(["人", "车"]))


class ParseTests(unittest.TestCase):
    def test_a_ref_labels_every_following_box_until_the_next_ref(self):
        text = ("<ref>person</ref><box><100><200><300><400></box><box><500><500><600><700></box>"
                "<ref>dog</ref><box><1><2><3><4></box>")
        out = lp.parse_output(text)
        self.assertEqual([d["label"] for d in out], ["person", "person", "dog"])
        self.assertEqual(out[0]["coords"], [100, 200, 300, 400])
        self.assertTrue(all(d["kind"] == "box" for d in out))

    def test_box_without_ref_uses_the_fallback_label(self):
        out = lp.parse_output("<box><1><2><3><4></box>", fallback_label="cat")
        self.assertEqual(out[0]["label"], "cat")

    def test_points_and_junk(self):
        out = lp.parse_output("<ref>x</ref><box><10><20></box> noise <box></box>")
        self.assertEqual([d["kind"] for d in out], ["point"])
        self.assertEqual(lp.parse_output(""), [])
        self.assertEqual(lp.parse_output(None), [])

    def test_decimal_coordinates_and_spaces(self):
        out = lp.parse_output("<ref>a</ref><box>< 10.5 ><20><30><40></box>")
        self.assertEqual(out[0]["coords"], [10.5, 20, 30, 40])


class PixelBoxTests(unittest.TestCase):
    def parsed(self, *boxes, label="a"):
        return [{"kind": "box", "coords": list(b), "label": label} for b in boxes]

    def test_grid_is_scaled_to_the_original_image(self):
        out = lp.to_pixel_boxes(self.parsed((100, 200, 500, 800)), width=2000, height=1000)
        self.assertEqual((out[0]["x1"], out[0]["y1"], out[0]["x2"], out[0]["y2"]), (200, 200, 1000, 800))

    def test_reversed_and_out_of_range_boxes_are_fixed(self):
        out = lp.to_pixel_boxes(self.parsed((900, 900, 1200, -50)), width=1000, height=1000)
        self.assertEqual((out[0]["x1"], out[0]["y1"], out[0]["x2"], out[0]["y2"]), (900, 0, 1000, 900))

    def test_tiny_boxes_are_dropped(self):
        self.assertEqual(lp.to_pixel_boxes(self.parsed((10, 10, 10.5, 10.5)), 1000, 1000), [])

    def test_repeated_boxes_collapse_per_label(self):
        same = (100, 100, 300, 300)
        out = lp.to_pixel_boxes(self.parsed(same, same, same), 1000, 1000)
        self.assertEqual(len(out), 1)
        other = lp.to_pixel_boxes(self.parsed(same) + self.parsed(same, label="b"), 1000, 1000)
        self.assertEqual(len(other), 2, "the same box with another label is a different detection")

    def test_points_are_ignored_and_count_is_capped(self):
        parsed = [{"kind": "point", "coords": [5, 5], "label": "a"}]
        self.assertEqual(lp.to_pixel_boxes(parsed, 100, 100), [])
        many = self.parsed(*[(i, i, i + 50, i + 50) for i in range(0, 400, 1)])
        self.assertEqual(len(lp.to_pixel_boxes(many, 1000, 1000, iou_dedup=1.01, max_boxes=10)), 10)


class MergePassesTests(unittest.TestCase):
    def box(self, x1, y1, x2, y2, label="a"):
        return {"label": label, "x1": x1, "y1": y1, "x2": x2, "y2": y2}

    def test_agreeing_passes_average_and_count_votes(self):
        merged = lp.merge_passes([[self.box(0, 0, 100, 100)], [self.box(10, 10, 110, 110)], []])
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["votes"], 2)
        self.assertAlmostEqual(merged[0]["score"], round(2 / 3, 3))
        self.assertEqual((merged[0]["x1"], merged[0]["x2"]), (5, 105))

    def test_min_votes_drops_unconfirmed_boxes(self):
        merged = lp.merge_passes([[self.box(0, 0, 10, 10), self.box(500, 500, 600, 600)],
                                  [self.box(0, 0, 10, 10)]], min_votes=2)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["x1"], 0)

    def test_different_labels_never_merge(self):
        merged = lp.merge_passes([[self.box(0, 0, 10, 10, "a")], [self.box(0, 0, 10, 10, "b")]])
        self.assertEqual(len(merged), 2)


class MapToWorkspaceTests(unittest.TestCase):
    def boxes(self, *labels):
        return [{"label": l, "x1": 1.4, "y1": 2.6, "x2": 30, "y2": 40, "score": 0.5} for l in labels]

    def test_prompts_map_to_classes_ignoring_case_and_spaces(self):
        targets = [{"prompt": "White  Horse", "map_to": "horse"}, {"prompt": "rider", "map_to": "person"}]
        out = lp.map_to_workspace(self.boxes("white horse", "RIDER"), targets)
        self.assertEqual([p["cls"] for p in out], ["horse", "person"])
        self.assertEqual(out[0]["rect"], (1, 3, 30, 40))
        self.assertEqual(out[0]["conf"], 0.5)

    def test_unknown_label_is_dropped_when_there_are_several_targets(self):
        targets = [{"prompt": "a", "map_to": "x"}, {"prompt": "b", "map_to": "y"}]
        self.assertEqual(lp.map_to_workspace(self.boxes("c"), targets), [])

    def test_unknown_label_goes_to_the_only_target(self):
        out = lp.map_to_workspace(self.boxes("something else"), [{"prompt": "a", "map_to": "x"}])
        self.assertEqual([p["cls"] for p in out], ["x"])

    def test_targets_without_a_class_or_prompt_are_ignored(self):
        out = lp.map_to_workspace(self.boxes("a"), [{"prompt": "a", "map_to": ""}, {"prompt": " ", "map_to": "x"}])
        self.assertEqual(out, [])


if __name__ == "__main__":
    unittest.main()
