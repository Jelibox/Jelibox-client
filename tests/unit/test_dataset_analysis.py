"""Analyze Dataset numbers: reading the VOC XML, the calculated resize, the small-object warning."""
import os
import tempfile
import unittest

from utils import dataset_analysis as da


def xml(path, size, objects):
    """objects: (class, (xmin, ymin, xmax, ymax)) for a box, (class, [(x, y), ...]) for a polygon."""
    rows = []
    if size:
        rows.append(f"<size><width>{size[0]}</width><height>{size[1]}</height><depth>3</depth></size>")
    for name, geo in objects:
        if isinstance(geo, tuple):
            x1, y1, x2, y2 = geo
            rows.append(f"<object><name>{name}</name><bndbox><xmin>{x1}</xmin><ymin>{y1}</ymin>"
                        f"<xmax>{x2}</xmax><ymax>{y2}</ymax></bndbox></object>")
        else:
            pts = "".join(f"<point><x>{x}</x><y>{y}</y></point>" for x, y in geo)
            rows.append(f"<object><name>{name}</name><polygon>{pts}</polygon></object>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("<annotation>" + "".join(rows) + "</annotation>")


class Workspace(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = self.tmp.name
        self.a, self.b, self.voc = (os.path.join(root, n) for n in ("ws-1", "ws-2", "voc"))
        for d in (self.a, self.b, self.voc):
            os.makedirs(d)
        for folder, names in ((self.a, ("p1", "p2", "p3")), (self.b, ("q1", "q2"))):
            for n in names:
                open(os.path.join(folder, n + ".jpg"), "wb").close()
        # p1: 1000x500 image with a car (200x100) and a polygon person (40..60 x 100..180)
        xml(os.path.join(self.voc, "p1.xml"), (1000, 500), [("car", (100, 100, 300, 200)),
                                                           ("person", [(40, 100), (60, 100), (60, 180), (40, 180)])])
        xml(os.path.join(self.voc, "p2.xml"), (640, 640), [("car", (0, 0, 64, 32))])
        xml(os.path.join(self.voc, "p3.xml"), (640, 640), [])                      # annotated, nothing in it
        xml(os.path.join(self.voc, "q1.xml"), None, [("car", (0, 0, 50, 50))])      # no size in the file
        xml(os.path.join(self.voc, "ghost.xml"), (640, 640), [("car", (0, 0, 10, 10))])   # no such image
        self.data = da.analyze("ws", {"ws-1": self.a, "ws-2": self.b}, self.voc)


class ReadingTests(Workspace):
    def test_counts(self):
        d = self.data
        self.assertEqual((d.n_images, d.n_labelled, d.n_unlabelled), (5, 3, 2))
        self.assertEqual(d.n_objects, 4)
        self.assertEqual(d.n_unmatched_xml, 1)
        self.assertEqual(d.n_no_size, 1)
        self.assertEqual([(i["name"], i["images"], i["labelled"]) for i in d.instances], [("ws-1", 3, 2), ("ws-2", 2, 1)])

    def test_classes_most_objects_first_and_counts(self):
        self.assertEqual(self.data.class_counts(), [("car", 3), ("person", 1)])

    def test_a_polygon_counts_through_its_bounding_box(self):
        v = da.view(self.data)
        i = list(v["cls"]).index("person")
        self.assertEqual((v["w"][i], v["h"][i]), (20, 80))
        self.assertIn("polygon", self.data.kind)

    def test_image_sizes_and_objects_per_image(self):
        self.assertEqual(dict(self.data.image_sizes), {(1000, 500): 1, (640, 640): 2})
        self.assertEqual(sorted(self.data.per_image), [1, 1, 2])

    def test_a_broken_xml_is_skipped(self):
        with open(os.path.join(self.voc, "p2.xml"), "w") as f:
            f.write("<annotation><oops")
        d = da.analyze("ws", {"ws-1": self.a, "ws-2": self.b}, self.voc)
        self.assertEqual(d.n_objects, 3)

    def test_a_workspace_without_annotations(self):
        d = da.analyze("ws", {"ws-1": self.a}, os.path.join(self.tmp.name, "nothing"))
        self.assertEqual((d.n_images, d.n_objects), (3, 0))
        self.assertEqual(da.small_summary(da.view(d)), None)
        self.assertIsNone(da.smallest(da.view(d)))


class ResizeTests(Workspace):
    def test_original_view_changes_nothing(self):
        v = da.view(self.data)
        self.assertEqual(sorted(v["w"]), [20, 50, 64, 200])

    def test_letterbox_scales_by_the_longest_side(self):
        v = da.view(self.data, 500)                     # 1000x500 -> 0.5; 640x640 -> 500/640
        by = {(c, round(float(w), 2)) for c, w in zip(v["cls"], v["w"])}
        self.assertIn(("car", 100.0), by)               # 200 * 0.5
        self.assertIn(("person", 10.0), by)             # 20 * 0.5
        self.assertIn(("car", round(64 * 500 / 640, 2)), by)

    def test_stretch_scales_width_and_height_separately(self):
        v = da.view(self.data, 640, da.STRETCH)
        i = [k for k, (c, w) in enumerate(zip(v["cls"], v["w"])) if c == "car" and w == 200 * 640 / 1000][0]
        self.assertAlmostEqual(v["h"][i], 100 * 640 / 500)

    def test_objects_of_an_image_without_a_size_are_left_out_of_the_simulation(self):
        self.assertEqual(len(da.view(self.data, 640)["w"]), 3)

    def test_smallest_is_by_area(self):
        # areas: car 20000, person 1600, car 2048, car 2500
        self.assertEqual(da.smallest(da.view(self.data)), (20.0, 80.0))

    def test_compare_has_one_row_per_size_and_shrinks_with_the_size(self):
        rows = da.compare(self.data, sizes=(320, 640, 1280))
        self.assertEqual([r["size"] for r in rows], [320, 640, 1280])
        sides = [min(r["smallest"]) for r in rows]
        self.assertEqual(sides, sorted(sides))
        self.assertAlmostEqual(sides[1] * 2, sides[2])


class SmallObjectTests(Workspace):
    def test_nothing_small_means_no_warning(self):
        import numpy as np
        v = {"cls": np.array(["car", "car"], dtype=object), "w": np.array([200.0, 90.0]), "h": np.array([100.0, 80.0])}
        self.assertIsNone(da.small_summary(v))

    def test_counts_objects_with_a_side_under_the_limit(self):
        info = da.small_summary(da.view(self.data), 32)
        # person 20x80 is under 32; the 64x32 car is not (32 is not under 32)
        self.assertEqual(info["count"], 1)
        self.assertEqual(info["by_class"], [("person", 1)])
        self.assertAlmostEqual(info["percent"], 25.0)

    def test_shrinking_the_images_makes_more_objects_small(self):
        before = da.small_summary(da.view(self.data), 32)["count"]
        after = da.small_summary(da.view(self.data, 320), 32)["count"]
        self.assertGreater(after, before)


class SourcesAndProgressTests(Workspace):
    def test_every_object_knows_its_xml_file(self):
        self.assertEqual(len(self.data.src), self.data.n_objects)
        self.assertEqual(sorted({os.path.basename(p) for p in self.data.src}), ["p1.xml", "p2.xml", "q1.xml"])

    def test_small_sources_lists_each_file_once(self):
        # only the 20 x 80 polygon person (p1) has a side under 32 px
        self.assertEqual([os.path.basename(p) for p in da.small_sources(self.data)], ["p1.xml"])
        self.assertEqual([os.path.basename(p) for p in da.small_sources(self.data, limit=60)], ["p1.xml", "p2.xml", "q1.xml"])

    def test_the_resize_decides_which_files_are_small_and_unknown_sizes_are_left_out(self):
        files = lambda size: sorted(os.path.basename(p) for p in da.small_sources(self.data, size))
        self.assertEqual(files(640), ["p1.xml"])         # q1 has no image size: never part of a simulation
        self.assertEqual(files(100), ["p1.xml", "p2.xml"])

    def test_analyze_reports_progress_up_to_the_end(self):
        seen = []
        da.analyze("ws", {"ws-1": self.a, "ws-2": self.b}, self.voc, lambda d, t, label=None: seen.append((d, t, label)))
        self.assertEqual(seen[0][2], "Counting images...")
        self.assertEqual(seen[-1][0], seen[-1][1])
        self.assertIn("Reading annotations", seen[-1][2])
        self.assertTrue(all(0 <= d <= t for d, t, _ in seen))


class SpreadTests(unittest.TestCase):
    def test_spread(self):
        self.assertEqual(da.spread([1, 2, 3, 10]), {"min": 1.0, "median": 2.5, "mean": 4.0, "max": 10.0})
        self.assertIsNone(da.spread([]))


if __name__ == "__main__":
    unittest.main()
