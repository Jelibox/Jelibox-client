"""Analyze Dataset -> "Remove these objects": the XML files and the YOLO labels lose the small objects, nothing else."""
import os
import tempfile
import unittest

from tests.helpers import isolated_workspace

ctx = isolated_workspace()

from utils import dataset_analysis as da                       # noqa: E402

CLASSES = ["car", "person"]


def write_xml(path, size, objects):
    """objects: (class, (xmin, ymin, xmax, ymax)) or (class, [(x, y), ...])."""
    rows = []
    for name, geo in objects:
        if isinstance(geo, tuple):
            x1, y1, x2, y2 = geo
            rows.append(f"<object><name>{name}</name><type>bbox</type><bndbox><xmin>{x1}</xmin><ymin>{y1}</ymin>"
                        f"<xmax>{x2}</xmax><ymax>{y2}</ymax></bndbox></object>")
        else:
            pts = "".join(f"<point><x>{x}</x><y>{y}</y></point>" for x, y in geo)
            rows.append(f"<object><name>{name}</name><type>polygon</type><polygon>{pts}</polygon></object>")
    head = f"<size><width>{size[0]}</width><height>{size[1]}</height><depth>3</depth></size>" if size else ""
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"<annotation><filename>x</filename>{head}{''.join(rows)}</annotation>")


class RemoveTests(unittest.TestCase):
    def setUp(self):
        root = tempfile.mkdtemp()
        self.images, self.voc, self.yolo = (os.path.join(root, n) for n in ("ws-1", "voc", "yolo"))
        for d in (self.images, self.voc, self.yolo):
            os.makedirs(d)
        for n in ("a", "b", "c", "d"):
            open(os.path.join(self.images, n + ".jpg"), "wb").close()
        # a: 1000x500 - a big car, a 20 px wide person, a small polygon person
        write_xml(os.path.join(self.voc, "a.xml"), (1000, 500), [
            ("car", (100, 100, 400, 300)), ("person", (10, 10, 30, 100)),
            ("person", [(500, 100), (520, 100), (520, 150), (500, 150)])])
        # b: only a tiny object -> left empty
        write_xml(os.path.join(self.voc, "b.xml"), (640, 640), [("car", (0, 0, 10, 10))])
        # c: nothing small
        write_xml(os.path.join(self.voc, "c.xml"), (640, 640), [("car", (0, 0, 100, 100))])
        # d: no image size in the file
        write_xml(os.path.join(self.voc, "d.xml"), None, [("car", (0, 0, 10, 10)), ("car", (0, 0, 200, 200))])
        self.data = da.analyze("ws", {"ws-1": self.images}, self.voc)

    def xml(self, name):
        with open(os.path.join(self.voc, name + ".xml"), encoding="utf-8") as f:
            return f.read()

    def label(self, name):
        with open(os.path.join(self.yolo, name + ".txt"), encoding="utf-8") as f:
            return f.read().split("\n")

    def remove(self, **kw):
        return da.remove_small(self.data, self.yolo, CLASSES, **kw)

    def test_original_sizes_small_objects_leave_the_xml_and_the_label(self):
        result = self.remove()
        self.assertEqual(result, {"objects": 4, "files": 3, "emptied": 1})        # a: 2, b: 1, d: 1
        self.assertEqual(self.xml("a").count("<object>"), 1)
        self.assertIn("<name>car</name>", self.xml("a"))
        self.assertNotIn("person", self.xml("a"))
        self.assertEqual(len(self.label("a")), 1)
        self.assertTrue(self.label("a")[0].startswith("0 "))                      # car = index 0
        self.assertEqual(self.label("b"), [""], "an image left with nothing gets an empty label file")

    def test_a_file_with_nothing_small_is_not_rewritten(self):
        path = os.path.join(self.voc, "c.xml")
        with open(path, "rb") as f:
            before = f.read()
        self.remove()
        with open(path, "rb") as f:
            self.assertEqual(f.read(), before)
        self.assertFalse(os.path.exists(os.path.join(self.yolo, "c.txt")))

    def test_an_image_without_a_size_loses_its_small_object_but_gets_no_label(self):
        self.remove()
        self.assertEqual(self.xml("d").count("<object>"), 1)
        self.assertFalse(os.path.exists(os.path.join(self.yolo, "d.txt")), "a label needs the image size")

    def test_simulated_view_removes_what_is_small_after_the_resize(self):
        # 1000x500 at 320 px letterbox: x0.32. car 300x200 -> 96x64 stays, person 20x90 -> 6.4 wide goes, the
        # polygon person 20x50 -> 6.4 goes. 640x640 at 320: x0.5, car 10x10 -> 5 goes, car 100x100 -> 50 stays.
        small = da.small_sources(self.data, 320, da.LETTERBOX)
        self.assertEqual(sorted(os.path.basename(p) for p in small), ["a.xml", "b.xml"])
        result = self.remove(size=320, mode=da.LETTERBOX)
        self.assertEqual(result["objects"], 3)
        self.assertIn("<name>car</name>", self.xml("a"))
        self.assertEqual(self.xml("d").count("<object>"), 2, "its size is unknown, so it is left alone in a simulation")

    def test_a_big_input_size_flags_nothing(self):
        self.assertEqual(da.small_sources(self.data, 4000, da.LETTERBOX), [])
        self.assertEqual(self.remove(size=4000), {"objects": 0, "files": 0, "emptied": 0})

    def test_stretch_scales_each_side_on_its_own(self):
        # a 400 x 90 box in a 1000 x 100 image at 100 px: letterbox x0.1 -> 40 x 9 (small), stretch x0.1 / x1 -> 40 x 90
        write_xml(os.path.join(self.voc, "c.xml"), (1000, 100), [("car", (0, 0, 400, 90))])
        self.data = da.analyze("ws", {"ws-1": self.images}, self.voc)
        names = lambda mode: [os.path.basename(p) for p in da.small_sources(self.data, 100, mode)]
        self.assertIn("c.xml", names(da.LETTERBOX))
        self.assertNotIn("c.xml", names(da.STRETCH))

    def test_progress_is_reported(self):
        seen = []
        self.remove(progress_cb=lambda done, total, label=None: seen.append((done, total, label)))
        self.assertEqual(seen[-1][0], seen[-1][1])
        self.assertTrue(any("Removing objects" in (s[2] or "") for s in seen))

    def test_analysis_after_the_removal_no_longer_sees_them(self):
        self.remove()
        again = da.analyze("ws", {"ws-1": self.images}, self.voc)
        self.assertEqual(da.small_sources(again), [])
        self.assertEqual(again.n_objects, 3)                                       # a: car, c: car, d: big car


if __name__ == "__main__":
    unittest.main()
