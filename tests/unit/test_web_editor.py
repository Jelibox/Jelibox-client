"""The annotation editor API: open, pictures, annotations (save, validate, conflicts, ZeroFill), classes, delete."""
import io
import os
import unittest

from tests.unit.test_web_app import H, WebBase

INSTANCE = "pets-1"


def make_dataset(datasets, instance=INSTANCE, names=("b.png", "a.png", "c.jpg", "note.txt", "d.bmp"), size=(320, 240)):
    from PIL import Image
    folder = os.path.join(datasets, instance)
    os.makedirs(folder, exist_ok=True)
    for name in names:
        path = os.path.join(folder, name)
        if name.lower().endswith((".png", ".jpg", ".bmp")):
            Image.new("RGB", size, (200, 200, 200)).save(path)
        else:
            with open(path, "w") as f:
                f.write("x")
    return folder


class EditorBase(WebBase):
    def setUp(self):
        super().setUp()
        from utils import workspace_config
        make_dataset(self.datasets)
        workspace_config.set_classes("pets", ["cat", "dog"])
        self.signed_in()
        self.base = f"/api/ws/{INSTANCE}"
        self.voc = os.path.join(self.tmp, "vocdataset", "pets")
        self.labels = os.path.join(self.tmp, "YOLOdataset", "pets", "labels")

    def save(self, name="a.png", bboxes=None, polygons=None, **extra):
        body = {"bboxes": bboxes or [], "polygons": polygons or [], **extra}
        return self.client.put(f"{self.base}/annotations/{name}", json=body, headers=H)

    def load(self, name="a.png"):
        return self.client.get(f"{self.base}/annotations/{name}").json()


class OpenTests(EditorBase):
    def test_lists_the_pictures_like_the_desktop_editor_and_the_classes(self):
        data = self.client.get(self.base).json()
        self.assertEqual(data["images"], ["a.png", "b.png", "c.jpg"])           # .bmp and .txt are not listed
        self.assertEqual(data["annotated"], [])
        self.assertEqual([c["name"] for c in data["classes"]], ["cat", "dog"])
        self.assertRegex(data["classes"][0]["color"], r"^#[0-9a-f]{6}$")
        self.assertEqual((data["instance"], data["workspace"]), (INSTANCE, "pets"))

    def test_a_workspace_without_classes_gets_the_default_one(self):
        from utils import workspace_config
        workspace_config.set_classes("pets", [])
        self.assertEqual([c["name"] for c in self.client.get(self.base).json()["classes"]], ["Object"])

    def test_missing_yolo_labels_are_made_from_the_xml_at_open(self):
        self.save("a.png", bboxes=[[10, 20, 110, 120, "dog"]])
        os.remove(os.path.join(self.labels, "a.txt"))
        self.client.get(self.base)
        with open(os.path.join(self.labels, "a.txt")) as f:
            self.assertTrue(f.read().startswith("1 "))

    def test_unknown_datasets_and_no_session(self):
        for name in ("ghost-1", "pets-9", "..", "pets-1%2F..%2Fpets-1"):
            self.assertIn(self.client.get(f"/api/ws/{name}").status_code, (404, 405), name)
        self.client.post("/api/logout", json={}, headers=H)
        self.assertEqual(self.client.get(self.base).status_code, 401)


class ImageTests(EditorBase):
    def test_serves_the_file(self):
        res = self.client.get(f"{self.base}/image/a.png")
        self.assertEqual((res.status_code, res.headers["content-type"]), (200, "image/png"))
        self.assertTrue(res.content.startswith(b"\x89PNG"))

    def test_only_listed_pictures_are_served(self):
        for name in ("note.txt", "d.bmp", "ghost.png", "..%2Fpets-1%2Fa.png", "%2e%2e", "a.png%00"):
            self.assertEqual(self.client.get(f"{self.base}/image/{name}").status_code in (404, 400), True, name)

    def test_the_size_follows_the_exif_rotation(self):
        from PIL import Image
        img = Image.new("RGB", (200, 100), (10, 10, 10))
        exif = img.getexif()
        exif[274] = 6                                       # rotated: shown as 100 x 200
        img.save(os.path.join(self.datasets, INSTANCE, "turned.jpg"), exif=exif)
        data = self.load("turned.jpg")
        self.assertEqual((data["width"], data["height"]), (100, 200))


class SaveTests(EditorBase):
    def test_save_writes_voc_and_yolo_and_load_returns_them(self):
        res = self.save("a.png", bboxes=[[32, 24, 160, 120, "dog"]], polygons=[[[[10, 10], [100, 10], [100, 90]], "cat"]])
        self.assertEqual(res.status_code, 200, res.text)
        self.assertEqual((res.json()["annotated"], res.json()["total"]), (1, 3))
        self.assertTrue(os.path.isfile(os.path.join(self.voc, "a.xml")))
        with open(os.path.join(self.labels, "a.txt")) as f:
            lines = f.read().splitlines()
        self.assertEqual(len(lines), 2)                                       # polygons present: segmentation format
        self.assertTrue(lines[0].startswith("1 ") and lines[1].startswith("0 "))
        data = self.load("a.png")
        self.assertEqual((data["width"], data["height"]), (320, 240))
        self.assertEqual(data["bboxes"], [[32, 24, 160, 120, "dog"]])
        self.assertEqual(data["polygons"], [[[[10.0, 10.0], [100.0, 10.0], [100.0, 90.0]], "cat"]])

    def test_boxes_only_make_detection_lines(self):
        self.save("a.png", bboxes=[[0, 0, 160, 120, "cat"]])
        with open(os.path.join(self.labels, "a.txt")) as f:
            self.assertEqual(f.read(), "0 0.250000 0.250000 0.500000 0.500000")

    def test_an_empty_save_marks_the_picture_as_done(self):
        self.save("b.png")
        self.assertEqual(self.client.get(self.base).json()["annotated"], ["b.png"])
        self.assertEqual(open(os.path.join(self.labels, "b.txt")).read(), "")

    def test_the_desktop_reads_what_the_web_wrote(self):
        from utils import annotation_io as aio
        import xml.etree.ElementTree as ET
        self.save("a.png", bboxes=[[5, 6, 70, 80, "cat"]])
        boxes, polygons = aio._parse_voc_objects(ET.parse(os.path.join(self.voc, "a.xml")).getroot())
        self.assertEqual((boxes, polygons), ([[5, 6, 70, 80, "cat"]], []))

    def test_values_are_checked_before_anything_is_written(self):
        bad = [
            {"bboxes": [[1, 2, 3, 4, "bird"]]},                                 # unknown class
            {"bboxes": [[1, 2, 3, "dog"]]},                                     # wrong length
            {"bboxes": [["x", 2, 30, 40, "dog"]]},
            {"bboxes": [[1, 2, float("nan"), 40, "dog"]]},
            {"bboxes": "nope"},
            {"polygons": [[[[1, 1], [2, 2]], "dog"]]},                          # two points
            {"polygons": [[[[1, 1], [2, 2], ["a", 3]], "dog"]]},
            {"polygons": [[[[1, 1], [2, 2], [3, 3]], "bird"]]},
            {"bboxes": [[1, 1, 9, 9, "dog"]] * 5001},
        ]
        for body in bad:
            res = self.client.put(f"{self.base}/annotations/a.png", content=__import__("json").dumps(body).replace("NaN", "null"),
                                  headers={**H, "Content-Type": "application/json"})
            self.assertEqual(res.status_code, 400, body if len(str(body)) < 200 else "many")
        self.assertEqual(os.listdir(self.voc), [])

    def test_boxes_are_kept_inside_the_picture_and_the_corners_sorted(self):
        self.save("a.png", bboxes=[[400, 300, -20, -10, "dog"], [10, 10, 10.2, 90, "cat"]])
        self.assertEqual(self.load("a.png")["bboxes"], [[0, 0, 320, 240, "dog"]])     # the 0.2 px sliver is dropped

    def test_unknown_picture_is_a_404(self):
        self.assertEqual(self.save("ghost.png").status_code, 404)
        self.assertEqual(self.client.get(f"{self.base}/annotations/ghost.png").status_code, 404)

    def test_changes_need_the_header(self):
        res = self.client.put(f"{self.base}/annotations/a.png", json={"bboxes": [], "polygons": []})
        self.assertEqual(res.status_code, 403)

    def test_a_damaged_xml_is_reported_not_a_crash(self):
        os.makedirs(self.voc, exist_ok=True)
        with open(os.path.join(self.voc, "a.xml"), "w") as f:
            f.write("<annotation><object>")
        res = self.client.get(f"{self.base}/annotations/a.png")
        self.assertEqual(res.status_code, 400)
        self.assertIn("damaged", res.json()["error"])


class ConflictTests(EditorBase):
    def test_a_stale_version_is_refused_and_force_overrides(self):
        first = self.load("a.png")
        self.assertEqual(first["version"], "0")
        mine = self.save("a.png", bboxes=[[1, 1, 50, 50, "cat"]], version=first["version"])
        self.assertEqual(mine.status_code, 200)
        other = self.save("a.png", bboxes=[[60, 60, 120, 120, "dog"]], version=first["version"])     # still the old one
        self.assertEqual(other.status_code, 409)
        self.assertTrue(other.json()["conflict"])
        self.assertEqual(self.load("a.png")["bboxes"], [[1, 1, 50, 50, "cat"]])
        forced = self.save("a.png", bboxes=[[60, 60, 120, 120, "dog"]], version=first["version"], force=True)
        self.assertEqual(forced.status_code, 200)
        self.assertEqual(self.load("a.png")["bboxes"], [[60, 60, 120, 120, "dog"]])

    def test_the_new_version_is_the_one_to_send_next(self):
        v1 = self.save("a.png", bboxes=[[1, 1, 50, 50, "cat"]], version="0").json()["version"]
        self.assertEqual(self.load("a.png")["version"], v1)
        self.assertEqual(self.save("a.png", bboxes=[], version=v1).status_code, 200)


class ZeroFillTests(EditorBase):
    def test_regions_are_burned_into_the_picture_file(self):
        from PIL import Image
        res = self.save("a.png", masks=[[[20, 20], [120, 20], [120, 100], [20, 100]]])
        self.assertEqual(res.status_code, 200, res.text)
        self.assertEqual(res.json()["masks"], 1)
        with Image.open(os.path.join(self.datasets, INSTANCE, "a.png")) as im:
            self.assertEqual(im.getpixel((60, 60)), (0, 0, 0))
            self.assertEqual(im.getpixel((200, 200)), (200, 200, 200))

    def test_a_region_needs_three_points(self):
        self.assertEqual(self.save("a.png", masks=[[[1, 1], [5, 5]]]).status_code, 400)
        self.assertEqual(self.save("a.png", masks="x").status_code, 400)

    def test_a_jpeg_stays_a_jpeg(self):
        self.assertEqual(self.save("c.jpg", masks=[[[0, 0], [50, 0], [50, 50]]]).status_code, 200)
        with open(os.path.join(self.datasets, INSTANCE, "c.jpg"), "rb") as f:
            self.assertEqual(f.read(3), b"\xff\xd8\xff")


class DeleteImageTests(EditorBase):
    def test_removes_the_picture_and_its_annotation_files(self):
        self.save("a.png", bboxes=[[1, 1, 50, 50, "cat"]])
        res = self.client.delete(f"{self.base}/images/a.png", headers=H)
        self.assertEqual(res.json()["images"], ["b.png", "c.jpg"])
        self.assertEqual(res.json()["annotated"], [])
        self.assertFalse(os.path.exists(os.path.join(self.datasets, INSTANCE, "a.png")))
        self.assertFalse(os.path.exists(os.path.join(self.voc, "a.xml")))
        self.assertFalse(os.path.exists(os.path.join(self.labels, "a.txt")))
        self.assertTrue(os.path.exists(os.path.join(self.datasets, INSTANCE, "note.txt")))     # not ours: untouched

    def test_needs_the_header_and_a_listed_name(self):
        self.assertEqual(self.client.delete(f"{self.base}/images/a.png").status_code, 403)
        self.assertEqual(self.client.delete(f"{self.base}/images/note.txt", headers=H).status_code, 404)
        self.assertTrue(os.path.exists(os.path.join(self.datasets, INSTANCE, "note.txt")))


class ClassTests(EditorBase):
    def test_add_and_list(self):
        res = self.client.post(f"{self.base}/classes", json={"name": "bird"}, headers=H)
        self.assertEqual([c["name"] for c in res.json()["classes"]], ["cat", "dog", "bird"])
        self.assertEqual([c["name"] for c in self.client.get(f"{self.base}/classes").json()["classes"]],
                         ["cat", "dog", "bird"])

    def test_bad_names_are_refused(self):
        for name in ("", "a b", "dog", "x" * 17, "a/b", None):
            res = self.client.post(f"{self.base}/classes", json={"name": name}, headers=H)
            self.assertEqual(res.status_code, 400, repr(name))

    def test_deleting_a_class_rewrites_every_picture(self):
        self.save("a.png", bboxes=[[1, 1, 50, 50, "cat"], [60, 60, 120, 120, "dog"]])
        self.save("b.png", bboxes=[[5, 5, 90, 90, "cat"]])
        res = self.client.delete(f"{self.base}/classes/cat", headers=H)
        self.assertEqual([c["name"] for c in res.json()["classes"]], ["dog"])
        self.assertEqual(self.load("a.png")["bboxes"], [[60, 60, 120, 120, "dog"]])
        self.assertEqual(self.load("b.png")["bboxes"], [])
        with open(os.path.join(self.labels, "a.txt")) as f:
            self.assertTrue(f.read().startswith("0 "))                          # dog moved from index 1 to 0

    def test_the_last_class_stays(self):
        self.client.delete(f"{self.base}/classes/cat", headers=H)
        res = self.client.delete(f"{self.base}/classes/dog", headers=H)
        self.assertEqual(res.status_code, 400)
        self.assertEqual(self.client.delete(f"{self.base}/classes/ghost", headers=H).status_code, 400)

    def test_needs_the_header(self):
        self.assertEqual(self.client.post(f"{self.base}/classes", json={"name": "bird"}).status_code, 403)
        self.assertEqual(self.client.delete(f"{self.base}/classes/cat").status_code, 403)

    def test_saving_a_deleted_class_is_refused_with_a_hint(self):
        self.client.delete(f"{self.base}/classes/cat", headers=H)
        res = self.save("a.png", bboxes=[[1, 1, 50, 50, "cat"]])
        self.assertEqual(res.status_code, 400)
        self.assertIn("Unknown class", res.json()["error"])


class SharedFilesTests(EditorBase):
    def test_annotation_io_writes_exactly_what_the_desktop_function_writes(self):
        from utils import annotation_io as aio
        boxes, polys = [[1, 2, 30, 40, "cat"]], [[[(5.0, 5.0), (50.0, 5.0), (50.0, 50.0)], "dog"]]
        a, b = os.path.join(self.tmp, "x"), os.path.join(self.tmp, "y")
        for folder in (a, b):
            os.makedirs(folder)
        aio.write_annotations(a, a, "p.png", (240, 320, 3), boxes, polys, ["cat", "dog"])
        with open(os.path.join(a, "p.xml")) as f:
            xml = f.read()
        self.assertIn("<name>cat</name>", xml)
        self.assertIn("<type>polygon</type>", xml)
        self.assertEqual(open(os.path.join(a, "p.txt")).read().count(chr(10)), 1)


if __name__ == "__main__":
    unittest.main()
