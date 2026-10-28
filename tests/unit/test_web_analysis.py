"""Analyze Dataset through the web API: the numbers for the charts, the simulated resize, the small-object limit and
removing the small objects. Skipped without starlette + httpx."""
import os
import time
import unittest
from unittest import mock

from tests.unit.test_web_app import H, TestClient
from tests.unit.test_web_editor import EditorBase


class AnalysisBase(EditorBase):
    def setUp(self):
        super().setUp()
        # a.png 320x240: a 100x100 cat and a 10x10 dog; b.png: a 200x50 dog; c.jpg and d.bmp have none (the
        # analysis counts every picture type the workspace folder holds, the editor lists only png / jpg)
        self.assertEqual(self.save("a.png", bboxes=[[0, 0, 100, 100, "cat"], [10, 10, 20, 20, "dog"]]).status_code, 200)
        self.assertEqual(self.save("b.png", bboxes=[[0, 0, 200, 50, "dog"]]).status_code, 200)

    def finish(self, started):
        started = started.json() if hasattr(started, "json") else started
        for _ in range(200):
            job = self.client.get(f"/api/jobs/{started['id']}").json()
            if job["state"] != "running":
                return job
            time.sleep(0.05)
        self.fail("the job did not end")

    def analyze(self, **body):
        res = self.client.post("/api/workspaces/pets/analysis", json=body, headers=H)
        self.assertEqual(res.status_code, 202, res.text)
        job = self.finish(res)
        self.assertEqual(job["state"], "done", job)
        return job["result"]["token"], job["result"]["view"]

    def view(self, token, **body):
        return self.client.post(f"/api/analysis/{token}/view", json=body, headers=H)

    def xml(self, name):
        with open(os.path.join(self.voc, name), encoding="utf-8") as f:
            return f.read()


class NumbersTests(AnalysisBase):
    def test_the_first_view_counts_images_objects_classes_and_the_smallest_object(self):
        _, v = self.analyze()
        self.assertEqual((v["images"], v["labelled"], v["unlabelled"], v["objects"], v["classes"]), (4, 2, 2, 3, 2))
        self.assertEqual({c["name"]: c["count"] for c in v["class_counts"]}, {"dog": 2, "cat": 1})
        self.assertEqual(v["class_counts"][0]["name"], "dog")                       # most objects first
        self.assertEqual(v["smallest"], [10, 10])
        self.assertEqual(v["smallest_original"], [10, 10])
        self.assertEqual(v["median"], [100, 50])
        self.assertEqual(v["instances"], [{"name": "pets-1", "images": 4, "labelled": 2}])
        self.assertEqual(v["resolutions"], [{"label": "320x240", "count": 2}])
        self.assertEqual(v["per_image"], [{"objects": 1, "images": 1}, {"objects": 2, "images": 1}])

    def test_objects_under_the_limit_are_reported_with_the_classes_they_hit(self):
        _, v = self.analyze()
        self.assertEqual(v["limit"], 32)
        self.assertEqual(v["small"]["count"], 1)
        self.assertEqual(v["small"]["by_class"], [{"name": "dog", "count": 1}])
        self.assertEqual(v["small"]["smallest"], [10, 10])
        self.assertAlmostEqual(v["small"]["percent"], 33.3, places=1)

    def test_charts_hold_every_object(self):
        _, v = self.analyze()
        self.assertEqual(sum(v["hist_w"]), 3)
        self.assertEqual(sum(v["hist_h"]), 3)
        self.assertEqual(len(v["hist_edges"]), 41)
        self.assertEqual(len(v["scatter"]), 3)
        self.assertEqual(v["size_classes"], ["cat", "dog"])
        self.assertEqual([b["name"] for b in v["boxes"]], ["cat", "dog"])
        dog = v["boxes"][1]                                                          # shortest sides 10 and 50
        self.assertEqual((dog["low"], dog["high"]), (10, 50))
        self.assertEqual(dog["median"], 30)

    def test_the_resize_table_has_a_row_per_input_size(self):
        _, v = self.analyze()
        self.assertEqual([r["size"] for r in v["table"]], [320, 416, 512, 640, 800, 960, 1280])
        row = [r for r in v["table"] if r["size"] == 640][0]
        self.assertEqual(row["smallest"], [20, 20])                                  # 320 -> 640 doubles everything
        self.assertEqual(row["small"], 1)

    def test_a_simulated_size_changes_the_objects_not_the_files(self):
        token, _ = self.analyze()
        before = self.xml("a.xml")
        v = self.view(token, size=640, mode="letterbox").json()
        self.assertEqual(v["smallest"], [20, 20])
        self.assertEqual(v["smallest_original"], [10, 10])
        self.assertEqual(v["size"], 640)
        v = self.view(token, size=640, mode="stretch").json()                         # 240 -> 640 is x2.67 in height
        self.assertEqual((v["smallest"][0], round(v["smallest"][1], 1)), (20, 26.7))
        v = self.view(token).json()                                                   # back to the original sizes
        self.assertEqual((v["size"], v["smallest"]), (None, [10, 10]))
        self.assertEqual(self.xml("a.xml"), before)

    def test_the_limit_changes_the_warning_and_is_remembered(self):
        from utils import app_settings
        token, _ = self.analyze()
        v = self.view(token, limit=5).json()
        self.assertIsNone(v["small"])
        self.assertEqual(app_settings.get("small_limit"), 5)
        _, again = self.analyze()                                                     # the next analysis starts with it
        self.assertEqual(again["limit"], 5)
        v = self.view(token, limit=60).json()
        self.assertEqual(v["small"]["count"], 2)                                      # 10x10 and the 200x50 dog

    def test_polygons_count_through_their_bounding_box(self):
        self.assertEqual(self.save("c.jpg", polygons=[[[[10, 10], [30, 10], [30, 25]], "cat"]]).status_code, 200)
        _, v = self.analyze()
        self.assertEqual(v["objects"], 4)
        self.assertEqual(v["smallest"], [10, 10])                                     # still the dog; the polygon is 20x15


class RefusalTests(AnalysisBase):
    def test_bad_input_is_explained(self):
        token, _ = self.analyze()
        for body, text in (({"size": 5}, "between 16 and 8192"), ({"size": "abc"}, "number of pixels"),
                           ({"mode": "zoom"}, "letterbox or stretch"), ({"limit": 0}, "between 1 and 1024"),
                           ({"limit": 5000}, "between 1 and 1024"), ({"limit": "x"}, "whole number")):
            res = self.view(token, **body)
            self.assertEqual(res.status_code, 400, body)
            self.assertIn(text, res.json()["error"])
        res = self.client.post("/api/workspaces/pets/analysis", json={"size": 1}, headers=H)
        self.assertEqual(res.status_code, 400)

    def test_an_unknown_workspace_or_token(self):
        self.assertEqual(self.client.post("/api/workspaces/nope/analysis", json={}, headers=H).status_code, 404)
        self.assertEqual(self.view("0" * 16).status_code, 404)
        self.assertEqual(self.client.post("/api/analysis/%s/remove-small" % ("0" * 16), json={}, headers=H).status_code, 404)

    def test_everything_needs_a_signed_in_user_and_the_header(self):
        token, _ = self.analyze()
        other = TestClient(self.app, base_url="http://testserver")
        for path in ("/api/workspaces/pets/analysis", f"/api/analysis/{token}/view", f"/api/analysis/{token}/remove-small"):
            self.assertEqual(other.post(path, json={}, headers=H).status_code, 401, path)
            self.assertIn(self.client.post(path, json={}).status_code, (400, 403), path)       # no X-Jelibox header

    def test_an_empty_workspace_gives_empty_charts(self):
        os.remove(os.path.join(self.voc, "a.xml"))
        os.remove(os.path.join(self.voc, "b.xml"))
        _, v = self.analyze()
        self.assertEqual((v["objects"], v["labelled"]), (0, 0))
        self.assertIsNone(v["smallest"])
        self.assertIsNone(v["median"])
        self.assertEqual((v["boxes"], v["scatter"], v["hist_w"], v["per_image"]), ([], [], [], []))


class RemoveSmallTests(AnalysisBase):
    def remove(self, token, **body):
        res = self.client.post(f"/api/analysis/{token}/remove-small", json=body, headers=H)
        return res if res.status_code != 202 else self.finish(res)

    def test_the_flagged_objects_leave_the_xml_and_the_yolo_label(self):
        token, _ = self.analyze()
        job = self.remove(token)
        self.assertEqual(job["state"], "done", job)
        self.assertEqual({k: job["result"][k] for k in ("objects", "files", "emptied")}, {"objects": 1, "files": 1, "emptied": 0})
        self.assertEqual(self.load("a.png")["bboxes"], [[0, 0, 100, 100, "cat"]])
        with open(os.path.join(self.labels, "a.txt")) as f:
            self.assertEqual(len(f.read().strip().splitlines()), 1)
        self.assertEqual(self.load("b.png")["bboxes"], [[0, 0, 200, 50, "dog"]])      # untouched
        view = job["result"]["view"]
        self.assertEqual((view["objects"], view["small"]), (2, None))
        self.assertEqual(self.view(token).json()["objects"], 2)                       # the kept analysis was refreshed

    def test_removing_at_a_simulated_size_uses_that_size(self):
        token, _ = self.analyze()
        job = self.remove(token, size=320, mode="letterbox", limit=60)                # 60 px at 320 = the original sizes
        self.assertEqual(job["result"]["objects"], 2)                                 # 10x10 and the 200x50 dog
        self.assertEqual(job["result"]["view"]["objects"], 1)

    def test_an_image_left_without_objects_is_counted(self):
        token, _ = self.analyze()
        job = self.remove(token, limit=60)
        self.assertEqual(job["result"]["emptied"], 1)                                 # b.png had only the 200x50 dog
        self.assertEqual(self.load("b.png")["bboxes"], [])

    def test_nothing_to_remove_is_refused(self):
        token, _ = self.analyze()
        res = self.remove(token, limit=5)
        self.assertEqual(res.status_code, 409)
        self.assertIn("No object is under the limit", res.json()["error"])

    def test_refused_while_that_workspace_trains(self):
        token, _ = self.analyze()
        with mock.patch("utils.process_guard.running_training", return_value={"workspace": "pets", "pid": 1}):
            res = self.remove(token)
        self.assertEqual(res.status_code, 409)
        self.assertIn("training", res.json()["error"])
        self.assertEqual(len(self.load("a.png")["bboxes"]), 2)


class ManyObjectsTests(unittest.TestCase):
    def test_a_big_scatter_is_sampled_but_the_total_is_kept(self):
        try:
            from utils.web import analysis
        except Exception:                                                              # no starlette / numpy here
            self.skipTest("web package not importable")
        from utils import dataset_analysis as da
        data = da.Analysis("big")
        data.instances = [{"name": "big-1", "images": 100, "labelled": 100}]
        data.n_images = data.n_labelled = 100
        for i in range(5000):
            data.cls.append("a" if i % 2 else "b")
            data.kind.append("bbox")
            data.w.append(10 + i % 90)
            data.h.append(20 + i % 70)
            data.iw.append(640)
            data.ih.append(480)
            data.src.append("x.xml")
        data.classes = ["a", "b"]
        data.per_image = [50] * 100
        data.image_sizes[(640, 480)] = 100
        out = analysis.payload(data, None, "letterbox", 32)
        self.assertEqual(len(out["scatter"]), analysis.SCATTER_POINTS)
        self.assertEqual(out["scatter_total"], 5000)
        self.assertEqual(sum(out["hist_w"]), 5000)
        self.assertEqual(out["per_image"][-1], {"objects": 40, "images": 100})          # 50 objects fall in the "40 or more" column


if __name__ == "__main__":
    unittest.main()
