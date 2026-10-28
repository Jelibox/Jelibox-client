"""Export Dataset, Export Model, the augmentation picker and downloads through the web API."""
import io
import json
import os
import time
import unittest
import zipfile
from unittest import mock

from tests.unit.test_web_app import H, TestClient
from tests.unit.test_web_editor import INSTANCE, EditorBase


class ExportBase(EditorBase):
    def setUp(self):
        super().setUp()
        # a, b, c are pictures; label two of them
        self.save("a.png", bboxes=[[10, 20, 110, 120, "cat"]])
        self.save("b.png", bboxes=[[30, 40, 130, 140, "dog"]])

    def post(self, path, body=None):
        return self.client.post(path, json=body or {}, headers=H)

    def finish(self, response):
        job = response.json() if hasattr(response, "json") else response
        for _ in range(400):
            data = self.client.get(f"/api/jobs/{job['id']}").json()
            if data["state"] != "running":
                return data
            time.sleep(0.03)
        self.fail("the job did not end")

    def export(self, **body):
        body = {"format": "yolo", "train_pct": 70, "valid_pct": 20, **body}
        res = self.post(f"{self.base}/export/dataset", body)
        self.assertEqual(res.status_code, 202, res.text)
        return self.finish(res)

    def target(self, job):
        """The scratch folder an export was built in (it is deleted once its .zip has been made)."""
        return self.app.state.downloads.get(job["result"]["download"], "admin")


class InfoTests(ExportBase):
    def test_counts_per_format(self):
        info = self.client.get(f"{self.base}/export/dataset").json()
        counts = {f["id"]: f["count"] for f in info["formats"]}
        self.assertEqual(counts, {"yolo": 2, "voc": 4, "coco": 4})                # YOLO exports only labelled pictures; d.bmp counts for the others
        for gone in ("default_folder", "dest", "missing", "next_name"):             # nothing is saved on the server any more
            self.assertNotIn(gone, info)
        self.assertIn("stretch", [m["id"] for m in info["resize"]["modes"]])

    def test_needs_a_session_and_a_dataset(self):
        self.assertEqual(self.client.get("/api/ws/ghost-1/export/dataset").status_code, 404)
        self.client.post("/api/logout", json={}, headers=H)
        self.assertEqual(self.client.get(f"{self.base}/export/dataset").status_code, 401)


class DatasetExportTests(ExportBase):
    def test_yolo_split_files_and_data_yml(self):
        job = self.export()
        self.assertEqual(job["state"], "done", job.get("error"))
        r = job["result"]
        target = self.target(job)
        self.assertEqual((r["format"], r["name"], os.path.basename(target)), ("YOLO", "pets-yolo.zip", "pets-yolo"))
        for split in ("train", "val", "test"):
            self.assertTrue(os.path.isdir(os.path.join(target, split, "images")), split)
        images = sum(len(os.listdir(os.path.join(target, s, "images"))) for s in ("train", "val", "test"))
        labels = sum(len(os.listdir(os.path.join(target, s, "labels"))) for s in ("train", "val", "test"))
        self.assertEqual((images, labels), (2, 2))
        yml = open(os.path.join(target, "data.yml"), encoding="utf-8").read()
        self.assertIn("nc:    2", yml)
        self.assertIn("0: cat", yml)

    def test_nothing_is_saved_on_the_server_outside_the_scratch_area(self):
        from utils import app_settings
        job = self.export()
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "export dataset")))
        self.assertIsNone(app_settings.get("export_dir"))
        self.assertTrue(os.path.abspath(self.target(job)).startswith(os.path.join(self.tmp, "temp", "downloads")))

    def test_every_export_gets_its_own_scratch_folder(self):
        first, second = self.export(), self.export()
        self.assertNotEqual(self.target(first), self.target(second))
        self.assertTrue(os.path.isdir(os.path.join(self.target(first), "train")))

    def test_the_folder_is_deleted_once_the_zip_exists_and_the_zip_can_be_fetched_again(self):
        job = self.export()
        folder = self.target(job)
        token = job["result"]["download"]
        first = self.client.get(f"/api/downloads/{token}")
        self.assertEqual(first.status_code, 200)
        self.assertFalse(os.path.exists(os.path.dirname(folder)), "the scratch folder is not kept next to its zip")
        second = self.client.get(f"/api/downloads/{token}")
        self.assertEqual((second.status_code, second.content), (200, first.content))
        self.assertIn("pets-yolo.zip", second.headers["content-disposition"])

    def test_old_leftovers_are_cleaned_when_the_next_export_starts(self):
        from utils.web import exporting
        root = os.path.join(self.tmp, "temp", "downloads")
        os.makedirs(os.path.join(root, "oldfolder"))
        open(os.path.join(root, "oldfolder", "x.txt"), "w").close()
        open(os.path.join(root, "old.zip"), "w").close()
        old = time.time() - exporting.STALE_SECONDS - 60
        for name in ("oldfolder", "old.zip"):
            os.utime(os.path.join(root, name), (old, old))
        fresh = os.path.join(root, "newer.zip")
        open(fresh, "w").close()
        self.export()
        self.assertFalse(os.path.exists(os.path.join(root, "oldfolder")))
        self.assertFalse(os.path.exists(os.path.join(root, "old.zip")))
        self.assertTrue(os.path.exists(fresh))

    def test_voc_and_coco(self):
        voc = self.export(format="voc")
        self.assertEqual(voc["state"], "done", voc.get("error"))
        target = self.target(voc)
        names = [n for s in ("train", "valid", "test") for n in os.listdir(os.path.join(target, s))]
        self.assertEqual(sum(n.endswith(".xml") for n in names), 2)
        self.assertEqual(sum(n.endswith((".png", ".jpg", ".bmp")) for n in names), 4)       # VOC exports every picture
        coco = self.export(format="coco")
        self.assertEqual(coco["state"], "done", coco.get("error"))
        total = 0
        for split in ("train", "val", "test"):
            data = json.load(open(os.path.join(self.target(coco), "annotations", f"instances_{split}.json"), encoding="utf-8"))
            self.assertEqual([c["name"] for c in data["categories"]], ["cat", "dog"])
            total += len(data["annotations"])
        self.assertEqual(total, 2)

    def test_preprocessing_resizes_the_pictures_and_moves_the_labels(self):
        from PIL import Image
        job = self.export(preprocess={"auto_orient": True, "resize": {"mode": "stretch", "width": 64, "height": 64}})
        self.assertEqual(job["state"], "done", job.get("error"))
        target = self.target(job)
        folder = next(os.path.join(target, s, "images") for s in ("train", "val", "test")
                      if os.listdir(os.path.join(target, s, "images")))
        with Image.open(os.path.join(folder, os.listdir(folder)[0])) as im:
            self.assertEqual(im.size, (64, 64))
        self.assertTrue(any("Stretch" in n for n in job["result"]["notes"]))

    def test_augmented_copies_double_the_training_pictures(self):
        augment = {"ops": {"flip_h": {"enabled": True, "probability": 100}}, "outputs": 2, "keep_original": True, "seed": 3}
        job = self.export(train_pct=100, valid_pct=0, augment=augment)
        self.assertEqual(job["state"], "done", job.get("error"))
        names = os.listdir(os.path.join(self.target(job), "train", "images"))
        self.assertEqual(len(names), 4)
        self.assertEqual(sum("_aug" in n for n in names), 2)
        self.assertTrue(any("Augmented copies added: 2" in n for n in job["result"]["notes"]))

    def test_one_output_that_is_the_original_adds_nothing_and_says_so(self):
        augment = {"ops": {"flip_h": {"enabled": True}}, "outputs": 1, "keep_original": True}
        job = self.export(augment=augment)
        self.assertTrue(any("adds nothing" in n for n in job["result"]["notes"]))

    def test_requests_are_checked_before_a_job_starts(self):
        base = {"format": "yolo", "train_pct": 70, "valid_pct": 20}
        cases = [({"format": "zip"}, "format"), ({"train_pct": 80, "valid_pct": 30}, "more than 100"),
                 ({"train_pct": "x"}, "whole number"),
                 ({"preprocess": {"resize": {"mode": "stretch", "width": 5, "height": 640}}}, "between"),
                 ({"preprocess": {"resize": {"mode": "warp", "width": 640, "height": 640}}}, "resize"),
                 ({"augment": {"ops": {"flip_h": {"enabled": True}}, "seed": "abc"}}, "seed")]
        for change, word in cases:
            res = self.post(f"{self.base}/export/dataset", {**base, **change})
            self.assertEqual(res.status_code, 400, change)
            self.assertIn(word, res.json()["error"].lower() if word.islower() else res.json()["error"], change)
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "temp", "downloads")))

    def test_no_picture_to_export_is_a_job_error(self):
        for name in ("a.png", "b.png"):
            os.remove(os.path.join(self.labels, name.replace(".png", ".txt")))
        job = self.export()
        self.assertEqual(job["state"], "error")
        self.assertIn("No labeled images", job["error"])

    def test_needs_the_header_and_a_session(self):
        self.assertEqual(self.client.post(f"{self.base}/export/dataset", json={}).status_code, 403)


class DownloadTests(ExportBase):
    def test_the_result_comes_as_a_zip_with_everything_in_it(self):
        job = self.export()
        res = self.client.get(f"/api/downloads/{job['result']['download']}")
        self.assertEqual(res.status_code, 200)
        self.assertIn("pets-yolo.zip", res.headers["content-disposition"])
        names = zipfile.ZipFile(io.BytesIO(res.content)).namelist()
        self.assertTrue(any(n.endswith("data.yml") for n in names))
        self.assertEqual(sum("/images/" in n for n in names), 2)

    def test_only_the_user_who_exported_can_download_it(self):
        job = self.export()
        self.post("/api/users", {"username": "ana", "password": "secret1"})
        other = TestClient(self.app, base_url="http://testserver")
        other.post("/api/login", json={"username": "ana", "password": "secret1"}, headers=H)
        self.assertEqual(other.get(f"/api/downloads/{job['result']['download']}").status_code, 404)
        self.assertEqual(self.client.get("/api/downloads/not-a-token").status_code, 404)
        self.client.post("/api/logout", json={}, headers=H)
        self.assertEqual(self.client.get(f"/api/downloads/{job['result']['download']}").status_code, 401)


class ModelExportTests(ExportBase):
    def put_model(self):
        folder = os.path.join(self.tmp, "models", "pets")
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, "modelAssistant.pt"), "wb") as f:
            f.write(b"model")

    def test_info_and_a_missing_model(self):
        info = self.client.get(f"{self.base}/export/model").json()
        self.assertFalse(info["has_model"])
        self.assertIn("onnx", [f["id"] for f in info["formats"]])
        res = self.post(f"{self.base}/export/model", {"format": "onnx"})
        self.assertEqual(res.status_code, 400)
        self.assertIn("no trained model", res.json()["error"])

    def test_export_runs_as_a_job_and_the_file_can_be_downloaded(self):
        self.put_model()

        def fake(model_path, **kw):
            out = os.path.join(kw["save_dir"], "modelAssistant.onnx")
            self.assertTrue(os.path.abspath(kw["save_dir"]).startswith(os.path.join(self.tmp, "temp", "downloads")))
            open(out, "wb").write(b"onnx-bytes")
            self.assertEqual((kw["fmt"], kw["imgsz"], kw["half"], kw["simplify"]), ("onnx", 480, True, False))
            return True, "Export completed", out
        with mock.patch("utils.web.exporting.ex.export_model", fake):
            res = self.post(f"{self.base}/export/model", {"format": "onnx", "imgsz": 480, "half": True})
            self.assertEqual(res.status_code, 202, res.text)
            job = self.finish(res)
        self.assertEqual(job["state"], "done", job.get("error"))
        dl = self.client.get(f"/api/downloads/{job['result']['download']}")
        self.assertEqual((dl.status_code, dl.content), (200, b"onnx-bytes"))
        self.assertEqual(job["result"]["name"], "modelAssistant.onnx")
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "export model")))

    def test_a_failed_export_is_the_job_message(self):
        self.put_model()
        with mock.patch("utils.web.exporting.ex.export_model", return_value=(False, "Export failed: no module named onnx\nTraceback x", None)):
            job = self.finish(self.post(f"{self.base}/export/model", {"format": "onnx"}))
        self.assertEqual(job["state"], "error")
        self.assertIn("no module named onnx", job["error"])
        self.assertNotIn("Traceback", job["error"])

    def test_bad_format_or_size(self):
        self.put_model()
        self.assertEqual(self.post(f"{self.base}/export/model", {"format": "exe"}).status_code, 400)
        self.assertEqual(self.post(f"{self.base}/export/model", {"format": "onnx", "imgsz": 8}).status_code, 400)


class AugmentationPickerTests(ExportBase):
    def test_the_catalog_lists_every_tile_and_greys_out_what_ultralytics_does_when_training(self):
        plain = self.client.get(f"{self.base}/augment/catalog").json()
        self.assertEqual(len(plain["tiles"]), 15)
        self.assertTrue(all(t["note"] is None for t in plain["tiles"]))
        training = self.client.get(f"{self.base}/augment/catalog", params={"training": 1}).json()
        noted = {t["key"] for t in training["tiles"] if t["note"]}
        self.assertEqual(noted, {"flip", "rotate", "shear", "hue", "saturation", "brightness"})
        self.assertIn("flip_h", plain["operations"])
        self.assertEqual(plain["outputs"], [1, 20])

    def test_thumbnails_and_previews_are_pictures(self):
        thumbs = self.client.post(f"{self.base}/augment/thumbnails", json={"ops": {}}, headers=H).json()["items"]
        self.assertEqual(len(thumbs), 15)
        self.assertTrue(all(v.startswith("data:image/jpeg;base64,") for v in thumbs.values()))
        shown = self.client.post(f"{self.base}/augment/previews", json={"tile": "brightness", "ops": {"brightness": {"magnitude": 40}}},
                                 headers=H).json()["items"]
        self.assertEqual([i["label"] for i in shown], ["original", "-40%", "40%"])
        self.assertEqual(self.client.post(f"{self.base}/augment/previews", json={"tile": "nope"}, headers=H).status_code, 400)

    def test_the_preview_picture_can_be_chosen_and_reset(self):
        self.assertEqual(self.client.get(f"{self.base}/augment/sample").json(), {"name": "a.png", "is_default": True})
        chosen = self.client.put(f"{self.base}/augment/sample", json={"name": "c.jpg"}, headers=H).json()
        self.assertEqual(chosen, {"name": "c.jpg", "is_default": False})
        self.assertEqual(self.client.put(f"{self.base}/augment/sample", json={"name": "ghost.png"}, headers=H).status_code, 404)
        reset = self.client.put(f"{self.base}/augment/sample", json={"name": ""}, headers=H).json()
        self.assertTrue(reset["is_default"])

    def test_an_operation_value_is_clamped_to_what_the_picker_allows(self):
        from utils.web import exporting
        state = exporting.build_state({"blur": {"enabled": True, "magnitude": 999, "probability": 500},
                                       "cutout": {"enabled": True, "options": {"count": 99}}, "bogus": {"enabled": True}})
        self.assertEqual((state["blur"].magnitude, state["blur"].probability), (25, 100.0))
        self.assertEqual(state["cutout"].options["count"], 10)
        self.assertNotIn("bogus", state)


if __name__ == "__main__":
    unittest.main()
