"""Add Workspace and delete through the web API: folder browser, uploads, zip files, import jobs, deletes."""
import io
import os
import time
import unittest
import zipfile

from tests.unit.test_web_app import H, TestClient, WebBase


def png_bytes(color=(40, 90, 150), size=(64, 48)):
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, "PNG")
    return buf.getvalue()


def write_yolo_dataset(folder, count=3):
    os.makedirs(folder, exist_ok=True)
    for i in range(1, count + 1):
        with open(os.path.join(folder, f"img{i}.png"), "wb") as f:
            f.write(png_bytes((i * 40, 90, 150)))
        with open(os.path.join(folder, f"img{i}.txt"), "w") as f:
            f.write("0 0.5 0.5 0.2 0.2\n1 0.3 0.3 0.1 0.1\n")
    with open(os.path.join(folder, "classes.txt"), "w") as f:
        f.write("cat\ndog\n")


class FlowBase(WebBase):
    def setUp(self):
        super().setUp()
        self.signed_in()

    def post(self, path, body=None, client=None):
        return (client or self.client).post(path, json=body or {}, headers=H)

    def finish(self, job, client=None):
        """Wait for a job (started with a 202 answer) and return its final state."""
        job = job.json() if hasattr(job, "json") else job
        for _ in range(200):
            data = (client or self.client).get(f"/api/jobs/{job['id']}").json()
            if data["state"] != "running":
                return data
            time.sleep(0.05)
        self.fail("the job did not end")

    def source(self, name="source"):
        folder = os.path.join(self.tmp, name)
        write_yolo_dataset(folder)
        return folder

    def upload(self, files, client=None):
        client = client or self.client
        upload_id = client.post("/api/uploads", json={}, headers=H).json()["id"]
        for rel, data in files.items():
            res = client.put(f"/api/uploads/{upload_id}/file", params={"path": rel}, content=data,
                             headers={**H, "Content-Type": "application/octet-stream"})
            self.assertEqual(res.status_code, 200, rel)
        return upload_id


class FolderBrowserTests(FlowBase):
    def test_lists_folders_with_their_parent(self):
        for name in ("beta", "Alpha", ".hidden"):
            os.makedirs(os.path.join(self.tmp, "browse", name))
        with open(os.path.join(self.tmp, "browse", "file.txt"), "w") as f:
            f.write("x")
        data = self.client.get("/api/folders", params={"path": os.path.join(self.tmp, "browse")}).json()
        self.assertEqual(data["folders"], ["Alpha", "beta"])
        self.assertEqual(os.path.normcase(data["parent"]), os.path.normcase(self.tmp))

    def test_no_path_gives_the_start_places_and_a_bad_path_an_error(self):
        data = self.client.get("/api/folders").json()
        self.assertTrue(data["roots"])
        self.assertEqual(self.client.get("/api/folders", params={"path": os.path.join(self.tmp, "nope")}).status_code, 400)

    def test_needs_a_session(self):
        self.post("/api/logout")
        self.assertEqual(self.client.get("/api/folders").status_code, 401)


class ScanTests(FlowBase):
    def test_a_server_folder_is_summarised_like_the_desktop_dialog(self):
        data = self.post("/api/scan", {"folder": self.source()}).json()
        self.assertEqual((data["images"], data["format"], data["annotation_files"]), (3, "yolo", 3))
        self.assertEqual(data["classes"], {"cat": 3, "dog": 3})
        self.assertFalse(data["yolo_unnamed"])

    def test_unreadable_or_empty_folders_are_explained(self):
        os.makedirs(os.path.join(self.tmp, "empty"))
        self.assertIn("No images", self.post("/api/scan", {"folder": os.path.join(self.tmp, "empty")}).json()["error"])
        self.assertIn("does not exist", self.post("/api/scan", {"folder": os.path.join(self.tmp, "x")}).json()["error"])
        self.assertEqual(self.post("/api/scan", {}).status_code, 400)

    def test_mixed_annotation_formats_are_refused(self):
        folder = self.source()
        with open(os.path.join(folder, "img1.xml"), "w") as f:
            f.write("<annotation/>")
        self.assertIn("more than one annotation format", self.post("/api/scan", {"folder": folder}).json()["error"])

    def test_an_existing_workspace_keeps_its_classes_and_lists_what_it_would_skip(self):
        self.finish(self.post("/api/import", {"name": "pets", "folder": self.source("one")}))
        other = self.source("two")
        with open(os.path.join(other, "classes.txt"), "w") as f:
            f.write("cat\nbird\n")
        data = self.post("/api/scan", {"folder": other, "workspace": "pets"}).json()
        self.assertEqual(data["locked_classes"], ["cat", "dog"])
        self.assertEqual(data["skipped_classes"], ["bird"])


class ImportTests(FlowBase):
    def test_import_from_a_server_folder_makes_the_workspace(self):
        start = self.post("/api/import", {"name": "pets", "folder": self.source(), "classes": ["cat", "dog"]})
        self.assertEqual(start.status_code, 202)
        job = self.finish(start)
        self.assertEqual(job["state"], "done", job.get("error"))
        self.assertEqual((job["result"]["instance_name"], job["result"]["image_count"], job["result"]["annotated_count"]),
                         ("pets-1", 3, 3))
        self.assertEqual(sorted(os.listdir(os.path.join(self.datasets, "pets-1"))), ["img1.png", "img2.png", "img3.png"])
        rows = self.client.get("/api/workspaces").json()["workspaces"]
        self.assertEqual([(w["name"], [i["name"] for i in w["instances"]]) for w in rows], [("pets", ["pets-1"])])

    def test_a_second_import_adds_an_instance_to_the_workspace(self):
        self.finish(self.post("/api/import", {"name": "pets", "folder": self.source("one")}))
        job = self.finish(self.post("/api/import", {"workspace": "pets", "folder": self.source("two"), "prefix": "b-"}))
        self.assertEqual(job["result"]["instance_name"], "pets-2")
        self.assertEqual(sorted(os.listdir(os.path.join(self.datasets, "pets-2"))), ["b-1.png", "b-2.png", "b-3.png"])

    def test_bad_input_is_refused_before_anything_starts(self):
        folder = self.source()
        cases = [({"name": "", "folder": folder}, "name"), ({"name": "a b", "folder": folder}, "letters"),
                 ({"name": "pets-3", "folder": folder}, "suffix"), ({"name": "pets"}, "folder"),
                 ({"name": "pets", "folder": folder, "prefix": "bad prefix!"}, "Prefix"),
                 ({"workspace": "ghost", "folder": folder}, "does not exist"),
                 ({"name": "pets", "folder": folder, "classes": ["x" * 20]}, "too long")]
        for body, word in cases:
            res = self.post("/api/import", body)
            self.assertEqual(res.status_code, 400, body)
            self.assertIn(word.lower(), res.json()["error"].lower(), body)
        self.assertEqual(os.listdir(self.datasets), [])

    def test_a_folder_inside_datasetsinput_is_refused_by_the_importer(self):
        inside = os.path.join(self.datasets, "already-1")
        write_yolo_dataset(inside)
        job = self.finish(self.post("/api/import", {"name": "again", "folder": inside}))
        self.assertEqual(job["state"], "error")
        self.assertIn("already inside", job["error"])

    def test_needs_a_session_and_the_header(self):
        self.assertEqual(self.client.post("/api/import", json={}).status_code, 403)
        self.post("/api/logout")
        self.assertEqual(self.post("/api/import", {}).status_code, 401)
        self.assertEqual(self.client.get("/api/jobs/abc").status_code, 401)

    def test_unknown_job(self):
        self.assertEqual(self.client.get("/api/jobs/doesnotexist").status_code, 404)


class UploadTests(FlowBase):
    def test_upload_scan_import_and_the_staging_folder_is_removed(self):
        files = {f"set/img{i}.png": png_bytes((i * 50, 20, 20)) for i in (1, 2)}
        files["set/classes.txt"] = b"cat\n"
        files["set/img1.txt"] = b"0 0.5 0.5 0.4 0.4\n"
        upload_id = self.upload(files)
        scanned = self.post("/api/scan", {"upload_id": upload_id}).json()
        self.assertEqual((scanned["images"], scanned["format"], scanned["classes"]), (2, "yolo", {"cat": 1}))
        job = self.finish(self.post("/api/import", {"name": "up", "upload_id": upload_id, "classes": ["cat"]}))
        self.assertEqual(job["state"], "done", job.get("error"))
        self.assertEqual(job["result"]["annotated_count"], 1)
        self.assertEqual(sorted(os.listdir(os.path.join(self.datasets, "up-1"))), ["img1.png", "img2.png"])
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "temp", "uploads", upload_id)))

    def test_a_zip_file_is_unpacked_and_unsafe_names_inside_it_are_skipped(self):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            z.writestr("data/a.png", png_bytes())
            z.writestr("data/b.png", png_bytes((9, 9, 9)))
            z.writestr("../../escape.png", png_bytes())
            z.writestr("/abs.png", png_bytes())              # a leading slash is just dropped: it lands inside
        upload_id = self.upload({"dataset.zip": buf.getvalue()})
        scanned = self.post("/api/scan", {"upload_id": upload_id}).json()
        self.assertEqual(scanned["images"], 3)                 # a, b and abs: the ".." entry was skipped
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "temp", "escape.png")))
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "escape.png")))

    def test_a_broken_zip_is_reported(self):
        upload_id = self.upload({"bad.zip": b"this is not a zip"})
        res = self.post("/api/scan", {"upload_id": upload_id})
        self.assertEqual(res.status_code, 400)
        self.assertIn("not a valid zip", res.json()["error"])

    def test_file_names_cannot_leave_the_staging_folder(self):
        upload_id = self.post("/api/uploads").json()["id"]
        for bad in ("../evil.png", "a/../../evil.png", "..\\evil.png", "C:/evil.png", "", "con.png",
                    "a/NUL", "x\x00y.png", "trailing."):
            res = self.client.put(f"/api/uploads/{upload_id}/file", params={"path": bad}, content=b"x",
                                  headers={**H, "Content-Type": "application/octet-stream"})
            self.assertEqual(res.status_code, 400, repr(bad))
        staging = os.path.join(self.tmp, "temp", "uploads")
        self.assertEqual(os.listdir(os.path.join(staging, upload_id)), [])
        self.assertEqual(os.listdir(staging), [upload_id])
        res = self.client.put(f"/api/uploads/{upload_id}/file", params={"path": "/etc/ok.png"}, content=b"x",
                              headers={**H, "Content-Type": "application/octet-stream"})
        self.assertEqual(res.status_code, 200)                  # an absolute name is made relative: inside staging
        self.assertTrue(os.path.isfile(os.path.join(staging, upload_id, "etc", "ok.png")))

    def test_an_upload_belongs_to_one_user(self):
        upload_id = self.upload({"a.png": png_bytes()})
        self.post("/api/users", {"username": "ana", "password": "secret1"})
        other = TestClient(self.app, base_url="http://testserver")
        self.assertEqual(other.post("/api/login", json={"username": "ana", "password": "secret1"}, headers=H).status_code, 200)
        self.assertEqual(self.post("/api/scan", {"upload_id": upload_id}, client=other).status_code, 400)
        self.assertEqual(other.delete(f"/api/uploads/{upload_id}", headers=H).status_code, 404)
        self.assertEqual(other.put(f"/api/uploads/{upload_id}/file", params={"path": "x.png"}, content=b"x",
                                   headers=H).status_code, 400)

    def test_an_invalid_upload_id_is_refused(self):
        for bad in ("../../etc", "zzzzzzzzzzzzzzzz", "abc"):
            self.assertEqual(self.post("/api/scan", {"upload_id": bad}).status_code, 400, bad)

    def test_discarding_an_upload_deletes_its_files(self):
        upload_id = self.upload({"a.png": png_bytes()})
        self.assertEqual(self.client.delete(f"/api/uploads/{upload_id}", headers=H).status_code, 200)
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "temp", "uploads", upload_id)))


class DeleteTests(FlowBase):
    def make(self, name="pets"):
        self.finish(self.post("/api/import", {"name": name, "folder": self.source("a_" + name)}))
        self.finish(self.post("/api/import", {"workspace": name, "folder": self.source("b_" + name), "prefix": "n-"}))

    def test_stats_for_the_confirmation(self):
        self.make()
        data = self.client.get("/api/workspaces/pets/stats").json()
        self.assertEqual((data["instances"], data["image_count"], data["xml_count"]), (["pets-1", "pets-2"], 6, 6))

    def test_delete_one_instance_keeps_the_others(self):
        self.make()
        job = self.finish(self.client.delete("/api/workspaces/pets/instances/pets-2", headers=H))
        self.assertEqual(job["state"], "done", job.get("error"))
        self.assertEqual(sorted(os.listdir(self.datasets)), ["pets-1"])

    def test_delete_a_workspace_removes_its_data_and_stored_counts(self):
        self.make()
        self.post("/api/workspaces/recount")
        job = self.finish(self.client.delete("/api/workspaces/pets", headers=H))
        self.assertEqual(job["state"], "done", job.get("error"))
        self.assertEqual(os.listdir(self.datasets), [])
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "vocdataset", "pets")))
        self.make()                                                      # same name again: no old numbers
        row = self.client.get("/api/workspaces").json()["workspaces"][0]
        self.assertIsNone(row["counted"])

    def test_unknown_names_and_paths_are_not_found(self):
        self.make()
        for path in ("/api/workspaces/ghost", "/api/workspaces/pets/instances/ghost-1",
                     "/api/workspaces/pets/instances/pets-1%2F..%2Fpets-2", "/api/workspaces/pets%2F..%2Fpets"):
            self.assertIn(self.client.delete(path, headers=H).status_code, (404, 405), path)
        self.assertEqual(sorted(os.listdir(self.datasets)), ["pets-1", "pets-2"])

    def test_deleting_needs_the_header_and_a_session(self):
        self.make()
        self.assertEqual(self.client.delete("/api/workspaces/pets").status_code, 403)
        self.post("/api/logout")
        self.assertEqual(self.client.delete("/api/workspaces/pets", headers=H).status_code, 401)
        self.assertEqual(sorted(os.listdir(self.datasets)), ["pets-1", "pets-2"])

    def test_a_workspace_that_is_training_cannot_be_deleted(self):
        from unittest import mock
        self.make()
        with mock.patch("utils.process_guard.running_training", return_value={"pid": 1, "workspace": "pets"}):
            self.assertEqual(self.client.delete("/api/workspaces/pets", headers=H).status_code, 409)
        self.assertEqual(sorted(os.listdir(self.datasets)), ["pets-1", "pets-2"])


if __name__ == "__main__":
    unittest.main()
