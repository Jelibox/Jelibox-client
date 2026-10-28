"""The Label Assistant through the web API, with fake predictors (no model is loaded here): settings, downloads that
need a yes, Infer, Auto-annotate all, the one-job-at-a-time rule and SAM 2's references."""
import os
import sys
import threading
import time
import types
import unittest
from unittest import mock

from tests.unit.test_web_app import H
from tests.unit.test_web_editor import INSTANCE, EditorBase


def fake_predictor(preds, polygon=False, delay=0.0, started=None, gate=None):
    """A predictor factory for assistant.build_predictor: predict(bgr) -> (preds, polygon)."""
    def build(ctx, status=None, conf=None, exclude=None, polygon_mode=False, **kw):
        def predict(bgr):
            if started:
                started.set()
            if gate:
                gate.wait(10)
            if delay:
                time.sleep(delay)
            return [dict(p) for p in preds], polygon
        return predict
    return build


CAT_BOX = {"rect": (10, 10, 70, 70), "conf": 0.9, "cls": "cat"}


class AssistBase(EditorBase):
    def setUp(self):
        super().setUp()
        from utils.web import assistant
        self.assistant = assistant
        assistant._world_cache.clear()
        self.addCleanup(lambda: [assistant.forget_references(w) for w in ("pets", "cats")])

    def post(self, path, body=None):
        return self.client.post(path, json=body or {}, headers=H)

    def finish(self, response):
        job = response.json() if hasattr(response, "json") else response
        for _ in range(300):
            data = self.client.get(f"/api/jobs/{job['id']}").json()
            if data["state"] != "running":
                return data
            time.sleep(0.03)
        self.fail("the job did not end")

    def run_infer(self, predictor, name="a.png", bboxes=None, polygons=None, **extra):
        with mock.patch.object(self.assistant, "build_predictor", predictor):
            res = self.post(f"{self.base}/infer", {"name": name, "bboxes": bboxes or [], "polygons": polygons or [],
                                                   "confirm_download": True, **extra})
            self.assertEqual(res.status_code, 202, res.text)
            return self.finish(res)


class SettingsTests(AssistBase):
    def settings(self):
        return self.client.get(f"{self.base}/assistant").json()

    def put(self, **body):
        return self.client.put(f"{self.base}/assistant", json=body, headers=H)

    def test_the_payload_has_what_every_mode_needs(self):
        data = self.settings()
        self.assertEqual((data["mode"], data["provider"], data["classes"]), ("yolo_world", "yolo_world", ["cat", "dog"]))
        self.assertIn("yolov8s-world", data["options"]["yolo_world_models"])
        self.assertEqual([m["id"] for m in data["options"]["sam2_models"]][0], "sam2.1_t.pt")
        self.assertEqual(data["references"], 0)

    def test_yolo_world_targets_are_validated_and_saved(self):
        ok = self.put(section="yolo_world", provider="yolo_world", model="yolov8m-worldv2", confidence=0.4,
                      targets=[{"prompt": "tabby  cat", "map_to": "cat"}])
        self.assertEqual(ok.status_code, 200, ok.text)
        saved = self.settings()
        self.assertEqual((saved["yolo_world"]["model"], saved["confidence"]), ("yolov8m-worldv2", 0.4))
        self.assertEqual(saved["yolo_world"]["target_classes"], [{"prompt": "tabby cat", "map_to": "cat"}])
        for targets, word in (([{"prompt": "cat 2", "map_to": "cat"}], "letters"), ([{"prompt": "x", "map_to": "bird"}], "class of this"),
                              ([{"prompt": "a", "map_to": "cat"}, {"prompt": "A", "map_to": "dog"}], "more than once"),
                              ([{"prompt": "", "map_to": "cat"}], "nothing to search")):
            res = self.put(section="yolo_world", provider="yolo_world", model="yolov8s-world", targets=targets)
            self.assertEqual(res.status_code, 400, targets)
            self.assertIn(word, res.json()["error"])

    def test_no_target_is_saved_with_a_warning(self):
        res = self.put(section="yolo_world", provider="yolo_world", model="yolov8s-world", targets=[])
        self.assertEqual(res.status_code, 200)
        self.assertIn("target class", res.json()["warnings"][0])

    def test_the_own_model_needs_an_existing_file_and_a_mapping(self):
        gone = self.put(section="yolo_world", provider="custom_model", model="yolov8s-world",
                        model_path=os.path.join(self.tmp, "nope.pt"), model_class_map={"x": "cat"})
        self.assertEqual(gone.status_code, 400)
        model = os.path.join(self.tmp, "mine.pt")
        open(model, "wb").close()
        self.assertEqual(self.put(section="yolo_world", provider="custom_model", model="yolov8s-world", model_path=model,
                                  model_class_map={"x": "bird"}).status_code, 400)          # maps to nothing real
        done = self.put(section="yolo_world", provider="custom_model", model="yolov8s-world", model_path=model,
                        model_class_map={"x": "dog", "y": "ghost"})
        self.assertEqual(done.status_code, 200, done.text)
        saved = self.settings()
        self.assertEqual((saved["provider"], saved["custom_model"]["class_map"]), ("custom_model", {"x": "dog"}))
        # "the workspace's own model": no path, nothing to map
        again = self.put(section="yolo_world", provider="custom_model", model="yolov8s-world", model_path="", model_class_map={})
        self.assertEqual(again.status_code, 200)

    def test_locate_anything_section(self):
        from utils import assistant_modes
        assistant_modes.set_mode(assistant_modes.MODE_LOCATE)
        good = dict(section="locate", targets=[{"prompt": "person", "map_to": "cat"}], generation_mode="fast", device="cpu",
                    short_side=512, passes=2, temperature=0.3)
        self.assertEqual(self.put(**good).status_code, 200)
        la = self.settings()["locate_anything"]
        self.assertEqual((la["generation_mode"], la["device"], la["short_side"], la["passes"], la["temperature"]),
                         ("fast", "cpu", 512, 2, 0.3))
        for change in ({"generation_mode": "warp"}, {"device": "tpu"}, {"short_side": "big"}, {"passes": None}):
            self.assertEqual(self.put(**{**good, **change}).status_code, 400, change)

    def test_head_section(self):
        head = os.path.join(self.tmp, "head_best.pt")
        open(head, "wb").close()
        good = dict(section="head", head_path=head, detector_weights="", detector_classes="person, bottle", iou=0.5,
                    min_head_conf=0.1, class_map={"sleeping": "cat", "x": "ghost"}, has_head_names=True, confidence=0.35)
        self.assertEqual(self.put(**good).status_code, 200)
        saved = self.settings()
        self.assertEqual((saved["custom_head"]["detector_classes"], saved["custom_head"]["class_map"], saved["confidence"]),
                         ([0, 39], {"sleeping": "cat"}, 0.35))
        for change, word in (({"head_path": ""}, "head checkpoint"), ({"head_path": os.path.join(self.tmp, "no.pt")}, "not found"),
                             ({"detector_classes": "unicorn"}, "not a class"), ({"class_map": {}}, "Map at least one"),
                             ({"iou": "high"}, "numbers")):
            res = self.put(**{**good, **change})
            self.assertEqual(res.status_code, 400, change)
            self.assertIn(word, res.json()["error"])

    def test_sam2_section_and_batch_option(self):
        good = dict(section="sam2", model="sam2.1_t.pt", imgsz=768, max_references=4, min_score=0.1)
        self.assertEqual(self.put(**good).status_code, 200)
        self.assertEqual(self.settings()["sam2_dynamic"], {"model": "sam2.1_t.pt", "imgsz": 768, "max_references": 4, "min_score": 0.1})
        for change in ({"model": "sam9.pt"}, {"imgsz": 100}, {"max_references": "many"}):
            self.assertEqual(self.put(**{**good, **change}).status_code, 400, change)
        self.assertEqual(self.put(section="batch", only_unlabeled=False).status_code, 200)
        self.assertFalse(self.settings()["batch"]["only_unlabeled"])
        self.assertEqual(self.put(section="nonsense").status_code, 400)

    def test_needs_the_header_and_a_session(self):
        self.assertEqual(self.client.put(f"{self.base}/assistant", json={"section": "batch"}).status_code, 403)
        self.client.post("/api/logout", json={}, headers=H)
        self.assertEqual(self.client.get(f"{self.base}/assistant").status_code, 401)

    def test_inspecting_model_files(self):
        with mock.patch.object(self.assistant.ap, "model_class_names", return_value=["", "car", "bus"]):
            data = self.post(f"{self.base}/assistant/inspect", {"kind": "model", "path": "x.pt"}).json()
        self.assertEqual((data["names"], data["all_names"]), (["car", "bus"], ["", "car", "bus"]))
        head = os.path.join(self.tmp, "h.pt")
        open(head, "wb").close()
        fake = types.SimpleNamespace(read_checkpoint=lambda p: {"names": ["a", "b"], "cfg": {"weights": "yolov9c.pt", "imgsz": (640, 640)}})
        with mock.patch.dict(sys.modules, {"utils.custom_head": fake}), mock.patch("utils.custom_head", fake, create=True):
            info = self.post(f"{self.base}/assistant/inspect", {"kind": "head", "path": head}).json()
        self.assertEqual(info["names"], ["a", "b"])
        self.assertIn("yolov9c.pt", info["info"])
        self.assertEqual(self.post(f"{self.base}/assistant/inspect", {"kind": "head", "path": "nope.pt"}).status_code, 400)
        self.assertEqual(self.post(f"{self.base}/assistant/inspect", {"kind": "zzz", "path": "x"}).status_code, 400)


class DownloadTests(AssistBase):
    def test_yolo_world_asks_before_downloading_its_weights(self):
        res = self.post(f"{self.base}/infer", {"name": "a.png"})
        self.assertEqual(res.status_code, 409)
        pending = res.json()["needs_confirm"]
        self.assertIn("YOLO-World", pending["title"])
        self.assertIn("github.com", pending["message"])
        self.assertFalse(self.assistant._busy.locked())                      # asking takes nothing
        # the weights are there: no question
        folder = os.path.join(self.tmp, "models", "_yolo_world")
        os.makedirs(folder)
        open(os.path.join(folder, "yolov8s-world.pt"), "wb").close()
        job = self.run_infer(fake_predictor([]))
        self.assertEqual(job["state"], "done")

    def test_sam2_asks_before_downloading(self):
        from utils import assistant_modes
        assistant_modes.set_mode(assistant_modes.MODE_SAM2)
        res = self.post(f"{self.base}/autoannotate", {})
        self.assertEqual(res.status_code, 409)
        self.assertIn("SAM 2", res.json()["needs_confirm"]["title"])
        folder = os.path.join(self.tmp, "models", "_sam2")
        os.makedirs(folder)
        open(os.path.join(folder, "sam2.1_b.pt"), "wb").close()
        with mock.patch.object(self.assistant, "build_predictor", fake_predictor([])):
            res = self.post(f"{self.base}/autoannotate", {})
            self.assertEqual(res.status_code, 202)
            self.finish(res)

    def test_the_own_model_and_the_head_never_download(self):
        from utils import assistant_modes
        assistant_modes.set_mode(assistant_modes.MODE_HEAD)
        with mock.patch.object(self.assistant, "build_predictor", fake_predictor([])):
            self.assertEqual(self.post(f"{self.base}/infer", {"name": "a.png"}).status_code, 202)


class InferTests(AssistBase):
    def test_returns_only_what_is_new_and_leaves_overlaps_to_the_existing_annotation(self):
        job = self.run_infer(fake_predictor([CAT_BOX, {"rect": (200, 100, 300, 200), "conf": 0.8, "cls": "dog"},
                                             {"rect": (5, 5, 60, 60), "conf": 0.5, "cls": "dog"}]),
                             bboxes=[[8, 8, 66, 66, "cat"]])
        self.assertEqual(job["state"], "done", job.get("error"))
        result = job["result"]
        self.assertEqual(result["bboxes"], [[200, 100, 300, 200, "dog"]])
        self.assertEqual((result["found"], result["skipped"], result["added"]), (3, 2, 1))
        self.assertIn("2 skipped", result["message"])
        self.assertFalse(result["polygon"])

    def test_classes_the_workspace_does_not_have_are_dropped(self):
        job = self.run_infer(fake_predictor([{"rect": (1, 1, 50, 50), "conf": 0.9, "cls": "bird"}]))
        self.assertEqual(job["result"]["bboxes"], [])
        self.assertEqual(job["result"]["added"], 0)

    def test_polygon_results_are_merged_against_polygon_boxes(self):
        poly = {"rect": (10, 10, 90, 90), "conf": 0.9, "cls": "dog", "poly": [(10, 10), (90, 10), (90, 90)]}
        job = self.run_infer(fake_predictor([poly], polygon=True), polygons=[[[[12, 12], [88, 12], [88, 88]], "cat"]])
        self.assertEqual(job["result"]["polygons"], [])                        # overlaps what is there
        job = self.run_infer(fake_predictor([poly], polygon=True))
        self.assertEqual(job["result"]["polygons"], [[[[10, 10], [90, 10], [90, 90]], "dog"]])
        self.assertTrue(job["result"]["polygon"])

    def test_nothing_is_written_to_disk_by_infer(self):
        self.run_infer(fake_predictor([CAT_BOX]))
        self.assertEqual(os.listdir(self.voc) if os.path.isdir(self.voc) else [], [])

    def test_a_problem_the_user_can_fix_is_the_job_message(self):
        def broken(ctx, status=None, **kw):
            raise self.assistant.AssistantError("YOLO-World can't work without a target class.")
        job = self.run_infer(broken)
        self.assertEqual(job["state"], "error")
        self.assertIn("target class", job["error"])
        self.assertFalse(self.assistant._busy.locked())                        # the model is free again

    def test_unknown_picture_and_dataset(self):
        self.assertEqual(self.post(f"{self.base}/infer", {"name": "ghost.png", "confirm_download": True}).status_code, 404)
        self.assertEqual(self.post("/api/ws/ghost-1/infer", {"name": "a.png"}).status_code, 404)

    def test_needs_the_header_and_a_session(self):
        self.assertEqual(self.client.post(f"{self.base}/infer", json={"name": "a.png"}).status_code, 403)
        self.client.post("/api/logout", json={}, headers=H)
        self.assertEqual(self.post(f"{self.base}/infer", {"name": "a.png"}).status_code, 401)


class AutoAnnotateTests(AssistBase):
    def run_all(self, predictor, **body):
        with mock.patch.object(self.assistant, "build_predictor", predictor):
            res = self.post(f"{self.base}/autoannotate", {"confirm_download": True, **body})
            self.assertEqual(res.status_code, 202, res.text)
            return self.finish(res)

    def test_labels_every_picture_and_saves_voc_and_yolo(self):
        job = self.run_all(fake_predictor([CAT_BOX]))
        self.assertEqual(job["state"], "done", job.get("error"))
        r = job["result"]
        self.assertEqual((r["total"], r["annotated"], r["added"], r["cancelled"]), (3, 3, 3, False))
        for name in ("a", "b", "c"):
            self.assertTrue(os.path.isfile(os.path.join(self.voc, name + ".xml")))
            self.assertTrue(os.path.isfile(os.path.join(self.labels, name + ".txt")))
        self.assertEqual(self.load("a.png")["bboxes"], [[10, 10, 70, 70, "cat"]])
        self.assertEqual(open(os.path.join(self.labels, "a.txt")).read().split()[0], "0")
        self.assertEqual(len(self.client.get(self.base).json()["annotated"]), 3)

    def test_only_unlabeled_skips_pictures_that_have_annotations(self):
        self.save("a.png", bboxes=[[100, 100, 200, 200, "dog"]])
        job = self.run_all(fake_predictor([CAT_BOX]), only_unlabeled=True)
        r = job["result"]
        self.assertEqual((r["skipped_labeled"], r["annotated"]), (1, 2))
        self.assertEqual(self.load("a.png")["bboxes"], [[100, 100, 200, 200, "dog"]])

    def test_without_that_option_new_boxes_are_merged_into_what_is_there(self):
        self.save("a.png", bboxes=[[100, 100, 200, 200, "dog"], [12, 12, 68, 68, "cat"]])
        job = self.run_all(fake_predictor([CAT_BOX]), only_unlabeled=False)
        self.assertEqual(job["result"]["skipped_labeled"], 0)
        self.assertEqual(len(self.load("a.png")["bboxes"]), 2)               # the prediction overlapped: the existing wins
        self.assertEqual(len(self.load("b.png")["bboxes"]), 1)

    def test_stop_ends_it_between_pictures(self):
        started, gate = threading.Event(), threading.Event()
        with mock.patch.object(self.assistant, "build_predictor", fake_predictor([CAT_BOX], started=started, gate=gate)):
            res = self.post(f"{self.base}/autoannotate", {"confirm_download": True})
            job = res.json()
            self.assertTrue(started.wait(10))
            self.assertEqual(self.post(f"/api/jobs/{job['id']}/cancel").status_code, 200)
            gate.set()
            done = self.finish(job)
        self.assertEqual(done["state"], "cancelled")
        self.assertTrue(done["result"]["cancelled"])
        self.assertLess(done["result"]["annotated"], 3)

    def test_progress_is_reported(self):
        gate, started = threading.Event(), threading.Event()
        with mock.patch.object(self.assistant, "build_predictor", fake_predictor([CAT_BOX], started=started, gate=gate)):
            job = self.post(f"{self.base}/autoannotate", {"confirm_download": True}).json()
            self.assertTrue(started.wait(10))
            data = self.client.get(f"/api/jobs/{job['id']}").json()
            self.assertEqual((data["state"], data["total"]), ("running", 3))
            gate.set()
            self.finish(job)

    def test_a_failing_picture_does_not_stop_the_rest(self):
        calls = {"n": 0}

        def build(ctx, status=None, **kw):
            def predict(bgr):
                calls["n"] += 1
                if calls["n"] == 2:
                    raise RuntimeError("boom")
                return [dict(CAT_BOX)], False
            return predict
        r = self.run_all(build)["result"]
        self.assertEqual((r["annotated"], r["failed_count"]), (2, 1))
        self.assertIn("boom", r["failed"][0][1])


class BusyTests(AssistBase):
    def test_one_model_job_at_a_time_and_the_message_names_who(self):
        started, gate = threading.Event(), threading.Event()
        with mock.patch.object(self.assistant, "build_predictor", fake_predictor([], started=started, gate=gate)):
            first = self.post(f"{self.base}/infer", {"name": "a.png", "confirm_download": True})
            self.assertEqual(first.status_code, 202)
            self.assertTrue(started.wait(10))
            second = self.post(f"{self.base}/autoannotate", {"confirm_download": True})
            self.assertEqual(second.status_code, 409)
            self.assertTrue(second.json()["busy"])
            self.assertIn("admin", second.json()["error"])
            gate.set()
            self.finish(first)
            third = self.post(f"{self.base}/autoannotate", {"confirm_download": True})
            self.assertEqual(third.status_code, 202)
            self.finish(third)


class SamReferenceTests(AssistBase):
    def store(self, ws="pets"):
        return self.assistant.sam_store(ws)

    def test_a_reviewed_save_becomes_a_reference_an_unreviewed_one_does_not(self):
        self.save("a.png", bboxes=[[10, 10, 110, 110, "cat"]])
        self.assertEqual(len(self.store()), 1)
        self.save("b.png", bboxes=[[10, 10, 110, 110, "cat"]], reviewed=False)
        self.assertEqual(len(self.store()), 1)
        self.save("a.png", bboxes=[[20, 20, 120, 120, "dog"]])             # the same picture again: replaced, not added
        self.assertEqual(len(self.store()), 1)
        path = os.path.join(self.datasets, INSTANCE, "a.png")
        self.assertEqual(self.store().items_for(path)[0][0], "dog")

    def test_too_small_boxes_are_not_references(self):
        self.save("a.png", bboxes=[[10, 10, 14, 14, "cat"]])
        self.assertEqual(len(self.store()), 0)

    def test_deleting_the_picture_forgets_it(self):
        self.save("a.png", bboxes=[[10, 10, 110, 110, "cat"]])
        self.client.delete(f"{self.base}/images/a.png", headers=H)
        self.assertEqual(len(self.store()), 0)

    def test_every_workspace_has_its_own_store(self):
        from utils import workspace_config
        from tests.unit.test_web_editor import make_dataset
        make_dataset(self.datasets, "cats-1", names=("x.png",))
        workspace_config.set_classes("cats", ["cat"])
        self.save("a.png", bboxes=[[10, 10, 110, 110, "cat"]])
        self.client.put("/api/ws/cats-1/annotations/x.png", json={"bboxes": [[5, 5, 90, 90, "cat"]], "polygons": []}, headers=H)
        self.assertEqual((len(self.store("pets")), len(self.store("cats"))), (1, 1))
        self.assertNotEqual(self.store("pets").select(5, ["cat"])[0][0], self.store("cats").select(5, ["cat"])[0][0])

    def test_learn_from_labeled_and_forget_all(self):
        for name in ("a.png", "b.png"):
            self.save(name, bboxes=[[10, 10, 110, 110, "cat"]], reviewed=False)
        self.assertEqual(len(self.store()), 0)
        res = self.post(f"{self.base}/assistant/references", {"action": "load"}).json()
        self.assertEqual((res["added"], res["references"]), (2, 2))
        self.assertEqual(self.client.get(f"{self.base}/assistant").json()["references"], 2)
        self.assertEqual(self.post(f"{self.base}/assistant/references", {"action": "forget"}).json()["references"], 0)
        self.assertEqual(self.post(f"{self.base}/assistant/references", {"action": "zzz"}).status_code, 400)

    def test_sam2_polygon_mode_without_polygon_references_explains(self):
        """The real provider, with SAM 2 itself left out: the message must tell what to do."""
        from utils import assistant_modes
        assistant_modes.set_mode(assistant_modes.MODE_SAM2)
        folder = os.path.join(self.tmp, "models", "_sam2")
        os.makedirs(folder)
        open(os.path.join(folder, "sam2.1_b.pt"), "wb").close()
        self.save("b.png", bboxes=[[10, 10, 110, 110, "cat"]])
        job = self.finish(self.post(f"{self.base}/infer", {"name": "a.png", "polygon": True}))
        self.assertEqual(job["state"], "error")
        self.assertTrue("boxes only" in job["error"] or "needs a recent ultralytics" in job["error"], job["error"])

    def test_sam2_without_any_reference_explains(self):
        from utils import assistant_modes
        assistant_modes.set_mode(assistant_modes.MODE_SAM2)
        folder = os.path.join(self.tmp, "models", "_sam2")
        os.makedirs(folder)
        open(os.path.join(folder, "sam2.1_b.pt"), "wb").close()
        job = self.finish(self.post(f"{self.base}/infer", {"name": "a.png"}))
        self.assertEqual(job["state"], "error")
        self.assertTrue("none yet" in job["error"] or "needs a recent ultralytics" in job["error"], job["error"])


class FileBrowserTests(AssistBase):
    def test_model_files_are_listed_when_asked_for(self):
        folder = os.path.join(self.tmp, "models_here")
        os.makedirs(os.path.join(folder, "sub"))
        for name in ("a.pt", "B.PT", "c.txt"):
            open(os.path.join(folder, name), "w").close()
        data = self.client.get("/api/folders", params={"path": folder, "ext": ".pt"}).json()
        self.assertEqual((data["folders"], data["files"]), (["sub"], ["a.pt", "B.PT"]))
        self.assertEqual(self.client.get("/api/folders", params={"path": folder}).json()["files"], [])


if __name__ == "__main__":
    unittest.main()
