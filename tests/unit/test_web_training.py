"""Training through the web API with a fake training process that speaks the real protocol (`@@ {json}` event lines):
the Train setup, validation, one training at a time, Finish early, Stop now, installing the best model."""
import json
import os
import sys
import time
import unittest

from tests.unit.test_web_app import H, TestClient
from tests.unit.test_web_editor import INSTANCE, EditorBase

FAKE = r'''
import json, os, sys, time
job = json.load(open(sys.argv[1]))
mode = os.environ.get("FAKE_MODE", "ok")
def emit(event, **kw):
    print("@@ " + json.dumps({"event": event, **kw}), flush=True)
kind = job.get("kind", "head")
epochs = int(job["epochs"])
emit("status", text="Preparing the training images ...")
emit("prepare", done=1, total=2, text="Step 3/3  Preparing the pictures")
if mode == "error":
    emit("error", message="boom: out of memory")
    sys.exit(1)
if kind == "yolo":
    emit("start", kind="yolo", train_images=8, val_images=2, test_images=0, background_images=0, classes=job["classes"],
         counts=[4, 4], epochs=epochs, device="cpu", batch=job["batch"], imgsz=[job["imgsz"], job["imgsz"]],
         model=os.path.basename(str(job["init_weights"])), freeze=job.get("freeze", 0))
else:
    emit("start", kind="head", train_images=8, val_images=2, classes=job["classes"], counts=[4, 4], epochs=epochs,
         device="cpu", batch=job["batch"], imgsz=job["imgsz"], head_params_m=0.1)
done = 0
stopped = False
for e in range(1, epochs + 1):
    if mode == "wait":
        while not os.path.exists(job["stop_file"]):
            time.sleep(0.05)
    elif os.path.exists(job["stop_file"]):
        stopped = True
        break
    time.sleep(float(os.environ.get("FAKE_DELAY", "0.05")))
    if kind == "yolo":
        emit("epoch", epoch=e, epochs=epochs, box_loss=1.0 / e, cls_loss=0.5, dfl_loss=0.4, precision=0.5, recall=0.5,
             map50=0.1 * e, map5095=0.05 * e, fitness=0.1 * e, best=(e == epochs), seconds=0.1)
    else:
        emit("epoch", epoch=e, epochs=epochs, loss=1.0 / e, acc=0.5, f1=0.1 * e, recall=[0.5, 0.5], best=(e == epochs), seconds=0.1)
    done = e
    if mode == "wait":
        stopped = True
        break
weights = None
if kind == "yolo":
    folder = os.path.join(job["out_dir"], "train_run", "weights")
    os.makedirs(folder, exist_ok=True)
    weights = os.path.join(folder, "best.pt")
    open(weights, "wb").write(b"best-model")
    emit("done", epochs_run=done, best_epoch=max(done, 1), map50=0.3, map5095=0.2, stopped=stopped, weights=weights)
else:
    open(os.path.join(job["out_dir"], "head_best.pt"), "wb").write(b"best-head")
    emit("done", epochs_run=done, best_epoch=max(done, 1), best_f1=0.4, stopped=stopped)
'''


class TrainBase(EditorBase):
    def setUp(self):
        super().setUp()
        script = os.path.join(self.tmp, "fake_train.py")
        with open(script, "w", encoding="utf-8") as f:
            f.write(FAKE)
        self.trainer = self.app.state.trainer
        self.trainer.command = lambda job_path: [sys.executable, "-u", script, job_path]
        self.addCleanup(self.trainer.stop_and_wait)
        self.addCleanup(lambda: os.environ.pop("FAKE_MODE", None))
        self.url = f"{self.base}/train"

    def post(self, path, body=None):
        return self.client.post(path, json=body or {}, headers=H)

    def yolo(self, **extra):
        return {"kind": "yolo", "base": "yolo11s.pt", "imgsz": 320, "epochs": 3, "batch": 4, "lr": 0.001, "freeze": 0,
                "weight_decay": 0.0005, "valid_pct": 20, "test_pct": 0, "confirm": True, **extra}

    def head(self, **extra):
        return {"kind": "head", "detector": "yolov8m.pt", "taps": "15, 18, 21", "height": 384, "width": 640, "epochs": 3,
                "batch": 4, "lr": 0.001, "val_share": 20, "confirm": True, **extra}

    def wait_end(self, limit=20):
        for _ in range(int(limit / 0.1)):
            state = self.client.get("/api/train").json()
            if state["run"] and state["run"]["state"] == "ended":
                return state["run"]
            time.sleep(0.1)
        self.fail("the training did not end: " + json.dumps(self.client.get("/api/train").json())[:400])

    def job_file(self, folder):
        with open(os.path.join(self.tmp, "models", "pets", folder, "job.json"), encoding="utf-8") as f:
            return json.load(f)


class InfoTests(TrainBase):
    def test_what_the_setup_shows(self):
        info = self.client.get(self.url).json()
        self.assertEqual((info["workspace"], info["task"], info["classes"]), ("pets", "detect", ["cat", "dog"]))
        self.assertEqual((info["min_labelled"], info["running"], info["run"]), (10, False, None))
        self.assertEqual(info["defaults"]["base"], "yolo11s.pt")
        self.assertIn("yolo26m.pt", info["stock"]["yolo"])
        self.assertEqual(info["taps"]["yolov9c.pt"], [15, 18, 21])
        self.assertIn(info["gpu"], (True, False, None))
        self.assertFalse(info["has_model"])
        seg = self.client.get(self.url, params={"polygon": 1}).json()
        self.assertEqual((seg["task"], seg["defaults"]["base"]), ("seg", "yolo11s-seg.pt"))
        self.assertTrue(all(m.endswith("-seg.pt") for m in seg["stock"]["yolo"]))

    def test_counting_is_only_done_on_request(self):
        self.assertIsNone(self.client.get(self.url).json()["counts"]["counted"])
        res = self.post(f"{self.url}/recount")
        self.assertEqual(res.status_code, 202)
        for _ in range(100):
            if self.client.get(f"/api/jobs/{res.json()['id']}").json()["state"] != "running":
                break
            time.sleep(0.05)
        counts = self.client.get(self.url).json()["counts"]
        self.assertIsNotNone(counts["counted"])

    def test_needs_a_session_and_the_header(self):
        self.assertEqual(self.client.post(self.url, json=self.yolo()).status_code, 403)
        self.client.post("/api/logout", json={}, headers=H)
        self.assertEqual(self.client.get(self.url).status_code, 401)
        self.assertEqual(self.client.get("/api/train").status_code, 401)


class ValidationTests(TrainBase):
    def test_every_number_is_checked_before_a_process_starts(self):
        cases = [({"epochs": 0}, "Epochs"), ({"epochs": 5000}, "Epochs"), ({"batch": 0}, "Batch"), ({"lr": 5}, "Learning rate"),
                 ({"lr": "fast"}, "Learning rate"), ({"imgsz": 100}, "multiple of 32"), ({"imgsz": 10}, "image size"),
                 ({"freeze": 500}, "Freeze"), ({"weight_decay": 3}, "Weight decay"), ({"valid_pct": 0}, "validation part"),
                 ({"valid_pct": 60, "test_pct": 40}, "training part"), ({"base": ""}, "base model"),
                 ({"base": os.path.join(self.tmp, "no.pt")}, "not found"), ({"kind": "other"}, "training type")]
        for change, word in cases:
            res = self.post(self.url, self.yolo(**change))
            self.assertEqual(res.status_code, 400, change)
            self.assertIn(word.lower(), res.json()["error"].lower(), change)
        self.assertIsNone(self.client.get("/api/train").json()["run"])

    def test_head_fields_are_checked_too(self):
        cases = [({"detector": ""}, "detector"), ({"taps": "1, 2"}, "Taps"), ({"height": 100}, "multiples of 32"),
                 ({"val_share": 90}, "validation share"), ({"detector": os.path.join(self.tmp, "no.pt")}, "not found")]
        for change, word in cases:
            res = self.post(self.url, self.head(**change))
            self.assertEqual(res.status_code, 400, change)
            self.assertIn(word.lower(), res.json()["error"].lower(), change)

    def test_too_few_labelled_pictures_once_they_were_counted(self):
        job = self.post(f"{self.url}/recount").json()
        for _ in range(100):
            if self.client.get(f"/api/jobs/{job['id']}").json()["state"] != "running":
                break
            time.sleep(0.05)
        res = self.post(self.url, self.yolo())
        self.assertEqual(res.status_code, 400)
        self.assertIn("At least 10", res.json()["error"])


class RunTests(TrainBase):
    def test_a_full_training_runs_and_installs_the_model(self):
        started = self.post(self.url, self.yolo())
        self.assertEqual(started.status_code, 200, started.text)
        run = self.wait_end()
        self.assertEqual((run["state"], run["final"]["kind"], run["final"]["title"]), ("ended", "ok", "Training finished"))
        self.assertEqual(len(run["rows"]), 3)
        self.assertEqual(run["rows"][-1]["epoch"], 3)
        self.assertEqual(run["best_epoch"], 3)
        self.assertEqual((run["bar"]["done"], run["bar"]["total"]), (3, 3))
        model = os.path.join(self.tmp, "models", "pets", "modelAssistant.pt")
        self.assertEqual(open(model, "rb").read(), b"best-model")
        self.assertEqual(run["final"]["installed"], "models/pets/modelAssistant.pt")
        self.assertIn("My trained model", run["final"]["text"])
        from utils import workspace_config as wcfg
        self.assertEqual(wcfg.get_assistant("pets")["provider"], "custom_model")
        job = self.job_file("yolo_run")
        self.assertEqual((job["kind"], job["init_weights"], job["imgsz"], job["epochs"], job["val_fraction"]),
                         ("yolo", "yolo11s.pt", 320, 3, 0.2))
        self.assertEqual(job["classes"], ["cat", "dog"])
        self.assertTrue(job["images_folders"][0].endswith(INSTANCE))
        self.assertFalse(self.client.get(self.url).json()["running"])
        self.assertFalse(self.trainer.is_running())

    def test_the_head_training_installs_the_head_and_points_the_assistant_at_it(self):
        self.assertEqual(self.post(self.url, self.head()).status_code, 200)
        run = self.wait_end()
        self.assertEqual(run["final"]["title"], "Training finished")
        self.assertEqual(open(os.path.join(self.tmp, "models", "pets", "head_best.pt"), "rb").read(), b"best-head")
        from utils import workspace_config as wcfg
        self.assertEqual(wcfg.get_assistant("pets")["custom_head"]["head_path"], "models/pets/head_best.pt")
        self.assertEqual(self.job_file("head_run")["taps"], [15, 18, 21])
        self.assertEqual(run["rows"][-1]["f1"], 0.30000000000000004)

    def test_the_run_stays_after_the_page_is_gone_and_can_be_dismissed(self):
        self.post(self.url, self.yolo())
        self.wait_end()
        again = self.client.get("/api/train").json()
        self.assertEqual(again["run"]["final"]["title"], "Training finished")
        self.assertEqual(self.post("/api/train/dismiss").status_code, 200)
        self.assertIsNone(self.client.get("/api/train").json()["run"])

    def test_only_the_new_rows_can_be_asked_for(self):
        self.post(self.url, self.yolo())
        self.wait_end()
        part = self.client.get("/api/train", params={"rows": 2}).json()["run"]
        self.assertEqual((len(part["rows"]), part["rows_total"], part["rows"][0]["epoch"]), (1, 3, 3))

    def test_an_error_event_ends_it_as_failed_and_installs_nothing(self):
        os.environ["FAKE_MODE"] = "error"
        self.post(self.url, self.yolo())
        run = self.wait_end()
        self.assertEqual((run["final"]["kind"], run["final"]["title"]), ("error", "Training failed"))
        self.assertIn("out of memory", run["final"]["text"])
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "models", "pets", "modelAssistant.pt")))

    def test_finish_early_ends_properly_and_keeps_the_best(self):
        os.environ["FAKE_MODE"] = "wait"
        self.post(self.url, self.yolo(epochs=5))
        for _ in range(100):
            if self.client.get("/api/train").json()["run"]["start_info"]:
                break
            time.sleep(0.05)
        res = self.post("/api/train/finish-early")
        self.assertEqual(res.status_code, 200)
        self.assertTrue(res.json()["finishing"])
        run = self.wait_end()
        self.assertEqual(run["final"]["title"], "Finished early")
        self.assertTrue(os.path.isfile(os.path.join(self.tmp, "models", "pets", "modelAssistant.pt")))
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "models", "pets", "yolo_run", "finish_early.flag")))

    def test_stop_now_kills_it_and_keeps_what_was_saved(self):
        os.environ["FAKE_MODE"] = "wait"
        self.post(self.url, self.yolo(epochs=5))
        for _ in range(100):
            if self.client.get("/api/train").json()["run"]["start_info"]:
                break
            time.sleep(0.05)
        self.assertEqual(self.post("/api/train/stop").status_code, 200)
        run = self.wait_end()
        self.assertTrue(run["stopping"])
        self.assertEqual(run["final"]["title"], "Stopped")                                 # killed before any epoch: nothing to keep
        self.assertFalse(self.trainer.is_running())

    def test_nothing_to_stop(self):
        for path in ("/api/train/stop", "/api/train/finish-early"):
            self.assertEqual(self.post(path).status_code, 400, path)


class RulesTests(TrainBase):
    def test_one_training_at_a_time(self):
        os.environ["FAKE_MODE"] = "wait"
        self.assertEqual(self.post(self.url, self.yolo(epochs=5)).status_code, 200)
        second = self.post(self.url, self.yolo())
        self.assertEqual(second.status_code, 409)
        self.assertTrue(second.json()["busy"])
        self.assertIn("pets", second.json()["error"])
        self.assertTrue(self.client.get(self.url).json()["running"])
        self.post("/api/train/stop")
        self.wait_end()

    def test_a_workspace_that_trains_cannot_be_deleted(self):
        os.environ["FAKE_MODE"] = "wait"
        self.post(self.url, self.yolo(epochs=5))
        self.assertEqual(self.client.delete("/api/workspaces/pets", headers=H).status_code, 409)
        self.post("/api/train/stop")
        self.wait_end()

    def test_replacing_a_trained_model_and_downloading_a_base_model_need_a_yes(self):
        folder = os.path.join(self.tmp, "models", "pets")
        os.makedirs(folder)
        open(os.path.join(folder, "modelAssistant.pt"), "wb").write(b"old")
        res = self.post(self.url, self.yolo(confirm=False))
        self.assertEqual(res.status_code, 409)
        self.assertIn("already has a trained model", res.json()["needs_confirm"]["message"])
        self.assertIsNone(self.client.get("/api/train").json()["run"])
        # continuing from it replaces nothing, so there is nothing to ask about
        ok = self.post(self.url, self.yolo(confirm=False, **{"continue": True}))
        self.assertEqual(ok.status_code, 200)
        self.wait_end()
        self.assertEqual(self.job_file("yolo_run")["init_weights"], os.path.join(folder, "modelAssistant.pt"))
        self.post("/api/train/dismiss")
        os.remove(os.path.join(folder, "modelAssistant.pt"))
        os.remove(os.path.join(folder, "modelAssistant.prev.pt")) if os.path.exists(os.path.join(folder, "modelAssistant.prev.pt")) else None
        first_use = self.post(self.url, self.yolo(confirm=False))
        self.assertEqual(first_use.status_code, 409)
        self.assertIn("first use", first_use.json()["needs_confirm"]["title"])
        self.assertIn("github.com", first_use.json()["needs_confirm"]["message"])

    def test_an_existing_head_asks_before_it_is_replaced(self):
        folder = os.path.join(self.tmp, "models", "pets")
        os.makedirs(folder)
        open(os.path.join(folder, "head_best.pt"), "wb").write(b"old head")
        res = self.post(self.url, self.head(confirm=False))
        self.assertEqual(res.status_code, 409)
        self.assertIn("head_best.prev.pt", res.json()["needs_confirm"]["message"])
        self.assertEqual(self.post(self.url, self.head()).status_code, 200)
        self.wait_end()
        self.assertEqual(open(os.path.join(folder, "head_best.prev.pt"), "rb").read(), b"old head")

    def test_a_model_on_disk_is_not_a_download(self):
        base = os.path.join(self.tmp, "mine.pt")
        open(base, "wb").close()
        res = self.post(self.url, self.yolo(base=base, confirm=False))
        self.assertEqual(res.status_code, 200, res.text)
        self.wait_end()


if __name__ == "__main__":
    unittest.main()
