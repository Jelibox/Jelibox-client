"""Full YOLO training (utils/yolo_train_job.py): the scratch dataset, the events sent after every epoch and what is
passed to Ultralytics. Ultralytics itself is faked, so no model is downloaded and nothing is trained."""
import contextlib
import io
import json
import os
import sys
import tempfile
import types
import unittest
from unittest import mock

from utils import yolo_train_job as yj

CLASSES = ["cat", "dog"]


class FakeTrainer:
    """The object Ultralytics hands to its callbacks; `stop` is the flag it checks between batches."""

    def __init__(self, epochs, save_dir):
        self.epochs, self.save_dir, self.stop = epochs, save_dir, False
        self.tloss = object()
        self.epoch, self.fitness, self.best_fitness, self.metrics = 0, 0.0, 0.0, {}

    def begin_epoch(self, epoch, fitness, best_fitness):
        self.epoch, self.fitness, self.best_fitness = epoch, fitness, best_fitness
        self.metrics = {"metrics/precision(B)": 0.5 + epoch / 10, "metrics/recall(B)": 0.4 + epoch / 10,
                        "metrics/mAP50(B)": fitness, "metrics/mAP50-95(B)": fitness / 2, "val/box_loss": 9.0}

    def label_loss_items(self, tloss, prefix="train"):
        return {f"{prefix}/box_loss": 1.0 - 0.1 * self.epoch, f"{prefix}/cls_loss": 0.5, f"{prefix}/dfl_loss": 0.9}


class FakeYOLO:
    """Stands in for ultralytics.YOLO: calls the registered callbacks the way a real run does, including leaving
    the batch loop and the epoch loop when `trainer.stop` is set, and the final evaluation afterwards."""
    fitnesses = [0.2, 0.5, 0.4]
    batches = 3
    on_batch = None                  # hook(epoch, batch, trainer) for a test that wants to act mid-epoch
    made = []

    def __init__(self, weights):
        self.weights, self.callbacks, self.kwargs, self.final_eval_ran = weights, {}, None, False
        FakeYOLO.made.append(self)

    def add_callback(self, event, fn):
        self.callbacks.setdefault(event, []).append(fn)

    def _run(self, event, trainer):
        for fn in self.callbacks.get(event, []):
            fn(trainer)

    def train(self, **kw):
        self.kwargs = kw
        save_dir = os.path.join(kw["project"], kw["name"])
        os.makedirs(os.path.join(save_dir, "weights"), exist_ok=True)
        with open(os.path.join(save_dir, "weights", "best.pt"), "wb") as f:
            f.write(b"best")
        trainer = FakeTrainer(kw["epochs"], save_dir)
        self.trainer = trainer
        self._run("on_train_start", trainer)
        best = -1.0
        for e in range(kw["epochs"]):
            fitness = self.fitnesses[e % len(self.fitnesses)]
            best = max(best, fitness)
            trainer.begin_epoch(e, fitness, best)
            for b in range(self.batches):
                if FakeYOLO.on_batch:
                    FakeYOLO.on_batch(e, b, trainer)
                self._run("on_train_batch_end", trainer)
                if trainer.stop:
                    break                                   # Ultralytics checks `stop` between batches
            self._run("on_fit_epoch_end", trainer)           # validation has run, even for an interrupted epoch
            if trainer.stop:
                break
        self.final_eval_ran = True


class Base(unittest.TestCase):
    def setUp(self):
        root = tempfile.mkdtemp()
        self.images, self.labels, self.out = (os.path.join(root, n) for n in ("images", "labels", "out"))
        os.makedirs(self.images)
        os.makedirs(self.labels)
        for i in range(1, 13):
            name = f"img_{i:02d}"
            with open(os.path.join(self.images, name + ".jpg"), "wb") as f:
                f.write(b"jpg")
            with open(os.path.join(self.labels, name + ".txt"), "w") as f:
                f.write(f"{i % 2} 0.5 0.5 0.4 0.4\n")
        FakeYOLO.made.clear()
        FakeYOLO.fitnesses = [0.2, 0.5, 0.4]
        FakeYOLO.on_batch = None
        self.addCleanup(setattr, FakeYOLO, "on_batch", None)

    def job(self, **kw):
        job = {"kind": "yolo", "images_folders": [self.images], "labels_folder": self.labels, "classes": CLASSES,
               "init_weights": "yolo11s.pt", "imgsz": 640, "epochs": 3, "batch": 8, "lr": 0.001, "freeze": 0,
               "val_fraction": 0.2, "out_dir": self.out, "scratch_dir": os.path.join(self.out, "dataset"),
               "workers": 1, "device": "cpu", "stop_file": os.path.join(self.out, "finish_early.flag")}
        job.update(kw)
        return job

    def run_job(self, job):
        events = []
        with mock.patch.object(yj, "load_yolo", lambda weights: FakeYOLO(weights)):
            summary = yj.train(job, emit=lambda event, **f: events.append({"event": event, **f}))
        return summary, events


class DatasetTests(Base):
    def test_labelled_images_are_copied_with_the_last_ones_as_validation(self):
        job = self.job()
        yaml_path, scratch, n_train, n_val, counts = yj.prepare_dataset(job)
        self.assertEqual((n_train, n_val), (10, 2))
        self.assertEqual(sorted(os.listdir(os.path.join(scratch, "val", "images"))), ["img_11.jpg", "img_12.jpg"])
        self.assertEqual(sorted(os.listdir(os.path.join(scratch, "val", "labels"))), ["img_11.txt", "img_12.txt"])
        self.assertEqual(len(os.listdir(os.path.join(scratch, "train", "images"))), 10)
        self.assertEqual(sum(counts), 10)

    def test_data_yaml_names_the_classes_and_the_two_folders(self):
        yaml_path = yj.prepare_dataset(self.job(classes=["cat", "dog food"]))[0]
        with open(yaml_path, encoding="utf-8") as f:
            text = f.read()
        self.assertIn("nc: 2", text)
        self.assertIn('  0: "cat"', text)
        self.assertIn('  1: "dog food"', text)
        self.assertIn("train:", text)
        self.assertIn("val:", text)

    def test_a_scratch_copy_left_by_a_stopped_run_is_replaced(self):
        job = self.job()
        os.makedirs(job["scratch_dir"])
        stale = os.path.join(job["scratch_dir"], "train", "images", "old.jpg")
        os.makedirs(os.path.dirname(stale))
        open(stale, "w").close()
        yj.prepare_dataset(job)
        self.assertFalse(os.path.exists(stale))

    def test_too_few_labelled_images_is_refused(self):
        for name in sorted(os.listdir(self.labels))[:5]:
            os.remove(os.path.join(self.labels, name))
        with self.assertRaisesRegex(ValueError, "Only 7 labelled images"):
            yj.prepare_dataset(self.job())


class AugmentTests(Base):
    """job["augment"]: the training part of the scratch dataset also gets augmented copies (never the validation part)."""

    def setUp(self):
        super().setUp()
        import cv2
        import numpy as np
        for name in os.listdir(self.images):                                   # real pictures, so they can be augmented
            os.remove(os.path.join(self.images, name))
        for i in range(1, 13):
            cv2.imwrite(os.path.join(self.images, f"img_{i:02d}.png"),
                        np.full((40, 60, 3), 20 * i, np.uint8))

    def augment(self, ops, **kw):
        from utils import augment_catalog as cat
        from utils import augmentation as aug
        state = cat.default_state()
        for name in ops:
            state[name].enabled = True
        return aug.config_to_dict(cat.to_config(state, kw.pop("outputs", 3), kw.pop("keep", True), seed=1, **kw))

    def test_the_training_part_grows_and_the_validation_part_does_not(self):
        job = self.job(augment=self.augment(["brightness"]))
        _yaml, scratch, n_train, n_val, counts = yj.prepare_dataset(job)
        self.assertEqual((n_train, n_val), (30, 2))                            # 10 pictures x (original + 2 copies)
        names = sorted(os.listdir(os.path.join(scratch, "train", "images")))
        self.assertIn("img_01.png", names)
        self.assertIn("img_01_aug2.png", names)
        self.assertEqual(sorted(os.listdir(os.path.join(scratch, "val", "images"))), ["img_11.png", "img_12.png"])
        self.assertEqual(sum(counts), 10, "the box counts are those of the real pictures")
        self.assertEqual(len(os.listdir(os.path.join(scratch, "train", "labels"))), 30)

    def test_without_the_original_only_the_copies_are_written(self):
        job = self.job(augment=self.augment(["brightness"], keep=False, outputs=2))
        _yaml, scratch, n_train, n_val, _ = yj.prepare_dataset(job)
        names = os.listdir(os.path.join(scratch, "train", "images"))
        self.assertEqual((n_train, len(names)), (20, 20))
        self.assertNotIn("img_01.png", names)
        self.assertEqual(sorted(os.listdir(os.path.join(scratch, "val", "images"))), ["img_11.png", "img_12.png"])

    def test_a_flip_moves_the_label_with_the_picture(self):
        label = os.path.join(self.labels, "img_01.txt")
        with open(label, "w") as f:
            f.write("0 0.25 0.5 0.2 0.4" + chr(10))
        job = self.job(augment=self.augment(["flip_h"], outputs=2))
        job["augment"]["ops"]["flip_h"]["probability"] = 100
        _yaml, scratch, *_ = yj.prepare_dataset(job)
        with open(os.path.join(scratch, "train", "labels", "img_01_aug1.txt")) as f:
            parts = f.read().split()
        self.assertEqual(parts[0], "0")
        self.assertAlmostEqual(float(parts[1]), 0.75, places=3)
        self.assertAlmostEqual(float(parts[2]), 0.5, places=3)

    def test_no_augment_or_no_copies_changes_nothing(self):
        for augment in (None, {}, self.augment(["brightness"], outputs=1)):
            job = self.job(augment=augment) if augment is not None else self.job()
            _yaml, scratch, n_train, n_val, _ = yj.prepare_dataset(job)
            self.assertEqual((n_train, n_val), (10, 2))

    def test_the_run_reports_the_grown_training_set(self):
        job = self.job(augment=self.augment(["brightness"]))
        _summary, events = self.run_job(job)
        start = next(e for e in events if e["event"] == "start")
        self.assertEqual((start["train_images"], start["val_images"]), (30, 2))


class TrainTests(Base):
    def test_one_event_per_epoch_with_the_losses_and_the_metrics(self):
        summary, events = self.run_job(self.job())
        kinds = [e["event"] for e in events]
        self.assertEqual(kinds, ["status", "status", "start", "epoch", "epoch", "epoch", "done"])
        start = events[2]
        self.assertEqual((start["kind"], start["train_images"], start["val_images"], start["epochs"]), ("yolo", 10, 2, 3))
        self.assertEqual((start["device"], start["batch"], start["imgsz"], start["model"]), ("cpu", 8, [640, 640], "yolo11s.pt"))
        first, second, third = events[3:6]
        self.assertEqual((first["epoch"], first["epochs"]), (1, 3))
        self.assertAlmostEqual(first["box_loss"], 1.0)
        self.assertAlmostEqual(first["cls_loss"], 0.5)
        self.assertAlmostEqual(first["dfl_loss"], 0.9)
        self.assertAlmostEqual(first["precision"], 0.5)
        self.assertAlmostEqual(first["map50"], 0.2)
        self.assertAlmostEqual(first["map5095"], 0.1)
        self.assertEqual([e["best"] for e in (first, second, third)], [True, True, False])
        self.assertGreaterEqual(first["seconds"], 0)

    def test_the_summary_points_at_the_best_weights(self):
        summary, events = self.run_job(self.job())
        self.assertEqual((summary["best_epoch"], summary["epochs_run"], summary["stopped"]), (2, 3, False))
        self.assertTrue(summary["has_model"])
        self.assertTrue(summary["weights"].endswith(os.path.join("train_run", "weights", "best.pt")))
        self.assertTrue(os.path.isfile(summary["weights"]))
        self.assertAlmostEqual(summary["map50"], 0.4)                  # the last epoch
        self.assertEqual(events[-1]["event"], "done")
        json.dumps(events)                                              # every event can go over stdout as JSON

    def test_ultralytics_gets_every_layer_the_chosen_size_and_the_scratch_data(self):
        self.run_job(self.job(freeze=0))
        kw = FakeYOLO.made[0].kwargs
        self.assertEqual(FakeYOLO.made[0].weights, "yolo11s.pt")
        self.assertEqual((kw["epochs"], kw["imgsz"], kw["batch"], kw["optimizer"]), (3, 640, 8, "AdamW"))
        self.assertEqual((kw["lr0"], kw["device"], kw["amp"], kw["plots"], kw["exist_ok"]), (0.001, "cpu", False, False, True))
        self.assertIsNone(kw["freeze"], "0 means every layer is trained")
        self.assertEqual((kw["project"], kw["name"]), (self.out, "train_run"))
        self.assertTrue(kw["data"].endswith("data.yaml"))

    def test_a_gpu_run_is_announced_as_cuda_not_as_a_bare_zero(self):
        _, events = self.run_job(self.job(device=0))
        self.assertEqual(events[2]["device"], "cuda")
        self.assertEqual(FakeYOLO.made[0].kwargs["device"], 0)

    def test_freeze_is_passed_on(self):
        self.run_job(self.job(freeze=10))
        self.assertEqual(FakeYOLO.made[0].kwargs["freeze"], 10)

    def test_the_scratch_copy_is_removed_afterwards(self):
        job = self.job()
        self.run_job(job)
        self.assertFalse(os.path.exists(job["scratch_dir"]))

    def test_an_error_in_training_still_removes_the_scratch_copy(self):
        job = self.job()
        with mock.patch.object(yj, "load_yolo", side_effect=RuntimeError("boom")):
            with self.assertRaisesRegex(RuntimeError, "boom"):
                yj.train(job, emit=lambda *a, **k: None)
        self.assertFalse(os.path.exists(job["scratch_dir"]))

    def test_the_image_size_must_be_a_multiple_of_32(self):
        with self.assertRaisesRegex(ValueError, "multiple of 32"):
            self.run_job(self.job(imgsz=100))

    def test_the_device_is_picked_automatically_when_the_job_names_none(self):
        _, events = self.run_job(self.job(device=None))
        gpu = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: True))
        cpu = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))
        with mock.patch.dict(sys.modules, {"torch": gpu}):
            self.assertEqual(yj.pick_device(), 0)
        with mock.patch.dict(sys.modules, {"torch": cpu}):
            self.assertEqual(yj.pick_device(), "cpu")
        with mock.patch.dict(sys.modules, {"torch": None}):               # not installed
            self.assertEqual(yj.pick_device(), "cpu")


class FinishEarlyTests(Base):
    """The Finish early button: a file appears, the next batch sees it and Ultralytics wraps the run up properly."""

    def flag_at(self, epoch, batch):
        flag = os.path.join(self.out, "finish_early.flag")

        def hook(e, b, trainer):
            if (e, b) == (epoch, batch):
                os.makedirs(self.out, exist_ok=True)
                with open(flag, "w") as f:
                    f.write("x")
        FakeYOLO.on_batch = hook

    def test_without_the_flag_the_run_goes_to_the_end(self):
        summary, events = self.run_job(self.job(epochs=3))
        self.assertFalse(summary["stopped"])
        self.assertEqual(summary["epochs_run"], 3)
        self.assertFalse(FakeYOLO.made[0].trainer.stop)

    def test_the_flag_ends_the_run_in_the_middle_of_an_epoch_but_it_is_still_validated_and_evaluated(self):
        self.flag_at(1, 1)                                   # epoch 2, second batch
        summary, events = self.run_job(self.job(epochs=5))
        model = FakeYOLO.made[0]
        self.assertTrue(model.trainer.stop, "Ultralytics was told to stop")
        self.assertTrue(model.final_eval_ran, "and then ran its usual final evaluation")
        self.assertEqual([e["epoch"] for e in events if e["event"] == "epoch"], [1, 2],
                         "the interrupted epoch is validated and reported like the others")
        self.assertTrue(summary["stopped"])
        self.assertEqual((summary["epochs_run"], summary["best_epoch"]), (2, 2))
        self.assertTrue(summary["has_model"])
        self.assertEqual(events[-1]["event"], "done")
        said = [e["text"] for e in events if e["event"] == "status"]
        self.assertTrue(any("Finishing early" in t for t in said), said)

    def test_a_flag_that_appears_during_validation_is_seen_after_the_epoch(self):
        FakeYOLO.batches = 1
        self.addCleanup(setattr, FakeYOLO, "batches", 3)
        flag = os.path.join(self.out, "finish_early.flag")
        seen = []

        def hook(e, b, trainer):
            if e == 0:
                os.makedirs(self.out, exist_ok=True)
                with open(flag, "w") as f:
                    f.write("x")
        FakeYOLO.on_batch = hook
        summary, _ = self.run_job(self.job(epochs=4))
        self.assertTrue(summary["stopped"])
        self.assertEqual(summary["epochs_run"], 1)

    def test_the_end_of_the_flag_is_announced_once(self):
        self.flag_at(0, 0)
        _, events = self.run_job(self.job(epochs=3))
        said = [e["text"] for e in events if e["event"] == "status" and "Finishing early" in e["text"]]
        self.assertEqual(len(said), 1)

    def test_a_job_without_a_stop_file_never_stops_early(self):
        job = self.job(epochs=2)
        del job["stop_file"]
        summary, _ = self.run_job(job)
        self.assertFalse(summary["stopped"])


class MainTests(Base):
    def run_main(self, job):
        path = os.path.join(self.out, "job.json")
        os.makedirs(self.out, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(job, f)
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            code = yj.main([path])
        events = [json.loads(line[len(yj.PREFIX):]) for line in out.getvalue().splitlines() if line.startswith(yj.PREFIX)]
        return code, events

    def test_main_prints_the_events_as_prefixed_json_lines(self):
        with mock.patch.object(yj, "load_yolo", lambda weights: FakeYOLO(weights)):
            code, events = self.run_main(self.job())
        self.assertEqual(code, 0)
        self.assertEqual([e["event"] for e in events].count("epoch"), 3)

    def test_main_reports_a_failure_as_an_error_event_and_a_non_zero_exit(self):
        with mock.patch.object(yj, "load_yolo", side_effect=RuntimeError("no such model")):
            code, events = self.run_main(self.job())
        self.assertEqual(code, 1)
        self.assertEqual(events[-1], {"event": "error", "message": "no such model"})

    def test_main_needs_exactly_one_argument(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(yj.main([]), 2)

    def test_the_module_imports_neither_torch_nor_ultralytics_at_the_top(self):
        with open(yj.__file__, encoding="utf-8") as f:
            top_level = [line for line in f.read().splitlines() if line.startswith(("import ", "from "))]
        self.assertFalse([line for line in top_level if "torch" in line or "ultralytics" in line], top_level)


if __name__ == "__main__":
    unittest.main()
