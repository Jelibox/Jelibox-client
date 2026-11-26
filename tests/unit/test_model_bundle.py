"""Model folders (the checkpoint carried to another session), the run state they hold and the two dataset helpers a
resume needs: the fixed split and the cheap "same dataset" check."""
import os
import shutil
import tempfile
import unittest

from utils import head_data, model_bundle, run_state


def write(path, data=b"x"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb" if isinstance(data, bytes) else "w") as f:
        f.write(data)


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.models = os.path.join(self.tmp, "models", "pets")
        self.run = os.path.join(self.models, "yolo_run")

    def a_run(self, status="paused", epoch=4, epochs=10):
        run_state.write(self.run, status=status, epoch=epoch, epochs=epochs, classes=["cat", "dog"], imgsz=640, batch=16,
                        split={"val": ["a.jpg"], "test": []}, dataset={"pictures": 10, "annotations": 25})
        write(run_state.resume_path(self.run), b"resume")
        write(os.path.join(self.run, "train_run", "weights", "best.pt"), b"best")
        write(os.path.join(self.run, "train_run", "weights", "last.pt"), b"stripped")
        write(os.path.join(self.run, "train_run", "results.csv"), "epoch,metrics/mAP50(B)\n1,0.1\n")


class StateTests(Base):
    def test_the_state_is_written_whole_and_merged(self):
        run_state.write(self.run, status="running", epoch=1)
        run_state.write(self.run, epoch=2)
        st = run_state.read(self.run)
        self.assertEqual((st["status"], st["epoch"], st["kind"]), ("running", 2, "yolo"))
        self.assertFalse([f for f in os.listdir(self.run) if f.endswith(".tmp")])

    def test_mark_changes_only_a_run_that_exists_and_knows_its_statuses(self):
        self.assertIsNone(run_state.mark(self.run, "stopped"))
        self.assertFalse(os.path.exists(self.run), "no state file is made for something that never was a full run")
        run_state.write(self.run, status="running")
        self.assertEqual(run_state.mark(self.run, "stopped", "Stop now")["reason"], "Stop now")
        with self.assertRaises(ValueError):
            run_state.mark(self.run, "sleeping")

    def test_a_file_that_is_not_ours_is_not_a_state(self):
        write(run_state.state_path(self.run), "{not json")
        self.assertIsNone(run_state.read(self.run))
        write(run_state.state_path(self.run), '{"kind": "head"}')
        self.assertIsNone(run_state.read(self.run))

    def test_which_runs_can_go_on(self):
        for status, offered in (("paused", "paused"), ("failed", "failed"), ("running", "interrupted"), ("finished", None),
                                ("stopped", None)):
            self.a_run(status)
            got = run_state.describe(self.run)
            self.assertEqual(got and got["status"], offered, status)

    def test_nothing_to_go_on_without_the_checkpoint_or_after_the_last_epoch(self):
        self.a_run()
        os.remove(run_state.resume_path(self.run))
        self.assertIsNone(run_state.describe(self.run))
        self.a_run(epoch=10)
        self.assertIsNone(run_state.describe(self.run))
        self.a_run(epoch=0)
        self.assertIsNone(run_state.describe(self.run))

    def test_forget_leaves_the_pause_flag_to_the_caller(self):
        self.a_run()
        write(run_state.pause_file_for(self.run), "x")
        run_state.forget(self.run)
        self.assertEqual((run_state.read(self.run), os.path.exists(run_state.resume_path(self.run))), (None, False))
        self.assertTrue(os.path.exists(run_state.pause_file_for(self.run)))

    def test_the_table_of_the_epochs_so_far_comes_from_results_csv(self):
        write(os.path.join(self.tmp, "r", "results.csv"),
              "epoch,time,train/box_loss,train/cls_loss,train/dfl_loss,metrics/precision(B),metrics/recall(B),metrics/mAP50(B),"
              "metrics/mAP50-95(B)\n1,5,1.0,0.5,0.9,0.4,0.3,0.2,0.1\n2,9,0.8,0.4,0.8,0.5,0.4,0.5,0.3\n3,13,0.7,0.4,0.7,0.5,0.4,0.4,0.2\n",
              )
        rows = run_state.rows_from_csv(os.path.join(self.tmp, "r"))
        self.assertEqual([r["epoch"] for r in rows], [1, 2, 3])
        self.assertEqual([r["best"] for r in rows], [True, True, False])
        self.assertEqual({r["epochs"] for r in run_state.rows_from_csv(os.path.join(self.tmp, "r"), epochs=10)}, {10})
        self.assertAlmostEqual(rows[1]["map50"], 0.5)
        self.assertEqual([r["epoch"] for r in run_state.rows_from_csv(os.path.join(self.tmp, "r"), upto=2)], [1, 2])
        self.assertEqual(run_state.rows_from_csv(os.path.join(self.tmp, "none")), [])


class BundleTests(Base):
    def test_the_checkpoint_folder_holds_a_marker_and_only_the_files_a_resume_needs(self):
        self.a_run()
        folder = model_bundle.build(self.models, "pets", os.path.join(self.tmp, "out"))
        names = sorted(os.path.relpath(os.path.join(d, f), folder).replace(os.sep, "/") for d, _x, fs in os.walk(folder) for f in fs)
        self.assertEqual(names, ["jelibox-model.json", "yolo_run/resume_point.pt", "yolo_run/run_state.json",
                                 "yolo_run/train_run/results.csv", "yolo_run/train_run/weights/best.pt"])
        marker = model_bundle.read_marker(folder)
        self.assertEqual((marker["workspace"], marker["epoch"], marker["epochs"], marker["classes"], marker["put_in"]),
                         ("pets", 4, 10, ["cat", "dog"], "models/pets"))

    def test_nothing_is_built_before_the_first_epoch_was_saved(self):
        with self.assertRaises(ValueError) as caught:
            model_bundle.build(self.models, "pets", os.path.join(self.tmp, "out"))
        self.assertIn("first epoch", str(caught.exception))

    def test_a_folder_with_the_marker_is_found_anywhere_and_the_most_advanced_comes_first(self):
        for name, epoch in (("a/early", 2), ("deep/er/late", 7)):
            write(os.path.join(self.tmp, "up", name, "jelibox-model.json"),
                  '{"jelibox": "%s", "epoch": %d}' % (model_bundle.KIND, epoch))
        write(os.path.join(self.tmp, "up", "other", "jelibox-model.json"), '{"jelibox": "something-else"}')
        found = model_bundle.find(os.path.join(self.tmp, "up"))
        self.assertEqual([m["epoch"] for _f, m in found], [7, 2])

    def test_install_puts_the_files_into_the_workspace_models_folder_and_makes_the_run_continuable(self):
        self.a_run()
        folder = model_bundle.build(self.models, "pets", os.path.join(self.tmp, "out"))
        target = os.path.join(self.tmp, "elsewhere", "models", "newname")
        info = model_bundle.install(folder, target, "newname")
        self.assertEqual((info["epoch"], info["epochs"], info["files"], info["classes"]), (4, 10, 4, ["cat", "dog"]))
        run = os.path.join(target, "yolo_run")
        st = run_state.read(run)
        self.assertEqual((st["workspace"], st["split"]["val"], st["dataset"]), ("newname", ["a.jpg"], {"pictures": 10, "annotations": 25}))
        self.assertEqual(run_state.describe(run)["epoch"], 4)
        self.assertTrue(os.path.isfile(os.path.join(run, "train_run", "weights", "best.pt")))

    def test_install_replaces_the_run_that_was_there(self):
        self.a_run()
        folder = model_bundle.build(self.models, "pets", os.path.join(self.tmp, "out"))
        old = os.path.join(self.tmp, "t", "models", "pets", "yolo_run")
        run_state.write(old, status="paused", epoch=9, epochs=10)
        write(os.path.join(old, "train_run", "weights", "leftover.pt"))
        model_bundle.install(folder, os.path.join(self.tmp, "t", "models", "pets"), "pets")
        self.assertEqual(run_state.read(old)["epoch"], 4)
        self.assertFalse(os.path.exists(os.path.join(old, "train_run", "weights", "leftover.pt")))

    def test_only_the_four_known_names_are_ever_installed(self):
        folder = os.path.join(self.tmp, "up", "m")
        write(os.path.join(folder, "jelibox-model.json"), '{"jelibox": "%s", "epoch": 1, "epochs": 5}' % model_bundle.KIND)
        write(os.path.join(folder, "yolo_run", "resume_point.pt"))
        write(os.path.join(folder, "evil.py"), "print(1)")
        write(os.path.join(folder, "yolo_run", "job.json"), "{}")
        write(os.path.join(folder, "modelAssistant.pt"))
        target = os.path.join(self.tmp, "models", "ws")
        info = model_bundle.install(folder, target, "ws")
        self.assertEqual(info["files"], 1)
        written = sorted(os.path.relpath(os.path.join(d, f), target).replace(os.sep, "/") for d, _x, fs in os.walk(target) for f in fs)
        self.assertEqual(written, ["yolo_run/resume_point.pt", "yolo_run/run_state.json"])
        self.assertEqual(run_state.read(os.path.join(target, "yolo_run"))["epoch"], 1, "a folder without a state: the marker knows enough")

    def test_a_folder_without_a_checkpoint_is_refused(self):
        folder = os.path.join(self.tmp, "up", "m")
        write(os.path.join(folder, "jelibox-model.json"), '{"jelibox": "%s"}' % model_bundle.KIND)
        with self.assertRaises(ValueError):
            model_bundle.install(folder, os.path.join(self.tmp, "models", "ws"), "ws")


class DatasetHelpersTests(unittest.TestCase):
    PAIRS = [(f"/x/{n}.jpg", f"/y/{n}.txt") for n in ("a", "b", "c", "d", "e")]

    def test_the_fixed_split_keeps_the_named_pictures_as_validation_and_test(self):
        train, val, test, missing = head_data.split_fixed(self.PAIRS, ["b.jpg"], ["e.jpg"])
        self.assertEqual([os.path.basename(p[0]) for p in train], ["a.jpg", "c.jpg", "d.jpg"])
        self.assertEqual(([os.path.basename(p[0]) for p in val], [os.path.basename(p[0]) for p in test], missing),
                         (["b.jpg"], ["e.jpg"], 0))

    def test_named_pictures_that_are_gone_are_counted(self):
        _t, val, _x, missing = head_data.split_fixed(self.PAIRS, ["b.jpg", "gone.jpg"], [])
        self.assertEqual((len(val), missing), (1, 1))

    def test_two_datasets_are_the_same_when_pictures_and_annotations_match(self):
        same = {"pictures": 10, "annotations": 25}
        self.assertIsNone(head_data.same_dataset(same, dict(same)))
        self.assertIsNone(head_data.same_dataset(None, same), "no saved numbers: nothing to compare")
        for now in ({"pictures": 10, "annotations": 26}, {"pictures": 11, "annotations": 25}):
            text = head_data.same_dataset(same, now)
            self.assertIn("not the one this run was trained with", text)
            self.assertIn(f"now {now['annotations']} annotations on {now['pictures']}", text)


if __name__ == "__main__":
    unittest.main()
