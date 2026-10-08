"""Moving the whole install ("Move Jelibox"): destination checks, requirement handling, move + rollback,
and a full headless run against a fake install (no network - the package list is stubbed out)."""
import os
import shutil
import tempfile
import unittest
from unittest import mock

from utils import relocate


def make_fake_install(root, with_data=True):
    for rel in ("utils/Annotator.py", "assets/jelibox.png", "README.md"):
        path = os.path.join(root, *rel.split("/"))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(rel)
    if with_data:
        for rel in ("datasetsInput/cats-1/a.jpg", "configs/cats.json", "models/cats/best.pt"):
            path = os.path.join(root, *rel.split("/"))
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                f.write("data:" + rel)


class ValidateTargetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="jelibox_rl_")
        self.src = os.path.join(self.tmp, "old", "Jelibox")
        os.makedirs(self.src)
        self.parent = os.path.join(self.tmp, "new")
        os.makedirs(self.parent)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_a_fresh_folder_is_fine(self):
        self.assertIsNone(relocate.validate_target(self.src, self.parent))
        self.assertEqual(relocate.target_for(self.parent), os.path.join(self.parent, "Jelibox"))

    def test_missing_folder(self):
        self.assertIn("not a folder", relocate.validate_target(self.src, os.path.join(self.tmp, "nope")))

    def test_same_location(self):
        self.assertIn("already", relocate.validate_target(self.src, os.path.dirname(self.src)))

    def test_inside_the_current_install(self):
        self.assertIn("inside", relocate.validate_target(self.src, self.src))

    def test_containing_the_current_install(self):
        nested_src = os.path.join(self.parent, "Jelibox", "deeper", "Jelibox")
        os.makedirs(nested_src)
        self.assertIn("contain", relocate.validate_target(nested_src, self.parent))

    def test_existing_non_empty_target_is_refused_but_empty_is_fine(self):
        target = os.path.join(self.parent, "Jelibox")
        os.makedirs(target)
        self.assertIsNone(relocate.validate_target(self.src, self.parent))
        open(os.path.join(target, "x"), "w").close()
        self.assertIn("not empty", relocate.validate_target(self.src, self.parent))


class RequirementTests(unittest.TestCase):
    def test_clean_requirements_drops_editable_and_local_installs(self):
        out = "numpy==1.26.4\n-e git+https://x/y#egg=y\nlocal @ file:///tmp/local\n\n# c\ntorch==2.5.1+cu121\n"
        self.assertEqual(relocate.clean_requirements(out), ["numpy==1.26.4", "torch==2.5.1+cu121"])

    def test_torch_index_follows_the_installed_build(self):
        self.assertEqual(relocate.torch_index_url(["numpy==1", "torch==2.5.1+cu121"]),
                         "https://download.pytorch.org/whl/cu121")
        self.assertEqual(relocate.torch_index_url(["torchvision==0.20.1+cpu"]),
                         "https://download.pytorch.org/whl/cpu")
        self.assertIsNone(relocate.torch_index_url(["torch==2.5.1", "numpy==1.26.4"]))


class MoveTreeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="jelibox_mv_")
        self.src = os.path.join(self.tmp, "src")
        self.dest = os.path.join(self.tmp, "dest")
        make_fake_install(self.src)
        os.makedirs(os.path.join(self.src, "venv"))
        os.makedirs(self.dest)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_moves_everything_except_the_skipped_venv(self):
        relocate.move_tree(self.src, self.dest, {"venv"}, lambda m: None)
        self.assertEqual(sorted(os.listdir(self.src)), ["venv"])
        with open(os.path.join(self.dest, "datasetsInput", "cats-1", "a.jpg")) as f:
            self.assertEqual(f.read(), "data:datasetsInput/cats-1/a.jpg")
        self.assertTrue(os.path.isfile(os.path.join(self.dest, "utils", "Annotator.py")))

    def test_failure_rolls_back_what_was_already_moved(self):
        real_move = shutil.move
        failing = {"on": True}

        def flaky(s, d):
            if failing["on"] and os.path.basename(s) == "models":
                failing["on"] = False           # only the forward move fails, not the rollback
                raise OSError("disk exploded")
            return real_move(s, d)

        with mock.patch("utils.relocate.shutil.move", side_effect=flaky):
            with self.assertRaises(relocate.RelocateError):
                relocate.move_tree(self.src, self.dest, {"venv"}, lambda m: None)
        self.assertEqual(os.listdir(self.dest), [])
        self.assertTrue(os.path.isfile(os.path.join(self.src, "configs", "cats.json")))
        self.assertTrue(os.path.isfile(os.path.join(self.src, "models", "cats", "best.pt")))
        self.assertTrue(os.path.isfile(os.path.join(self.src, "utils", "Annotator.py")))


class FullRunTests(unittest.TestCase):
    """Creates a real (empty) venv, so it needs no network but takes a few seconds."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="jelibox_full_")
        self.src = os.path.join(self.tmp, "old", "Jelibox")
        self.parent = os.path.join(self.tmp, "new")
        make_fake_install(self.src)
        os.makedirs(self.parent)
        self.dest = relocate.target_for(self.parent)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_everything_moves_and_the_old_folder_is_gone(self):
        steps = []
        with mock.patch("utils.relocate.freeze", return_value=[]), \
                mock.patch("utils.relocate.create_shortcuts"), \
                mock.patch("utils.relocate.wait_for_exit"):
            new_venv, leftovers = relocate.relocate(self.src, self.dest, None, None, 0, lambda m: None, steps.append)
        self.assertEqual(leftovers, [])
        self.assertEqual(steps, list(range(len(relocate.STEPS))))
        self.assertFalse(os.path.exists(self.src))
        self.assertTrue(os.path.isfile(relocate.venv_python(new_venv)))
        self.assertEqual(os.path.dirname(new_venv), self.dest)
        for rel in ("utils/Annotator.py", "datasetsInput/cats-1/a.jpg", "configs/cats.json", "models/cats/best.pt"):
            self.assertTrue(os.path.isfile(os.path.join(self.dest, *rel.split("/"))), rel)

    def test_a_failure_before_moving_leaves_the_old_install_untouched(self):
        with mock.patch("utils.relocate.freeze", side_effect=relocate.RelocateError("pip is broken")), \
                mock.patch("utils.relocate.wait_for_exit"):
            with self.assertRaises(relocate.RelocateError):
                relocate.relocate(self.src, self.dest, None, None, 0, lambda m: None, lambda i: None)
        self.assertTrue(os.path.isfile(os.path.join(self.src, "datasetsInput", "cats-1", "a.jpg")))
        self.assertFalse(os.path.exists(self.dest))      # the folder it created is cleaned up again

    def test_refuses_a_folder_that_is_not_a_jelibox_install(self):
        stray = os.path.join(self.tmp, "stray")
        os.makedirs(stray)
        with self.assertRaises(relocate.RelocateError):
            relocate.relocate(stray, self.dest, None, None, 0, lambda m: None, lambda i: None)
        self.assertTrue(os.path.isdir(stray))


if __name__ == "__main__":
    unittest.main()
