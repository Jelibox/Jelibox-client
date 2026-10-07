"""Move Jelibox when the install uses its own private Python (<install>/.python, see tools/setup_python.*).

Such a venv points at its Python by absolute path, so the move has to bring a copy of that Python along and
build the new venv from it - otherwise the new venv would point into the folder that is about to be deleted.
Everything here is faked (no network, no real Python download)."""
import os
import shutil
import tempfile
import unittest
from unittest import mock

from utils import relocate
from tests.unit.test_relocate import make_fake_install

IS_WIN = os.name == "nt"
PY_NAME = "cpython-3.12.10-test"


def make_private_python(root):
    """<root>/.python/<name> with a stand-in interpreter, plus a venv whose pyvenv.cfg points at it."""
    home = os.path.join(root, ".python", PY_NAME)
    exe = os.path.join(home, "python.exe") if IS_WIN else os.path.join(home, "bin", "python3")
    os.makedirs(os.path.dirname(exe))
    with open(exe, "w") as f:
        f.write("fake interpreter")
    with open(os.path.join(home, "marker.txt"), "w") as f:
        f.write("copied")
    venv = os.path.join(root, "venv" if IS_WIN else "jelibox")
    os.makedirs(venv)
    cfg_home = home if IS_WIN else os.path.dirname(exe)             # Linux venvs record the bin folder
    with open(os.path.join(venv, "pyvenv.cfg"), "w") as f:
        f.write(f"home = {cfg_home}\nimplementation = CPython\nversion_info = 3.12.10\n")
    return home, venv


class PrivatePythonDetectionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="jelibox_pp_")
        self.root = os.path.join(self.tmp, "Jelibox")
        os.makedirs(self.root)
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def test_finds_the_private_python_a_venv_was_built_from(self):
        home, venv = make_private_python(self.root)
        self.assertEqual(os.path.normcase(relocate.private_python_dir(venv, self.root)), os.path.normcase(home))

    def test_a_venv_built_from_a_system_python_is_not_private(self):
        venv = os.path.join(self.root, "venv")
        os.makedirs(venv)
        with open(os.path.join(venv, "pyvenv.cfg"), "w") as f:
            f.write(f"home = {os.path.dirname(os.sys.executable)}\n")
        self.assertIsNone(relocate.private_python_dir(venv, self.root))

    def test_missing_or_broken_config_is_not_an_error(self):
        self.assertIsNone(relocate.private_python_dir(None, self.root))
        self.assertIsNone(relocate.private_python_dir(os.path.join(self.root, "nope"), self.root))
        venv = os.path.join(self.root, "venv")
        os.makedirs(venv)
        with open(os.path.join(venv, "pyvenv.cfg"), "w") as f:
            f.write("garbage without a home line\n")
        self.assertIsNone(relocate.private_python_dir(venv, self.root))

    def test_python_that_lives_in_a_different_install_is_not_ours(self):
        other = os.path.join(self.tmp, "Other")
        home, venv = make_private_python(other)
        self.assertIsNone(relocate.private_python_dir(venv, self.root))

    def test_python_in_picks_the_right_executable(self):
        home, _ = make_private_python(self.root)
        exe = relocate.python_in(home)
        self.assertTrue(os.path.isfile(exe), exe)
        self.assertEqual(os.path.dirname(exe), home if IS_WIN else os.path.join(home, "bin"))


class MoveWithPrivatePythonTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="jelibox_ppm_")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.src = os.path.join(self.tmp, "old", "Jelibox")
        make_fake_install(self.src)
        self.home, self.venv = make_private_python(self.src)
        self.parent = os.path.join(self.tmp, "new")
        os.makedirs(self.parent)
        self.dest = relocate.target_for(self.parent)
        self.venv_calls = []

    def fake_run_command(self, cmd, log, cwd=None):
        """Records the interpreter used to build the venv and creates the folder like `python -m venv` would."""
        self.venv_calls.append(list(cmd))
        if cmd[1:3] == ["-m", "venv"]:
            os.makedirs(os.path.join(cmd[3], "Scripts" if IS_WIN else "bin"))
            with open(relocate.venv_python(cmd[3]), "w") as f:
                f.write("new venv python")

    def run_move(self, run_command=None):
        with mock.patch("utils.relocate.freeze", return_value=[]), \
                mock.patch("utils.relocate.create_shortcuts"), \
                mock.patch("utils.relocate.wait_for_exit"), \
                mock.patch("utils.relocate.run_command", side_effect=run_command or self.fake_run_command):
            return relocate.relocate(self.src, self.dest, self.venv, None, 0, lambda m: None, lambda i: None)

    def test_new_venv_is_built_from_a_copy_of_python_inside_the_new_install(self):
        new_venv, leftovers = self.run_move()
        self.assertEqual(leftovers, [])
        new_home = os.path.join(self.dest, ".python", PY_NAME)
        self.assertTrue(os.path.isfile(os.path.join(new_home, "marker.txt")), "the Python was copied along")
        builder = self.venv_calls[0][0]
        self.assertEqual(os.path.normcase(builder), os.path.normcase(relocate.python_in(new_home)),
                         "venv must be built from the NEW copy, never from the folder about to be deleted")
        self.assertFalse(os.path.exists(self.src), "old install, including its old .python, is gone")
        self.assertTrue(os.path.isfile(os.path.join(self.dest, "datasetsInput", "cats-1", "a.jpg")))

    def test_the_old_python_is_not_moved_over_the_new_copy(self):
        self.run_move()
        entries = os.listdir(os.path.join(self.dest, ".python"))
        self.assertEqual(entries, [PY_NAME])

    def test_a_failure_removes_the_python_copy_again_and_leaves_the_old_install_alone(self):
        def boom(cmd, log, cwd=None):
            raise relocate.RelocateError("venv creation failed")
        with self.assertRaises(relocate.RelocateError):
            self.run_move(run_command=boom)
        self.assertTrue(os.path.isfile(os.path.join(self.home, "marker.txt")))
        self.assertTrue(os.path.isfile(os.path.join(self.src, "configs", "cats.json")))
        self.assertFalse(os.path.exists(self.dest), "nothing is left behind in the target")

    def test_an_install_without_a_private_python_is_moved_the_old_way(self):
        shutil.rmtree(os.path.join(self.src, ".python"))
        with open(os.path.join(self.venv, "pyvenv.cfg"), "w") as f:
            f.write(f"home = {os.path.dirname(os.sys.executable)}\n")
        self.run_move()
        self.assertFalse(os.path.exists(os.path.join(self.dest, ".python")))
        self.assertEqual(self.venv_calls[0][0], relocate.base_python())


if __name__ == "__main__":
    unittest.main()
