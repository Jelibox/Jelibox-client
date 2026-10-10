"""Export Dataset: where the dataset is saved (default folder, or a new folder inside a folder the user chose)."""
import os
import tempfile
import unittest
from unittest import mock

from utils import export


class PlanTests(unittest.TestCase):
    def folder_with(self, *names):
        root = tempfile.mkdtemp()
        for n in names:
            os.makedirs(os.path.join(root, n))
        return root

    def test_no_folder_chosen_uses_the_default_and_replaces_it(self):
        self.assertEqual(export.plan_export_target("D", None, "ws", "YOLO"), ("D", True))
        self.assertEqual(export.plan_export_target("D", "", "ws", "YOLO"), ("D", True))

    def test_an_empty_or_missing_folder_starts_at_v1_and_nothing_is_replaced(self):
        root = tempfile.mkdtemp()
        self.assertEqual(export.plan_export_target("D", root, "ws", "YOLO"), (os.path.join(root, "ws-v1"), False))
        missing = os.path.join(root, "not", "there")
        self.assertEqual(export.plan_export_target("D", missing, "ws")[0], os.path.join(missing, "ws-v1"))

    def test_it_counts_on_from_the_highest_version_already_there(self):
        root = self.folder_with("ws-v1", "ws-v2", "ws-v4")             # v3 was deleted: still v5 next
        self.assertEqual(os.path.basename(export.plan_export_target("D", root, "ws")[0]), "ws-v5")

    def test_other_workspaces_and_odd_names_do_not_count(self):
        root = self.folder_with("other-v9", "ws2-v7", "ws-v", "ws-vx", "ws-v3-old", "xws-v8", "ws-v2")
        open(os.path.join(root, "ws-v6"), "w").close()                 # a file, not a folder
        self.assertEqual(export.next_version_name(root, "ws"), "ws-v3")

    def test_workspace_names_with_dashes_and_regex_characters(self):
        root = self.folder_with("my-ws-v2", "my.ws-v5")
        self.assertEqual(export.next_version_name(root, "my-ws"), "my-ws-v3")
        self.assertEqual(export.next_version_name(root, "my.ws"), "my.ws-v6")
        self.assertEqual(export.next_version_name(root, "myxws"), "myxws-v1")


class DestinationTests(unittest.TestCase):
    def test_a_usable_folder_is_created_and_left_clean(self):
        root = tempfile.mkdtemp()
        folder = os.path.join(root, "new", "archive")
        self.assertIsNone(export.check_destination(folder, 10))
        self.assertEqual(os.listdir(folder), [], "the write test leaves nothing behind")

    def test_a_folder_that_cannot_be_created_is_explained(self):
        root = tempfile.mkdtemp()
        blocker = os.path.join(root, "afile")
        open(blocker, "w").close()
        problem = export.check_destination(os.path.join(blocker, "inside"))
        self.assertIn("Cannot save to this folder", problem)

    def test_not_enough_free_space_is_explained_with_sizes(self):
        usage = mock.Mock(free=2 * 1024 ** 3)
        with mock.patch("utils.export.shutil.disk_usage", return_value=usage):
            problem = export.check_destination(tempfile.mkdtemp(), 5 * 1024 ** 3)
        self.assertIn("5.0 GB", problem)
        self.assertIn("2.0 GB", problem)
        with mock.patch("utils.export.shutil.disk_usage", return_value=usage):
            self.assertIsNone(export.check_destination(tempfile.mkdtemp(), 1 * 1024 ** 3))

    def test_files_size_adds_up_and_ignores_missing_files(self):
        root = tempfile.mkdtemp()
        a, b = os.path.join(root, "a"), os.path.join(root, "b")
        with open(a, "wb") as f:
            f.write(b"12345")
        with open(b, "wb") as f:
            f.write(b"123")
        self.assertEqual(export.files_size([a, b, os.path.join(root, "missing")]), 8)


if __name__ == "__main__":
    unittest.main()
