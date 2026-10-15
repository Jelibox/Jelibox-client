"""Reading ten thousand label files can take a minute on a slow disk, so head_data.labelled_pairs remembers what it read."""
import os
import tempfile
import unittest
from unittest import mock

from utils import head_data


class LabelCacheTests(unittest.TestCase):
    def setUp(self):
        root = tempfile.mkdtemp()
        self.images = os.path.join(root, "images")
        self.labels = os.path.join(root, "labels")
        os.makedirs(self.images)
        os.makedirs(self.labels)
        for i in range(6):
            open(os.path.join(self.images, f"p{i}.jpg"), "wb").close()
            with open(os.path.join(self.labels, f"p{i}.txt"), "w") as f:
                f.write("0 0.5 0.5 0.2 0.2" if i != 3 else "")        # p3 has no label row
        self.cache = os.path.join(root, head_data.CACHE_NAME)

    def reads(self):
        """How many label files a call opens."""
        with mock.patch.object(head_data, "read_label_rows", wraps=head_data.read_label_rows) as spy:
            pairs = head_data.labelled_pairs([self.images], self.labels)
        return len(pairs), spy.call_count

    def test_the_first_call_reads_every_file_and_the_next_ones_none(self):
        self.assertEqual(self.reads(), (5, 6))
        self.assertTrue(os.path.isfile(self.cache))
        self.assertEqual(self.reads(), (5, 0), "only the files' dates are looked at")

    def test_a_changed_label_is_read_again_and_the_answer_follows(self):
        self.reads()
        with open(os.path.join(self.labels, "p3.txt"), "w") as f:
            f.write("1 0.4 0.4 0.3 0.3 ")
        self.assertEqual(self.reads(), (6, 1))
        os.remove(os.path.join(self.labels, "p0.txt"))
        self.assertEqual(self.reads(), (5, 0))

    def test_a_label_emptied_later_leaves_the_pairs(self):
        self.reads()
        with open(os.path.join(self.labels, "p1.txt"), "w") as f:
            f.write("")
        self.assertEqual(self.reads(), (4, 1))

    def test_a_damaged_or_missing_cache_file_is_just_rebuilt(self):
        self.reads()
        with open(self.cache, "w") as f:
            f.write("{ not json")
        self.assertEqual(self.reads(), (5, 6))
        os.remove(self.cache)
        self.assertEqual(self.reads(), (5, 6))

    def test_the_boxes_per_class_come_from_the_cache_too(self):
        with open(os.path.join(self.labels, "p1.txt"), "w") as f:
            f.write("1 0.5 0.5 0.2 0.2" + chr(10) + "1 0.3 0.3 0.1 0.1" + chr(10) + "0 0.6 0.6 0.2 0.2")
        pairs = head_data.labelled_pairs([self.images], self.labels)
        expected = head_data.class_counts(pairs, 3)
        self.assertEqual(expected, [5, 2, 0])
        head_data._box_counts.clear()                                    # a new process: only the cache file is left
        pairs = head_data.labelled_pairs([self.images], self.labels)
        with mock.patch.object(head_data, "read_label_rows", side_effect=AssertionError("a file was opened")):
            self.assertEqual(head_data.class_counts(pairs, 3), expected)
            self.assertEqual(head_data.class_counts(pairs, 3), expected)

    def test_a_label_that_was_not_scanned_is_still_counted(self):
        head_data._box_counts.clear()
        path = os.path.join(self.labels, "p2.txt")
        self.assertEqual(head_data.class_counts([("x", path)], 2), [1, 0])

    def test_the_pairs_are_the_same_as_without_a_cache(self):
        first = head_data.labelled_pairs([self.images], self.labels)
        again = head_data.labelled_pairs([self.images], self.labels)
        self.assertEqual(first, again)
        self.assertEqual([os.path.basename(i) for i, _ in first], ["p0.jpg", "p1.jpg", "p2.jpg", "p4.jpg", "p5.jpg"])


if __name__ == "__main__":
    unittest.main()
