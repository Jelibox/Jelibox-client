import unittest

from tests import run


class RunnerHelpersTests(unittest.TestCase):
    def test_app_chatter_is_filtered_out_of_failure_reports(self):
        text = "[GUI] Loaded 2 classes\n      Original: 1280x720px\nFAIL: test_x\nAssertionError: boom\n"
        cleaned = run.clean_output(text)
        self.assertNotIn("[GUI]", cleaned)
        self.assertNotIn("Original:", cleaned)
        self.assertIn("AssertionError: boom", cleaned)

    def test_github_annotation_is_a_single_line_with_escaped_newlines(self):
        out = run.github_annotation("repo", "FAIL: a\n100% broken\r\nline3\n")
        self.assertTrue(out.startswith("::error title=Jelibox tests: repo failed::"))
        self.assertNotIn("\n", out)
        self.assertNotIn("\r", out)
        self.assertIn("%0A", out)
        self.assertIn("100%25 broken", out)

    def test_github_annotation_keeps_the_end_of_long_output(self):
        out = run.github_annotation("gui", "x" * 10000 + "\nTHE-REAL-ERROR", limit=200)
        self.assertIn("THE-REAL-ERROR", out)
        self.assertLess(len(out), 400)


if __name__ == "__main__":
    unittest.main()
