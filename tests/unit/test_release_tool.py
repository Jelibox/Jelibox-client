import unittest

from tools import release as R

BASE = """# Changelog

## [Unreleased]

### Added
- A shiny thing.

### Fixed
- A crash.

[Unreleased]: https://github.com/o/r/commits/main
"""


class VersionTests(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(R.parse_version("1.2.3"), (1, 2, 3))
        self.assertEqual(R.parse_version("v0.10.0"), (0, 10, 0))
        for bad in ("1.2", "1.2.3.4", "one", "1.2.x", ""):
            with self.assertRaises(ValueError, msg=bad):
                R.parse_version(bad)

    def test_bumps(self):
        self.assertEqual(R.next_version("0.3.9", "patch"), "0.3.10")
        self.assertEqual(R.next_version("0.3.9", "minor"), "0.4.0")
        self.assertEqual(R.next_version("0.3.9", "major"), "1.0.0")
        self.assertEqual(R.next_version("0.3.9", "0.5.0"), "0.5.0")

    def test_explicit_version_must_go_up(self):
        for bad in ("0.3.9", "0.3.8", "0.2.0"):
            with self.assertRaises(ValueError, msg=bad):
                R.next_version("0.3.9", bad)

    def test_read_and_write_version(self):
        src = '"""doc"""\n\n__version__ = "0.0.0"   # note\n'
        self.assertEqual(R.read_version(src), "0.0.0")
        out = R.write_version(src, "1.4.2")
        self.assertEqual(R.read_version(out), "1.4.2")
        self.assertIn("# note", out)
        with self.assertRaises(ValueError):
            R.read_version("no version here")


class ChangelogTests(unittest.TestCase):
    URL = "https://github.com/o/r"

    def test_unreleased_body(self):
        self.assertIn("A shiny thing.", R.unreleased_body(BASE))
        self.assertEqual(R.unreleased_body("## [Unreleased]\n\n## [0.1.0] - 2026-01-01\n- x\n"), "")

    def test_first_release_moves_entries_and_empties_unreleased(self):
        out = R.release_changelog(BASE, "0.1.0", "2026-10-04", self.URL)
        self.assertIn("## [Unreleased]\n\n## [0.1.0] - 2026-10-04\n\n### Added\n- A shiny thing.", out)
        self.assertEqual(R.unreleased_body(out), "")
        self.assertIn("[Unreleased]: https://github.com/o/r/compare/v0.1.0...HEAD", out)
        self.assertIn("[0.1.0]: https://github.com/o/r/releases/tag/v0.1.0", out)

    def test_second_release_links_compare_range(self):
        first = R.release_changelog(BASE, "0.1.0", "2026-10-04", self.URL)
        again = first.replace("## [Unreleased]\n", "## [Unreleased]\n\n### Fixed\n- Another crash.\n", 1)
        out = R.release_changelog(again, "0.1.1", "2026-11-01", self.URL)
        self.assertIn("[0.1.1]: https://github.com/o/r/compare/v0.1.0...v0.1.1", out)
        self.assertIn("[Unreleased]: https://github.com/o/r/compare/v0.1.1...HEAD", out)
        self.assertIn("## [0.1.0] - 2026-10-04", out, "older sections are untouched")
        self.assertEqual(R.extract_notes(out, "0.1.1").strip(), "### Fixed\n- Another crash.")

    def test_empty_unreleased_refuses(self):
        with self.assertRaises(ValueError):
            R.release_changelog("## [Unreleased]\n\n## [0.1.0] - 2026-01-01\n- x\n", "0.1.1", "2026-02-01", self.URL)

    def test_extract_notes(self):
        out = R.release_changelog(BASE, "0.1.0", "2026-10-04", self.URL)
        notes = R.extract_notes(out, "0.1.0")
        self.assertTrue(notes.startswith("### Added"))
        self.assertNotIn("[Unreleased]:", notes)
        with self.assertRaises(ValueError):
            R.extract_notes(out, "9.9.9")

    def test_crlf_files_stay_crlf(self):
        crlf = BASE.replace("\n", "\r\n")
        out = R.release_changelog(crlf, "0.1.0", "2026-10-04", self.URL)
        self.assertIn("\r\n", out)
        self.assertNotIn("\n", out.replace("\r\n", ""))


if __name__ == "__main__":
    unittest.main()
