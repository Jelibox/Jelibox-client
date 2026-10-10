"""The release machinery must stay consistent with itself."""
import os
import re
import subprocess
import sys
import unittest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
VERSION_FILE = '__version__ = "0.4.0"\n'


def read(*parts):
    with open(os.path.join(REPO, *parts), encoding="utf-8", errors="ignore") as f:
        return f.read()


class ReleaseProcessTests(unittest.TestCase):
    def test_version_is_semver(self):
        from utils import __version__
        self.assertRegex(__version__, r"^\d+\.\d+\.\d+$")

    def test_changelog_has_an_unreleased_section_and_the_current_version(self):
        from utils import __version__
        log = read("CHANGELOG.md")
        self.assertIn("## [Unreleased]", log)
        if __version__ != "0.0.0":                      # 0.0.0 = nothing released yet
            self.assertRegex(log, rf"## \[{re.escape(__version__)}\] - \d{{4}}-\d{{2}}-\d{{2}}")

    def test_changelog_versions_are_newest_first(self):
        versions = [tuple(map(int, v.split("."))) for v in re.findall(r"^## \[(\d+\.\d+\.\d+)\]", read("CHANGELOG.md"), re.M)]
        self.assertEqual(versions, sorted(versions, reverse=True))

    def test_release_workflow_is_wired_to_tags_and_the_helper(self):
        wf = read(".github", "workflows", "release.yml")
        self.assertIn("tags:", wf)
        self.assertIn("v[0-9]+.[0-9]+.[0-9]+", wf)
        self.assertIn("tools/release.py --check-tag", wf)
        self.assertIn("tools/release.py --notes", wf)
        self.assertIn("gh release create", wf)
        self.assertIn("contents: write", wf)

    def test_both_workflows_install_what_the_unit_tests_import(self):
        # several unit test modules import numpy / cv2 at the top; without them the whole unit group fails to load
        for name in ("ci.yml", "release.yml"):
            wf = read(".github", "workflows", name)
            for package in ("pillow", "numpy", "opencv-python-headless"):
                self.assertIn(package, wf, f"{name} must pip install {package}")

    def test_ci_workflow_runs_the_test_runner_on_prs(self):
        wf = read(".github", "workflows", "ci.yml")
        self.assertIn("pull_request", wf)
        self.assertIn("tests/run.py", wf)
        self.assertIn("windows-latest", wf)

    def test_docs_and_template_reference_the_real_commands(self):
        doc = read("RELEASING.md")
        for needle in ("tools/release.py", "tests/run.py", "CHANGELOG.md", "--dry-run", "Semantic Versioning"):
            self.assertIn(needle, doc)
        self.assertIn("tests/run.py", read(".github", "pull_request_template.md"))

    # The two tests below run the release tool against COPIES of the version file and
    # changelog, so they pass no matter what the real [Unreleased] section holds today.
    def _tool_on_copies(self, changelog_text):
        import contextlib
        import io
        import tempfile
        from tools import release as R
        tmp = tempfile.mkdtemp()
        init, log = os.path.join(tmp, "__init__.py"), os.path.join(tmp, "CHANGELOG.md")
        with open(init, "w", encoding="utf-8", newline="") as f:
            f.write(VERSION_FILE)
        with open(log, "w", encoding="utf-8", newline="") as f:
            f.write(changelog_text)
        old = R.INIT, R.CHANGELOG
        R.INIT, R.CHANGELOG = init, log
        out = io.StringIO()
        try:
            with contextlib.redirect_stdout(out):
                code = R.main(["minor", "--dry-run"])
        finally:
            R.INIT, R.CHANGELOG = old
        with open(init, encoding="utf-8", newline="") as f:
            init_after = f.read()
        with open(log, encoding="utf-8", newline="") as f:
            log_after = f.read()
        return code, out.getvalue(), init_after, log_after

    def test_dry_run_never_modifies_anything(self):
        log = "\n".join(["## [Unreleased]", "", "### Added", "- Something.", "",
                         "[Unreleased]: https://example.com/compare/v0.4.0...HEAD", ""])
        code, out, init_after, log_after = self._tool_on_copies(log)
        self.assertEqual(code, 0, out)
        self.assertIn("0.4.0 -> 0.5.0", out)
        self.assertIn("dry run", out)
        self.assertEqual(init_after, VERSION_FILE)
        self.assertEqual(log_after, log)

    def test_empty_unreleased_gives_a_clear_refusal_not_a_crash(self):
        log = "\n".join(["## [Unreleased]", "", "## [0.4.0] - 2026-01-01", "- x", ""])
        code, out, *_ = self._tool_on_copies(log)
        self.assertEqual(code, 1)
        self.assertIn("Cannot release", out)
        self.assertIn("[Unreleased] is empty", out)

    def test_failing_tests_undo_the_version_and_changelog_edits(self):
        import contextlib
        import io
        import tempfile
        from unittest import mock
        from tools import release as R
        tmp = tempfile.mkdtemp()
        init, log = os.path.join(tmp, "__init__.py"), os.path.join(tmp, "CHANGELOG.md")
        changelog = "\n".join(["## [Unreleased]", "", "- Something.", ""])
        for path, text in ((init, VERSION_FILE), (log, changelog)):
            with open(path, "w", encoding="utf-8", newline="") as f:
                f.write(text)
        fake_git = lambda *a, **k: {"rev-parse": "main"}.get(a[0], "")       # on main, clean tree, no tag yet
        failing_suite = mock.Mock(return_value=mock.Mock(returncode=1))
        old = R.INIT, R.CHANGELOG
        R.INIT, R.CHANGELOG = init, log
        try:
            with mock.patch.object(R, "git", fake_git), mock.patch.object(R.subprocess, "run", failing_suite), \
                    contextlib.redirect_stdout(io.StringIO()) as out:
                code = R.main(["minor"])
        finally:
            R.INIT, R.CHANGELOG = old
        self.assertEqual(code, 1)
        self.assertIn("undone", out.getvalue())
        failing_suite.assert_called_once()
        with open(init, encoding="utf-8", newline="") as f:
            self.assertEqual(f.read(), VERSION_FILE)
        with open(log, encoding="utf-8", newline="") as f:
            self.assertEqual(f.read(), changelog)

    def test_check_tag_rejects_a_mismatch(self):
        from utils import __version__
        ok = subprocess.run([sys.executable, "tools/release.py", "--check-tag", f"v{__version__}"], cwd=REPO, capture_output=True)
        bad = subprocess.run([sys.executable, "tools/release.py", "--check-tag", "v99.0.0"], cwd=REPO, capture_output=True)
        self.assertEqual(ok.returncode, 0)
        self.assertNotEqual(bad.returncode, 0)


if __name__ == "__main__":
    unittest.main()
