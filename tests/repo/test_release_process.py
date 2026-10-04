"""The release machinery must stay consistent with itself."""
import hashlib
import os
import re
import subprocess
import sys
import unittest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


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

    def test_dry_run_never_modifies_anything(self):
        def snapshot():
            h = hashlib.sha1()
            for rel in ("utils/__init__.py", "CHANGELOG.md"):
                h.update(open(os.path.join(REPO, rel), "rb").read())
            return h.hexdigest()
        before = snapshot()
        r = subprocess.run([sys.executable, "tools/release.py", "patch", "--dry-run"], cwd=REPO,
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("dry run", r.stdout)
        self.assertEqual(snapshot(), before)

    def test_check_tag_rejects_a_mismatch(self):
        from utils import __version__
        ok = subprocess.run([sys.executable, "tools/release.py", "--check-tag", f"v{__version__}"], cwd=REPO, capture_output=True)
        bad = subprocess.run([sys.executable, "tools/release.py", "--check-tag", "v99.0.0"], cwd=REPO, capture_output=True)
        self.assertEqual(ok.returncode, 0)
        self.assertNotEqual(bad.returncode, 0)


if __name__ == "__main__":
    unittest.main()
