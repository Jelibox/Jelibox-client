"""The three modes are wired into the installers and documented; the pins that keep them compatible are in place."""
import os
import re
import unittest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def read(name):
    with open(os.path.join(REPO, name), encoding="utf-8", errors="ignore") as f:
        return f.read()


class InstallerPackageTests(unittest.TestCase):
    def test_both_installers_install_from_requirements_txt(self):
        self.assertIn('-m pip install -r "%~dp0requirements.txt"', read("jelibox_windows_installation.bat"))
        self.assertIn('-m pip install -r "$APP_PATH/requirements.txt"', read("jelibox_linux_installation.bash"))

    def test_nothing_is_installed_twice_with_a_conflicting_inline_list(self):
        for name in ("jelibox_windows_installation.bat", "jelibox_linux_installation.bash"):
            for line in read(name).splitlines():
                if " -m pip install" not in line or line.lstrip().startswith(("echo", "::", "#")):
                    continue
                line = re.sub(r"https?://\S+", "", line)           # the CLIP archive URL is not a package name
                for pkg in ("ultralytics", "streamlit", "yt-dlp", "pyinstaller", "transformers", "huggingface"):
                    self.assertNotIn(pkg, line, f"{name}: package lists live in requirements.txt")

    def test_torch_is_still_chosen_by_the_installer(self):
        for name in ("jelibox_windows_installation.bat", "jelibox_linux_installation.bash"):
            text = read(name)
            self.assertIn("download.pytorch.org/whl/cpu", text)
            self.assertIn("download.pytorch.org/whl/cu121", text)


class DocumentationTests(unittest.TestCase):
    def test_readme_describes_the_modes_and_their_requirements(self):
        readme = read("README.md")
        for needle in ("## Modes", "YOLO-World", "LocateAnything", "Custom head", "Shift+G", "transformers==4.57.6",
                       "ultralytics>=8.4.68", "huggingface_hub>=0.34.0,<1.0", "models/_huggingface"):
            self.assertIn(needle, readme, needle)

    def test_changelog_lists_the_modes(self):
        unreleased = read("CHANGELOG.md").split("## [0.2.0]")[0]
        for needle in ("LocateAnything", "Custom head", "Auto-annotate all"):
            self.assertIn(needle, unreleased, needle)

    def test_locateanything_code_is_pinned_and_trusted_code_is_explained(self):
        text = read(os.path.join("utils", "locate_anything.py"))
        self.assertRegex(text, r'MODEL_REVISION = "[0-9a-f]{40}"')
        self.assertIn("trust_remote_code", text)
        self.assertIn("revision=MODEL_REVISION", text)

    def test_the_big_model_is_never_loaded_at_import_or_start(self):
        for name in ("locate_anything.py", "assistant_providers.py", "inferenceObjectDetection.py", "ModeDialogs.py",
                     "AutoAnnotateDialog.py", "WorkspacePicker.py", "AnnotationGUI.py", "HeadTrainDialog.py", "head_data.py",
                     "dialog_scroll.py"):
            text = read(os.path.join("utils", name))
            self.assertNotRegex(text, r"^(import|from) (transformers|torch)\b.*$" if name in (
                "locate_anything.py", "assistant_providers.py", "WorkspacePicker.py", "ModeDialogs.py",
                "AutoAnnotateDialog.py", "HeadTrainDialog.py", "head_data.py", "dialog_scroll.py") else r"^\0$", name)
            self.assertNotRegex(text, r"^engine\.load\(", name)


if __name__ == "__main__":
    unittest.main()
