"""Repository-level checks: things that silently rot between releases."""
import os
import re
import shutil
import struct
import subprocess
import tempfile
import unittest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
NEW_REPO_URL = "github.com/Jelibox/Jelibox-client"

SKIP_DIRS = {"venv", ".git", "__pycache__", "datasetsInput", "datasetsOutput", "vocdataset", "YOLOdataset",
             "models", "configs", "export dataset", "export model", "jelibox", "boxify", "VC_redist"}
BINARY = (".png", ".ico", ".exe", ".lnk", ".pt", ".jpg", ".jpeg", ".pyc")


def source_files(exts=None):
    for dirpath, dirs, files in os.walk(REPO):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for f in files:
            if f.endswith(BINARY):
                continue
            if exts is None or f.endswith(exts):
                yield os.path.join(dirpath, f)


def read(path):
    with open(path, encoding="utf-8", errors="ignore") as f:
        return f.read()


class CodeHealthTests(unittest.TestCase):
    def test_every_python_file_compiles(self):
        bad = []
        for p in source_files((".py",)):
            try:
                compile(read(p), p, "exec")
            except SyntaxError as e:
                bad.append(f"{os.path.relpath(p, REPO)}:{e.lineno}: {e.msg}")
        self.assertEqual(bad, [])

    def test_no_machine_specific_paths_or_personal_names(self):
        pat = re.compile(r"([A-Za-z]:[\\/]Users[\\/][\w .-]*|/home/\w+|/Users/\w+|ai-server|milha)")
        offenders = []
        for p in source_files():
            rel = os.path.relpath(p, REPO)
            if rel.startswith("tests") or rel in ("LICENSE", "Jelibox.desktop"):
                continue                       # Jelibox.desktop is generated + git-ignored
            for i, line in enumerate(read(p).splitlines(), 1):
                m = pat.search(line)
                if m and "googleapis" not in line:
                    offenders.append(f"{rel}:{i}: {m.group(0)}")
        self.assertEqual(offenders, [])

    def test_no_neon_leftovers_in_the_theme(self):
        theme = read(os.path.join(REPO, "utils", "theme.py")).lower()
        for neon in ("#00d4ff", "#00e676", "#ff1744", "#0a0e1a"):
            self.assertFalse(neon in theme, f"old neon color {neon} is back in theme.py")

    def test_gui_modules_do_not_hardcode_colors_outside_the_theme(self):
        offenders = []
        pat = re.compile(r"(bg|fg|background|foreground)\s*=\s*['\"]#[0-9a-fA-F]{6}['\"]")
        for name in ("AnnotationGUI.py", "WorkspacePicker.py", "LabelAssistantDialog.py", "ScreenGuard.py"):
            for i, line in enumerate(read(os.path.join(REPO, "utils", name)).splitlines(), 1):
                if pat.search(line):
                    offenders.append(f"{name}:{i}: {line.strip()[:90]}")
        self.assertEqual(offenders, [], "use the C_* roles from utils/theme.py instead of literal colors")


class BrandingTests(unittest.TestCase):
    USER_FACING = ["README.md", "index.html", "jelibox_windows_installation.bat",
                   "jelibox_linux_installation.bash", "LICENSE"]

    def test_old_name_is_gone_from_user_facing_files(self):
        files = [os.path.join(REPO, f) for f in self.USER_FACING] + list(source_files((".py",)))
        offenders = []
        for p in files:
            rel = os.path.relpath(p, REPO)
            if rel.startswith("tests"):
                continue
            for i, line in enumerate(read(p).splitlines(), 1):
                if re.search(r"boxify", line, re.I):
                    offenders.append(f"{rel}:{i}: {line.strip()[:80]}")
        self.assertEqual(offenders, [])

    def test_renamed_files_exist_and_old_ones_do_not(self):
        for name in ("jelibox_windows_installation.bat", "jelibox_linux_installation.bash",
                     "assets/jelibox.png", "assets/jelibox.ico", "assets/jelibox.svg"):
            self.assertTrue(os.path.exists(os.path.join(REPO, name)), name)
        for name in ("boxify_windows_installation.bat", "boxify_linux_installation.bash",
                     "assets/boxify.png", "assets/boxify.ico", "Boxify Launcher.lnk"):
            self.assertFalse(os.path.exists(os.path.join(REPO, name)), name)

    def test_license_is_apache_2(self):
        text = read(os.path.join(REPO, "LICENSE"))
        self.assertTrue("Apache License" in text and "Version 2.0" in text)
        self.assertTrue("Apache" in read(os.path.join(REPO, "README.md")), "README must state the license")
        self.assertFalse("MIT licensed" in read(os.path.join(REPO, "index.html")))

    def test_repo_links_point_at_the_new_repository(self):
        for name in ("README.md", "index.html"):
            text = read(os.path.join(REPO, name))
            self.assertTrue(NEW_REPO_URL in text, f"{name} must link to the new repo")
            self.assertFalse("BoxifyAnnotationTools" in text, f"{name} still links to the old repo")


class AssetTests(unittest.TestCase):
    def test_png_is_512_square(self):
        from PIL import Image
        with Image.open(os.path.join(REPO, "assets", "jelibox.png")) as im:
            self.assertEqual(im.size, (512, 512))
            self.assertEqual(im.mode, "RGBA")

    def test_ico_has_the_sizes_windows_needs(self):
        with open(os.path.join(REPO, "assets", "jelibox.ico"), "rb") as f:
            reserved, kind, count = struct.unpack("<HHH", f.read(6))
            sizes = []
            for _ in range(count):
                w, _h = struct.unpack("<BB", f.read(2))
                f.read(14)
                sizes.append(w or 256)
        self.assertEqual((reserved, kind), (0, 1))
        for needed in (16, 20, 24, 32, 48, 256):
            self.assertIn(needed, sizes)

    def test_site_references_only_existing_local_assets(self):
        html = read(os.path.join(REPO, "index.html"))
        refs = set(re.findall(r'(?:src|href)="(assets/[^"]+)"', html))
        self.assertTrue(refs)
        for r in refs:
            self.assertTrue(os.path.exists(os.path.join(REPO, r)), r)

    def test_site_makes_no_overblown_claims_and_is_responsive(self):
        html = read(os.path.join(REPO, "index.html")).lower()
        for phrase in ("local-first", "<strong>100%", "for yolo</span>"):
            self.assertFalse(phrase in html, f"index.html still contains the claim {phrase!r}")
        self.assertTrue('name="viewport"' in html, "site must be mobile friendly")
        self.assertTrue("prefers-color-scheme:dark" in html.replace(" ", ""), "site must support dark mode")


class InstallerTests(unittest.TestCase):
    def test_linux_installer_syntax_and_content(self):
        text = read(os.path.join(REPO, "jelibox_linux_installation.bash"))
        for needle in ("jelibox", "ultralytics", "ftfy", "CLIP/archive", "Jelibox.desktop"):
            self.assertTrue(needle in text, f"installer is missing {needle!r}")
        bash = shutil.which("bash")
        if not bash:
            self.skipTest("bash not available")
        # check an LF-normalised copy: a Windows checkout may have converted the line endings
        with tempfile.NamedTemporaryFile("w", suffix=".bash", delete=False, newline="\n", encoding="utf-8") as tmp:
            tmp.write(text.replace("\r\n", "\n"))
        try:
            r = subprocess.run([bash, "-n", tmp.name], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
        finally:
            os.remove(tmp.name)

    def test_line_endings_are_pinned_per_file_type(self):
        attrs = read(os.path.join(REPO, ".gitattributes"))
        self.assertRegex(attrs, r"\*\.bat\s+text eol=crlf")
        self.assertRegex(attrs, r"\*\.bash\s+text eol=lf")

    def test_windows_installer_content(self):
        text = read(os.path.join(REPO, "jelibox_windows_installation.bat"))
        for needle in ("jelibox.ico", "Jelibox Launcher.lnk", "CLIP/archive", "CLIP_STATUS"):
            self.assertTrue(needle in text, f"installer is missing {needle!r}")

    def test_generated_launchers_are_git_ignored(self):
        text = read(os.path.join(REPO, ".gitignore"))
        for entry in ("Jelibox.desktop", "Jelibox-launcher.sh", "*.lnk", "__pycache__/", "configs/*"):
            self.assertTrue(entry in text, f".gitignore should contain {entry}")


if __name__ == "__main__":
    unittest.main()
