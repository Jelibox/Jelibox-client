"""The one-line installers (install.ps1 / install.sh) and the pieces they depend on.

Nothing here downloads from the internet or installs Python/torch: the scripts are run against a
local archive built from the working tree, with JELIBOX_NO_INSTALL=1 (unpack only)."""
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
import zipfile

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RAW = "https://raw.githubusercontent.com/Jelibox/Jelibox-client/main"
PS_CMD = f"irm {RAW}/install.ps1 | iex"
SH_CMD = f"curl -fsSL {RAW}/install.sh | bash"
MAX_FILE = 3 * 1024 * 1024


def read(name):
    with open(os.path.join(REPO, name), encoding="utf-8", errors="ignore") as f:
        return f.read()


def tracked_and_new_files():
    """What a release archive would contain: tracked files plus new, non-ignored ones."""
    out = subprocess.check_output(["git", "ls-files", "--cached", "--others", "--exclude-standard"],
                                  cwd=REPO, text=True, encoding="utf-8")
    return [f for f in out.splitlines()
            if os.path.isfile(os.path.join(REPO, f)) and os.path.getsize(os.path.join(REPO, f)) < MAX_FILE]


def build_archives(folder):
    files = tracked_and_new_files()
    tgz, zp = os.path.join(folder, "jelibox.tar.gz"), os.path.join(folder, "jelibox.zip")
    with tarfile.open(tgz, "w:gz") as t:
        for f in files:
            t.add(os.path.join(REPO, f), arcname="Jelibox-client-test/" + f)
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(os.path.join(REPO, f), "Jelibox-client-test/" + f)
    return tgz, zp


class ScriptContentTests(unittest.TestCase):
    def test_scripts_exist_at_the_repo_root(self):
        for name in ("install.ps1", "install.sh"):
            self.assertTrue(os.path.isfile(os.path.join(REPO, name)), name)

    def test_readme_and_site_show_the_exact_commands(self):
        for doc in ("README.md", "index.html"):
            text = read(doc)
            self.assertIn(PS_CMD, text, doc)
            self.assertIn(SH_CMD, text, doc)

    def test_readme_documents_every_option(self):
        readme = read("README.md")
        for var in ("JELIBOX_VERSION", "JELIBOX_HOME", "JELIBOX_NO_INSTALL"):
            self.assertIn(var, readme)
            self.assertIn(var, read("install.ps1"))
            self.assertIn(var, read("install.sh"))

    def test_scripts_only_talk_to_the_jelibox_repo(self):
        for name in ("install.ps1", "install.sh"):
            text = read(name)
            self.assertIn("Jelibox/Jelibox-client", text)
            self.assertNotIn("BoxifyAnnotationTools", text)

    def test_powershell_script_is_windows_powershell_5_safe(self):
        text = read("install.ps1")
        self.assertIn("Tls12", text)                              # Windows PowerShell 5.1 defaults to old TLS
        self.assertIn("SilentlyContinue", text)                   # progress bar makes 5.1 downloads crawl
        self.assertNotIn("Invoke-Expression", text.replace("irm", ""))   # never executes downloaded text
        self.assertNotIn("&&", text)                              # not valid in 5.1
        self.assertNotIn(" ?? ", text)

    def test_bash_script_survives_being_piped_into_bash(self):
        text = read("install.sh")
        self.assertTrue(text.rstrip().endswith('main "$@"'), "everything must be inside main() so a piped script is fully read first")
        self.assertIn("set -euo pipefail", text)
        self.assertIn('trap "rm -rf', text)                       # expands $tmp when set, not when it fires
        self.assertIn("/dev/tty", text)                           # sudo prompt needs the keyboard

    def test_bash_syntax(self):
        bash = shutil.which("bash")
        if not bash:
            self.skipTest("bash not available")
        with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False, newline="\n", encoding="utf-8") as tmp:
            tmp.write(read("install.sh").replace("\r\n", "\n"))
        try:
            r = subprocess.run([bash, "-n", tmp.name], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
        finally:
            os.remove(tmp.name)

    @unittest.skipUnless(sys.platform == "win32", "Windows PowerShell only")
    def test_powershell_parses_without_errors(self):
        ps = shutil.which("powershell")
        if not ps:
            self.skipTest("powershell not available")
        cmd = ("$e=$null; [void][System.Management.Automation.Language.Parser]::ParseFile("
               f"'{os.path.join(REPO, 'install.ps1')}',[ref]$null,[ref]$e); $e.Count")
        r = subprocess.run([ps, "-NoProfile", "-Command", cmd], capture_output=True, text=True)
        self.assertEqual(r.stdout.strip(), "0", r.stdout + r.stderr)


class WindowsInstallerEditsTests(unittest.TestCase):
    def setUp(self):
        self.bat = read("jelibox_windows_installation.bat")

    def test_install_folder_with_spaces_is_supported(self):
        self.assertIn('cd /d "%~dp0"', self.bat)

    def test_one_line_install_skips_the_yes_no_question(self):
        self.assertIn(".install-yes", self.bat)
        self.assertIn('if "%ASSUME_YES%"=="1" goto CONFIRMED', self.bat)
        self.assertIn(":CONFIRMED", self.bat)
        self.assertIn(".install-yes", read("install.ps1"))        # the bootstrap writes the same marker

    def test_visual_cpp_runtime_is_checked_with_the_right_registry_key(self):
        self.assertIn(r'reg query "HKLM\SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64" /v Installed', self.bat)
        self.assertIn(r"VC_redist\VC_redist.x64.exe", self.bat)
        self.assertTrue(os.path.exists(os.path.join(REPO, "VC_redist", "VC_redist.x64.exe")))

    def test_no_stray_control_characters_in_text_files(self):
        """A mangled escape (e.g. \\14 turning into a form-feed) once slipped into the .bat."""
        bad = []
        for name in ("jelibox_windows_installation.bat", "jelibox_linux_installation.bash", "install.ps1", "install.sh",
                     "README.md", "index.html", "RELEASING.md"):
            for i, line in enumerate(read(name).splitlines(), 1):
                if any(ord(c) < 32 and c not in "\t" for c in line):
                    bad.append(f"{name}:{i}")
        self.assertEqual(bad, [])

    def test_confirmation_is_the_only_interactive_step_before_install(self):
        self.assertEqual(self.bat.count("choice /c YN"), 1)


class BootstrapRunTests(unittest.TestCase):
    """Run the real scripts against a local archive (no network, no dependency install)."""

    @classmethod
    def setUpClass(cls):
        cls.work = tempfile.mkdtemp(prefix="jelibox_qi_")
        cls.tgz, cls.zip = build_archives(cls.work)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.work, ignore_errors=True)

    def env(self, archive, home):
        e = dict(os.environ, JELIBOX_ARCHIVE=archive, JELIBOX_HOME=home, JELIBOX_NO_INSTALL="1")
        e.pop("JELIBOX_VERSION", None)
        return e

    def check_install_and_update(self, run, home):
        r = run()
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        for rel in ("utils/theme.py", "jelibox_linux_installation.bash", "jelibox_windows_installation.bat", "assets/jelibox.ico"):
            self.assertTrue(os.path.exists(os.path.join(home, rel)), rel)
        # user data must survive an update
        data = os.path.join(home, "vocdataset", "mine")
        os.makedirs(data, exist_ok=True)
        with open(os.path.join(data, "a.xml"), "w") as f:
            f.write("keep me")
        r = run()
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        with open(os.path.join(data, "a.xml")) as f:
            self.assertEqual(f.read(), "keep me")

    @unittest.skipUnless(shutil.which("bash") and shutil.which("tar"), "bash/tar not available")
    def test_bash_install_and_update_when_piped_like_curl_does(self):
        home = os.path.join(self.work, "sh_home")
        script = os.path.join(REPO, "install.sh")
        run = lambda: subprocess.run(["bash", "-c", 'cat "$1" | bash', "_", script], env=self.env(self.tgz, home),
                                     capture_output=True, text=True, encoding="utf-8", errors="replace")
        self.check_install_and_update(run, home)

    @unittest.skipUnless(shutil.which("bash") and shutil.which("tar"), "bash/tar not available")
    def test_bash_fails_clearly_without_an_archive(self):
        r = subprocess.run(["bash", os.path.join(REPO, "install.sh")], env=self.env(os.path.join(self.work, "nope.tar.gz"),
                           os.path.join(self.work, "x")), capture_output=True, text=True, encoding="utf-8", errors="replace")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("not found", r.stderr)

    @unittest.skipUnless(sys.platform == "win32" and shutil.which("powershell"), "Windows PowerShell only")
    def test_powershell_install_and_update_when_piped_into_iex(self):
        home = os.path.join(self.work, "ps_home")
        script = os.path.join(REPO, "install.ps1")
        command = f"Get-Content -Raw -LiteralPath '{script}' | Invoke-Expression"      # same path as `irm ... | iex`
        run = lambda: subprocess.run(["powershell", "-NoProfile", "-Command", command], env=self.env(self.zip, home),
                                     capture_output=True, text=True, encoding="utf-8", errors="replace")
        self.check_install_and_update(run, home)

    @unittest.skipUnless(sys.platform == "win32" and shutil.which("powershell"), "Windows PowerShell only")
    def test_powershell_leaves_no_temp_folders_behind(self):
        before = {d for d in os.listdir(tempfile.gettempdir()) if d.startswith("jelibox_") and len(d) == 40}
        home = os.path.join(self.work, "ps_home2")
        command = f"Get-Content -Raw -LiteralPath '{os.path.join(REPO, 'install.ps1')}' | Invoke-Expression"
        subprocess.run(["powershell", "-NoProfile", "-Command", command], env=self.env(self.zip, home), capture_output=True)
        after = {d for d in os.listdir(tempfile.gettempdir()) if d.startswith("jelibox_") and len(d) == 40}
        self.assertEqual(after - before, set())


if __name__ == "__main__":
    unittest.main()
