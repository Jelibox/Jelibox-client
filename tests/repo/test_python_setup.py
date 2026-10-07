"""The private Python: tools/setup_python.ps1 and tools/setup_python.sh, and how both installers use them.

Nothing here downloads anything. The helpers are run against a fake `uv` that records how it was
called and builds a real (stdlib) virtual environment, so the logic around it is what gets tested."""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

from tests.shell import find_bash

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PS_HELPER = os.path.join(REPO, "tools", "setup_python.ps1")
SH_HELPER = os.path.join(REPO, "tools", "setup_python.sh")
HAVE_PS = sys.platform == "win32" and bool(shutil.which("powershell"))

FAKE_UV_PY = r'''
import json, os, sys, venv
args = sys.argv[1:]
with open(os.environ["FAKE_UV_LOG"], "a") as f:
    f.write(json.dumps({"args": args, "install_dir": os.environ.get("UV_PYTHON_INSTALL_DIR")}) + "\n")
if os.environ.get("FAKE_UV_FAIL") == args[0]:
    sys.exit(3)
if args[0] == "venv":
    venv.create(args[1], with_pip=False)
'''

# the shell version of the same fake: venv layout is bin/python, a wrapper around the real interpreter
FAKE_UV_SH = r'''#!/usr/bin/env bash
printf '{"args": ["%s"], "install_dir": "%s"}\n' "$*" "${UV_PYTHON_INSTALL_DIR:-}" >> "$FAKE_UV_LOG"
[ "${FAKE_UV_FAIL:-}" = "$1" ] && exit 3
if [ "$1" = "venv" ]; then
    mkdir -p "$2/bin"
    printf '#!/usr/bin/env bash\nexec "%s" "$@"\n' "$FAKE_PY" > "$2/bin/python"
    chmod +x "$2/bin/python"
fi
exit 0
'''


def log_lines(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


class Workspace(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="jelibox_py_")
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.log = os.path.join(self.root, "uv.log")
        self.home = os.path.join(self.root, "app")
        os.makedirs(self.home)

    def calls(self):
        return log_lines(self.log)


@unittest.skipUnless(HAVE_PS, "Windows PowerShell only")
class PowerShellHelperTests(Workspace):
    def setUp(self):
        super().setUp()
        with open(os.path.join(self.root, "fake_uv.py"), "w") as f:
            f.write(FAKE_UV_PY)
        self.fake_uv = os.path.join(self.root, "fake_uv.cmd")
        with open(self.fake_uv, "w") as f:
            f.write(f'@"{sys.executable}" "{os.path.join(self.root, "fake_uv.py")}" %*\r\n')

    def run_helper(self, venv="venv", **extra):
        env = dict(os.environ, JELIBOX_UV=self.fake_uv, FAKE_UV_LOG=self.log)
        env.pop("JELIBOX_PYTHON_VERSION", None)
        env.update(extra)
        r = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", PS_HELPER,
                            "-Root", self.home, "-Venv", venv],
                           env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
        return r, r.stdout + r.stderr

    def venv_python(self):
        return os.path.join(self.home, "venv", "Scripts", "python.exe")

    def test_fresh_setup_asks_uv_for_the_pinned_python_and_keeps_everything_in_the_folder(self):
        r, out = self.run_helper()
        self.assertEqual(r.returncode, 0, out)
        install, venv = self.calls()
        self.assertEqual(install["args"], ["python", "install", "3.12.10", "--no-bin", "--no-config"])
        self.assertEqual(venv["args"][0], "venv")
        self.assertEqual(os.path.normcase(venv["args"][1]), os.path.normcase(os.path.join(self.home, "venv")))
        for flag in ("--seed", "--managed-python", "--no-config"):
            self.assertIn(flag, venv["args"])
        self.assertEqual(venv["args"][venv["args"].index("--python") + 1], "3.12.10")
        for call in (install, venv):                      # Python lands inside the Jelibox folder, never elsewhere
            self.assertEqual(os.path.normcase(call["install_dir"]), os.path.normcase(os.path.join(self.home, ".python")))
        self.assertTrue(os.path.exists(self.venv_python()))

    def test_a_working_environment_is_left_alone(self):
        self.assertEqual(self.run_helper()[0].returncode, 0)
        before = len(self.calls())
        r, out = self.run_helper()
        self.assertEqual(r.returncode, 0, out)
        self.assertEqual(len(self.calls()), before, "no uv call, no download when the environment already works")
        self.assertIn("ready", out)

    def test_a_broken_environment_is_rebuilt(self):
        os.makedirs(os.path.dirname(self.venv_python()))
        open(self.venv_python(), "w").close()                      # python.exe that cannot run
        marker = os.path.join(self.home, "venv", "old.txt")
        open(marker, "w").close()
        r, out = self.run_helper()
        self.assertEqual(r.returncode, 0, out)
        self.assertIn("broken", out)
        self.assertFalse(os.path.exists(marker))
        self.assertEqual(len(self.calls()), 2)

    def test_python_version_can_be_overridden(self):
        r, out = self.run_helper(JELIBOX_PYTHON_VERSION="3.12.99")
        self.assertEqual(self.calls()[0]["args"][2], "3.12.99")

    def test_failures_are_reported_with_a_nonzero_exit_code(self):
        for stage in ("python", "venv"):
            shutil.rmtree(os.path.join(self.home, "venv"), ignore_errors=True)
            r, out = self.run_helper(FAKE_UV_FAIL=stage)
            self.assertNotEqual(r.returncode, 0, f"{stage}: {out}")
            self.assertIn("[!]", out)

    def test_missing_uv_override_is_a_clear_error(self):
        r, out = self.run_helper(JELIBOX_UV=os.path.join(self.root, "nope.exe"))
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("does not exist", out)

    def test_script_parses_cleanly(self):
        cmd = ("$e=$null; [void][System.Management.Automation.Language.Parser]::ParseFile("
               f"'{PS_HELPER}',[ref]$null,[ref]$e); $e.Count")
        r = subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True, text=True)
        self.assertEqual(r.stdout.strip(), "0", r.stdout + r.stderr)


@unittest.skipUnless(find_bash(), "no working bash available")
class ShellHelperTests(Workspace):
    def setUp(self):
        super().setUp()
        self.fake_uv = os.path.join(self.root, "fake_uv")
        with open(self.fake_uv, "w", newline="\n") as f:
            f.write(FAKE_UV_SH)
        os.chmod(self.fake_uv, 0o755)

    def run_helper(self, **extra):
        env = dict(os.environ, JELIBOX_UV=self.fake_uv.replace("\\", "/"), FAKE_UV_LOG=self.log.replace("\\", "/"),
                   FAKE_PY=sys.executable.replace("\\", "/"))
        env.pop("JELIBOX_PYTHON_VERSION", None)
        env.update(extra)
        r = subprocess.run([find_bash(), SH_HELPER.replace("\\", "/"), self.home.replace("\\", "/"), "jelibox"],
                           env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
        return r, r.stdout + r.stderr

    def test_fresh_setup_asks_uv_for_the_pinned_python(self):
        r, out = self.run_helper()
        self.assertEqual(r.returncode, 0, out)
        install, venv = self.calls()
        self.assertEqual(install["args"], ["python install 3.12.10 --no-bin --no-config"])
        self.assertIn("--seed --python 3.12.10 --managed-python --no-config", venv["args"][0])
        self.assertTrue(venv["args"][0].startswith("venv "))
        self.assertTrue(install["install_dir"].replace("\\", "/").endswith("/app/.python"))
        self.assertTrue(os.path.exists(os.path.join(self.home, "jelibox", "bin", "python")))

    def test_a_working_environment_is_left_alone(self):
        self.assertEqual(self.run_helper()[0].returncode, 0)
        before = len(self.calls())
        r, out = self.run_helper()
        self.assertEqual(r.returncode, 0, out)
        self.assertEqual(len(self.calls()), before)

    def test_a_broken_environment_is_rebuilt(self):
        broken = os.path.join(self.home, "jelibox", "bin")
        os.makedirs(broken)
        with open(os.path.join(broken, "python"), "w") as f:
            f.write("not a program")
        r, out = self.run_helper()
        self.assertEqual(r.returncode, 0, out)
        self.assertIn("broken", out)
        self.assertEqual(len(self.calls()), 2)

    def test_failures_are_reported_with_a_nonzero_exit_code(self):
        for stage in ("python", "venv"):
            shutil.rmtree(os.path.join(self.home, "jelibox"), ignore_errors=True)
            r, out = self.run_helper(FAKE_UV_FAIL=stage)
            self.assertNotEqual(r.returncode, 0, f"{stage}: {out}")
            self.assertIn("[!]", out)

    def test_bash_syntax(self):
        for script in (SH_HELPER, os.path.join(REPO, "jelibox_linux_installation.bash")):
            r = subprocess.run([find_bash(), "-n", script.replace("\\", "/")], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, script + r.stderr)


def read(name):
    with open(os.path.join(REPO, name), encoding="utf-8", errors="ignore") as f:
        return f.read()


class InstallerWiringTests(unittest.TestCase):
    def test_windows_installer_uses_the_private_python_and_only_asks_for_admin_when_needed(self):
        bat = read("jelibox_windows_installation.bat")
        self.assertIn(r'tools\setup_python.ps1', bat)
        self.assertIn("JELIBOX_PYTHON", bat)
        self.assertIn(":REQUIRE_ADMIN", bat)
        head = bat.split(":: 0. INTERNET CHECK")[0]
        self.assertNotIn("net session", head, "no administrator prompt before anything needs it")
        self.assertNotIn("-Verb RunAs", head)
        self.assertIn("/yes", bat)                                  # the elevated restart must not ask Y/N again
        # the system-wide Python install (the only thing that needs admin besides the C++ runtime) asks first
        self.assertLess(bat.index("call :REQUIRE_ADMIN", bat.index(":PYTHON_SYSTEM")), bat.index("InstallAllUsers=1"))

    def test_every_batch_label_that_is_jumped_to_exists(self):
        import re
        bat = read("jelibox_windows_installation.bat")
        labels = {m.upper() for m in re.findall(r"^:([A-Za-z_]+)\s*$", bat, re.M)}
        jumps = {m.upper() for m in re.findall(r"(?:goto|call)\s+:?([A-Za-z_]+)", bat, re.I)} - {"EOF"}
        self.assertEqual(jumps - labels, set())

    def test_batch_file_keeps_windows_line_endings(self):
        with open(os.path.join(REPO, "jelibox_windows_installation.bat"), "rb") as f:
            data = f.read()
        self.assertEqual(data.count(b"\n"), data.count(b"\r\n"))

    def test_gpu_fallback_command_is_valid_powershell(self):
        # an escaped ")" inside the quoted command once made this a syntax error
        self.assertNotIn("TESLA'^)", read("jelibox_windows_installation.bat"))

    def test_linux_installer_uses_the_private_python_and_keeps_system_python_as_a_fallback(self):
        sh = read("jelibox_linux_installation.bash")
        self.assertIn("tools/setup_python.sh", sh)
        self.assertIn("JELIBOX_PYTHON", sh)
        self.assertIn("use_system_python", sh)
        # the fallback function is only *called* after the private Python was tried
        self.assertLess(sh.index("tools/setup_python.sh"), sh.rindex("use_system_python"))
        # sudo must only be asked for inside the fallback function, not up front
        self.assertGreater(sh.index("sudo -v"), sh.index("use_system_python()"))

    def test_helpers_pin_the_same_python_as_each_other(self):
        for name in ("tools/setup_python.ps1", "tools/setup_python.sh"):
            text = read(name)
            self.assertIn("3.12.10", text, name)
            self.assertIn("--managed-python", text, name)
            self.assertIn("--no-bin", text, name)                    # no shim outside the folder
            self.assertIn("UV_PYTHON_INSTALL_DIR", text, name)
            self.assertIn("sha256", text.lower(), name)              # the uv download is verified
            self.assertIn("tkinter", text, name)

    def test_downloaded_state_is_not_committed(self):
        ignore = read(".gitignore")
        for entry in (".python/", ".uv/"):
            self.assertIn(entry, ignore)

    def test_one_line_installers_no_longer_promise_a_system_python(self):
        for name in ("install.ps1", "install.sh"):
            self.assertIn("JELIBOX_PYTHON", read(name), name)

    def test_readme_explains_the_private_python(self):
        readme = read("README.md")
        self.assertIn("JELIBOX_PYTHON", readme)
        self.assertIn("uv", readme)


if __name__ == "__main__":
    unittest.main()
