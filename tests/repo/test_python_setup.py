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
        self.assertRegex(out, "ready|keeping it")    # "keeping it" when the Python running the tests is not 3.12

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
    def test_windows_installer_uses_the_private_python_and_never_runs_elevated_as_a_whole(self):
        bat = read("jelibox_windows_installation.bat")
        self.assertIn(r'tools\setup_python.ps1', bat)
        self.assertNotIn("InstallAllUsers", bat, "no system-wide Python")
        self.assertNotIn("PrependPath", bat, "Python must never be put on PATH")
        self.assertNotIn("python.org", bat)
        self.assertNotIn("activate.bat", bat, "the environment is used by path, never activated")
        self.assertNotIn(":REQUIRE_ADMIN", bat, "the whole installer is not restarted as administrator")
        self.assertNotIn("%SELF%", bat)
        # the only elevation is the Visual C++ runtime installer, and only when it is missing
        self.assertEqual(bat.count("-Verb RunAs"), 1)
        self.assertIn(":INSTALL_VCREDIST", bat)
        self.assertIn("-Verb RunAs", bat.split(":INSTALL_VCREDIST")[-1])
        self.assertIn("VC\\Runtimes\\x64", bat.split(":: 2. PYTHON")[0])      # checked before it is installed
        head = bat.split(":: 1b. VISUAL C++ RUNTIME")[0]
        self.assertNotIn("net session", head, "no administrator prompt before anything needs it")
        self.assertNotIn("-Verb RunAs", head)

    def test_windows_installer_calls_the_environments_python_directly(self):
        bat = read("jelibox_windows_installation.bat")
        pips = [line for line in bat.splitlines() if " -m pip install" in line and not line.lstrip().startswith(("echo", "::"))]
        self.assertGreaterEqual(len(pips), 5)
        for line in pips:
            self.assertTrue(line.startswith('"%VENV_PY%" -m pip'), line)

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

    def test_linux_installer_has_no_system_python_and_asks_for_sudo_only_for_system_libraries(self):
        sh = read("jelibox_linux_installation.bash")
        self.assertIn("tools/setup_python.sh", sh)
        self.assertNotIn("use_system_python", sh)
        self.assertNotIn("deadsnakes", sh)
        self.assertNotIn("add-apt-repository", sh)
        self.assertNotIn("activate", sh.replace('never "activated"', ""), "the environment is used by path, never activated")
        self.assertNotIn("JELIBOX_PYTHON", sh)
        # every sudo is after the Python packages are installed, inside the system-library step
        self.assertEqual(sh.count("sudo -v"), 1)
        self.assertGreater(sh.index("sudo -v"), sh.index("-r \"$APP_PATH/requirements.txt\""))
        self.assertGreater(sh.index("sudo -v"), sh.index("SYSTEM LIBRARIES"))
        for line in sh.splitlines():
            if " -m pip install" in line:
                self.assertTrue(line.lstrip().startswith('"$VENV_PY" -m pip'), line)

    def test_helpers_install_uv_with_the_official_installer_only_when_it_is_missing(self):
        ps, sh = read("tools/setup_python.ps1"), read("tools/setup_python.sh")
        self.assertIn('-ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"', ps)
        self.assertIn("curl -LsSf https://astral.sh/uv/install.sh | sh", sh)
        self.assertIn("wget -qO- https://astral.sh/uv/install.sh | sh", sh)
        self.assertLess(ps.index("$uv = Find-Uv"), ps.index("install.ps1 | iex"))
        self.assertLess(sh.index('uv="$(find_uv)"'), sh.index("install.sh | sh"))
        for text in (ps, sh):
            self.assertNotIn("sha256", text.lower(), "uv is no longer downloaded and unpacked by Jelibox itself")
            self.assertNotIn(".uv", text.replace("https://astral.sh/uv", ""))

    def test_helpers_keep_python_private(self):
        for name in ("tools/setup_python.ps1", "tools/setup_python.sh"):
            text = read(name)
            self.assertIn("3.12.10", text, name)
            self.assertIn("--managed-python", text, name)            # never pick the user's own Python
            self.assertIn("--no-bin", text, name)                    # no python shim on PATH
            self.assertIn("UV_PYTHON_INSTALL_DIR", text, name)       # inside the Jelibox folder
            self.assertIn("tkinter", text, name)
        self.assertIn("UV_PYTHON_INSTALL_REGISTRY", read("tools/setup_python.ps1"))   # `py -3.12` must not find it

    def test_every_installer_asks_about_the_nvidia_gpu_and_cpu_installs_get_no_cuda_packages(self):
        for name in ("install.ps1", "install.sh", "jelibox_windows_installation.bat", "jelibox_linux_installation.bash"):
            text = read(name)
            self.assertIn("JELIBOX_GPU", text, name)
            self.assertIn("discrete NVIDIA graphics card", text, name)
        # the one-line installers ask before they download anything, and hand the answer to the installer script
        ps, sh = read("install.ps1"), read("install.sh")
        self.assertLess(ps.index("discrete NVIDIA graphics card"), ps.index("Invoke-WebRequest -UseBasicParsing -Uri $url"))
        self.assertLess(sh.index("discrete NVIDIA graphics card"), sh.index('download "$url" "$archive"'))
        self.assertIn("$env:JELIBOX_GPU = $gpu", ps)
        self.assertIn('export JELIBOX_GPU="$gpu"', sh)
        # PyPI's default Linux wheel is the CUDA build: the CPU choice must use the CPU index, the GPU choice the CUDA one
        for name in ("jelibox_windows_installation.bat", "jelibox_linux_installation.bash"):
            text = read(name)
            self.assertIn("download.pytorch.org/whl/cpu", text, name)
            self.assertIn("download.pytorch.org/whl/cu121", text, name)
        bat = read("jelibox_windows_installation.bat")
        self.assertIn("if defined JELIBOX_GPU goto GPU_PRESET", bat)
        self.assertLess(bat.index(":GPU_DONE"), bat.index("[5/6] INSTALL PYTORCH"))

    def test_downloaded_state_is_not_committed(self):
        ignore = read(".gitignore")
        for entry in (".python/", ".uv/"):
            self.assertIn(entry, ignore)

    def test_one_line_installers_no_longer_offer_a_system_python(self):
        for name in ("install.ps1", "install.sh", "README.md", "index.html"):
            self.assertNotIn("JELIBOX_PYTHON=system", read(name), name)
            self.assertNotIn("JELIBOX_PYTHON      system", read(name), name)

    def test_docs_do_not_promise_the_wrong_permissions(self):
        for name in ("README.md", "index.html"):
            text = read(name)
            self.assertNotIn("Run as administrator", text, name)
            self.assertNotIn("asks for your sudo password at the beginning", text, name)
            self.assertNotIn("asks for administrator permission once", text, name)
        self.assertIn("uv", read("README.md"))


if __name__ == "__main__":
    unittest.main()
