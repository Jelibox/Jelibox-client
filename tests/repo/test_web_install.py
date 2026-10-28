"""The web version in the installers: the "App + web / App only" question, the web packages, and the services
(tools/web_service.sh = systemd user unit, tools/web_service.ps1 = Windows Startup shortcut).

Nothing here installs a package or touches the real Startup folder / systemd: the shell service is run against a
fake systemctl / loginctl and a temporary folder, the PowerShell one against a temporary Startup folder."""
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import unittest

from tests.shell import find_bash

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SERVICE_SH = os.path.join(REPO, "tools", "web_service.sh")
SERVICE_PS = os.path.join(REPO, "tools", "web_service.ps1")
HAVE_PS = sys.platform == "win32" and bool(shutil.which("powershell"))


def read(name):
    with open(os.path.join(REPO, name), encoding="utf-8", errors="ignore") as f:
        return f.read()


class WiringTests(unittest.TestCase):
    def test_web_packages_are_not_part_of_the_app(self):
        app = [line.split("#")[0].strip().lower() for line in read("requirements.txt").splitlines()]
        web = [line.split("#")[0].strip().lower() for line in read("requirements-web.txt").splitlines()]
        for pkg in ("starlette", "uvicorn", "python-multipart"):
            self.assertIn(pkg, web)
            self.assertNotIn(pkg, app, "the app only install must not get the web server")

    def test_every_installer_knows_the_question_and_the_one_line_ones_ask_before_downloading(self):
        for name in ("install.ps1", "install.sh", "jelibox_windows_installation.bat", "jelibox_linux_installation.bash"):
            self.assertIn("JELIBOX_WEB", read(name), name)
        ps, sh = read("install.ps1"), read("install.sh")
        self.assertLess(ps.index("What do you want to install?"), ps.index("Invoke-WebRequest -UseBasicParsing -Uri $url"))
        self.assertLess(sh.index("What do you want to install?"), sh.index('download "$url" "$archive"'))
        self.assertIn("$env:JELIBOX_WEB = $web", ps)
        self.assertIn('export JELIBOX_WEB="$web"', sh)
        # the answer is asked again on every install and update, and the previous answer is the default
        self.assertIn(".jelibox-web", ps)
        self.assertIn(".jelibox-web", sh)

    def test_an_update_stops_the_old_server_before_files_and_packages_change(self):
        ps, sh = read("install.ps1"), read("install.sh")
        self.assertLess(ps.index("-Action stop"), ps.index("Copy-Item -Path"))
        self.assertLess(sh.index("web_service.sh\" stop"), sh.index('cp -a "$top/." "$dest/"'))
        bat, bash = read("jelibox_windows_installation.bat"), read("jelibox_linux_installation.bash")
        self.assertLess(bat.index("-Action stop"), bat.index("[2/6] Preparing"))
        self.assertLess(bash.index("web_service.sh\" stop"), bash.index("tools/setup_python.sh"))

    def test_windows_installer_installs_web_packages_and_the_service_only_when_asked(self):
        bat = read("jelibox_windows_installation.bat")
        self.assertEqual(bat.count("choice /c YN"), 3)           # continue? + NVIDIA GPU? + web?
        self.assertIn("if defined JELIBOX_WEB goto WEB_PRESET", bat)
        self.assertIn('"%VENV_PY%" -m pip install -r "%~dp0requirements-web.txt"', bat)
        self.assertLess(bat.index("requirements-web.txt"), bat.index("-Action install"))
        self.assertIn("-Action remove", bat)                      # "app only" removes an earlier server
        self.assertIn(".jelibox-web", bat)
        self.assertEqual(bat.count("-Verb RunAs"), 1, "the Visual C++ runtime stays the only elevation")

    def test_linux_installer_installs_web_packages_and_the_service_only_when_asked(self):
        sh = read("jelibox_linux_installation.bash")
        self.assertIn('-m pip install -r "$APP_PATH/requirements-web.txt"', sh)
        self.assertLess(sh.index("requirements-web.txt"), sh.index("web_service.sh\" install"))
        self.assertIn('web_service.sh" remove', sh)
        self.assertIn(".jelibox-web", sh)
        self.assertEqual(sh.count("sudo -v"), 1, "the installer's own sudo stays the system-library step")

    def test_services_need_no_administrator_and_touch_nothing_else(self):
        ps, sh = read("tools/web_service.ps1"), read("tools/web_service.sh")
        self.assertNotIn("RunAs", ps)
        self.assertNotIn("RunAs", sh)
        for word in ("HKLM", "HKCU", "schtasks", "Register-ScheduledTask", "netsh", "Set-ItemProperty"):
            self.assertNotIn(word, ps, word)
        for line in ps.splitlines():                              # the firewall command is only printed, never run
            if "New-NetFirewallRule" in line and not line.lstrip().startswith("#"):
                self.assertIn("Say ", line)
        for line in sh.splitlines():                              # sudo only in messages and the one linger command
            if "sudo" in line and not line.lstrip().startswith("#"):
                self.assertTrue("say " in line or "enable-linger" in line or "command -v sudo" in line, line)
        self.assertIn("--user", sh)

    def test_docs_describe_the_web_version_and_the_question_truthfully(self):
        readme, site = read("README.md"), read("index.html")
        for text in (readme, site):
            self.assertIn("App + web", text)
            self.assertIn("App only", text)
            self.assertIn("8420", text)
        self.assertIn("JELIBOX_WEB", readme)
        self.assertIn("## Web version", readme)
        self.assertIn('id="web"', site)
        for name in ("web-login.png", "web-workspaces.png"):                  # the screenshots the docs show exist
            self.assertTrue(os.path.isfile(os.path.join(REPO, "assets", name)), name)
            self.assertIn("assets/" + name, readme)
            self.assertIn("assets/" + name, site)
        # what the scripts do is what the README says
        self.assertIn("Startup folder", readme)
        self.assertIn("systemd", readme)
        self.assertIn("enable-linger", readme)
        self.assertIn("never changes your firewall", readme)
        self.assertNotIn("Save to", read("utils/web/static/export.js"))      # the web downloads; the README says so

    def test_script_endings_and_encoding(self):
        with open(SERVICE_SH, "rb") as f:
            self.assertNotIn(b"\r", f.read())
        with open(SERVICE_PS, "rb") as f:
            data = f.read()
        data.decode("ascii")                                       # Windows PowerShell 5.1 reads BOM-less files as ANSI

    @unittest.skipUnless(HAVE_PS, "Windows PowerShell only")
    def test_powershell_service_parses_cleanly(self):
        cmd = ("$e=$null; [void][System.Management.Automation.Language.Parser]::ParseFile("
               f"'{SERVICE_PS}',[ref]$null,[ref]$e); $e.Count")
        r = subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True, text=True)
        self.assertEqual(r.stdout.strip(), "0", r.stdout + r.stderr)

    @unittest.skipUnless(find_bash(), "no working bash available")
    def test_bash_syntax(self):
        for name in ("tools/web_service.sh", "install.sh", "jelibox_linux_installation.bash"):
            r = subprocess.run([find_bash(), "-n", os.path.join(REPO, name)], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, name + r.stderr)


FAKE_SYSTEMCTL = r'''#!/usr/bin/env bash
echo "systemctl $*" >> "$FAKE_LOG"
[ "${FAKE_NO_SYSTEMD:-}" = "1" ] && exit 1
case "$*" in
    *is-failed*) exit 1 ;;
    *is-enabled*) [ -f "$FAKE_UNIT" ] && exit 0 || exit 1 ;;
esac
exit 0
'''

FAKE_LOGINCTL = r'''#!/usr/bin/env bash
echo "loginctl $*" >> "$FAKE_LOG"
case "$1" in
    show-user) echo "${FAKE_LINGER:-no}"; exit 0 ;;
    enable-linger) [ "${FAKE_LINGER_DENIED:-}" = "1" ] && exit 1; exit 0 ;;
esac
'''


@unittest.skipUnless(find_bash(), "no working bash available")
class SystemdServiceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="jelibox_svc_")
        self.root = os.path.join(self.tmp, "Jelibox with space")
        self.units = os.path.join(self.tmp, "units")
        self.bin = os.path.join(self.tmp, "bin")
        for d in (os.path.join(self.root, "utils"), os.path.join(self.root, "jelibox", "bin"),
                  os.path.join(self.root, "configs"), self.bin):
            os.makedirs(d)
        open(os.path.join(self.root, "utils", "server.py"), "w").close()
        self.py = os.path.join(self.root, "jelibox", "bin", "python")
        with open(self.py, "w", newline="\n") as f:
            f.write("#!/usr/bin/env bash\n")
        os.chmod(self.py, 0o755)
        for name, body in (("systemctl", FAKE_SYSTEMCTL), ("loginctl", FAKE_LOGINCTL)):
            path = os.path.join(self.bin, name)
            with open(path, "w", newline="\n") as f:
                f.write(body)
            os.chmod(path, 0o755)
        self.log = os.path.join(self.tmp, "calls.log")
        self.unit = os.path.join(self.units, "jelibox-web.service")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def run_service(self, action, **env):
        e = dict(os.environ, JELIBOX_SYSTEMD_DIR=self.units, JELIBOX_WEB_WAIT="1", FAKE_LOG=self.log, FAKE_UNIT=self.unit,
                 PATH=self.bin + os.pathsep + os.environ["PATH"])
        e.update(env)
        return subprocess.run([find_bash(), SERVICE_SH, action, self.root], env=e, capture_output=True, text=True,
                              encoding="utf-8", errors="replace", stdin=subprocess.DEVNULL)

    def calls(self):
        if not os.path.exists(self.log):
            return []
        with open(self.log) as f:
            return [line.strip() for line in f]

    def listen(self):
        """Pretend to be the server on the configured port; returns the socket (close it to stop)."""
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        sock.listen(5)
        with open(os.path.join(self.root, "configs", "_app.json"), "w") as f:
            f.write('{"web_port": %d}' % sock.getsockname()[1])
        self.addCleanup(sock.close)
        return sock

    def test_install_writes_a_user_unit_enables_it_and_restarts_it(self):
        self.listen()
        r = self.run_service("install", FAKE_LINGER="yes")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        with open(self.unit) as f:
            text = f.read()
        root = subprocess.run([find_bash(), "-c", 'cd "$1" && pwd', "_", self.root], capture_output=True, text=True).stdout.strip()   # as bash spells it
        self.assertIn(f'WorkingDirectory={root}', text)
        self.assertIn(f'ExecStart="{root}/jelibox/bin/python" -u "{root}/utils/server.py"', text)     # quoted: the folder has a space
        self.assertIn("WantedBy=default.target", text)
        self.assertIn("Restart=on-failure", text)
        calls = self.calls()
        for expected in ("systemctl --user daemon-reload", "systemctl --user enable jelibox-web.service",
                         "systemctl --user restart jelibox-web.service"):
            self.assertIn(expected, calls)
        self.assertLess(calls.index("systemctl --user enable jelibox-web.service"),
                        calls.index("systemctl --user restart jelibox-web.service"))
        self.assertIn("answers at:  http://localhost:", r.stdout)
        self.assertFalse([c for c in calls if "sudo" in c])

    def test_linger_is_turned_on_without_sudo_when_the_system_allows_it(self):
        self.listen()
        r = self.run_service("install", FAKE_LINGER="no")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertTrue([c for c in self.calls() if c.startswith("loginctl enable-linger")])
        self.assertIn("Start at boot is on", r.stdout)

    def test_when_linger_is_refused_it_says_how_instead_of_failing(self):
        self.listen()
        r = self.run_service("install", FAKE_LINGER="no", FAKE_LINGER_DENIED="1", PATH=self.bin + os.pathsep + "/nonexistent")
        out = r.stdout
        self.assertEqual(r.returncode, 0, out + r.stderr)
        self.assertIn("Start at boot is OFF", out)
        self.assertIn("loginctl enable-linger", out)

    def test_install_says_so_when_the_server_does_not_come_up(self):
        free = socket.socket()                                              # a port nothing listens on (8420 may be in use)
        free.bind(("127.0.0.1", 0))
        port = free.getsockname()[1]
        free.close()
        with open(os.path.join(self.root, "configs", "_app.json"), "w") as f:
            f.write('{"web_port": %d}' % port)
        r = self.run_service("install", FAKE_LINGER="yes")
        self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
        self.assertIn("journalctl --user -u jelibox-web", r.stdout)

    def test_a_system_without_user_systemd_gets_a_clear_message(self):
        r = self.run_service("install", FAKE_NO_SYSTEMD="1")
        self.assertEqual(r.returncode, 1)
        self.assertIn("systemd user services are not available", r.stdout)
        self.assertFalse(os.path.exists(self.unit))

    def test_remove_disables_and_deletes_the_unit(self):
        self.listen()
        self.run_service("install", FAKE_LINGER="yes")
        self.assertTrue(os.path.exists(self.unit))
        r = self.run_service("remove", FAKE_LINGER="yes")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertFalse(os.path.exists(self.unit))
        self.assertIn("systemctl --user disable --now jelibox-web.service", self.calls())

    def test_remove_without_an_install_is_harmless(self):
        r = self.run_service("remove")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)

    def test_status_reports_running_and_automatic(self):
        sock = self.listen()
        self.run_service("install", FAKE_LINGER="yes")
        r = self.run_service("status")
        self.assertEqual(r.returncode, 0)
        self.assertIn("running on port", r.stdout)
        self.assertIn("Starts automatically: yes", r.stdout)
        sock.close()
        r = self.run_service("status")
        self.assertEqual(r.returncode, 1)
        self.assertIn("not running", r.stdout)


@unittest.skipUnless(HAVE_PS, "Windows PowerShell only")
class StartupShortcutTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="jelibox_ps_")
        self.root = os.path.join(self.tmp, "Jelibox with space")
        self.startup = os.path.join(self.tmp, "Startup")
        os.makedirs(os.path.join(self.root, "utils"))
        open(os.path.join(self.root, "utils", "server.py"), "w").close()
        self.link = os.path.join(self.startup, "Jelibox Web.lnk")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def run_service(self, action):
        env = dict(os.environ, JELIBOX_STARTUP_DIR=self.startup)
        return subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", SERVICE_PS,
                               "-Action", action, "-Root", self.root], env=env, capture_output=True, text=True,
                              encoding="utf-8", errors="replace")

    def shortcut(self):
        cmd = ("$s=(New-Object -ComObject WScript.Shell).CreateShortcut('%s'); "
               "$s.TargetPath; $s.Arguments; $s.WorkingDirectory; $s.WindowStyle" % self.link)
        out = subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True, text=True).stdout
        return out.strip().splitlines()

    def test_install_makes_a_hidden_startup_shortcut_to_the_environments_pythonw(self):
        r = self.run_service("install")                    # no environment in this folder: the server cannot start
        self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
        self.assertIn("Python was not found", r.stdout)
        self.assertTrue(os.path.exists(self.link))
        target, args, workdir, style = self.shortcut()
        self.assertEqual(target, os.path.join(self.root, "venv", "Scripts", "pythonw.exe"))
        self.assertEqual(args, '"%s"' % os.path.join(self.root, "utils", "server.py"))
        self.assertEqual(workdir, self.root)
        self.assertEqual(style, "7")

    def test_remove_deletes_the_shortcut_and_status_says_what_is_going_on(self):
        self.run_service("install")
        r = self.run_service("status")
        self.assertIn("Starts at sign in: yes", r.stdout)
        r = self.run_service("remove")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertFalse(os.path.exists(self.link))
        self.assertIn("Starts at sign in: no", self.run_service("status").stdout)

    def test_stop_with_nothing_running_is_harmless(self):
        r = self.run_service("stop")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("was not running", r.stdout)


class InstallerRefusesAnUnknownAnswer(unittest.TestCase):
    """JELIBOX_WEB is checked before anything is unpacked, like JELIBOX_GPU."""

    def check(self, cmd, env_extra):
        work = tempfile.mkdtemp(prefix="jelibox_web_")
        self.addCleanup(shutil.rmtree, work, ignore_errors=True)
        home = os.path.join(work, "home")
        archive = os.path.join(work, "x.tar.gz")
        open(archive, "w").close()
        env = dict(os.environ, JELIBOX_HOME=home, JELIBOX_GPU="cpu", JELIBOX_WEB="maybe", JELIBOX_ARCHIVE=archive)
        env.pop("JELIBOX_NO_INSTALL", None)
        env.update(env_extra)
        r = subprocess.run(cmd, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace",
                           stdin=subprocess.DEVNULL)
        out = r.stdout + r.stderr
        self.assertNotEqual(r.returncode, 0, out)
        self.assertIn("JELIBOX_WEB must be 'yes' or 'no'", out)
        self.assertFalse(os.path.exists(os.path.join(home, "utils")))

    @unittest.skipUnless(find_bash(), "no working bash available")
    def test_bash(self):
        self.check([find_bash(), "-c", 'cat "$1" | bash', "_", os.path.join(REPO, "install.sh")], {})

    @unittest.skipUnless(HAVE_PS, "Windows PowerShell only")
    def test_powershell(self):
        script = os.path.join(REPO, "install.ps1")
        self.check(["powershell", "-NoProfile", "-Command", f"Get-Content -Raw -LiteralPath '{script}' | Invoke-Expression"], {})


class AnswerReachesTheInstallerScript(unittest.TestCase):
    """The one-line installers hand the answer on in JELIBOX_WEB. The installer script in the archive is a stub that
    writes down what it saw (the real one would install packages)."""

    def make_archive(self, kind):
        import tarfile
        import zipfile
        work = tempfile.mkdtemp(prefix="jelibox_ans_")
        self.addCleanup(shutil.rmtree, work, ignore_errors=True)
        if kind == "ps":
            archive = os.path.join(work, "a.zip")
            with zipfile.ZipFile(archive, "w") as z:
                z.writestr("Top/jelibox_windows_installation.bat",
                           '@echo off\r\necho %JELIBOX_WEB%> "%~dp0web_seen.txt"\r\n')
                z.writestr("Top/utils/AnnotationGUI.py", "")
        else:
            archive = os.path.join(work, "a.tar.gz")
            stub = os.path.join(work, "stub.bash")
            with open(stub, "w", newline="\n") as f:
                f.write('echo "$JELIBOX_WEB" > "$(dirname "$0")/web_seen.txt"\n')
            with tarfile.open(archive, "w:gz") as t:
                t.add(stub, arcname="Top/jelibox_linux_installation.bash")
        return work, archive

    def seen(self, home):
        import time
        path = os.path.join(home, "web_seen.txt")
        for _ in range(60):                                        # the .bat is started in its own window
            if os.path.exists(path) and os.path.getsize(path):
                with open(path) as f:
                    return f.read().strip()
            time.sleep(0.25)
        return None

    def env(self, archive, home, **extra):
        env = dict(os.environ, JELIBOX_ARCHIVE=archive, JELIBOX_HOME=home, JELIBOX_GPU="cpu")
        for name in ("JELIBOX_NO_INSTALL", "JELIBOX_WEB", "JELIBOX_VERSION"):
            env.pop(name, None)
        env.update(extra)
        return env

    @unittest.skipUnless(HAVE_PS, "Windows PowerShell only")
    def test_powershell_default_is_the_previous_answer_and_a_preset_wins(self):
        work, archive = self.make_archive("ps")
        home = os.path.join(work, "home")
        script = os.path.join(REPO, "install.ps1")
        run = lambda **extra: subprocess.run(
            ["powershell", "-NoProfile", "-Command", f"Get-Content -Raw -LiteralPath '{script}' | Invoke-Expression"],
            env=self.env(archive, home, **extra), capture_output=True, text=True, encoding="utf-8", errors="replace",
            stdin=subprocess.DEVNULL)
        r = run()                                                  # fresh install, nobody to ask: app + web
        self.assertIn("Installing: app + web version", r.stdout, r.stdout + r.stderr)
        self.assertEqual(self.seen(home), "yes")
        os.remove(os.path.join(home, "web_seen.txt"))
        with open(os.path.join(home, ".jelibox-web"), "w") as f:   # the last install was app only
            f.write("no\n")
        r = run()
        self.assertIn("Installing: app only", r.stdout, r.stdout + r.stderr)
        self.assertEqual(self.seen(home), "no")
        os.remove(os.path.join(home, "web_seen.txt"))
        r = run(JELIBOX_WEB="yes")
        self.assertEqual(self.seen(home), "yes", r.stdout + r.stderr)

    @unittest.skipUnless(find_bash() and shutil.which("tar"), "no working bash/tar available")
    def test_bash_passes_a_preset_answer_on(self):
        work, archive = self.make_archive("sh")
        home = os.path.join(work, "home")
        r = subprocess.run([find_bash(), "-c", 'cat "$1" | bash', "_", os.path.join(REPO, "install.sh")],
                           env=self.env(archive, home, JELIBOX_WEB="no"), capture_output=True, text=True,
                           encoding="utf-8", errors="replace", stdin=subprocess.DEVNULL)
        self.assertIn("Installing: app only", r.stdout, r.stdout + r.stderr)
        self.assertEqual(self.seen(home), "no")


if __name__ == "__main__":
    unittest.main()
