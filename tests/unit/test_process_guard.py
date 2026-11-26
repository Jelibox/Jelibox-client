"""A training worker must not outlive Jelibox, and only one training may run at a time."""
import os
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest

from utils import process_guard as pg

PARENT = textwrap.dedent("""
    import subprocess, sys, time
    from utils import process_guard as pg
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"], **pg.popen_extra())
    pg.bind_to_parent(child)
    print(child.pid, flush=True)
    time.sleep(120)
""")


def wait_until(check, seconds=15):
    end = time.time() + seconds
    while time.time() < end:
        if check():
            return True
        time.sleep(0.2)
    return check()


class OrphanTests(unittest.TestCase):
    def test_the_worker_dies_when_its_parent_is_killed(self):
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        parent = subprocess.Popen([sys.executable, "-c", PARENT], cwd=root, stdout=subprocess.PIPE, text=True)
        child_pid = None
        try:
            child_pid = int(parent.stdout.readline())
            self.assertTrue(pg.pid_alive(child_pid), "the worker started")
            parent.kill()                                          # a crash or a forced close: no chance to clean up
            parent.wait(timeout=10)
            self.assertTrue(wait_until(lambda: not pg.pid_alive(child_pid)),
                            "the worker is still running after its parent died")
        finally:
            parent.kill()
            if child_pid and pg.pid_alive(child_pid):
                if os.name == "nt":
                    subprocess.run(["taskkill", "/PID", str(child_pid), "/F"], stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL)
                else:
                    os.kill(child_pid, 9)

    def test_pid_alive_knows_a_dead_process(self):
        done = subprocess.Popen([sys.executable, "-c", "pass"])
        done.wait()
        self.assertFalse(pg.pid_alive(done.pid))
        self.assertTrue(pg.pid_alive(os.getpid()))
        self.assertFalse(pg.pid_alive(None))
        self.assertFalse(pg.pid_alive(0))


class SpawnTests(unittest.TestCase):
    """Linux ties a child to the thread that started it: a web server's pool threads come and go, so the launcher must not be one."""

    def start_from_a_short_lived_thread(self):
        import threading
        made = []

        def work():
            made.append(pg.spawn(lambda: subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"],
                                                          **pg.popen_extra())))
        thread = threading.Thread(target=work)
        thread.start()
        thread.join()                                           # the thread that asked is gone now
        self.addCleanup(lambda: (made[0].kill(), made[0].wait()))
        return made[0]

    @unittest.skipIf(os.name == "nt", "Windows binds the worker with a job object, not to a thread")
    def test_a_worker_started_from_a_pool_thread_survives_that_thread(self):
        child = self.start_from_a_short_lived_thread()
        time.sleep(1.0)
        self.assertIsNone(child.poll(), "the child was ended with the thread that started it")

    def test_the_caller_gets_what_the_launch_raised(self):
        import threading
        caught = []

        def work():
            try:
                pg.spawn(lambda: subprocess.Popen(["/definitely/not/a/program"]))
            except OSError as exc:
                caught.append(exc)
        thread = threading.Thread(target=work)
        thread.start()
        thread.join()
        self.assertEqual(len(caught), 1)

    def test_the_main_thread_starts_the_process_itself(self):
        child = pg.spawn(lambda: subprocess.Popen([sys.executable, "-c", "pass"]))
        self.assertEqual(child.wait(10), 0)


class LockTests(unittest.TestCase):
    def setUp(self):
        self.base = tempfile.mkdtemp()

    def test_a_running_worker_holds_the_lock_and_a_dead_one_does_not(self):
        self.assertIsNone(pg.running_training(self.base))
        pg.claim(self.base, "cats", os.getpid())
        info = pg.running_training(self.base)
        self.assertEqual((info["workspace"], info["pid"]), ("cats", os.getpid()))
        done = subprocess.Popen([sys.executable, "-c", "pass"])
        done.wait()
        pg.claim(self.base, "dogs", done.pid)
        self.assertIsNone(pg.running_training(self.base), "its worker is gone: the lock is stale")
        self.assertFalse(os.path.exists(os.path.join(self.base, "temp", "training.lock")), "and is removed")

    def test_only_the_owner_releases_it(self):
        pg.claim(self.base, "cats", os.getpid())
        pg.release(self.base, os.getpid() + 1)
        self.assertIsNotNone(pg.running_training(self.base))
        pg.release(self.base, os.getpid())
        self.assertIsNone(pg.running_training(self.base))

    def test_a_damaged_lock_file_is_ignored(self):
        os.makedirs(os.path.join(self.base, "temp"))
        with open(os.path.join(self.base, "temp", "training.lock"), "w") as f:
            f.write("{ not json")
        self.assertIsNone(pg.running_training(self.base))
        pg.release(self.base, 1)


if __name__ == "__main__":
    unittest.main()
