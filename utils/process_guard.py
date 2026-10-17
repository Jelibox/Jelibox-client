"""Keep a training worker from outliving Jelibox, and allow only one training at a time.

A training is its own process (hours long, and it fills the GPU memory). It used to keep running after its window was
closed, after going back to the workspace list, and after Jelibox was closed or crashed:

- `bind_to_parent` makes the operating system end the worker when Jelibox ends, however it ends (Windows: a job object
  that kills its processes when its last handle closes, which also covers the data-loader children; Linux: the child
  gets SIGTERM when its parent dies).
- `training_lock` is a small file (`temp/training.lock`) naming the worker that is running, so a second Jelibox window
  (another workspace) says so instead of starting a second training on the same GPU.
"""
import json
import os
import time

_jobs = []                       # the job handles must stay open: closing one is what kills the worker


def _pdeathsig():
    """preexec_fn on Linux: SIGTERM for the child when its parent thread dies."""
    try:
        import ctypes
        import signal
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGTERM)       # PR_SET_PDEATHSIG
    except Exception:
        pass


def popen_extra():
    """Extra Popen arguments that tie the child to its parent where the platform wants them at launch time."""
    return {} if os.name == "nt" else {"preexec_fn": _pdeathsig}


def bind_to_parent(proc):
    """Windows: put the started process in a kill-on-close job object. True when it worked (always False elsewhere)."""
    if os.name != "nt":
        return False
    try:
        import ctypes
        from ctypes import wintypes

        class BASIC(ctypes.Structure):
            _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                        ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                        ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                        ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                        ("SchedulingClass", wintypes.DWORD)]

        class IO(ctypes.Structure):
            _fields_ = [(n, ctypes.c_uint64) for n in ("ReadOps", "WriteOps", "OtherOps", "ReadBytes", "WriteBytes",
                                                       "OtherBytes")]

        class EXTENDED(ctypes.Structure):
            _fields_ = [("Basic", BASIC), ("Io", IO), ("ProcessMemoryLimit", ctypes.c_size_t),
                        ("JobMemoryLimit", ctypes.c_size_t), ("PeakProcessMemoryUsed", ctypes.c_size_t),
                        ("PeakJobMemoryUsed", ctypes.c_size_t)]

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.CreateJobObjectW.restype = wintypes.HANDLE
        k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        job = k32.CreateJobObjectW(None, None)
        if not job:
            return False
        info = EXTENDED()
        info.Basic.LimitFlags = 0x2000                     # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not k32.SetInformationJobObject(wintypes.HANDLE(job), 9, ctypes.byref(info), ctypes.sizeof(info)):
            return False
        if not k32.AssignProcessToJobObject(wintypes.HANDLE(job), wintypes.HANDLE(int(proc._handle))):
            return False
        _jobs.append(job)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------- one training at a time
def pid_alive(pid):
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        handle = k32.OpenProcess(0x1000, False, pid)        # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            return bool(k32.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259   # STILL_ACTIVE
        finally:
            k32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except PermissionError:
        return True
    except OSError:
        return False


def _lock_path(base_dir):
    return os.path.join(base_dir, "temp", "training.lock")


def running_training(base_dir):
    """{"pid", "workspace", "started"} of the training that is running now, or None. A lock whose worker is gone
    (crash, kill) is stale and removed."""
    path = _lock_path(base_dir)
    try:
        with open(path, "r", encoding="utf-8") as f:
            info = json.load(f)
    except (OSError, ValueError):
        return None
    if isinstance(info, dict) and pid_alive(info.get("pid")):
        return info
    try:
        os.remove(path)
    except OSError:
        pass
    return None


def claim(base_dir, workspace, pid):
    path = _lock_path(base_dir)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"pid": int(pid), "workspace": workspace, "started": time.time()}, f)
    except OSError:
        pass


def release(base_dir, pid):
    """Remove the lock if it is this worker's."""
    path = _lock_path(base_dir)
    try:
        with open(path, "r", encoding="utf-8") as f:
            info = json.load(f)
        if int(info.get("pid", -1)) == int(pid):
            os.remove(path)
    except (OSError, ValueError, TypeError):
        pass
