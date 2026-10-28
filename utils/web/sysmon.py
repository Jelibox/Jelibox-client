"""
Resource monitor of the computer that runs the server: CPU and RAM use, CPU and GPU temperature, GPU use, storage.

Windows and Linux. psutil gives the load, memory and disks (it ships with ultralytics; without it those numbers are
null). Temperatures: Linux reads the kernel's hwmon sensors through psutil; Windows reads the "Thermal Zone
Information" performance counter (no administrator rights, but only some machines expose it). NVIDIA GPUs are asked
through nvidia-smi; on Linux an AMD GPU is read from sysfs. Whatever a machine cannot tell is null and the page says
"not available".

`snapshot(base_dir)` is cheap to call every couple of seconds: the load is a delta since the last call, the slow
sources (PowerShell, nvidia-smi) are cached for a few seconds.
"""
import glob
import os
import shutil
import subprocess
import sys
import threading
import time

FAST_TTL = 1.0                       # seconds a whole snapshot is reused (several browsers poll at once)
GPU_TTL = 2.0
CPU_TEMP_TTL = 5.0                   # PowerShell takes about a second
CPU_TEMP_NAMES = ("coretemp", "k10temp", "zenpower", "cpu_thermal", "cpu-thermal", "acpitz", "soc_thermal")

_lock = threading.Lock()
_cache = {"snap": None, "at": 0.0, "gpu": (0.0, None), "cpu_temp": (0.0, None), "primed": False}


# ------------------------------------------------------------------ pure parsing (tested without the hardware)
def _number(text):
    try:
        return float(str(text).strip())
    except (TypeError, ValueError):
        return None


def parse_nvidia_smi(text):
    """[{name, percent, temp, mem_used, mem_total}] from `nvidia-smi --query-gpu=name,utilization.gpu,temperature.gpu,
    memory.used,memory.total --format=csv,noheader,nounits` (memory in MiB -> bytes). '[N/A]' becomes None."""
    gpus = []
    for line in (text or "").splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5 or not parts[0]:
            continue
        used, total = _number(parts[3]), _number(parts[4])
        gpus.append({"name": parts[0], "percent": _number(parts[1]), "temp": _number(parts[2]),
                     "mem_used": None if used is None else int(used * 1024 * 1024),
                     "mem_total": None if total is None else int(total * 1024 * 1024)})
    return gpus


def pick_cpu_temp(sensors):
    """The CPU temperature (Celsius) out of psutil.sensors_temperatures(): the package / hottest core of a CPU sensor
    chip, else the hottest reading of any chip. None when there is nothing."""
    if not sensors:
        return None
    for want in CPU_TEMP_NAMES:
        for chip, readings in sensors.items():
            if chip.lower() == want:
                values = [r.current for r in readings if r.current is not None and 0 < r.current < 150]
                if values:
                    return max(values)
    values = [r.current for readings in sensors.values() for r in readings if r.current is not None and 0 < r.current < 150]
    return max(values) if values else None


def parse_thermal_zones(text):
    """Celsius from the Windows counter output (one Kelvin value per line); the hottest sensible one, else None."""
    values = []
    for line in (text or "").splitlines():
        kelvin = _number(line)
        if kelvin is not None and 233 < kelvin < 423:               # -40 .. 150 C: anything else is not a temperature
            values.append(kelvin - 273.15)
    return round(max(values), 1) if values else None


def disk_rows(partitions, usage, here):
    """[{mount, percent, used, total, here}] for the real drives; `here` is the path of the Jelibox folder, its drive is
    marked and listed first."""
    rows, seen = [], set()
    here = os.path.abspath(here)
    for part in partitions:
        mount = part["mount"]
        if mount in seen or part.get("fstype") in ("squashfs", "tmpfs", "devtmpfs", "overlay", "") and not part.get("fixed"):
            continue
        seen.add(mount)
        try:
            used, total = usage(mount)
        except OSError:
            continue
        if not total:
            continue
        folder, root = os.path.normcase(here), os.path.normcase(mount)
        inside = folder == root.rstrip(os.sep) or folder.startswith(root.rstrip(os.sep) + os.sep)     # /data holds /data/x, not /database
        rows.append({"mount": mount, "percent": round(100.0 * used / total, 1), "used": used, "total": total,
                     "here": inside, "_len": len(mount) if inside else -1})
    best = max((r["_len"] for r in rows), default=-1)                 # the deepest mount that holds the folder
    for r in rows:
        r["here"] = r["_len"] == best and best >= 0
        del r["_len"]
    rows.sort(key=lambda r: (not r["here"], r["mount"]))
    return rows


# ------------------------------------------------------------------ the real sources
def _quiet():
    return {"creationflags": 0x08000000} if sys.platform == "win32" else {}      # no console window under pythonw


def _run(command, timeout=6):
    try:
        out = subprocess.run(command, capture_output=True, text=True, timeout=timeout, **_quiet())
        return out.stdout if out.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def _psutil():
    try:
        import psutil
        return psutil
    except Exception:
        return None


def _amd_gpus():
    """AMD GPUs on Linux (amdgpu driver): load from gpu_busy_percent, temperature from hwmon."""
    found = []
    for card in sorted(glob.glob("/sys/class/drm/card[0-9]*/device")):
        try:
            with open(os.path.join(card, "vendor")) as f:
                if f.read().strip() != "0x1002":
                    continue
            busy = None
            if os.path.exists(os.path.join(card, "gpu_busy_percent")):
                with open(os.path.join(card, "gpu_busy_percent")) as f:
                    busy = _number(f.read())
            temp = None
            for path in glob.glob(os.path.join(card, "hwmon", "hwmon*", "temp1_input")):
                with open(path) as f:
                    value = _number(f.read())
                temp = None if value is None else value / 1000.0
            used = total = None
            for name, store in (("mem_info_vram_used", "u"), ("mem_info_vram_total", "t")):
                if os.path.exists(os.path.join(card, name)):
                    with open(os.path.join(card, name)) as f:
                        value = _number(f.read())
                    used, total = (value, total) if store == "u" else (used, value)
            found.append({"name": "AMD GPU", "percent": busy, "temp": temp,
                          "mem_used": None if used is None else int(used), "mem_total": None if total is None else int(total)})
        except OSError:
            continue
    return found


def read_gpus():
    exe = shutil.which("nvidia-smi")
    if exe:
        gpus = parse_nvidia_smi(_run([exe, "--query-gpu=name,utilization.gpu,temperature.gpu,memory.used,memory.total",
                                      "--format=csv,noheader,nounits"]))
        if gpus:
            return gpus
    return _amd_gpus() if sys.platform.startswith("linux") else []


def read_cpu_temp():
    ps = _psutil()
    if ps is not None and hasattr(ps, "sensors_temperatures"):
        try:
            temp = pick_cpu_temp(ps.sensors_temperatures())
            if temp is not None:
                return round(temp, 1)
        except Exception:
            pass
    if sys.platform == "win32":
        script = ("try { (Get-Counter '\\Thermal Zone Information(*)\\Temperature' -ErrorAction Stop).CounterSamples "
                  "| ForEach-Object { $_.CookedValue } } catch { }")
        return parse_thermal_zones(_run(["powershell", "-NoProfile", "-NonInteractive", "-Command", script], timeout=10))
    return None


def _slow(key, ttl, read):
    now = time.time()
    at, value = _cache[key]
    if now - at < ttl:
        return value
    value = read()
    _cache[key] = (now, value)
    return value


def snapshot(base_dir):
    """Everything the page shows, as JSON-friendly data."""
    with _lock:
        now = time.time()
        if _cache["snap"] is not None and now - _cache["at"] < FAST_TTL:
            return _cache["snap"]
        ps = _psutil()
        cpu = {"percent": None, "cores": os.cpu_count(), "temp": _slow("cpu_temp", CPU_TEMP_TTL, read_cpu_temp)}
        ram = {"percent": None, "used": None, "total": None}
        disks = []
        if ps is not None:
            if not _cache["primed"]:                                  # the first reading has nothing to compare with
                ps.cpu_percent(interval=0.25)
                _cache["primed"] = True
            cpu["percent"] = ps.cpu_percent(interval=None)
            memory = ps.virtual_memory()
            ram = {"percent": memory.percent, "used": memory.total - memory.available, "total": memory.total}
            parts = [{"mount": p.mountpoint, "fstype": p.fstype, "fixed": "fixed" in (p.opts or "")}
                     for p in ps.disk_partitions(all=False)]
            disks = disk_rows(parts, lambda mount: (lambda u: (u.used, u.total))(ps.disk_usage(mount)), base_dir)
        else:
            try:
                usage = shutil.disk_usage(base_dir)
                disks = [{"mount": os.path.abspath(base_dir), "percent": round(100.0 * usage.used / usage.total, 1),
                          "used": usage.used, "total": usage.total, "here": True}]
            except OSError:
                pass
        snap = {"time": now, "cpu": cpu, "ram": ram, "gpus": _slow("gpu", GPU_TTL, read_gpus), "disks": disks,
                "platform": "windows" if sys.platform == "win32" else "linux" if sys.platform.startswith("linux") else sys.platform}
        _cache["snap"], _cache["at"] = snap, now
        return snap
