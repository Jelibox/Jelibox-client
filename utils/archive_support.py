"""
Archives in a dataset folder: .zip, .rar and .7z, also split into parts, wherever they sit in the folder tree.

  find_archives   the archives of a folder tree, a split one counted once (its parts are grouped; a missing part is an error)
  unpack_all      unpacks every archive (and the archives found inside them) and returns the folders it made

.zip is read with Python's zipfile. .rar and .7z (and any split set) are read by 7-Zip: the `py7zz` package brings
the 7-Zip program (7zz) with it, so nothing is installed system-wide; a 7zz / 7z found on PATH is the fallback.
No tkinter, web or torch in here: the desktop picker and the web upload both use it.

Split sets it understands: name.part1.rar, name.part2.rar ... (also part01) / name.rar + name.r00, name.r01 ... /
name.7z.001, name.7z.002 ... (also .zip.001 and .rar.001). Give it any one part's folder: the first part is the one
that is opened, 7-Zip reads the others by itself.
"""
import importlib.util
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile

RESERVED = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}
SPARE_BYTES = 1 << 30                     # always leave 1 GB free on the disk
MAX_FILES = 300_000
MAX_DEPTH = 3                             # archives inside archives inside archives
STALE_SECONDS = 24 * 3600
ARCHIVE_EXTENSIONS = (".zip", ".rar", ".7z")

_PART_RAR = re.compile(r"^(.+)\.part0*(\d+)\.rar$")
_SPLIT = re.compile(r"^(.+\.(?:7z|zip|rar))\.(\d{3})$")
_OLD_VOLUME = re.compile(r"^(.+)\.r(\d{2})$")


def safe_parts(path):
    """The folders and file name of a relative path from the browser, or ValueError. No '..', no drive letters, no
    control characters, no Windows device names."""
    parts = []
    for part in str(path or "").replace("\\", "/").split("/"):
        if part in ("", "."):
            continue
        if part == ".." or ":" in part or any(ord(c) < 32 for c in part) or len(part) > 200:
            raise ValueError(f"Not an allowed file name: {path!r}")
        if part.split(".")[0].upper() in RESERVED or part != part.rstrip(" ."):
            raise ValueError(f"Not an allowed file name: {path!r}")
        parts.append(part)
    if not parts:
        raise ValueError("The file has no name.")
    return parts


def free_bytes(folder):
    try:
        return shutil.disk_usage(folder).free - SPARE_BYTES
    except OSError:
        return 0


# ------------------------------------------------------------------ finding them

def _group_key(dirpath, name, lower_names):
    """(group key, part number) of a file that belongs to an archive, else None."""
    low = name.lower()
    m = _PART_RAR.match(low)
    if m:
        return (dirpath, m.group(1) + ".part.rar"), int(m.group(2))
    m = _SPLIT.match(low)
    if m:
        return (dirpath, m.group(1)), int(m.group(2))
    m = _OLD_VOLUME.match(low)
    if m:
        if m.group(1) + ".rar" not in lower_names:
            raise ValueError(f"{name} is a part of a split archive, but {m.group(1)}.rar (its first part) is missing.")
        return (dirpath, m.group(1) + ".rar"), int(m.group(2)) + 2        # name.r00 is the second part
    if low.endswith(ARCHIVE_EXTENSIONS):
        return (dirpath, low), 1
    return None


def find_archives(folder):
    """[(first_file, [every part's file])] of the archives under `folder`, sorted. Raises ValueError for a split
    archive that has a part missing."""
    groups = {}
    for dirpath, _dirs, files in os.walk(folder):
        lower_names = {f.lower() for f in files}
        for name in files:
            found = _group_key(dirpath, name, lower_names)
            if found:
                key, number = found
                groups.setdefault(key, {})[number] = os.path.join(dirpath, name)
    result = []
    for (dirpath, key), parts in groups.items():
        top = max(parts)
        missing = [n for n in range(1, top + 1) if n not in parts]
        if missing:
            shown = ", ".join(str(n) for n in missing[:6])
            raise ValueError(f"The split archive '{_stem(parts[min(parts)])}' has part(s) missing (part {shown}); "
                             f"found {len(parts)} of {top}. Put all the parts in the same folder.")
        result.append((parts[1], [parts[n] for n in sorted(parts)]))
    return sorted(result)


def _stem(path):
    """name.part1.rar / name.rar / name.7z.001 -> name."""
    name = os.path.basename(path)
    low = name.lower()
    for pattern in (_PART_RAR, _SPLIT, _OLD_VOLUME):
        m = pattern.match(low)
        if m:
            name = name[:len(m.group(1))]
            break
    else:
        name = os.path.splitext(name)[0]
    base, ext = os.path.splitext(name)
    return base if ext.lower() in ARCHIVE_EXTENSIONS else name


# ------------------------------------------------------------------ the 7-Zip program

def find_7z():
    """The 7-Zip console program: the one py7zz brings along, else a 7zz / 7z on PATH, else None."""
    spec = importlib.util.find_spec("py7zz")
    if spec and spec.origin:
        folder = os.path.join(os.path.dirname(spec.origin), "bin")
        for name in ("7zz.exe", "7zz", "7z.exe"):
            path = os.path.join(folder, name)
            if os.path.isfile(path):
                return path
    for name in ("7zz", "7z"):
        path = shutil.which(name)
        if path:
            return path
    return None


def _run_7z(args, tick=None):
    """(returncode, output) of the 7-Zip program. The output goes to a file, not a pipe (no reader threads), and
    `tick()` is called while it works so a window can stay alive."""
    exe = find_7z()
    if not exe:
        raise ValueError("Opening .rar and .7z files needs the 7-Zip engine that comes with the 'py7zz' package. "
                         "Run the Jelibox installer again to add it.")
    flags = 0x08000000 if os.name == "nt" else 0                      # no console window flashing up
    with tempfile.TemporaryFile() as out:
        proc = subprocess.Popen([exe, *args], stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT,
                                creationflags=flags)
        while proc.poll() is None:
            if tick:
                tick()
            time.sleep(0.05)
        out.seek(0)
        return proc.returncode, out.read().decode("utf-8", "replace")


def _problem(name, output):
    low = output.lower()
    if "enter password" in low or "wrong password" in low or "encrypted = +" in low or output == "password":
        return f"{name} is protected with a password. Jelibox cannot open it: unpack it yourself first."
    if "unsupported method" in low:
        return (f"{name} uses a compression the 7-Zip program found here cannot read. "
                f"Run the Jelibox installer again to get the one that comes with Jelibox (py7zz).")
    if "unexpected end" in low or "cannot find volume" in low or "missing volume" in low:
        return f"{name} is incomplete (a part is missing or the download was cut short)."
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    return f"Could not unpack {name}: {lines[-1] if lines else 'unknown error'}"


def _list_7z(entry, tick=None):
    """(file count, bytes) an archive holds; ValueError when it cannot be read or is encrypted."""
    name = os.path.basename(entry)
    code, text = _run_7z(["l", "-slt", "-bd", "--", entry], tick)
    if code >= 2:
        raise ValueError(_problem(name, text))
    body = text.split("\n----------", 1)
    if len(body) < 2:
        return 0, 0
    count = size = 0
    for block in body[1].split("\n\n"):
        fields = dict(line.split(" = ", 1) for line in block.splitlines() if " = " in line)
        if "Path" not in fields or fields.get("Folder") == "+":
            continue
        if fields.get("Encrypted") == "+":
            raise ValueError(_problem(name, "password"))
        count += 1
        try:
            size += int(fields.get("Size") or 0)
        except ValueError:
            pass
    return count, size


# ------------------------------------------------------------------ unpacking

def _check_room(name, count, size, where):
    if count > MAX_FILES:
        raise ValueError(f"{name} holds too many files.")
    if size > free_bytes(where):
        raise ValueError(f"Not enough free disk space to unpack {name}.")


def _unzip(path, target):
    name = os.path.basename(path)
    try:
        archive = zipfile.ZipFile(path)
    except zipfile.BadZipFile:
        raise ValueError(f"{name} is not a valid zip file.")
    with archive:
        infos = [i for i in archive.infolist() if not i.is_dir()]
        _check_room(name, len(infos), sum(i.file_size for i in infos), os.path.dirname(path))
        root = os.path.realpath(target)
        for info in infos:
            try:
                out = os.path.join(target, *safe_parts(info.filename))
            except ValueError:
                continue                                    # a name that is not allowed is skipped, never written
            if os.path.commonpath([os.path.realpath(out), root]) != root:
                continue
            os.makedirs(os.path.dirname(out), exist_ok=True)
            with archive.open(info) as src, open(out, "wb") as dst:
                shutil.copyfileobj(src, dst)


def _unpack_7z(entry, target, tick=None):
    name = os.path.basename(entry)
    count, size = _list_7z(entry, tick)
    _check_room(name, count, size, os.path.dirname(os.path.abspath(target)))
    os.makedirs(target, exist_ok=True)
    code, text = _run_7z(["x", "-y", "-bd", "-aoa", "-o" + os.path.abspath(target), "--", os.path.abspath(entry)], tick)
    if code >= 2:                                                   # 1 = a warning only
        raise ValueError(_problem(name, text))
    _drop_links(target)


def _drop_links(target):
    """Links an archive may carry are removed: a name inside the unpacked folder must never lead outside it."""
    for dirpath, dirs, files in os.walk(target):
        for name in dirs + files:
            path = os.path.join(dirpath, name)
            if os.path.islink(path):
                os.remove(path)


def _free_name(path):
    while os.path.exists(path):
        path += "_x"
    return path


def unpack_all(folder, dest=None, remove=False, tick=None, log=None):
    """Unpack every archive under `folder`, then the archives inside those, up to MAX_DEPTH levels.

    dest=None: each archive is unpacked into a folder of its name next to it (the web's staging folder);
    dest=<folder>: into numbered folders inside `dest` and `folder` is not touched (the desktop's temp folder).
    remove=True deletes the archive files once unpacked. `log(text)` hears what is being unpacked.
    Returns the folders that were made (to be scanned together with `folder`). Raises ValueError with a message for
    the person: missing part, password, damaged file, no room, no 7-Zip."""
    made, queue, number = [], [(folder, 0)], 0
    while queue:
        root, depth = queue.pop(0)
        archives = find_archives(root)
        if archives and depth >= MAX_DEPTH:
            raise ValueError("Archives are nested too deeply inside each other.")
        for entry, parts in archives:
            number += 1
            if log:
                log(f"Unpacking {os.path.basename(entry)}...")
            if dest is None:
                target = _free_name(os.path.join(os.path.dirname(entry), _stem(entry)))
            else:
                os.makedirs(dest, exist_ok=True)
                target = os.path.join(dest, f"{number:02d}_{_stem(entry)[:60]}")
            if entry.lower().endswith(".zip") and len(parts) == 1:
                os.makedirs(target, exist_ok=True)
                _unzip(entry, target)
            else:
                _unpack_7z(entry, target, tick)
            made.append(target)
            queue.append((target, depth + 1))
            if remove:
                for part in parts:
                    os.remove(part)
    return made


def forget_stale(folder):
    """Delete the unpacked leftovers in `folder` that are older than a day (a window that was never closed properly)."""
    try:
        names = os.listdir(folder)
    except OSError:
        return
    now = time.time()
    for name in names:
        path = os.path.join(folder, name)
        try:
            if now - os.path.getmtime(path) > STALE_SECONDS:
                shutil.rmtree(path, ignore_errors=True)
        except OSError:
            pass
