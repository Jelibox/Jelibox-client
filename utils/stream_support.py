"""
What the Stream feature shares between the desktop (utils/stream_app.py) and the web (utils/web/streaming.py): the
folder the videos are kept in, the YouTube download and the cache cleaning. No tkinter, torch or web framework here.

Videos live in <Jelibox>/temp/stream. A YouTube video is DOWNLOADED first and read from that file: OpenCV cannot open
the address yt-dlp finds (the address expires and needs the downloader's headers), which is why streaming straight from
the link never worked.
"""
import os
import re
import shutil
import sys
import time
from urllib.parse import urlparse

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v")
YOUTUBE_HOSTS = ("youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com", "youtu.be", "www.youtu.be")
MAX_DOWNLOAD_BYTES = 4 << 30
# One file with picture and sound, so no ffmpeg is needed to join them.
YOUTUBE_FORMAT = "best[height<=720][ext=mp4]/best[ext=mp4]/best[height<=720]/best"
RESERVED = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}
# Leftovers of other features in temp/ that are junk once they are old (same ages the web server uses itself):
# (folder, seconds). A dataset upload nobody imported, an export nobody downloaded.
STALE_FOLDERS = (("uploads", 24 * 3600), ("downloads", 6 * 3600))
STALE_FILES = ("server.log.old",)


class StreamError(Exception):
    """Something the person can read: a bad link, a download that failed."""


class Cancelled(Exception):
    pass


def stream_folder(base_dir):
    return os.path.join(base_dir, "temp", "stream")


def is_video_name(name):
    return str(name).lower().endswith(VIDEO_EXTS)


# ------------------------------------------------------------------ names
def clean_video_name(name, folder):
    """A safe, unused file name for an uploaded video (ValueError when it is not a video). Only the last part of a path
    is kept, odd characters become '_', and a name that is taken gets _2, _3 ..."""
    base = os.path.basename(str(name or "").replace("\\", "/"))
    stem, ext = os.path.splitext(base)
    if ext.lower() not in VIDEO_EXTS:
        raise ValueError("That is not a video file (" + ", ".join(e[1:] for e in VIDEO_EXTS) + ").")
    stem = re.sub(r"[^A-Za-z0-9._ ()-]", "_", stem).strip(" .")[:100] or "video"
    if stem.split(".")[0].upper() in RESERVED:
        stem = "_" + stem
    candidate, n = stem + ext.lower(), 1
    while os.path.exists(os.path.join(folder, candidate)):
        n += 1
        candidate = f"{stem}_{n}{ext.lower()}"
    return candidate


def video_path(base_dir, name):
    """The file of a video that is in the stream folder, or ValueError (no folders, no '..', must exist)."""
    name = str(name or "")
    if not name or os.path.basename(name) != name or name.startswith(".") or not is_video_name(name):
        raise ValueError("Choose a video first.")
    path = os.path.join(stream_folder(base_dir), name)
    if not os.path.isfile(path):
        raise ValueError("That video is not on the server any more.")
    return path


def list_videos(base_dir):
    """[{name, bytes, mtime}] newest first."""
    folder, rows = stream_folder(base_dir), []
    try:
        names = os.listdir(folder)
    except OSError:
        return rows
    for name in names:
        path = os.path.join(folder, name)
        try:
            if is_video_name(name) and os.path.isfile(path):
                rows.append({"name": name, "bytes": os.path.getsize(path), "mtime": os.path.getmtime(path)})
        except OSError:
            pass
    return sorted(rows, key=lambda r: -r["mtime"])


# ------------------------------------------------------------------ YouTube
def check_youtube_url(url):
    """The cleaned address, or ValueError. Only YouTube: the server fetches this address, so it must not be able to
    reach other computers on the network."""
    url = str(url or "").strip()
    try:
        parsed = urlparse(url)
        host = (parsed.hostname or "").lower()
    except ValueError:
        host, parsed = "", None
    if parsed is None or parsed.scheme not in ("http", "https") or host not in YOUTUBE_HOSTS:
        raise ValueError("Paste a YouTube link (youtube.com or youtu.be).")
    return url


def friendly_error(exc):
    """What went wrong with a download, in words. The last line of yt-dlp's message is kept for the details."""
    text = re.sub(r"\x1b\[[0-9;]*m", "", str(exc)).strip()
    detail = text.splitlines()[-1].replace("ERROR: ", "")[:200] if text else exc.__class__.__name__
    low = text.lower()
    if "403" in low or "forbidden" in low:
        return ("YouTube refused to send this video (HTTP 403). yt-dlp may be out of date "
                "(update it with: python -m pip install -U yt-dlp), or YouTube is limiting this computer. "
                "You can upload the video file instead.")
    if "requested format is not available" in low:
        return ("YouTube offered no format of this video that can be downloaded here. Usually yt-dlp is out of date "
                "(python -m pip install -U yt-dlp) or has no JavaScript runtime. You can upload the video file instead.")
    if "not a bot" in low or "sign in to confirm" in low:
        return "YouTube wants this computer to prove it is not a robot, so it will not hand over this video. Upload the video file instead."
    if "private video" in low or "unavailable" in low or "members-only" in low or ("age" in low and "restrict" in low):
        return f"YouTube says this video cannot be watched here: {detail}"
    if "urlopen error" in low or "getaddrinfo" in low or "name or service not known" in low or "timed out" in low:
        return f"This computer could not reach YouTube ({detail}). Check its internet connection and DNS."
    return f"The video could not be downloaded: {detail}"


def deno_path():
    """The deno of the `deno` pip package (requirements.txt), which lives next to this environment's python. The
    environment is never activated, so that folder is not on PATH and yt-dlp is pointed at it. None: yt-dlp looks on PATH."""
    path = os.path.join(os.path.dirname(sys.executable), "deno.exe" if os.name == "nt" else "deno")
    return path if os.path.isfile(path) else None


class _Log:
    """yt-dlp's logger: keeps its warnings, so a failed download can pass on what yt-dlp itself said was missing."""

    def __init__(self):
        self.warnings = []

    def debug(self, message):
        pass

    def info(self, message):
        pass

    def warning(self, message):
        self.warnings.append(str(message))

    def error(self, message):
        pass

    def hint(self):
        if any("JavaScript runtime" in w for w in self.warnings):
            return (" yt-dlp also says no JavaScript runtime was found, and YouTube downloads need one (deno): run the "
                    "Jelibox installer again (it installs deno into Jelibox's own environment), "
                    "see https://github.com/yt-dlp/yt-dlp/wiki/EJS")
        return ""


def download_youtube(url, folder, progress=None, cancelled=None):
    """Download a YouTube video (up to 720p, one file) into `folder` and return its path.
    progress(percent, text) is called while it downloads; when cancelled() turns true, Cancelled is raised and the
    unfinished files are removed. Raises StreamError with a readable message."""
    url = check_youtube_url(url)
    try:
        import yt_dlp
    except ImportError as exc:
        raise StreamError("yt-dlp is not installed on this computer (python -m pip install yt-dlp).") from exc
    os.makedirs(folder, exist_ok=True)
    before = set(os.listdir(folder))

    def hook(d):
        if cancelled and cancelled():
            raise Cancelled()
        if progress and d.get("status") == "downloading":
            done, total = d.get("downloaded_bytes") or 0, d.get("total_bytes") or d.get("total_bytes_estimate") or 0
            if total:
                progress(min(99, 100.0 * done / total), f"Downloading from YouTube: {done / 1e6:.0f} of {total / 1e6:.0f} MB")
            else:
                progress(0, f"Downloading from YouTube: {done / 1e6:.0f} MB")

    log = _Log()
    options = {"format": YOUTUBE_FORMAT, "outtmpl": os.path.join(folder, "yt_%(id)s.%(ext)s"), "noplaylist": True,
               "logger": log, "noprogress": True, "restrictfilenames": True,
               "max_filesize": MAX_DOWNLOAD_BYTES, "socket_timeout": 30, "retries": 3, "progress_hooks": [hook]}
    deno = deno_path()
    if deno:
        options["js_runtimes"] = {"deno": {"path": deno}}
    try:
        with yt_dlp.YoutubeDL(options) as ydl:
            info = ydl.extract_info(url, download=True)
            done = (info.get("requested_downloads") or [{}])[0].get("filepath") or ydl.prepare_filename(info)
    except Exception as exc:
        for name in set(os.listdir(folder)) - before:                     # the unfinished pieces
            try:
                os.remove(os.path.join(folder, name))
            except OSError:
                pass
        if isinstance(exc, Cancelled) or (cancelled and cancelled()):
            raise Cancelled() from exc
        raise StreamError(friendly_error(exc) + log.hint()) from exc
    if not done or not os.path.isfile(done):
        raise StreamError(f"The video was not saved: it is larger than {MAX_DOWNLOAD_BYTES >> 30} GB, or YouTube sent nothing.")
    return done


# ------------------------------------------------------------------ the cache: videos and junk in temp/
def _size(path):
    if os.path.islink(path):
        return 0
    if os.path.isfile(path):
        try:
            return os.path.getsize(path)
        except OSError:
            return 0
    total = 0
    for folder, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(folder, name))
            except OSError:
                pass
    return total


def _targets(base_dir, keep=()):
    """(videos, junk, kept): paths that clear_cache removes, and the names it leaves because they are in use."""
    temp, videos, junk, kept = os.path.join(base_dir, "temp"), [], [], []
    folder = stream_folder(base_dir)
    try:
        names = sorted(os.listdir(folder))
    except OSError:
        names = []
    for name in names:
        if name in keep:
            kept.append(name)
        elif is_video_name(name) and os.path.isfile(os.path.join(folder, name)):
            videos.append(os.path.join(folder, name))
        else:                                                              # .part, .ytdl, thumbnails ...
            junk.append(os.path.join(folder, name))
    now = time.time()
    for sub, seconds in STALE_FOLDERS:
        try:
            entries = os.listdir(os.path.join(temp, sub))
        except OSError:
            continue
        for name in entries:
            path = os.path.join(temp, sub, name)
            try:
                if now - os.path.getmtime(path) > seconds:
                    junk.append(path)
            except OSError:
                pass
    for name in STALE_FILES:
        if os.path.isfile(os.path.join(temp, name)):
            junk.append(os.path.join(temp, name))
    return videos, junk, kept


def cache_summary(base_dir, keep=()):
    """What clear_cache would remove now: {videos, video_bytes, junk, junk_bytes, kept}."""
    videos, junk, kept = _targets(base_dir, keep)
    return {"videos": len(videos), "video_bytes": sum(_size(p) for p in videos),
            "junk": len(junk), "junk_bytes": sum(_size(p) for p in junk), "kept": kept}


def clear_cache(base_dir, keep=()):
    """Delete every video in temp/stream (but the names in `keep`: the one that plays now) and the junk: unfinished
    downloads there, and old leftovers of uploads and exports in temp/. Nothing else in temp/ is touched (not the
    training lock, not server.log). Returns {removed, bytes, failed, kept}."""
    videos, junk, kept = _targets(base_dir, keep)
    removed = freed = failed = 0
    root = os.path.realpath(os.path.join(base_dir, "temp"))
    for path in videos + junk:
        if os.path.commonpath([os.path.realpath(os.path.dirname(path)), root]) != root:
            failed += 1                                                    # never delete outside temp/
            continue
        size = _size(path)
        try:
            if os.path.isdir(path) and not os.path.islink(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
        except OSError:
            failed += 1
            continue
        removed += 1
        freed += size
    return {"removed": removed, "bytes": freed, "failed": failed, "kept": kept}
