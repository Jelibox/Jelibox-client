"""
Stream for the web server: the workspace's trained model runs on a video (an uploaded file or a downloaded YouTube
video, both kept in temp/stream) and the annotated pictures go to the browser as a motion-JPEG feed.

ONE stream runs at a time and it holds the Label Assistant's model slot (`assistant.acquire`), so it never runs next
to an Infer or an Auto-annotate. The frames are made on a thread of their own, newest frame only: a slow browser skips
frames, it never makes the model wait. A stream nobody has looked at for IDLE_SECONDS stops by itself (a closed tab).
Nothing here imports torch, ultralytics or cv2 until a stream starts.
"""
import collections
import contextlib
import importlib.util
import os
import secrets
import threading
import time

from .. import stream_support as ss
from . import assistant
from .editing import EditError

IDLE_SECONDS = 45
KEEP_SECONDS = 600                       # an ended stream can still be read for ten minutes (its last numbers)
MAX_WIDTH = 1280                         # wider pictures are shrunk before they are sent
CONF_RANGE = (0.05, 0.95)


class Session:
    def __init__(self, user, workspace, video, path, model_file, conf):
        self.id = secrets.token_hex(8)
        self.user, self.workspace, self.video = user, workspace, video
        self.path, self.model_file, self.conf = path, model_file, conf
        self.state = "loading"           # loading | running | ended | error
        self.error = None
        self.note = None
        self.frames = self.detections = self.loops = 0
        self.fps = 0.0
        self.counts = {}                 # classes seen in the newest picture
        self.size = (0, 0)
        self.jpeg, self.seq = None, 0
        self.last_seen = time.time()
        self.finished = None
        self._stop = threading.Event()

    @property
    def alive(self):
        return self.state in ("loading", "running")

    def touch(self):
        self.last_seen = time.time()

    def stop(self):
        self._stop.set()

    def stopped(self):
        return self._stop.is_set()

    def public(self):
        return {"id": self.id, "user": self.user, "workspace": self.workspace, "video": self.video, "state": self.state,
                "error": self.error, "note": self.note, "frames": self.frames, "detections": self.detections,
                "loops": self.loops, "fps": round(self.fps, 1), "counts": self.counts, "conf": self.conf,
                "width": self.size[0], "height": self.size[1]}


def default_loader(path):
    from ultralytics import YOLO
    return YOLO(path)


def _cleanup():
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass


def run_session(session, load_model):
    """The stream thread: read a picture, run the model, keep the annotated JPEG. Ends on stop, idle or an error."""
    import cv2
    cap = None
    try:
        model = load_model(session.model_file)
        cap = cv2.VideoCapture(session.path)
        if not cap.isOpened():
            raise ss.StreamError("This video could not be opened. It may be damaged, or this computer cannot decode it.")
        fps = cap.get(cv2.CAP_PROP_FPS)
        step = 1.0 / min(60.0, max(1.0, fps if fps and fps == fps else 25.0))  # play at the video's own speed
        session.state = "running"
        stamps, next_at, pass_frames = collections.deque(maxlen=30), time.time(), 0
        while not session.stopped():
            if time.time() - session.last_seen > IDLE_SECONDS:
                session.note = "Nobody was watching, so the stream stopped."
                break
            ok, frame = cap.read()
            if not ok:
                if pass_frames == 0:
                    raise ss.StreamError("No picture could be read from this video.")
                cap.release()                                                  # the video ended: start it again
                cap = cv2.VideoCapture(session.path)
                if not cap.isOpened():
                    raise ss.StreamError("The video file is gone.")
                session.loops += 1
                pass_frames = 0
                continue
            pass_frames += 1
            result = model(frame, conf=session.conf, verbose=False)[0]
            picture = result.plot()
            counts = {}
            boxes = getattr(result, "boxes", None)
            if boxes is not None and len(boxes):
                for cls in boxes.cls:
                    name = str(result.names[int(cls)])
                    counts[name] = counts.get(name, 0) + 1
            height, width = picture.shape[:2]
            if width > MAX_WIDTH:
                picture = cv2.resize(picture, (MAX_WIDTH, round(height * MAX_WIDTH / width)))
            done, jpeg = cv2.imencode(".jpg", picture, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not done:
                continue
            now = time.time()
            stamps.append(now)
            session.counts, session.size = counts, (width, height)
            session.detections += sum(counts.values())
            session.frames += 1
            if len(stamps) > 1 and stamps[-1] > stamps[0]:
                session.fps = (len(stamps) - 1) / (stamps[-1] - stamps[0])
            session.jpeg = jpeg.tobytes()
            session.seq += 1
            next_at = max(next_at + step, now)                                 # never "catch up" in a burst
            if next_at > now:
                session._stop.wait(next_at - now)
        session.state = "ended"
    except Exception as exc:                                                   # the page shows the message
        session.error = str(exc) or exc.__class__.__name__
        session.state = "error"
    finally:
        if cap is not None:
            cap.release()
        session.finished = time.time()
        session.jpeg = None
        assistant.release()
        _cleanup()


class Streams:
    def __init__(self, base_dir, load_model=None):
        self.base_dir = base_dir
        self.folder = ss.stream_folder(base_dir)
        self.load_model = load_model or default_loader
        self._sessions = {}
        self._lock = threading.Lock()
        self._transfers = 0

    # ---------------------------------------------------------------- what the window shows
    def current(self):
        with self._lock:
            return next((s for s in self._sessions.values() if s.alive), None)

    def _keep(self):
        session = self.current()
        return {session.video} if session else set()

    def info(self, ws):
        session = self.current()
        model = assistant.Ctx(ws).custom_model_file()
        return {"model": os.path.basename(model) if model else None, "videos": ss.list_videos(self.base_dir),
                "cache": ss.cache_summary(self.base_dir, self._keep()), "session": session.public() if session else None,
                "youtube": importlib.util.find_spec("yt_dlp") is not None}

    # ---------------------------------------------------------------- the stream
    def start(self, user, ws, video, conf):
        try:
            path = ss.video_path(self.base_dir, video)
        except ValueError as exc:
            raise EditError(str(exc)) from exc
        try:
            conf = 0.4 if conf is None else float(conf)
        except (TypeError, ValueError):
            raise EditError("The confidence must be a number.") from None
        if not CONF_RANGE[0] <= conf <= CONF_RANGE[1]:
            raise EditError(f"The confidence must be between {CONF_RANGE[0]} and {CONF_RANGE[1]}.")
        model_file = assistant.Ctx(ws).custom_model_file()
        if not model_file:
            raise EditError("This workspace has no trained model yet. Train one first (the Train button), or choose a "
                            "model in the Label Assistant.")
        assistant.acquire(user, "Stream")                                      # raises assistant.Busy
        try:
            session = Session(user, ws.name, video, path, model_file, round(conf, 2))
            with self._lock:
                self._prune()
                self._sessions[session.id] = session
            threading.Thread(target=run_session, args=(session, self.load_model), name="stream", daemon=True).start()
        except Exception:
            assistant.release()
            raise
        return session

    def get(self, session_id):
        with self._lock:
            session = self._sessions.get(session_id)
        if session is None:
            raise EditError("That stream does not exist any more.")
        return session

    def _prune(self):
        now = time.time()
        for sid in [i for i, s in self._sessions.items() if s.finished and now - s.finished > KEEP_SECONDS]:
            del self._sessions[sid]

    # ---------------------------------------------------------------- videos and the cache
    @contextlib.contextmanager
    def transfer(self):
        """An upload or a download is writing into the stream folder: the cache is not cleared meanwhile."""
        with self._lock:
            self._transfers += 1
        try:
            yield
        finally:
            with self._lock:
                self._transfers -= 1

    def download(self, job, url):
        """Job body: YouTube -> temp/stream. Returns {name, bytes}, or None when the person stopped it."""
        with self.transfer():
            try:
                path = ss.download_youtube(url, self.folder, lambda pct, text: job.progress(done=int(pct), total=100, text=text),
                                           job.cancelled)
            except ss.Cancelled:
                return None
        return {"name": os.path.basename(path), "bytes": os.path.getsize(path)}

    def clear_cache(self):
        with self._lock:
            if self._transfers:
                raise EditError("An upload or a download is running. Try again when it has finished.")
        return ss.clear_cache(self.base_dir, self._keep())
