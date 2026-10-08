"""
Auto-annotate all images: run the Label Assistant over every image of the current dataset folder in one go.

The runner is independent of the GUI and of the workspace globals: everything it needs (how to read an image,
how to predict, how to load / save annotations) is passed in, so it is easy to test and the GUI can run it on a
worker thread. It never touches Tk.

For every image it
  1. skips it when it already has annotations and `only_unlabeled` is set,
  2. predicts,
  3. MERGES the predictions into the annotations already on the image (same rule as the G key: an existing
     annotation always wins over an overlapping prediction),
  4. saves the VOC XML + YOLO label - only when something was added, so images with no detection stay untouched.
"""
import threading
import time
from dataclasses import dataclass, field

from .annotation_merge import merge, poly_rect


@dataclass
class BatchResult:
    total: int = 0
    processed: int = 0          # images the assistant looked at
    annotated: int = 0          # images that received at least one new annotation
    added: int = 0              # new annotations in total
    skipped_labeled: int = 0    # images skipped because they already had annotations
    no_detection: int = 0       # looked at, nothing found (or everything overlapped existing annotations)
    failed: list = field(default_factory=list)      # [(image name, message)]
    cancelled: bool = False
    seconds: float = 0.0

    def summary(self):
        parts = [f"{self.annotated} of {self.total} images annotated, {self.added} new annotations"]
        if self.skipped_labeled:
            parts.append(f"{self.skipped_labeled} already labelled and skipped")
        if self.no_detection:
            parts.append(f"{self.no_detection} with nothing found")
        if self.failed:
            parts.append(f"{len(self.failed)} failed")
        if self.cancelled:
            parts.append("stopped before the end")
        return "; ".join(parts) + "."


def run_batch(images, *, read_image, predict, load_annotations, save_annotations, classes,
              only_unlabeled=True, progress=None, should_cancel=None):
    """
    images            image file names, in the order to process
    read_image(name)  -> BGR image array, or None when unreadable
    predict(bgr)      -> (predictions, is_polygon) as produced by assistant_providers / inferenceObjectDetection
    load_annotations(name)               -> (bboxes, polygons) currently on disk
    save_annotations(name, shape, bboxes, polygons)
    classes           the workspace's classes; predictions of any other class are dropped
    progress(done, total, name, result)  called before the first image (done=0) and after each one
    should_cancel()   -> True to stop before the next image
    """
    res = BatchResult(total=len(images))
    started = time.time()
    allowed = set(classes)
    if progress:
        progress(0, res.total, "", res)

    for n, name in enumerate(images, 1):
        if should_cancel and should_cancel():
            res.cancelled = True
            break
        try:
            bboxes, polygons = load_annotations(name)
            if only_unlabeled and (bboxes or polygons):
                res.skipped_labeled += 1
            else:
                img = read_image(name)
                if img is None:
                    raise ValueError("could not read the image")
                preds, is_polygon = predict(img)
                preds = [p for p in preds if p["cls"] in allowed]
                res.processed += 1
                if is_polygon:
                    kept = merge(preds, [poly_rect(p[0]) for p in polygons])
                    polygons = list(polygons) + [[p["poly"], p["cls"]] for p in kept]
                else:
                    kept = merge(preds, [tuple(b[:4]) for b in bboxes])
                    bboxes = list(bboxes) + [[*p["rect"], p["cls"]] for p in kept]
                if kept:
                    save_annotations(name, img.shape, bboxes, polygons)
                    res.annotated += 1
                    res.added += len(kept)
                else:
                    res.no_detection += 1
        except Exception as exc:                                   # one bad image must not stop the run
            res.failed.append((name, str(exc)))
        if progress:
            progress(n, res.total, name, res)

    res.seconds = time.time() - started
    return res


class BatchWorker:
    """Runs run_batch on a background thread. The GUI polls `snapshot()` (thread-safe) from its own thread.

    work(status, cancel_flag) is any callable that performs the whole job and returns its result; `status(text)`
    reports a line of text, and `progress(done, total, name, result)` reports image progress.
    """

    def __init__(self, work):
        self._work = work
        self._lock = threading.Lock()
        self._status = ""
        self._progress = (0, 0, "")
        self._result = None
        self._error = None
        self._done = False
        self.cancel_flag = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def cancel(self):
        self.cancel_flag.set()

    def status(self, text):
        with self._lock:
            self._status = text

    def progress(self, done, total, name, _result=None):
        with self._lock:
            self._progress = (done, total, name)

    def _run(self):
        try:
            result = self._work(self.status, self.progress, self.cancel_flag)
            error = None
        except Exception as exc:
            result, error = None, exc
        with self._lock:
            self._result, self._error, self._done = result, error, True

    def snapshot(self):
        """(done, status text, progress (done, total, name), result, error)"""
        with self._lock:
            return self._done, self._status, self._progress, self._result, self._error
