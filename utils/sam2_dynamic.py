"""
SAM 2 Dynamic mode: Ultralytics' SAM2DynamicInteractivePredictor, a training-free SAM 2 that learns from
reference images. Every image you annotate and leave becomes a reference; the next time you press Infer, SAM 2
finds the same kinds of objects in the new image.

How it works, and the one rule this module keeps:

  * The predictor has a memory bank: one entry per reference image, holding the objects of that image
    (one slot per workspace class). Querying an image without prompts conditions it on every entry.
  * ReferenceStore keeps ONE record per image, keyed by the image's path. Leaving an image replaces its
    record whole, so the same annotation can never be counted twice, however often you go back and forth.
  * The memory bank is never appended to blindly: it is rebuilt from the store each time (one entry per
    stored image), and encoded entries are cached per image so only new or edited images are re-encoded.

What a reference tells SAM 2: for every class, ONE mask that covers ALL the objects of that class in the image.
Polygons are filled exactly. A box is first turned into the object's silhouette by SAM 2 itself (its normal
one-image mode, box prompt), because a plain rectangle also teaches the background and the trouser hem inside it
as "part of the object" - measured: loose boxes then gave sloppy or merged results. Marking every object matters:
SAM 2 remembers whatever is not marked in a reference as "not the object", so an unmarked second worker would be
skipped later.

Modes follow the Mode button: in polygon mode the references are the polygons and the output is polygons;
in box mode the references are the boxes (a polygon counts through its bounding box) and the output is boxes.
One object slot per class: when a class appears several times, SAM 2 segments them as pieces of one mask and
each separate piece becomes its own annotation.

Dependency-free on purpose until a predictor is built: torch and ultralytics are imported lazily.
"""
import gc
import os
from collections import OrderedDict

from .annotation_merge import poly_rect

MODELS = ["sam2.1_t.pt", "sam2.1_s.pt", "sam2.1_b.pt", "sam2.1_l.pt"]
MODEL_INFO = {                      # file -> (download size in MB, what it is for)
    "sam2.1_t.pt": (75, "tiny - fastest, fine on CPU"),
    "sam2.1_s.pt": (92, "small"),
    "sam2.1_b.pt": (162, "base - balanced"),
    "sam2.1_l.pt": (224, "large - most accurate, wants a GPU"),
}
IMAGE_SIZES = [512, 768, 1024]
MIN_SIDE = 8                        # a reference box smaller than this (px) is noise, not an object
MIN_PIECE_FRACTION = 0.0004         # a piece of a predicted mask smaller than this share of the image is dropped
MAX_PIECES_PER_CLASS = 20


IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')


def workspace_images(datasets_root, workspace):
    """[{'source': image path, 'base': name without extension}] of every dataset folder of the workspace."""
    records = []
    prefix = workspace + "-"
    if not os.path.isdir(datasets_root):
        return records
    for folder in sorted(os.listdir(datasets_root)):
        full = os.path.join(datasets_root, folder)
        if folder.startswith(prefix) and os.path.isdir(full):
            for name in sorted(os.listdir(full)):
                if name.lower().endswith(IMAGE_EXTS):
                    records.append({"source": os.path.join(full, name), "base": os.path.splitext(name)[0]})
    return records


def weights_path(base_dir, model):
    return os.path.join(base_dir, "models", "_sam2", model)


def model_size_mb(model):
    return MODEL_INFO.get(model, (0, ""))[0]


def _read_bgr(path):
    """BGR image or None (also works for non-ASCII Windows paths)."""
    try:
        import cv2
        import numpy as np
        data = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
    except OSError:
        return None


# ----------------------------------------------------------------------
#  The references
# ----------------------------------------------------------------------
def _key(path):
    return os.path.normcase(os.path.abspath(path))


class ReferenceStore:
    """{image: [(class name, 'box', (x1, y1, x2, y2)) | (class name, 'poly', [(x, y), ...]), ...]}
    One record per image, newest last."""

    def __init__(self):
        self._records = OrderedDict()           # key -> (path, items)

    def __len__(self):
        return len(self._records)

    def __contains__(self, path):
        return _key(path) in self._records

    def clear(self):
        self._records.clear()

    def forget(self, path):
        self._records.pop(_key(path), None)

    def items_for(self, path):
        record = self._records.get(_key(path))
        return list(record[1]) if record else []

    def remember(self, path, bboxes, polygons):
        """Store `path`'s annotations, replacing whatever was stored for it before. An image without usable
        annotations is dropped from the store. Returns the number of annotations kept."""
        items = []
        for box in bboxes:
            x1, y1, x2, y2, cls = box[:5]
            items.append((cls, "box", (float(x1), float(y1), float(x2), float(y2))))
        for poly in polygons:
            points, cls = poly[0], poly[1]
            if len(points) >= 3:
                items.append((cls, "poly", [(float(x), float(y)) for x, y in points]))
        items = [it for it in items if _big_enough(it)]
        key = _key(path)
        self._records.pop(key, None)            # re-inserting puts it last: the newest reference
        if items:
            self._records[key] = (path, items)
        return len(items)

    def select(self, limit, classes, exclude=None, polygon=False):
        """The newest `limit` stored images as [(path, items)], keeping only annotations of `classes`.
        polygon=True keeps polygons only; otherwise boxes (a polygon then counts through its bounding box).
        `exclude` (the image about to be annotated) is never its own reference."""
        skip = _key(exclude) if exclude else None
        picked = []
        for key, (path, items) in reversed(self._records.items()):
            if key == skip:
                continue
            kept = []
            for cls, kind, geometry in items:
                if cls not in classes:
                    continue
                if polygon:
                    if kind == "poly":
                        kept.append((cls, kind, geometry))
                elif kind == "box":
                    kept.append((cls, kind, geometry))
                else:
                    kept.append((cls, "box", tuple(float(v) for v in poly_rect(geometry))))
            if kept:
                picked.append((path, kept))
            if len(picked) >= limit:
                break
        picked.reverse()
        return picked

    def load_labeled(self, records, voc_folder, limit):
        """Fill the store from images that are already annotated on disk (newest XML first).
        records: [{'source': image path, 'base': file name without extension}]. Returns how many were added."""
        from .file_handler import _parse_voc_objects
        import xml.etree.ElementTree as ET
        by_base = {r["base"]: r["source"] for r in records}
        candidates = []
        for name in os.listdir(voc_folder) if os.path.isdir(voc_folder) else []:
            base, ext = os.path.splitext(name)
            if ext.lower() == ".xml" and base in by_base:
                candidates.append((os.path.getmtime(os.path.join(voc_folder, name)), base))
        added = 0
        for _mtime, base in sorted(candidates)[-limit:]:           # the newest `limit`, oldest first
            try:
                root = ET.parse(os.path.join(voc_folder, base + ".xml")).getroot()
            except (ET.ParseError, OSError):
                continue
            boxes, polygons = _parse_voc_objects(root)
            if self.remember(by_base[base], boxes, polygons):
                added += 1
        return added


def _big_enough(item):
    _cls, kind, geometry = item
    x1, y1, x2, y2 = geometry if kind == "box" else poly_rect(geometry)
    return x2 - x1 >= MIN_SIDE and y2 - y1 >= MIN_SIDE


def items_signature(items, classes):
    """Hashable summary of what a reference image says (used to know when its memory entry is out of date)."""
    return tuple(sorted((classes.index(c), kind, tuple(round(v) for v in (g if kind == "box" else
                                                                           [n for p in g for n in p])))
                        for c, kind, g in items if c in classes))


def reference_masks(items, classes, shape, refine=None):
    """(masks, obj_ids) for one reference image of the given (height, width): for every class ONE mask holding
    all its objects. Polygons are filled exactly. Boxes become silhouettes when `refine(boxes)` (-> one boolean mask
    per box, or None) is given and answers; any box it cannot refine is filled as a rectangle.
    obj id = the class's position in the workspace class list (SAM 2 keeps one object slot per class).
    (None, []) when nothing is usable."""
    import cv2
    import numpy as np
    h, w = shape[:2]
    per_class = {}
    boxes_of = {}
    for cls, kind, geometry in items:
        if cls not in classes:
            continue
        mask = per_class.setdefault(cls, np.zeros((h, w), np.uint8))
        if kind == "poly":
            cv2.fillPoly(mask, [np.round(np.array(geometry)).astype(np.int32)], 1)
        else:
            boxes_of.setdefault(cls, []).append(geometry)
    for cls, boxes in boxes_of.items():
        silhouettes = refine(boxes) if refine else None
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            sil = silhouettes[i] if silhouettes is not None and i < len(silhouettes) else None
            if sil is not None and sil.shape == (h, w) and sil.any():
                per_class[cls] |= sil.astype(np.uint8)
            else:
                per_class[cls][max(int(round(y1)), 0):max(int(round(y2)), 0),
                               max(int(round(x1)), 0):max(int(round(x2)), 0)] = 1
    ids = sorted(classes.index(c) for c in per_class if per_class[c].any())
    if not ids:
        return None, []
    return np.stack([per_class[classes[i]] for i in ids]), ids


# ----------------------------------------------------------------------
#  Masks -> predictions
# ----------------------------------------------------------------------
def masks_to_predictions(masks, scores, slots, classes, min_score, polygon):
    """masks: (N, H, W) booleans, scores: N values in [0, 1], slots: the class position of each mask.
    Every connected piece of a mask becomes one prediction {'rect', 'conf', 'cls', 'poly'?}."""
    import cv2
    import numpy as np
    preds = []
    for mask, score, slot in zip(masks, scores, slots):
        if score < min_score or slot >= len(classes):
            continue
        h, w = mask.shape[:2]
        found, _ = cv2.findContours(np.asarray(mask, dtype=np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        found = sorted(found, key=cv2.contourArea, reverse=True)[:MAX_PIECES_PER_CLASS]
        for contour in found:
            if cv2.contourArea(contour) < max(16, MIN_PIECE_FRACTION * h * w):
                continue
            x, y, bw, bh = cv2.boundingRect(contour)
            pred = {"rect": (int(x), int(y), int(x + bw), int(y + bh)), "conf": float(score), "cls": classes[slot]}
            if polygon:
                eps = 0.003 * cv2.arcLength(contour, True)
                pts = cv2.approxPolyDP(contour, eps, True).reshape(-1, 2).astype(float).tolist()
                if len(pts) < 3:
                    continue
                pred["poly"] = pts
            preds.append(pred)
    return preds


# ----------------------------------------------------------------------
#  The engine
# ----------------------------------------------------------------------
def predictor_problem():
    """Why SAM 2 Dynamic cannot run here (a sentence for the user), or None."""
    try:
        from ultralytics.models.sam import SAM2DynamicInteractivePredictor  # noqa: F401
    except Exception as exc:
        return (f"SAM 2 Dynamic needs a recent ultralytics ({exc}). Run the Jelibox installer again to update it.")
    return None


class Sam2Engine:
    """One loaded SAM 2 predictor and the encoded reference entries it has seen."""

    def __init__(self):
        self.predictor = None
        self.refiner = None                     # SAM 2 in its normal one-image mode: boxes -> silhouettes
        self.key = None
        self.entries = OrderedDict()            # (path, file stamp, annotations) -> (memory entry, object slots)

    def release(self):
        self.predictor = None
        self.refiner = None
        self.key = None
        self.entries.clear()
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    @property
    def loaded(self):
        return self.predictor is not None

    def _load(self, weights, imgsz, slots, status):
        key = (weights, imgsz, slots)
        if self.predictor is not None and self.key == key:
            return
        self.release()
        if not os.path.isfile(weights):
            os.makedirs(os.path.dirname(weights), exist_ok=True)
            if status:
                status(f"Downloading {os.path.basename(weights)} (once) ...")
            from ultralytics.utils.downloads import attempt_download_asset
            attempt_download_asset(weights)
            if not os.path.isfile(weights):
                raise RuntimeError(f"{os.path.basename(weights)} could not be downloaded (no internet?).")
        if status:
            status("Loading SAM 2 ...")
        from ultralytics.models.sam import SAM2DynamicInteractivePredictor
        # conf=0: the predictor drops objects whose score is 0; our own min_score decides what is kept
        overrides = dict(task="segment", mode="predict", imgsz=imgsz, model=weights, verbose=False, save=False,
                         conf=0.0)
        self.predictor = SAM2DynamicInteractivePredictor(overrides=overrides, max_obj_num=slots)
        self.key = key

    def _boxes_refiner(self, image):
        """refine(boxes) -> one boolean silhouette per box, using the standard Ultralytics SAM 2 box prompt.
        None (rectangles are used instead) when that is not possible."""
        def refine(boxes):
            try:
                if self.refiner is None:
                    from ultralytics import SAM
                    self.refiner = SAM(self.key[0])
                result = self.refiner(image, bboxes=[[float(v) for v in b] for b in boxes], verbose=False)[0]
                if result.masks is None or len(result.masks.data) != len(boxes):
                    return None
                return [m > 0 for m in result.masks.data.cpu().numpy()]
            except Exception as exc:
                print(f"[SAM2] Could not refine the reference boxes, using rectangles: {exc}")
                return None
        return refine

    def _entry_for(self, path, items, classes, status):
        """The encoded memory entry of one reference image (cached while the image and its annotations are
        unchanged). None when the image cannot be read or has nothing usable."""
        try:
            stat = os.stat(path)
            stamp = (stat.st_mtime_ns, stat.st_size)
        except OSError:
            return None
        sig = (_key(path), stamp, items_signature(items, classes))
        cached = self.entries.get(sig)
        if cached is not None:
            self.entries.move_to_end(sig)
            return cached
        image = _read_bgr(path)
        if image is None:
            return None
        masks, ids = reference_masks(items, classes, image.shape, self._boxes_refiner(image))
        if masks is None:
            return None
        if status:
            status(f"Learning from {os.path.basename(path)} ...")
        p = self.predictor
        p.memory_bank.clear()                   # so memory_bank[-1] below is exactly this image's entry
        p.obj_idx_set.clear()
        p(source=image, masks=masks, obj_ids=ids, update_memory=True)
        cached = (p.memory_bank[-1], set(p.obj_idx_set))
        self.entries[sig] = cached
        return cached

    def build(self, weights, imgsz, classes, references, min_score, polygon, status=None):
        """Load the model, put one memory entry per reference image into it, and return predict(bgr)."""
        self._load(weights, imgsz, len(classes), status)
        p = self.predictor
        wanted, slots = [], set()
        for path, items in references:
            entry = self._entry_for(path, items, classes, status)
            if entry is not None:
                wanted.append(entry)
        if not wanted:
            raise RuntimeError("None of the reference images could be read.")
        for _entry, used in wanted:
            slots |= used
        while len(self.entries) > max(2 * len(wanted), 8):          # forget the oldest encodings
            self.entries.popitem(last=False)
        # the memory is exactly the wanted entries: one per image, nothing counted twice
        p.memory_bank = [entry for entry, _ in wanted]
        p.obj_idx_set = set(slots)

        def predict(bgr):
            order = list(p.obj_idx_set)                             # row i of the output is slot order[i]
            results = p(source=bgr)
            preds = []
            for res in results:
                if res.masks is None or res.boxes is None or len(res.boxes) == 0:
                    continue
                rows = [int(v) for v in res.boxes.cls.tolist()]
                preds += masks_to_predictions(res.masks.data.cpu().numpy(), res.boxes.conf.tolist(),
                                              [order[r] for r in rows], classes, min_score, polygon)
            return preds, polygon

        return predict


store = ReferenceStore()
engine = Sam2Engine()
