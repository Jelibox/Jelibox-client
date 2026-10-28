"""
Export Dataset, Export Model and the augmentation picker for the web server, on top of the pieces the desktop already
has (dataset_export, export_preprocess, augmentation, augment_catalog, augment_sample, export). Everything takes the
workspace it works for; nothing reads the desktop's globals and nothing here imports tkinter, torch or ultralytics at
import time.

Nothing is saved for good on the server: an export is built in a scratch folder under temp/downloads and handed to the
browser as a download (a file, or a .zip when it is a folder). `Downloads` owns that folder; what nobody fetched is
removed after a few hours, and a folder is deleted as soon as its .zip exists.
"""
import base64
import os
import random
import secrets
import shutil
import threading
import time
import zipfile

from .. import augment_catalog as cat
from .. import augment_sample as sample
from .. import augmentation as aug
from .. import dataset_export as de
from .. import export as ex
from .. import export_preprocess as pre
from .. import workspace_manager as wm
from ..augmentation import Augmenter
from .editing import EditError

IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
DATASET_FORMATS = [("yolo", "YOLO"), ("voc", "Pascal VOC (XML)"), ("coco", "COCO")]
MODEL_FORMATS = [("TorchScript", "torchscript"), ("ONNX", "onnx"), ("OpenVINO", "openvino"), ("TensorRT", "engine"),
                 ("CoreML", "coreml"), ("TF SavedModel", "saved_model"), ("TFLite", "tflite"), ("TFJS", "tfjs"),
                 ("Paddle", "paddle"), ("MNN", "mnn"), ("NCNN", "ncnn"), ("IMX500", "imx"), ("RKNN", "rknn"),
                 ("ExecuTorch", "executorch"), ("Axelera", "axelera")]
MODEL_OPTIONS = ("optimize", "keras", "half", "int8", "dynamic", "simplify", "end2end")
STALE_SECONDS = 6 * 3600


# ------------------------------------------------------------------ downloads
class Downloads:
    """What a finished job offers to the browser: a file, or a folder that is zipped when it is asked for."""

    def __init__(self, root):
        self.root = root
        self.items = {}
        self.lock = threading.Lock()
        self.zip_lock = threading.Lock()

    def scratch(self):
        """A new empty folder for one export (older leftovers are cleaned up first)."""
        os.makedirs(self.root, exist_ok=True)
        self._forget_stale()
        folder = os.path.join(self.root, secrets.token_hex(6))
        os.makedirs(folder)
        return folder

    def add(self, path, user):
        token = secrets.token_urlsafe(12)
        with self.lock:
            self.items[token] = {"path": path, "user": user, "archive": None, "name": None}
        return token

    def _item(self, token, user):
        with self.lock:
            item = self.items.get(token)
        if not item or item["user"] != user:
            raise EditError("That download does not exist any more.")
        return item

    def get(self, token, user):
        item = self._item(token, user)
        for path in (item["archive"], item["path"]):
            if path and os.path.exists(path):
                return path
        raise EditError("That download does not exist any more.")

    def prepare(self, token, user):
        """(file to send, name to show). A folder is packed into <root>/<token>.zip first (once), then deleted."""
        item = self._item(token, user)
        with self.zip_lock:
            if item["archive"] and os.path.isfile(item["archive"]):
                return item["archive"], item["name"]
            path = item["path"]
            if not os.path.exists(path):
                raise EditError("That download does not exist any more.")
            if os.path.isfile(path):
                return path, os.path.basename(path)
            os.makedirs(self.root, exist_ok=True)
            archive = os.path.join(self.root, token + ".zip")
            tmp = archive + ".part"
            base = os.path.basename(os.path.normpath(path))
            with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED, allowZip64=True) as z:   # pictures are compressed already
                for folder, _dirs, files in os.walk(path):
                    for name in files:
                        full = os.path.join(folder, name)
                        z.write(full, os.path.join(base, os.path.relpath(full, path)))
            os.replace(tmp, archive)
            root = os.path.abspath(self.root) + os.sep               # only what lives under root is ever deleted
            scratch = os.path.dirname(os.path.abspath(path))         # the export's own folder
            if scratch.startswith(root):
                shutil.rmtree(scratch, ignore_errors=True)
            item["archive"], item["name"] = archive, base + ".zip"
            return archive, item["name"]

    def _forget_stale(self):
        now = time.time()
        try:
            for name in os.listdir(self.root):
                full = os.path.join(self.root, name)
                if now - os.path.getmtime(full) > STALE_SECONDS:
                    shutil.rmtree(full, ignore_errors=True) if os.path.isdir(full) else os.remove(full)
        except OSError:
            pass


# ------------------------------------------------------------------ Export Dataset
def dataset_info(ws):
    """What the Export Dataset window shows before anything is chosen."""
    records = de.find_workspace_images(wm.DATASETS_ROOT, ws.name, IMAGE_EXTS)
    labels = set()
    if os.path.isdir(ws.labels):
        labels = {os.path.splitext(f)[0] for f in os.listdir(ws.labels) if f.endswith(".txt")}
    return {
        "workspace": ws.name,
        "formats": [{"id": k, "label": label,
                     "count": sum(1 for r in records if r["base"] in labels) if k == "yolo" else len(records)}
                    for k, label in DATASET_FORMATS],
        "resize": {"modes": [{"id": m, "label": pre.RESIZE_LABELS[m], "hint": pre.RESIZE_HINTS[m]}
                             for m in ("stretch", "fit", "letterbox")], "range": list(pre.SIZE_RANGE)},
    }


def _int(value, what, low, high):
    try:
        number = int(value)
    except (TypeError, ValueError):
        raise EditError(f"{what} must be a whole number.")
    if not low <= number <= high:
        raise EditError(f"{what} must be between {low} and {high}.")
    return number


def build_preprocess(data):
    data = data if isinstance(data, dict) else {}
    resize = None
    if data.get("resize"):
        r = data["resize"]
        lo, hi = pre.SIZE_RANGE
        if r.get("mode") not in pre.RESIZE_MODES:
            raise EditError("Choose how to resize.")
        w, h = _int(r.get("width"), "The width", lo, hi), _int(r.get("height"), "The height", lo, hi)
        resize = (r["mode"], w, h)
    return pre.Preprocess(bool(data.get("auto_orient", True)), resize)


def build_state(ops):
    """The picker's {operation: OpSetting} from what the page sends (checked and clamped)."""
    state = cat.default_state()
    for name, raw in (ops or {}).items():
        if name not in aug.OPERATIONS or not isinstance(raw, dict):
            continue
        spec = aug.OPERATIONS[name]
        lo, hi = spec["range"]
        try:
            magnitude = float(raw.get("magnitude", spec["default"]))
            probability = float(raw.get("probability", 50.0))
        except (TypeError, ValueError):
            raise EditError(f"The numbers of {spec['label']} are not valid.")
        magnitude = min(max(magnitude, lo), hi) if hi else spec["default"]
        options = dict(state[name].options)
        for key, value in (raw.get("options") or {}).items():
            if key in options:
                options[key] = max(1, min(10, int(value))) if key == "count" and not isinstance(value, bool) else bool(value)
        state[name] = aug.OpSetting(bool(raw.get("enabled")), magnitude, min(max(probability, 0.0), 100.0), options)
    return state


def build_augment(data):
    """AugmentConfig from the picker's request, or None when nothing is picked."""
    if not data or not isinstance(data, dict):
        return None
    state = build_state(data.get("ops"))
    if not cat.selected_tiles(state):
        return None
    seed = data.get("seed")
    try:
        seed = None if seed in (None, "") else int(seed)
    except (TypeError, ValueError):
        raise EditError("The seed must be a whole number, or empty.")
    splits = ("train",) + tuple(s for s in ("val", "test") if s in (data.get("splits") or ()))
    return cat.to_config(state, _int(data.get("outputs", cat.DEFAULT_OUTPUTS), "Outputs per training image",
                                     *cat.OUTPUTS_RANGE), bool(data.get("keep_original", True)), seed, splits)


def check_dataset_request(ws, req):
    """Validate everything before a job is started; returns the parsed pieces."""
    fmt = req.get("format")
    if fmt not in dict(DATASET_FORMATS):
        raise EditError("Choose a format.")
    train = _int(req.get("train_pct"), "The training share", 0, 100)
    valid = _int(req.get("valid_pct"), "The validation share", 0, 100)
    if train + valid > 100:
        raise EditError("Train + valid cannot be more than 100 %.")
    preprocess = build_preprocess(req.get("preprocess"))
    config = build_augment(req.get("augment"))
    return fmt, train, valid, preprocess, config


def run_dataset_export(ws, job, req, parsed, downloads, user):
    fmt, train, valid, preprocess, config = parsed
    label = dict(DATASET_FORMATS)[fmt]
    augmenter = None
    note = ""
    if config is not None:
        if config.enabled:
            augmenter = Augmenter(config, post=preprocess.post if preprocess.resize else None)
        else:
            note = ("One output per image that is the original adds nothing, so no augmented copies were made. Raise "
                    "\"Outputs per training image\" to make some.")
    use_pre = preprocess if preprocess.active else None
    records = de.find_workspace_images(wm.DATASETS_ROOT, ws.name, IMAGE_EXTS)

    job.progress(0, 0, "Preparing the export ...")
    problem = ex.check_destination(downloads.root if os.path.isdir(downloads.root) else ws.base_dir,
                                   ex.files_size(r["source"] for r in records))
    if problem:
        raise EditError(problem)
    target = os.path.join(downloads.scratch(), f"{ws.name}-{fmt}")
    os.makedirs(target)

    seed = config.seed if config is not None else None
    rng = random.Random(seed)
    progress = lambda done, total, text=None: job.progress(done, total, text or f"Exporting ... {done} / {total}")
    if fmt == "yolo":                                           # read-only on the annotations: saving goes on meanwhile
        de.export_yolo(records, ws.labels, ws.name, ws.classes.get_classes(), target, train, valid, progress,
                       augment=augmenter, preprocess=use_pre, rng=rng)
    elif fmt == "voc":
        de.export_voc(records, ws.voc, ws.name, target, train, valid, progress, augment=augmenter,
                      preprocess=use_pre, rng=rng)
    else:
        de.export_coco(records, ws.voc, ws.name, ws.classes.get_classes(), target, train, valid, progress,
                       augment=augmenter, preprocess=use_pre, rng=rng)
    extra = [note] if note else []
    if use_pre is not None:
        extra.append("Preprocessing: " + ", ".join(use_pre.describe()))
    if augmenter is not None:
        extra.append(f"Augmented copies added: {augmenter.created}"
                     + (f" ({augmenter.skipped} skipped: no effect fired)" if augmenter.skipped else ""))
    files = sum(len(names) for _d, _s, names in os.walk(target))
    return {"name": os.path.basename(target) + ".zip", "format": label, "files": files, "notes": extra,
            "download": downloads.add(target, user)}


# ------------------------------------------------------------------ Export Model
def model_info(ws):
    path = os.path.join(ws.base_dir, "models", ws.name, "modelAssistant.pt")
    return {"has_model": os.path.isfile(path), "model": os.path.relpath(path, ws.base_dir).replace("\\", "/"),
            "formats": [{"label": a, "id": b} for a, b in MODEL_FORMATS], "options": list(MODEL_OPTIONS)}


def check_model_request(ws, req):
    path = os.path.join(ws.base_dir, "models", ws.name, "modelAssistant.pt")
    if not os.path.isfile(path):
        raise EditError("This workspace has no trained model yet. Train one first (the Train button).")
    if req.get("format") not in dict((b, a) for a, b in MODEL_FORMATS):
        raise EditError("Choose a format.")
    return path, _int(req.get("imgsz", 640), "The image size", 32, 8192)


def run_model_export(ws, job, req, parsed, downloads, user):
    path, imgsz = parsed
    job.progress(0, 0, "Exporting the model (this can take a few minutes) ...")
    opts = {k: bool(req.get(k)) for k in MODEL_OPTIONS}
    ok, message, out = ex.export_model(path, fmt=req["format"], imgsz=imgsz, save_dir=downloads.scratch(), **opts)
    if not ok:
        raise EditError(message.split("\nTraceback")[0].strip() or "The export failed.")
    return {"name": os.path.basename(os.path.normpath(out)) + ("" if os.path.isfile(out) else ".zip") if out and os.path.exists(out) else None,
            "message": message, "download": downloads.add(out, user) if out and os.path.exists(out) else None}


# ------------------------------------------------------------------ the augmentation picker
def catalog(training=False):
    ops = {}
    for name, spec in aug.OPERATIONS.items():
        ops[name] = {"label": spec["label"], "unit": spec["unit"], "default": spec["default"], "range": list(spec["range"]),
                     "options": dict(spec.get("options") or {}), "note": spec.get("fixed_note", "")}
    tiles = []
    for tile, note in cat.usable_tiles(training):
        tiles.append({"key": tile["key"], "title": tile["title"], "description": tile["description"], "ops": list(tile["ops"]),
                      "slider": list(tile["slider"]) if tile["slider"] else None,
                      "choices": [list(c) for c in tile["choices"]], "count": list(tile["count"]) if tile.get("count") else None,
                      "note": note})
    return {"tiles": tiles, "operations": ops, "outputs": list(cat.OUTPUTS_RANGE), "default_outputs": cat.DEFAULT_OUTPUTS}


def _data_url(img):
    import cv2
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def sample_info(ws):
    path, is_default = sample.pick_sample(ws.name)
    return {"name": os.path.basename(path) if path else "", "is_default": is_default}


def choose_sample(ws, name):
    """Use one picture of this dataset as the preview picture (the first picture again when `name` is empty)."""
    if not name:
        sample.forget_sample(ws.name)
        return sample_info(ws)
    sample.remember_sample(ws.name, ws.path(name))
    return sample_info(ws)


def previews(ws, tile_key, ops):
    """The settings window of one tile: the original and its examples at the current setting, as data URLs."""
    tile = cat.TILE_BY_KEY.get(tile_key)
    if tile is None:
        raise EditError("Unknown augmentation.")
    path, _default = sample.pick_sample(ws.name)
    image = sample.load_sample(path)
    if image is None:
        raise EditError("This workspace has no picture to preview with.")
    return {"items": [{"label": label, "image": _data_url(img)} for label, img in cat.previews(tile, build_state(ops), image)]}


def thumbnails(ws, ops):
    """The picture of every tile for the grid (each tile's strongest example at its current setting)."""
    path, _default = sample.pick_sample(ws.name)
    image = sample.load_sample(path)
    if image is None:
        return {"items": {}}
    state = build_state(ops)
    return {"items": {t["key"]: _data_url(cat.thumbnail(t, state, image)) for t in cat.TILES}}
