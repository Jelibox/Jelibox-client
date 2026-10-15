"""Dataset augmentation (the exporter, and the training window's "augmented copies").

Two kinds of operation:
  * photometric (brightness, blur, cutout ...) change pixel values only, so the
    annotations of the copy are identical to the original's;
  * geometric (flip, rotate, 90 degree turn, shear) move the pixels, so the copy comes
    with a GeoTransform that the exporters use to move the annotations the same way
    (see augment_geometry.py).

Each operation has an on/off switch, a *magnitude* (its limit - the amount
actually used is drawn at random up to that limit for every image), a *probability*
(a separate coin flip per image and per operation) and a few `options` (for example
"darken only" for brightness).
"""
import os
from dataclasses import dataclass, field

import cv2
import numpy as np

from .augment_geometry import GeoTransform

SPLITS = ("train", "val", "test")
MAX_REROLLS = 20

# name -> label, magnitude unit, default magnitude, (min, max) the UI allows.
# range (0, 0) = no adjustable value. "group" is the kind of operation. "options" = the yes/no choices an operation
# has (all on by default); "choices" shows them in the settings window.
OPERATIONS = {
    "brightness": {"group": "color", "label": "Brightness", "unit": "± %",  "default": 25, "range": (1, 100),
                   "options": {"brighten": True, "darken": True}},
    "exposure":   {"group": "color", "label": "Exposure",   "unit": "± %",  "default": 25, "range": (1, 100),
                   "options": {"brighten": True, "darken": True}},
    "contrast":   {"group": "color", "label": "Contrast",   "unit": "± %",  "default": 25, "range": (1, 100)},
    "saturation": {"group": "color", "label": "Saturation", "unit": "± %",  "default": 25, "range": (1, 100)},
    "hue":        {"group": "color", "label": "Hue",        "unit": "± °",  "default": 15, "range": (1, 90)},
    "blur":       {"group": "color", "label": "Blur",       "unit": "px",   "default": 3,  "range": (3, 25)},
    "motion_blur": {"group": "color", "label": "Motion blur", "unit": "px", "default": 15, "range": (3, 60)},
    "noise":      {"group": "color", "label": "Noise",      "unit": "σ",    "default": 10, "range": (1, 50)},
    "cutout":     {"group": "color", "label": "Cutout",     "unit": "% of the side", "default": 10, "range": (2, 50),
                   "options": {"count": 3}},
    "grayscale":  {"group": "color", "label": "Grayscale",  "unit": "",     "default": 0,  "range": (0, 0)},
    "jpeg":       {"group": "color", "label": "JPEG compression", "unit": "", "default": 70, "range": (0, 0),
                   "fixed_note": "quality 70 (fixed)"},
    "flip_h":     {"group": "geometry", "label": "Flip horizontal", "unit": "", "default": 0, "range": (0, 0),
                   "fixed_note": "mirror left-right"},
    "flip_v":     {"group": "geometry", "label": "Flip vertical",   "unit": "", "default": 0, "range": (0, 0),
                   "fixed_note": "mirror top-bottom"},
    "rot90":      {"group": "geometry", "label": "90° rotate",      "unit": "", "default": 0, "range": (0, 0),
                   "options": {"cw": True, "ccw": True, "upside_down": True}},
    "rotate":     {"group": "geometry", "label": "Rotation",        "unit": "± °", "default": 15, "range": (1, 90)},
    "shear":      {"group": "geometry", "label": "Shear",           "unit": "± °", "default": 10, "range": (1, 45),
                   "options": {"horizontal": True, "vertical": True}},
}
GEOMETRIC = ("flip_h", "flip_v", "rot90", "rotate", "shear")
# What the output of these operations looks like on a canvas of another size (the labels follow with a GeoTransform).
CHANGES_SIZE = ("rot90", "rotate", "shear")


@dataclass
class OpSetting:
    enabled: bool = False
    magnitude: float = 0
    probability: float = 50.0  # percent
    options: dict = field(default_factory=dict)   # yes/no choices (see OPERATIONS[name]["options"])

    def option(self, name, default=True):
        return self.options.get(name, default)


@dataclass
class AugmentConfig:
    ops: dict = field(default_factory=dict)       # name -> OpSetting
    copies: int = 1                                # augmented versions per original image
    splits: tuple = ("train",)                     # which splits get augmented copies
    seed: int = None                               # None = different every run
    keep_original: bool = True                     # False: the training originals are replaced by their copies

    def active_ops(self):
        return {n: s for n, s in self.ops.items()
                if n in OPERATIONS and s.enabled and s.probability > 0}

    @property
    def enabled(self):
        return self.copies > 0 and bool(self.active_ops()) and bool(self.splits)


def config_to_dict(config):
    """A JSON-able copy of a config (the training job file carries it to the training process)."""
    return {
        "ops": {n: {"enabled": s.enabled, "magnitude": s.magnitude, "probability": s.probability,
                    "options": dict(s.options)} for n, s in config.ops.items()},
        "copies": int(config.copies), "splits": list(config.splits), "seed": config.seed,
        "keep_original": bool(config.keep_original),
    }


def config_from_dict(data):
    ops = {n: OpSetting(bool(v.get("enabled")), float(v.get("magnitude", 0)), float(v.get("probability", 50.0)),
                        dict(v.get("options") or {}))
           for n, v in (data.get("ops") or {}).items() if n in OPERATIONS}
    seed = data.get("seed")
    return AugmentConfig(ops=ops, copies=int(data.get("copies", 1)), splits=tuple(data.get("splits") or ("train",)),
                         seed=None if seed is None else int(seed), keep_original=bool(data.get("keep_original", True)))


# ----------------------------------------------------------------------
#  The operations. All take/return uint8 BGR images, a numpy Generator and the operation's options.
# ----------------------------------------------------------------------
def _clip(arr):
    return np.clip(arr, 0, 255).astype(np.uint8)


def _signed(rng, mag, opts, up="brighten", down="darken"):
    """A random amount in [-mag, +mag], or only one side of it when the options ask for it."""
    up_on, down_on = opts.get(up, True), opts.get(down, True)
    if not up_on and not down_on:
        up_on = down_on = True
    return rng.uniform(-mag if down_on else 0.0, mag if up_on else 0.0)


def _brightness(img, mag, rng, opts):
    factor = 1.0 + _signed(rng, mag, opts) / 100.0
    return _clip(img.astype(np.float32) * factor)


def _exposure(img, mag, rng, opts):
    """Gamma: brighter pictures lift the dark parts most, darker ones deepen them (unlike plain brightness)."""
    factor = max(0.05, 1.0 + _signed(rng, mag, opts) / 100.0)
    return _clip(255.0 * (img.astype(np.float32) / 255.0) ** (1.0 / factor))


def _contrast(img, mag, rng, opts):
    factor = 1.0 + rng.uniform(-mag, mag) / 100.0
    f = img.astype(np.float32)
    mean = f.mean()
    return _clip((f - mean) * factor + mean)


def _saturation(img, mag, rng, opts):
    factor = 1.0 + rng.uniform(-mag, mag) / 100.0
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def _hue(img, mag, rng, opts):
    shift = rng.uniform(-mag, mag) / 2.0          # OpenCV hue runs 0-179 for 0-358 degrees
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + shift) % 180
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def _blur(img, mag, rng, opts):
    top = max(3, int(mag))
    sizes = list(range(3, top + 1, 2)) or [3]     # odd kernels only, up to the limit
    k = int(rng.choice(sizes))
    return cv2.GaussianBlur(img, (k, k), 0)


def _motion_blur(img, mag, rng, opts):
    """A straight smear of up to `mag` pixels in a random direction, like a moving camera or object."""
    length = max(3, int(round(rng.uniform(max(3, mag / 3.0), max(3, mag)))))
    angle = float(rng.uniform(0.0, 180.0))
    kernel = np.zeros((length, length), np.float32)
    kernel[length // 2, :] = 1.0
    rot = cv2.getRotationMatrix2D((length / 2.0 - 0.5, length / 2.0 - 0.5), angle, 1.0)
    kernel = cv2.warpAffine(kernel, rot, (length, length))
    total = kernel.sum()
    if total <= 0:
        return img
    return cv2.filter2D(img, -1, kernel / total, borderType=cv2.BORDER_REFLECT)


def _noise(img, mag, rng, opts):
    sigma = rng.uniform(0, mag)
    noise = rng.normal(0.0, sigma, img.shape)
    return _clip(img.astype(np.float32) + noise)


def _cutout(img, mag, rng, opts):
    """Black rectangles (up to `mag` % of the picture's side) at random places; labels are left as they are."""
    h, w = img.shape[:2]
    out = img.copy()
    for _ in range(max(1, int(opts.get("count", 3)))):
        bw = max(2, int(round(w * rng.uniform(max(1.0, mag / 2.0), max(1.0, mag)) / 100.0)))
        bh = max(2, int(round(h * rng.uniform(max(1.0, mag / 2.0), max(1.0, mag)) / 100.0)))
        x = int(rng.uniform(0, max(0, w - bw)))
        y = int(rng.uniform(0, max(0, h - bh)))
        out[y:y + bh, x:x + bw] = 0
    return out


def _grayscale(img, mag, rng, opts):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _jpeg(img, mag, rng, opts):
    quality = int(mag)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR) if ok else img


_FUNCS = {
    "brightness": _brightness, "exposure": _exposure, "contrast": _contrast, "saturation": _saturation,
    "hue": _hue, "blur": _blur, "motion_blur": _motion_blur, "noise": _noise, "cutout": _cutout,
    "grayscale": _grayscale, "jpeg": _jpeg,
}
# Order matters a little: colour changes first, degradations (blur/noise/jpeg) last, cutout after them.
_ORDER = ["brightness", "exposure", "contrast", "saturation", "hue", "grayscale", "blur", "motion_blur", "noise",
          "jpeg", "cutout"]


class FixedRng:
    """Stands in for the random generator so a preview shows an operation at its limit: every uniform(lo, hi) gives
    `hi` (or `lo`), so "± 30 %" shows +30 % (or -30 %), and every coin flip comes up "yes"."""

    def __init__(self, high=True, seed=3):
        self.high = high
        self._real = np.random.default_rng(seed)

    def uniform(self, lo=0.0, hi=1.0, size=None):
        return hi if self.high else lo

    def random(self, *a, **k):
        return 0.0

    def choice(self, seq, *a, **k):
        return seq[-1] if self.high else seq[0]

    def normal(self, *a, **k):
        return self._real.normal(*a, **k)


def _turn(setting, rng):
    """The angle of the 90 degree turn: one of the allowed ones (counter-clockwise is positive on screen)."""
    allowed = [a for key, a in (("cw", -90.0), ("ccw", 90.0), ("upside_down", 180.0)) if setting.option(key)]
    return float(rng.choice(allowed or [-90.0, 90.0, 180.0]))


def _geometry(out, applied, active, rng):
    """The flips / rotations / shear of an image: (new image, GeoTransform or None)."""
    if not any(n in applied for n in GEOMETRIC):
        return out, None
    angle = 0.0
    if "rotate" in applied:
        m = active["rotate"].magnitude
        angle += rng.uniform(-m, m)
    if "rot90" in applied:
        angle += _turn(active["rot90"], rng)
    shear_x = shear_y = 0.0
    if "shear" in applied:
        s = active["shear"]
        m = s.magnitude
        if s.option("horizontal") or not s.option("vertical"):
            shear_x = rng.uniform(-m, m)
        if s.option("vertical") or not s.option("horizontal"):
            shear_y = rng.uniform(-m, m)
    h, w = out.shape[:2]
    tf = GeoTransform(w, h, flip_h="flip_h" in applied, flip_v="flip_v" in applied, angle=angle,
                      shear_x=shear_x, shear_y=shear_y)
    return tf.warp(out), tf


def apply_operations(img, applied, active, rng):
    """Run the operations named in `applied` (settings in `active`): pixel changes first, then the geometry.
    Returns (image, GeoTransform or None)."""
    out = img
    for name in _ORDER:                          # pixel values first, so rotation fill stays black
        if name in applied:
            s = active[name]
            out = _FUNCS[name](out, s.magnitude, rng, s.options)
    return _geometry(out, applied, active, rng)


def preview_operation(img, name, setting, high=True):
    """One operation at its limit, for the preview pictures of the settings window. Returns the image."""
    rng = FixedRng(high) if name not in ("cutout", "noise", "motion_blur") else np.random.default_rng(5)
    out, _ = apply_operations(img, [name], {name: setting}, rng)
    return out


def augment_image(img, config, rng):
    """Apply the configured operations, each with its own probability.

    Returns (image, [names applied], GeoTransform or None). The transform is
    only set when a flip/rotation/shear happened; the exporters use it to move the
    annotations. An "augmented version" that nothing touched would just be a
    duplicate, so when every dice roll misses the whole draw is repeated (up
    to MAX_REROLLS times); if it still misses, (None, [], None) is returned
    and the caller skips that copy.
    """
    active = config.active_ops()
    if not active:
        return None, [], None
    for _ in range(MAX_REROLLS):
        applied = [n for n in _ORDER + list(GEOMETRIC)
                   if n in active and rng.random() * 100.0 < active[n].probability]
        if applied:
            break
    else:
        return None, [], None
    out, tf = apply_operations(img, applied, active, rng)
    return out, applied, tf


# ----------------------------------------------------------------------
#  Image I/O that survives non-ASCII Windows paths (cv2.imread does not)
# ----------------------------------------------------------------------
def read_image(path):
    try:
        data = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except OSError:
        return None


def write_image(path, img):
    ext = os.path.splitext(path)[1] or ".png"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise ValueError(f"Could not encode {path}")
    buf.tofile(path)


class Augmenter:
    """What the dataset exporters talk to: hand it a source image and a split,
    get back the augmented copies to write next to the original."""

    def __init__(self, config, post=None):
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.created = 0
        self.skipped = 0
        self.post = post              # post(image, transform, source (h, w)) -> (image, transform): e.g. the export's Resize

    def wants(self, split_name):
        split = "val" if split_name == "valid" else split_name
        return self.config.enabled and split in self.config.splits

    def keeps_original(self, split_name):
        """False when this split's originals are replaced by their augmented versions."""
        return not self.wants(split_name) or self.config.keep_original

    def copies(self, src_path, split_name):
        """Yield (suffix, image, transform) for each augmented version, e.g.
        ("_aug1", array, GeoTransform-or-None). Move the annotations with the
        transform when it is not None; otherwise they are copied unchanged."""
        if not self.wants(split_name):
            return
        img = read_image(src_path)
        if img is None:
            return
        for i in range(1, self.config.copies + 1):
            out, _, tf = augment_image(img, self.config, self.rng)
            if out is None:
                self.skipped += 1
                continue
            if self.post:
                out, tf = self.post(out, tf, img.shape[:2])
            self.created += 1
            yield f"_aug{i}", out, tf
