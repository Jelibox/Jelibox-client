"""Dataset augmentation for the exporter.

Two kinds of operation:
  * photometric (brightness, blur, ...) change pixel values only, so the
    annotations of the copy are identical to the original's;
  * geometric (flip, rotate) move the pixels, so the copy comes with a
    GeoTransform that the exporters use to move the annotations the same way
    (see augment_geometry.py).

Each operation has an on/off switch, a *magnitude* (its limit - the amount
actually used is drawn at random up to that limit for every image) and a
*probability* (a separate coin flip per image and per operation).
"""
import os
from dataclasses import dataclass, field

import cv2
import numpy as np

from .augment_geometry import GeoTransform

SPLITS = ("train", "val", "test")
MAX_REROLLS = 20

# name -> label, magnitude unit, default magnitude, (min, max) the UI allows.
# range (0, 0) = no adjustable value. "group" decides the tab in the export dialog.
OPERATIONS = {
    "brightness": {"group": "color", "label": "Brightness", "unit": "± %",  "default": 25, "range": (1, 100)},
    "contrast":   {"group": "color", "label": "Contrast",   "unit": "± %",  "default": 25, "range": (1, 100)},
    "saturation": {"group": "color", "label": "Saturation", "unit": "± %",  "default": 25, "range": (1, 100)},
    "hue":        {"group": "color", "label": "Hue",        "unit": "± °",  "default": 15, "range": (1, 90)},
    "blur":       {"group": "color", "label": "Blur",       "unit": "px",   "default": 3,  "range": (3, 25)},
    "noise":      {"group": "color", "label": "Noise",      "unit": "σ",    "default": 10, "range": (1, 50)},
    "grayscale":  {"group": "color", "label": "Grayscale",  "unit": "",     "default": 0,  "range": (0, 0)},
    "jpeg":       {"group": "color", "label": "JPEG compression", "unit": "", "default": 70, "range": (0, 0),
                   "fixed_note": "quality 70 (fixed)"},
    "flip_h":     {"group": "geometry", "label": "Flip horizontal", "unit": "", "default": 0, "range": (0, 0),
                   "fixed_note": "mirror left-right"},
    "flip_v":     {"group": "geometry", "label": "Flip vertical",   "unit": "", "default": 0, "range": (0, 0),
                   "fixed_note": "mirror top-bottom"},
    "rotate":     {"group": "geometry", "label": "Rotation",        "unit": "± °", "default": 15, "range": (1, 90)},
}
GEOMETRIC = ("flip_h", "flip_v", "rotate")


@dataclass
class OpSetting:
    enabled: bool = False
    magnitude: float = 0
    probability: float = 50.0  # percent


@dataclass
class AugmentConfig:
    ops: dict = field(default_factory=dict)       # name -> OpSetting
    copies: int = 1                                # augmented versions per original image
    splits: tuple = ("train",)                     # which splits get augmented copies
    seed: int = None                               # None = different every run

    def active_ops(self):
        return {n: s for n, s in self.ops.items()
                if n in OPERATIONS and s.enabled and s.probability > 0}

    @property
    def enabled(self):
        return self.copies > 0 and bool(self.active_ops()) and bool(self.splits)


# ----------------------------------------------------------------------
#  The operations. All take/return uint8 BGR images and a numpy Generator.
# ----------------------------------------------------------------------
def _clip(arr):
    return np.clip(arr, 0, 255).astype(np.uint8)


def _brightness(img, mag, rng):
    factor = 1.0 + rng.uniform(-mag, mag) / 100.0
    return _clip(img.astype(np.float32) * factor)


def _contrast(img, mag, rng):
    factor = 1.0 + rng.uniform(-mag, mag) / 100.0
    f = img.astype(np.float32)
    mean = f.mean()
    return _clip((f - mean) * factor + mean)


def _saturation(img, mag, rng):
    factor = 1.0 + rng.uniform(-mag, mag) / 100.0
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def _hue(img, mag, rng):
    shift = rng.uniform(-mag, mag) / 2.0          # OpenCV hue runs 0-179 for 0-358 degrees
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + shift) % 180
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def _blur(img, mag, rng):
    top = max(3, int(mag))
    sizes = list(range(3, top + 1, 2)) or [3]     # odd kernels only, up to the limit
    k = int(rng.choice(sizes))
    return cv2.GaussianBlur(img, (k, k), 0)


def _noise(img, mag, rng):
    sigma = rng.uniform(0, mag)
    noise = rng.normal(0.0, sigma, img.shape)
    return _clip(img.astype(np.float32) + noise)


def _grayscale(img, mag, rng):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _jpeg(img, mag, rng):
    quality = int(mag)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR) if ok else img


_FUNCS = {
    "brightness": _brightness, "contrast": _contrast, "saturation": _saturation,
    "hue": _hue, "blur": _blur, "noise": _noise, "grayscale": _grayscale, "jpeg": _jpeg,
}
# Order matters a little: colour changes first, degradations (blur/noise/jpeg) last.
_ORDER = ["brightness", "contrast", "saturation", "hue", "grayscale", "blur", "noise", "jpeg"]


def augment_image(img, config, rng):
    """Apply the configured operations, each with its own probability.

    Returns (image, [names applied], GeoTransform or None). The transform is
    only set when a flip/rotation happened; the exporters use it to move the
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

    out = img
    for name in _ORDER:                          # pixel values first, so rotation fill stays black
        if name in applied:
            out = _FUNCS[name](out, active[name].magnitude, rng)

    tf = None
    if any(n in applied for n in GEOMETRIC):
        angle = rng.uniform(-active["rotate"].magnitude, active["rotate"].magnitude)             if "rotate" in applied else 0.0
        h, w = out.shape[:2]
        tf = GeoTransform(w, h, flip_h="flip_h" in applied, flip_v="flip_v" in applied, angle=angle)
        out = tf.warp(out)
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

    def __init__(self, config):
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.created = 0
        self.skipped = 0

    def wants(self, split_name):
        split = "val" if split_name == "valid" else split_name
        return self.config.enabled and split in self.config.splits

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
            self.created += 1
            yield f"_aug{i}", out, tf
