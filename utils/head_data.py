"""
Data side of training a custom head (no torch here, so the GUI can use it before anything heavy is imported).

A workspace already has what `TOBEADDED/custom_dual_heads/train.py` wants: one YOLO label file per image, one row per
object, `class cx cy w h` (normalized), where the class is the behavior / brand and the box is the person / bottle.
This module pairs labels with their images and splits them into train and validation.

The split is by position, not random: frames of a video are almost the same picture, so a random split puts
near-duplicates on both sides and the validation score looks better than it is. The last part of the (sorted)
images becomes validation.
"""
import os
import re

IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
MIN_LABELLED = 10

# Neck layers (strides 8 / 16 / 32) of the detectors the head is known to work on - see custom_dual_heads/README.md.
KNOWN_TAPS = (
    ("yolov5", (17, 20, 23)),
    ("yolov8", (15, 18, 21)),
    ("yolov9", (15, 18, 21)),
    ("yolo11", (16, 19, 22)),
    ("yolo26", (16, 19, 22)),
)
DEFAULT_TAPS = (15, 18, 21)


def taps_for(weights):
    """(taps, known): the neck layers to read for this detector, and whether the model family is one we know."""
    # a Windows path (C:\models\yolov9c.pt) must still be read on Linux, where os.path.basename
    # does not treat the backslash as a separator
    name = re.split(r"[\\/]", str(weights))[-1].lower()
    for prefix, taps in KNOWN_TAPS:
        if name.startswith(prefix):
            return taps, True
    return DEFAULT_TAPS, False


def _natural(text):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", text)]


def index_images(folders):
    """{file base name: image path} over the dataset folders. A later folder wins on a name clash."""
    found = {}
    for folder in folders:
        if not os.path.isdir(folder):
            continue
        for name in os.listdir(folder):
            if name.lower().endswith(IMAGE_EXTS):
                found[os.path.splitext(name)[0]] = os.path.join(folder, name)
    return found


def read_label_rows(path):
    """[(class_id, cx, cy, w, h)] normalized. A segmentation row (class x1 y1 x2 y2 ...) becomes its bounding box."""
    rows = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()
    except OSError:
        return rows
    for line in lines:
        parts = line.split()
        try:
            cls, vals = int(float(parts[0])), [float(v) for v in parts[1:]]
        except (ValueError, IndexError):
            continue
        if len(vals) == 4:
            cx, cy, w, h = vals
        elif len(vals) >= 6 and len(vals) % 2 == 0:
            xs, ys = vals[0::2], vals[1::2]
            cx, cy, w, h = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, max(xs) - min(xs), max(ys) - min(ys)
        else:
            continue
        if w > 0 and h > 0:
            rows.append((cls, cx, cy, w, h))
    return rows


def labelled_pairs(images_folders, labels_dir):
    """[(image path, label path)] of the images that have at least one label row, in natural name order."""
    images = index_images(images_folders)
    pairs = []
    if os.path.isdir(labels_dir):
        for name in os.listdir(labels_dir):
            base, ext = os.path.splitext(name)
            path = os.path.join(labels_dir, name)
            if ext.lower() == ".txt" and base in images and read_label_rows(path):
                pairs.append((images[base], path))
    pairs.sort(key=lambda p: _natural(os.path.basename(p[0])))
    return pairs


def split_contiguous(pairs, val_fraction=0.2):
    """(train, val): the last `val_fraction` of the pairs, as one block, is validation. Both sides get at least one."""
    if len(pairs) < 2:
        raise ValueError("Need at least 2 labelled images to split into train and validation.")
    n_val = min(len(pairs) - 1, max(1, int(round(len(pairs) * val_fraction))))
    return pairs[:len(pairs) - n_val], pairs[len(pairs) - n_val:]


def class_counts(pairs, n_classes):
    counts = [0] * n_classes
    for _, label in pairs:
        for cls, *_ in read_label_rows(label):
            if 0 <= cls < n_classes:
                counts[cls] += 1
    return counts
