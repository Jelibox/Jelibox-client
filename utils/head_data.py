"""
Data side of training a custom head (no torch here, so the GUI can use it before anything heavy is imported).

A workspace already has what `TOBEADDED/custom_dual_heads/train.py` wants: one YOLO label file per image, one row per
object, `class cx cy w h` (normalized), where the class is the behavior / brand and the box is the person / bottle.
This module pairs labels with their images and splits them into train and validation.

The split is by position, not random: frames of a video are almost the same picture, so a random split puts
near-duplicates on both sides and the validation score looks better than it is. The last part of the (sorted)
images becomes validation.
"""
import json
import os
import re
import threading

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


CACHE_NAME = ".label_cache.json"          # next to the labels folder: {label file name: [mtime_ns, size, has rows, boxes per class]}
_cache_lock = threading.Lock()
_box_counts = {}                          # label path -> {class id: boxes}, filled by labelled_pairs for class_counts


def _read_cache(labels_dir):
    try:
        with open(os.path.join(os.path.dirname(os.path.normpath(labels_dir)), CACHE_NAME), encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _write_cache(labels_dir, cache):
    path = os.path.join(os.path.dirname(os.path.normpath(labels_dir)), CACHE_NAME)
    try:
        with open(path + ".tmp", "w", encoding="utf-8") as f:
            json.dump(cache, f)
        os.replace(path + ".tmp", path)
    except OSError:
        pass                                   # a read-only folder just means no cache


def _is_blank(path, size):
    """True for a label file with nothing in it: the picture was looked at and has no object."""
    if size == 0:
        return True
    try:
        with open(path, "r", encoding="utf-8") as f:
            return not f.read().strip()
    except OSError:
        return False


def scan_labels(images_folders, labels_dir):
    """([(image, label)] with at least one box, [(image, label)] whose label file is empty), both in natural name order.

    The empty ones are pictures the user looked at and saved with no object in them (a "null" picture): they teach a
    detector what is NOT there. A picture with no label file at all was never annotated and is in neither list.

    Opening ten thousand small files takes a minute on some disks, so what was read is remembered (by file size and
    modification time) in a small cache file: the next call only looks at the files' dates."""
    images = index_images(images_folders)
    pairs, empty = [], []
    if os.path.isdir(labels_dir):
        with _cache_lock:                      # one scan at a time: the background warm-up and a window may overlap
            cache = _read_cache(labels_dir)
            fresh, changed = {}, False
            for entry in os.scandir(labels_dir):
                base, ext = os.path.splitext(entry.name)
                if ext.lower() != ".txt" or base not in images:
                    continue
                try:
                    st = entry.stat()
                except OSError:
                    continue
                hit = cache.get(entry.name)
                if hit and len(hit) >= 5 and hit[0] == st.st_mtime_ns and hit[1] == st.st_size:
                    has_rows, per_class, blank = bool(hit[2]), {int(k): v for k, v in hit[3].items()}, bool(hit[4])
                else:
                    rows, changed = read_label_rows(entry.path), True
                    has_rows, per_class = bool(rows), {}
                    for cls, *_ in rows:
                        per_class[cls] = per_class.get(cls, 0) + 1
                    blank = not has_rows and _is_blank(entry.path, st.st_size)
                _box_counts[entry.path] = per_class
                fresh[entry.name] = [st.st_mtime_ns, st.st_size, has_rows, {str(k): v for k, v in per_class.items()}, blank]
                if has_rows:
                    pairs.append((images[base], entry.path))
                elif blank:
                    empty.append((images[base], entry.path))
            if changed or len(fresh) != len(cache):
                _write_cache(labels_dir, fresh)
    pairs.sort(key=lambda p: _natural(os.path.basename(p[0])))
    empty.sort(key=lambda p: _natural(os.path.basename(p[0])))
    return pairs, empty


def labelled_pairs(images_folders, labels_dir):
    """[(image path, label path)] of the images that have at least one label row, in natural name order."""
    return scan_labels(images_folders, labels_dir)[0]


def split_contiguous(pairs, val_fraction=0.2):
    """(train, val): the last `val_fraction` of the pairs, as one block, is validation. Both sides get at least one."""
    if len(pairs) < 2:
        raise ValueError("Need at least 2 labelled images to split into train and validation.")
    n_val = min(len(pairs) - 1, max(1, int(round(len(pairs) * val_fraction))))
    return pairs[:len(pairs) - n_val], pairs[len(pairs) - n_val:]


def split_three(pairs, val_fraction=0.2, test_fraction=0.0, shuffle=False, seed=None):
    """(train, val, test). By position (the default) they are three blocks in this order: the first pictures train, then
    the validation block, then the test block, so near-duplicate video frames do not leak between them; with `shuffle`
    the pictures are mixed first. Train and validation get at least one picture; test may be empty."""
    if len(pairs) < 2:
        raise ValueError("Need at least 2 labelled images to split into train and validation.")
    items = list(pairs)
    if shuffle:
        import random
        random.Random(seed).shuffle(items)
    n = len(items)
    n_test = min(n - 2, int(round(n * test_fraction))) if test_fraction > 0 else 0
    n_val = min(n - n_test - 1, max(1, int(round(n * val_fraction))))
    n_train = n - n_val - n_test
    return items[:n_train], items[n_train:n_train + n_val], items[n_train + n_val:]


def class_counts(pairs, n_classes):
    counts = [0] * n_classes
    for _, label in pairs:
        per_class = _box_counts.get(label)
        if per_class is None:                  # a label that labelled_pairs did not scan (or changed since)
            per_class = {}
            for cls, *_ in read_label_rows(label):
                per_class[cls] = per_class.get(cls, 0) + 1
        for cls, n in per_class.items():
            if 0 <= cls < n_classes:
                counts[cls] += n
    return counts
