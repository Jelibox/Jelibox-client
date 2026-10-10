"""
Dataset analysis (the numbers behind the Analyze Dataset dashboard). No matplotlib / Tk here, so it can be tested
anywhere and the window can import it cheaply.

Everything is read from the workspace's VOC XML files (vocdataset/<workspace>/*.xml): the image size is in the file,
so no image is ever opened. A polygon counts through its bounding box. Resizing is only calculated: `view(...)`
turns every object's width / height into what it would be after the image is resized to the model's input size.

Letterbox (what Ultralytics does): scale = size / longest side, the rest is padded.
Stretch: the image is squeezed to size x size, so width and height scale separately.
"""
import os
import xml.etree.ElementTree as ET
from collections import Counter

SMALL_LIMIT = 32                      # px: an object with a side below this is flagged
COMPARE_SIZES = (320, 416, 512, 640, 800, 960, 1280)
LETTERBOX, STRETCH = "letterbox", "stretch"


class Analysis:
    def __init__(self, workspace):
        self.workspace = workspace
        self.instances = []           # [{"name", "images", "labelled"}]
        self.n_images = 0
        self.n_labelled = 0           # images with at least one object
        self.n_xml = 0
        self.n_unmatched_xml = 0      # annotation files without an image in the workspace
        self.n_no_size = 0            # annotation files that carry no image size
        self.classes = []             # names, most objects first
        self.cls = []                 # per object: class name
        self.w, self.h = [], []       # per object: width / height in the original image (px)
        self.iw, self.ih = [], []     # per object: size of its image (0 when the file has none)
        self.kind = []                # per object: "bbox" | "polygon"
        self.src = []                 # per object: the XML file it came from
        self.image_sizes = Counter()  # (width, height) -> annotated images
        self.per_image = []           # objects in each labelled image

    @property
    def n_objects(self):
        return len(self.w)

    @property
    def n_unlabelled(self):
        return self.n_images - self.n_labelled

    def class_counts(self):
        counts = Counter(self.cls)
        return [(name, counts[name]) for name in self.classes]


def _num(node, tag):
    try:
        return float(node.findtext(tag))
    except (TypeError, ValueError):
        return None


def _object_geometry(obj):
    """(class, kind, width, height) of one VOC <object> (a polygon through its bounding box), or None."""
    name = (obj.findtext("name") or "").strip()
    box = obj.find("bndbox")
    poly = obj.find("polygon")
    if box is not None:
        vals = [_num(box, t) for t in ("xmin", "ymin", "xmax", "ymax")]
        if None in vals:
            return None
        xs, ys, kind = (vals[0], vals[2]), (vals[1], vals[3]), "bbox"
    elif poly is not None:
        pts = [(_num(p, "x"), _num(p, "y")) for p in poly.findall("point")]
        pts = [p for p in pts if None not in p]
        if len(pts) < 3:
            return None
        xs, ys, kind = [p[0] for p in pts], [p[1] for p in pts], "polygon"
    else:
        return None
    w, h = max(xs) - min(xs), max(ys) - min(ys)
    return (name, kind, w, h) if name and w > 0 and h > 0 else None


def read_xml(path):
    """(image width, image height, [(class, kind, width, height)]); sizes are 0 when the file has none.
    None when the file cannot be parsed."""
    try:
        root = ET.parse(path).getroot()
    except (ET.ParseError, OSError):
        return None
    size = root.find("size")
    iw = (_num(size, "width") or 0) if size is not None else 0
    ih = (_num(size, "height") or 0) if size is not None else 0
    objects = [g for g in (_object_geometry(obj) for obj in root.iter("object")) if g]
    return iw, ih, objects


def _image_names(folder):
    from .workspace_manager import IMAGE_EXTENSIONS
    try:
        return {os.path.splitext(f)[0] for f in os.listdir(folder) if f.lower().endswith(IMAGE_EXTENSIONS)}
    except OSError:
        return set()


def analyze(workspace, instance_folders, voc_dir, progress_cb=None):
    """Analysis of one workspace. `instance_folders` = {instance name: image folder}, `voc_dir` = its XML folder.
    progress_cb(done, total, label), if given, is called while the files are read."""
    data = Analysis(workspace)
    owner = {}                                        # image base name -> instance
    if progress_cb:
        progress_cb(0, 1, "Counting images...")
    for name, folder in instance_folders.items():
        stems = _image_names(folder)
        data.instances.append({"name": name, "images": len(stems), "labelled": 0})
        data.n_images += len(stems)
        for stem in stems:
            owner.setdefault(stem, data.instances[-1])
    xmls = sorted(f for f in os.listdir(voc_dir) if f.lower().endswith(".xml")) if os.path.isdir(voc_dir) else []
    total = len(xmls)
    for i, fname in enumerate(xmls):
        if progress_cb and i % 25 == 0:
            progress_cb(i, total, f"Reading annotations... {i}/{total}")
        parsed = read_xml(os.path.join(voc_dir, fname))
        if parsed is None:
            continue
        data.n_xml += 1
        iw, ih, objects = parsed
        instance = owner.get(os.path.splitext(fname)[0])
        if instance is None:
            data.n_unmatched_xml += 1
            continue
        if not (iw > 0 and ih > 0):
            iw = ih = 0
            data.n_no_size += 1
        else:
            data.image_sizes[(int(iw), int(ih))] += 1
        if not objects:
            continue
        instance["labelled"] += 1
        data.n_labelled += 1
        data.per_image.append(len(objects))
        for cls, kind, w, h in objects:
            data.cls.append(cls)
            data.kind.append(kind)
            data.src.append(os.path.join(voc_dir, fname))
            data.w.append(w)
            data.h.append(h)
            data.iw.append(iw)
            data.ih.append(ih)
    data.classes = [name for name, _ in Counter(data.cls).most_common()]
    if progress_cb:
        progress_cb(total, total, f"Reading annotations... {total}/{total}")
    return data


def analyze_workspace(workspace, progress_cb=None):
    from . import workspace_manager as wm
    folders = {i: wm.instance_path(i) for i in wm.list_workspaces().get(workspace, [])}
    return analyze(workspace, folders, os.path.join(wm.BASE_DIR, "vocdataset", workspace), progress_cb)


def yolo_dir_for(workspace):
    from . import workspace_manager as wm
    return os.path.join(wm.BASE_DIR, "YOLOdataset", workspace, "labels")


def view(data, size=None, mode=LETTERBOX):
    """{"cls", "w", "h"} numpy arrays of every object. With `size`, the width / height the object would have after
    its image is resized to size x size (objects whose image size is unknown are left out)."""
    import numpy as np
    cls = np.array(data.cls, dtype=object)
    w, h = np.array(data.w, float), np.array(data.h, float)
    if size is None:
        return {"cls": cls, "w": w, "h": h}
    iw, ih = np.array(data.iw, float), np.array(data.ih, float)
    known = (iw > 0) & (ih > 0)
    cls, w, h, iw, ih = cls[known], w[known], h[known], iw[known], ih[known]
    if mode == STRETCH:
        return {"cls": cls, "w": w * size / iw, "h": h * size / ih}
    scale = size / np.maximum(iw, ih)
    return {"cls": cls, "w": w * scale, "h": h * scale}


def smallest(v):
    """(w, h) of the object with the smallest area, or None."""
    import numpy as np
    if not len(v["w"]):
        return None
    i = int(np.argmin(v["w"] * v["h"]))
    return float(v["w"][i]), float(v["h"][i])


def small_summary(v, limit=SMALL_LIMIT):
    """What is under the limit: {"count", "percent", "by_class": [(class, count)], "smallest"} - None when nothing
    has a side below `limit`."""
    import numpy as np
    small = np.minimum(v["w"], v["h"]) < limit
    count = int(small.sum())
    if not count:
        return None
    by_class = Counter(v["cls"][small].tolist()).most_common()
    return {"count": count, "percent": 100.0 * count / len(small), "by_class": by_class, "smallest": smallest(v)}


def spread(values):
    """min / median / mean / max of a list of numbers (None for an empty one)."""
    import numpy as np
    if not len(values):
        return None
    a = np.asarray(values, float)
    return {"min": float(a.min()), "median": float(np.median(a)), "mean": float(a.mean()), "max": float(a.max())}


def compare(data, sizes=COMPARE_SIZES, mode=LETTERBOX, limit=SMALL_LIMIT):
    """One row per input size: the smallest object, the median shortest side, and how many are under the limit."""
    import numpy as np
    rows = []
    for size in sizes:
        v = view(data, size, mode)
        n = len(v["w"])
        short = np.minimum(v["w"], v["h"])
        rows.append({
            "size": size,
            "objects": n,
            "smallest": smallest(v),
            "median_side": float(np.median(short)) if n else None,
            "small": int((short < limit).sum()),
            "small_percent": 100.0 * float((short < limit).sum()) / n if n else 0.0,
        })
    return rows


# ---- removing the small objects from the dataset ----------------------------------------------------------------

def _scale_for(iw, ih, size, mode):
    """(sx, sy) from an image of iw x ih to the model input size, or None when the image size is unknown."""
    if size is None:
        return 1.0, 1.0
    if not (iw > 0 and ih > 0):
        return None
    if mode == STRETCH:
        return size / iw, size / ih
    return size / max(iw, ih), size / max(iw, ih)


def small_sources(data, size=None, mode=LETTERBOX, limit=SMALL_LIMIT):
    """The XML files that hold at least one object with a side under `limit` in this view (original sizes, or after
    the calculated resize to `size`)."""
    import numpy as np
    w, h = np.array(data.w, float), np.array(data.h, float)
    iw, ih = np.array(data.iw, float), np.array(data.ih, float)
    known = (iw > 0) & (ih > 0)
    if size is None:
        sx = sy = np.ones(len(w))
    elif mode == STRETCH:
        sx, sy = size / np.where(known, iw, 1.0), size / np.where(known, ih, 1.0)
    else:
        sx = sy = size / np.maximum(np.where(known, iw, 1.0), np.where(known, ih, 1.0))
    small = (np.minimum(w * sx, h * sy) < limit) & (known if size is not None else True)
    seen, files = set(), []
    for flag, src in zip(small.tolist(), data.src):
        if flag and src not in seen:
            seen.add(src)
            files.append(src)
    return files


def remove_small(data, yolo_dir, classes, size=None, mode=LETTERBOX, limit=SMALL_LIMIT, progress_cb=None):
    """Delete from the dataset every object with a side under `limit` in this view. Each affected XML is rewritten
    without those objects and its YOLO label is written again from what is left (with `classes` = the workspace's
    class list). Images are never touched. Returns {"objects", "files", "emptied"} (emptied = images left with no
    object at all)."""
    from .file_handler import prettify_xml, _parse_voc_objects, _parse_voc_size, _yolo_lines_from_annotations
    files = small_sources(data, size, mode, limit)
    removed = changed = emptied = 0
    for i, path in enumerate(files, start=1):
        if progress_cb:
            progress_cb(i - 1, len(files), f"Removing objects... {i}/{len(files)}")
        try:
            root = ET.parse(path).getroot()
        except (ET.ParseError, OSError):
            continue
        shape = _parse_voc_size(root)                                   # (height, width) or None
        ih, iw = shape if shape else (0, 0)
        scale = _scale_for(iw, ih, size, mode)
        if scale is None:
            continue
        gone = 0
        for obj in list(root.findall("object")):
            geometry = _object_geometry(obj)
            if geometry and min(geometry[2] * scale[0], geometry[3] * scale[1]) < limit:
                root.remove(obj)
                gone += 1
        if not gone:
            continue
        with open(path, "w", encoding="utf-8") as f:
            f.write(prettify_xml(root))
        if shape:                                                       # a YOLO label needs the image size
            boxes, polygons = _parse_voc_objects(root)
            base = os.path.splitext(os.path.basename(path))[0]
            lines = _yolo_lines_from_annotations(boxes, polygons, list(classes), iw, ih)
            os.makedirs(yolo_dir, exist_ok=True)
            with open(os.path.join(yolo_dir, base + ".txt"), "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
        removed += gone
        changed += 1
        if not root.findall("object"):
            emptied += 1
    if progress_cb:
        progress_cb(len(files), len(files), "Done")
    return {"objects": removed, "files": changed, "emptied": emptied}
