"""
Universal dataset importer for Jelibox.

Rather than building a format-specific importer (e.g. "Import from
Roboflow") that only understands one export layout, this scans an arbitrary
folder tree - recursively, regardless of how it's organized internally -
for images plus at most one annotation format:

  - Pascal VOC   (.xml)
  - YOLO         (.txt)
  - COCO         (.json)

...and converts whatever it finds into Jelibox's own native storage, the
same two representations every other part of the app already expects:

  datasetsInput/<workspace>-<N>/   the images (copied in)
  vocdataset/<workspace>/          VOC XML (source of truth for the editor)
  YOLOdataset/<workspace>/labels/  YOLO .txt (used for training/export)
  configs/<workspace>.json         class list (only written if the
                                    workspace doesn't already have one)

A folder that mixes more than one annotation format is rejected - Jelibox
has no way to know which one is authoritative for a given image.

No tkinter/config imports here on purpose, same as workspace_manager - this
must be usable from the workspace picker before any workspace is open.
"""

import os
import json
import shutil
import xml.etree.ElementTree as ET

from . import workspace_config
from .workspace_manager import (
    DATASETS_ROOT, CONFIGS_ROOT, BASE_DIR, IMAGE_EXTENSIONS,
    next_instance_name, existing_classes_for_workspace,
)
from .file_handler import build_voc_xml, prettify_xml, _yolo_lines_from_annotations

CLASS_LIST_FILENAMES = {"classes.txt", "obj.names"}


# ============================================================
#  Scanning
# ============================================================

def _walk_files(root_folder):
    for dirpath, _dirnames, filenames in os.walk(root_folder):
        for fname in filenames:
            yield os.path.join(dirpath, fname)


def _looks_like_coco(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    return isinstance(data, dict) and "images" in data and "annotations" in data


def scan_dataset_folder(root_folder):
    """
    Recursively scan root_folder for images and annotation files, wherever
    they're nested. Returns:
      {
        'images': {basename: path},
        'voc': {basename: path},          # .xml
        'yolo': {basename: path},         # .txt, excluding class-list files
        'coco': [path, ...],              # .json that look like COCO
        'yolo_classes_file': path or None,  # classes.txt / obj.names / data.yaml
      }
    """
    images, voc, yolo, coco = {}, {}, {}, []
    yolo_classes_file = None

    for path in _walk_files(root_folder):
        fname = os.path.basename(path)
        base, ext = os.path.splitext(fname)
        ext = ext.lower()

        if ext in IMAGE_EXTENSIONS:
            images[base] = path
        elif ext == ".xml":
            voc[base] = path
        elif ext == ".txt":
            if fname.lower() in CLASS_LIST_FILENAMES:
                yolo_classes_file = path
            else:
                yolo[base] = path
        elif ext == ".json":
            if _looks_like_coco(path):
                coco.append(path)
        elif ext in (".yaml", ".yml") and yolo_classes_file is None:
            yolo_classes_file = path

    return {
        "images": images, "voc": voc, "yolo": yolo, "coco": coco,
        "yolo_classes_file": yolo_classes_file,
    }


def sanitize_filename_prefix(prefix):
    """Validate a filename prefix (e.g. 'test-' -> test-1.jpg, test-2.jpg,
    ...). Empty/None means "keep original filenames", not an error."""
    prefix = (prefix or "").strip()
    if not prefix:
        return ""
    if not all(c.isalnum() or c in "_-" for c in prefix):
        raise ValueError("Prefix can only contain letters, numbers, underscore (_), and dash (-).")
    return prefix


def build_rename_map(image_bases, prefix):
    """
    {original_basename: new_basename} for every image, numbered in a stable
    (sorted) order - e.g. prefix 'test-' turns pic_a.jpg/pic_b.jpg into
    test-1.jpg/test-2.jpg, and the matching annotation files follow suit.
    Empty prefix means no renaming (identity map).
    """
    if not prefix:
        return {base: base for base in image_bases}
    return {base: f"{prefix}{i}" for i, base in enumerate(sorted(image_bases), start=1)}


def detect_dataset_format(scan):
    """
    Returns 'voc', 'yolo', 'coco', or None (images only). Raises ValueError
    if more than one annotation format is present in the same folder.
    """
    present = []
    if scan["voc"]:
        present.append(("voc", "Pascal VOC (.xml)"))
    if scan["yolo"]:
        present.append(("yolo", "YOLO (.txt)"))
    if scan["coco"]:
        present.append(("coco", "COCO (.json)"))

    if len(present) > 1:
        names = " and ".join(label for _key, label in present)
        raise ValueError(
            f"This folder contains more than one annotation format ({names}). "
            "A dataset can only use a single annotation format at a time - "
            "please import a folder that contains just one."
        )

    return present[0][0] if present else None


# ============================================================
#  Class name resolution
# ============================================================

def _parse_yolo_class_names(path):
    """Read class names (index order) from classes.txt / obj.names / a
    data.yaml|yml's `names:` field. Returns [] if unrecognized/unreadable."""
    if path is None:
        return []

    fname = os.path.basename(path).lower()
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    except OSError:
        return []

    if fname.endswith((".yaml", ".yml")):
        try:
            import yaml
            data = yaml.safe_load(text)
            names = data.get("names") if isinstance(data, dict) else None
            if isinstance(names, dict):
                return [names[k] for k in sorted(names, key=lambda k: int(k))]
            if isinstance(names, list):
                return [str(n) for n in names]
        except Exception:
            pass
        return []

    # classes.txt / obj.names: one class per line
    return [line.strip() for line in text.splitlines() if line.strip()]


def yolo_classes_resolved(scan):
    """True if a usable class-name source (classes.txt / obj.names /
    data.yaml) was found for a YOLO dataset. False means _import_yolo_annotations
    will fall back to generic names (class_0, class_1, ...) since the .txt
    labels only carry numeric indices, not names."""
    return bool(_parse_yolo_class_names(scan.get("yolo_classes_file")))


# ============================================================
#  Image size lookup
# ============================================================

def _image_size(path):
    """(width, height) of an image file, or None if it can't be read."""
    try:
        from PIL import Image
        with Image.open(path) as img:
            return img.size  # (w, h)
    except Exception:
        return None


# ============================================================
#  VOC import - already Jelibox's native annotation format, so this mostly
#  copies XML in (our loader already tolerates plain Pascal VOC bndboxes,
#  our own <type> tag, and Roboflow-style <polygon><x1/><y1/>...), then
#  backfills YOLO labels from it.
# ============================================================

def _import_voc_annotations(scan, voc_dir, yolo_dir, class_order, rename_map, progress_cb, done, total):
    from .file_handler import _parse_voc_size, _parse_voc_objects

    class_set = set(class_order)
    annotated = 0

    # Pass 1: copy every VOC XML that has a matching image, and collect
    # every class name mentioned - the class list has to be complete before
    # any YOLO label (which stores classes as an index) can be written.
    matched = [(base, path) for base, path in scan["voc"].items() if base in scan["images"]]
    parsed = {}
    for base, xml_path in matched:
        try:
            root = ET.parse(xml_path).getroot()
        except ET.ParseError:
            continue
        boxes, polygons = _parse_voc_objects(root)
        size = _parse_voc_size(root)
        parsed[base] = (root, boxes, polygons, size)
        for cls in [b[4] for b in boxes] + [p[1] for p in polygons]:
            if cls not in class_set:
                class_set.add(cls)
                class_order.append(cls)

    for base, (root, boxes, polygons, size) in parsed.items():
        new_base = rename_map.get(base, base)

        # Keep <filename> in sync with the renamed image, so the XML we
        # write doesn't point at a file that no longer exists under that name.
        filename_elem = root.find("filename")
        if filename_elem is not None:
            ext = os.path.splitext(scan["images"][base])[1]
            filename_elem.text = new_base + ext

        dest_xml = os.path.join(voc_dir, new_base + ".xml")
        with open(dest_xml, "w", encoding="utf-8") as f:
            f.write(prettify_xml(root))

        if size is not None:
            h, w = size
            lines = _yolo_lines_from_annotations(boxes, polygons, class_order, w, h)
            with open(os.path.join(yolo_dir, new_base + ".txt"), "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
            annotated += 1

        done += 1
        if progress_cb:
            progress_cb(done, total, f"Converting annotations... {new_base}")

    return class_order, annotated, done


# ============================================================
#  YOLO import - normalized coords need image dimensions to convert back
#  to the absolute-pixel VOC XML Jelibox's editor expects.
# ============================================================

def _import_yolo_annotations(scan, voc_dir, yolo_dir, class_order, rename_map, progress_cb, done, total):
    yolo_class_names = _parse_yolo_class_names(scan["yolo_classes_file"])

    matched = [(base, path) for base, path in scan["yolo"].items() if base in scan["images"]]

    # Figure out how many classes are actually referenced so unnamed ones
    # still get *some* usable name instead of failing the import outright.
    max_idx = -1
    for _base, txt_path in matched:
        try:
            with open(txt_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.split()
                    if parts:
                        max_idx = max(max_idx, int(float(parts[0])))
        except (OSError, ValueError):
            continue

    index_to_name = {}
    class_set = set(class_order)
    for i in range(max_idx + 1):
        name = yolo_class_names[i] if i < len(yolo_class_names) else f"class_{i}"
        index_to_name[i] = name
        if name not in class_set:
            class_set.add(name)
            class_order.append(name)

    annotated = 0
    for base, txt_path in matched:
        img_path = scan["images"][base]
        size = _image_size(img_path)
        if size is None:
            done += 1
            if progress_cb:
                progress_cb(done, total, f"Skipped unreadable image: {base}")
            continue
        w, h = size

        with open(txt_path, "r", encoding="utf-8") as f:
            raw_lines = [ln.split() for ln in f if ln.strip()]

        boxes, polygons = [], []
        for parts in raw_lines:
            try:
                idx = int(float(parts[0]))
                nums = [float(v) for v in parts[1:]]
            except ValueError:
                continue
            cls = index_to_name.get(idx, f"class_{idx}")

            if len(nums) == 4:
                cx, cy, bw, bh = nums
                abs_w, abs_h = bw * w, bh * h
                abs_cx, abs_cy = cx * w, cy * h
                x1, y1 = abs_cx - abs_w / 2, abs_cy - abs_h / 2
                boxes.append([x1, y1, x1 + abs_w, y1 + abs_h, cls])
            elif len(nums) >= 6 and len(nums) % 2 == 0:
                points = [(nums[i] * w, nums[i + 1] * h) for i in range(0, len(nums), 2)]
                polygons.append([points, cls])

        new_base = rename_map.get(base, base)
        ext = os.path.splitext(img_path)[1]
        img_name = new_base + ext
        ann = build_voc_xml(img_name, (h, w, 3), boxes, polygons)
        with open(os.path.join(voc_dir, new_base + ".xml"), "w", encoding="utf-8") as f:
            f.write(prettify_xml(ann))
        shutil.copy2(txt_path, os.path.join(yolo_dir, new_base + ".txt"))
        annotated += 1

        done += 1
        if progress_cb:
            progress_cb(done, total, f"Converting annotations... {new_base}")

    return class_order, annotated, done


# ============================================================
#  COCO import - self-describing (categories are embedded), and boxes /
#  polygon segmentations are already in absolute pixel coordinates.
# ============================================================

def _load_coco_files(paths):
    """Merge one or more COCO JSON files into a single
    {image_file_name: {'width', 'height', 'objects': [(bbox_or_None, polygon_or_None, class_name), ...]}}"""
    merged = {}
    for path in paths:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue

        categories = {c["id"]: c["name"] for c in data.get("categories", [])}
        images_by_id = {img["id"]: img for img in data.get("images", [])}

        for ann in data.get("annotations", []):
            img = images_by_id.get(ann.get("image_id"))
            if img is None:
                continue
            file_name = os.path.basename(img.get("file_name", ""))
            if not file_name:
                continue

            entry = merged.setdefault(file_name, {
                "width": img.get("width"), "height": img.get("height"), "objects": []
            })
            cls = categories.get(ann.get("category_id"), str(ann.get("category_id")))

            seg = ann.get("segmentation")
            polygon = None
            if isinstance(seg, list) and seg and isinstance(seg[0], (int, float)):
                flat = seg
                polygon = [(flat[i], flat[i + 1]) for i in range(0, len(flat) - 1, 2)]
            elif isinstance(seg, list) and seg and isinstance(seg[0], list):
                flat = seg[0]
                polygon = [(flat[i], flat[i + 1]) for i in range(0, len(flat) - 1, 2)]
            # RLE segmentation (dict with 'counts') isn't supported - falls
            # back to the bbox below instead of being dropped entirely.

            bbox = None
            if polygon is None and "bbox" in ann:
                x, y, bw, bh = ann["bbox"]
                bbox = (x, y, x + bw, y + bh)

            if bbox is not None or polygon is not None:
                entry["objects"].append((bbox, polygon, cls))

    return merged


def _import_coco_annotations(scan, voc_dir, yolo_dir, class_order, rename_map, progress_cb, done, total):
    coco_data = _load_coco_files(scan["coco"])

    # Match COCO's file_name entries to the images we actually found, by
    # basename - COCO exports commonly nest images under images/<split>/.
    by_basename = {}
    for file_name, entry in coco_data.items():
        base = os.path.splitext(file_name)[0]
        if base in scan["images"]:
            by_basename[base] = entry

    class_set = set(class_order)
    for entry in by_basename.values():
        for _bbox, _polygon, cls in entry["objects"]:
            if cls not in class_set:
                class_set.add(cls)
                class_order.append(cls)

    annotated = 0
    for base, entry in by_basename.items():
        img_path = scan["images"][base]
        w, h = entry.get("width"), entry.get("height")
        if not w or not h:
            size = _image_size(img_path)
            if size is None:
                done += 1
                if progress_cb:
                    progress_cb(done, total, f"Skipped unreadable image: {base}")
                continue
            w, h = size

        boxes, polygons = [], []
        for bbox, polygon, cls in entry["objects"]:
            if polygon is not None:
                polygons.append([polygon, cls])
            elif bbox is not None:
                boxes.append([bbox[0], bbox[1], bbox[2], bbox[3], cls])

        new_base = rename_map.get(base, base)
        ext = os.path.splitext(img_path)[1]
        img_name = new_base + ext
        ann = build_voc_xml(img_name, (h, w, 3), boxes, polygons)
        with open(os.path.join(voc_dir, new_base + ".xml"), "w", encoding="utf-8") as f:
            f.write(prettify_xml(ann))

        lines = _yolo_lines_from_annotations(boxes, polygons, class_order, w, h)
        with open(os.path.join(yolo_dir, new_base + ".txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        annotated += 1

        done += 1
        if progress_cb:
            progress_cb(done, total, f"Converting annotations... {new_base}")

    return class_order, annotated, done


# ============================================================
#  Orchestrator
# ============================================================

def import_dataset(source_folder, workspace_name, prefix=None, progress_cb=None):
    """
    Import an external folder (any layout, searched recursively) as a new
    datasetsInput/<workspace>-<N> instance, converting whichever single
    annotation format it contains (or none) into Jelibox's native VOC XML +
    YOLO label storage.

    If `prefix` is given, every imported image (and its matching annotation
    file) is renamed to "<prefix><N>" in sorted order - e.g. prefix "test-"
    turns two images into test-1.jpg/test-2.jpg (and test-1.xml/test-1.txt
    etc. alongside them) instead of keeping their original filenames.

    progress_cb(done, total, label), if given, is called repeatedly while
    images are copied and annotations converted.

    Returns a summary dict. Raises ValueError on bad or conflicting input.
    """
    if not source_folder or not os.path.isdir(source_folder):
        raise ValueError(f"'{source_folder}' is not a folder.")

    prefix = sanitize_filename_prefix(prefix)  # raises ValueError on bad chars

    scan = scan_dataset_folder(source_folder)
    if not scan["images"]:
        raise ValueError("No images found in that folder (all subfolders were searched).")

    fmt = detect_dataset_format(scan)  # raises on conflicting formats
    rename_map = build_rename_map(scan["images"].keys(), prefix)

    instance_name = next_instance_name(workspace_name)
    dest_images = os.path.join(DATASETS_ROOT, instance_name)
    voc_dir = os.path.join(BASE_DIR, "vocdataset", workspace_name)
    yolo_dir = os.path.join(BASE_DIR, "YOLOdataset", workspace_name, "labels")
    os.makedirs(dest_images, exist_ok=True)
    os.makedirs(voc_dir, exist_ok=True)
    os.makedirs(yolo_dir, exist_ok=True)

    n_images = len(scan["images"])
    total = n_images + (len(scan["voc"]) or len(scan["yolo"]) or len(scan["coco"]) or 0)
    done = 0

    for base, path in scan["images"].items():
        new_base = rename_map.get(base, base)
        ext = os.path.splitext(path)[1]
        shutil.copy2(path, os.path.join(dest_images, new_base + ext))
        done += 1
        if progress_cb:
            progress_cb(done, total, f"Copying images... {done}/{n_images}")

    class_order = list(existing_classes_for_workspace(workspace_name) or [])
    annotated = 0

    if fmt == "voc":
        class_order, annotated, done = _import_voc_annotations(
            scan, voc_dir, yolo_dir, class_order, rename_map, progress_cb, done, total)
    elif fmt == "yolo":
        class_order, annotated, done = _import_yolo_annotations(
            scan, voc_dir, yolo_dir, class_order, rename_map, progress_cb, done, total)
    elif fmt == "coco":
        class_order, annotated, done = _import_coco_annotations(
            scan, voc_dir, yolo_dir, class_order, rename_map, progress_cb, done, total)

    if class_order and existing_classes_for_workspace(workspace_name) is None:
        workspace_config.set_classes(workspace_name, class_order)

    format_labels = {"voc": "Pascal VOC", "yolo": "YOLO", "coco": "COCO", None: None}
    return {
        "instance_name": instance_name,
        "image_count": n_images,
        "annotated_count": annotated,
        "format": format_labels[fmt],
        "classes": class_order,
    }
