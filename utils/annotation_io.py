"""
Pure annotation file helpers: Pascal VOC XML and YOLO label text, built from and parsed into plain lists.

No tkinter, no utils.config, no workspace state: the desktop file_handler, the dataset importer, the analysis and
the web server all use these. Coordinates are always in ORIGINAL image pixels.
"""
import os
import xml.etree.ElementTree as ET
from xml.dom import minidom


def prettify_xml(elem):
    """Convert XML to pretty-printed string"""
    return minidom.parseString(ET.tostring(elem)).toprettyxml(indent="   ")

def build_voc_xml(img_name, img_shape, bboxes, polygons):
    """Build a Pascal VOC <annotation> element (Jelibox's schema: bbox/polygon
    objects, each with a <type>) from bboxes/polygons in ORIGINAL image
    coordinates. Pure - no config/state dependency, so it's reusable by the
    dataset importer as well as save_pascal_voc."""
    ann = ET.Element("annotation")
    ET.SubElement(ann, "folder").text = "dataset"
    ET.SubElement(ann, "filename").text = img_name
    size = ET.SubElement(ann, "size")
    ET.SubElement(size, "width").text = str(img_shape[1])
    ET.SubElement(size, "height").text = str(img_shape[0])
    ET.SubElement(size, "depth").text = str(img_shape[2] if len(img_shape) > 2 else 3)

    for bbox in bboxes:
        x1 = int(round(bbox[0]))
        y1 = int(round(bbox[1]))
        x2 = int(round(bbox[2]))
        y2 = int(round(bbox[3]))
        cls = bbox[4]
        obj = ET.SubElement(ann, "object")
        ET.SubElement(obj, "name").text = cls
        ET.SubElement(obj, "type").text = "bbox"
        bnd = ET.SubElement(obj, "bndbox")
        ET.SubElement(bnd, "xmin").text = str(max(0, x1))
        ET.SubElement(bnd, "ymin").text = str(max(0, y1))
        ET.SubElement(bnd, "xmax").text = str(max(0, x2))
        ET.SubElement(bnd, "ymax").text = str(max(0, y2))

    for polygon_data in polygons:
        points_orig = polygon_data[0]  # ORIGINAL coordinates (float)
        cls = polygon_data[1]

        obj = ET.SubElement(ann, "object")
        ET.SubElement(obj, "name").text = cls
        ET.SubElement(obj, "type").text = "polygon"

        poly_elem = ET.SubElement(obj, "polygon")
        for x, y in points_orig:
            x_int = int(round(x))
            y_int = int(round(y))
            pt = ET.SubElement(poly_elem, "point")
            ET.SubElement(pt, "x").text = str(max(0, x_int))
            ET.SubElement(pt, "y").text = str(max(0, y_int))

    return ann


def _yolo_lines_from_annotations(bboxes, polygons, classList, w, h):
    """
    Build YOLO-format label lines from a set of bboxes/polygons (all in
    ORIGINAL image coordinates).
    - If ONLY bboxes: detection format (class_id cx cy width height)
    - If polygons exist: ALL annotations use segmentation format
      (bboxes get converted to a 4-point polygon)
    """
    lines = []
    has_polygons = len(polygons) > 0

    if has_polygons:
        # Convert bboxes to polygon format
        for bbox in bboxes:
            x1, y1, x2, y2, cls = bbox
            if cls not in classList:
                continue
            idx = classList.index(cls)

            bbox_polygon_points = [
                (x1, y1),  # top-left
                (x2, y1),  # top-right
                (x2, y2),  # bottom-right
                (x1, y2),  # bottom-left
            ]
            normalized_points = []
            for x, y in bbox_polygon_points:
                x_norm = max(0, min(1, x / w))
                y_norm = max(0, min(1, y / h))
                normalized_points.append(f"{x_norm:.6f} {y_norm:.6f}")
            lines.append(f"{idx} " + " ".join(normalized_points))

        # Polygons in YOLO segmentation format
        for polygon_data in polygons:
            points_orig, cls = polygon_data[0], polygon_data[1]
            if cls not in classList:
                continue
            idx = classList.index(cls)

            normalized_points = []
            for x, y in points_orig:
                x_norm = max(0, min(1, x / w))
                y_norm = max(0, min(1, y / h))
                normalized_points.append(f"{x_norm:.6f} {y_norm:.6f}")
            if normalized_points:
                lines.append(f"{idx} " + " ".join(normalized_points))

    else:
        # Detection format only (no polygons)
        for bbox in bboxes:
            x1, y1, x2, y2, cls = bbox
            if cls not in classList:
                continue
            idx = classList.index(cls)

            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            cx = (x1 + x2) / 2 / w
            cy = (y1 + y2) / 2 / h
            lines.append(f"{idx} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

    return lines


def _read_voc_size(xml_path):
    """Read (height, width) from a VOC XML's <size> block, or None if missing/invalid."""
    try:
        root = ET.parse(xml_path).getroot()
        return _parse_voc_size(root)
    except (ET.ParseError, OSError):
        return None


def _parse_voc_size(root):
    """Extract (height, width) from an already-parsed VOC XML root, or None."""
    size = root.find("size")
    if size is None:
        return None
    try:
        w = int(float(size.find("width").text))
        h = int(float(size.find("height").text))
    except (AttributeError, TypeError, ValueError):
        return None
    if w <= 0 or h <= 0:
        return None
    return h, w


def _parse_voc_objects(root):
    """Extract (boxes, polygons) from an already-parsed VOC XML root.
    Same format handling as load_annotation_local, without the per-object
    logging - used for bulk backfill where thousands of files get parsed."""
    boxes = []
    polygons = []

    for obj in root.findall("object"):
        cls = obj.find("name").text

        poly_elem = obj.find("polygon")
        is_polygon = False
        points = []

        if poly_elem is not None:
            point_elems = poly_elem.findall("point")
            if point_elems:
                for point_elem in point_elems:
                    points.append((float(point_elem.find("x").text), float(point_elem.find("y").text)))
                is_polygon = len(points) >= 3

            if not is_polygon:
                roboflow_points = []
                i = 1
                while True:
                    xi_elem = poly_elem.find(f"x{i}")
                    yi_elem = poly_elem.find(f"y{i}")
                    if xi_elem is None or yi_elem is None:
                        break
                    roboflow_points.append((float(xi_elem.text), float(yi_elem.text)))
                    i += 1
                if len(roboflow_points) >= 3:
                    points = roboflow_points
                    is_polygon = True

        if is_polygon:
            polygons.append([points, cls])
        else:
            bb = obj.find("bndbox")
            if bb is not None:
                boxes.append([
                    int(round(float(bb.find("xmin").text))),
                    int(round(float(bb.find("ymin").text))),
                    int(round(float(bb.find("xmax").text))),
                    int(round(float(bb.find("ymax").text))),
                    cls
                ])

    return boxes, polygons


def write_annotations(voc_dir, labels_dir, img_name, img_shape, bboxes, polygons, classes):
    """Write one image's VOC XML (voc_dir) and YOLO label (labels_dir) from explicit lists. `img_shape` is the
    original (height, width[, depth])."""
    base = os.path.splitext(img_name)[0]
    with open(os.path.join(voc_dir, base + ".xml"), "w") as f:
        f.write(prettify_xml(build_voc_xml(img_name, img_shape, bboxes, polygons)))
    h, w = img_shape[:2]
    lines = _yolo_lines_from_annotations(bboxes, polygons, classes, w, h)
    with open(os.path.join(labels_dir, base + ".txt"), "w") as f:
        f.write("\n".join(lines))


def sync_yolo_labels(voc_dir, labels_dir, classes, progress_cb=None):
    """Make sure every VOC XML in voc_dir has a YOLO label in labels_dir, generating the missing ones (after a
    migration, or a vocdataset/ restored from a backup). Cheap to call at every open: only XML files without a .txt
    are parsed, each once. progress_cb(done, total) is called after each generated file. Returns how many."""
    if not os.path.isdir(voc_dir):
        return 0
    xml_files = [f for f in os.listdir(voc_dir) if f.lower().endswith(".xml")]
    if not xml_files:
        return 0
    os.makedirs(labels_dir, exist_ok=True)
    existing = set(os.listdir(labels_dir))
    missing = [f for f in xml_files if os.path.splitext(f)[0] + ".txt" not in existing]
    generated = 0
    for xml_file in missing:
        base = os.path.splitext(xml_file)[0]
        try:
            root = ET.parse(os.path.join(voc_dir, xml_file)).getroot()
        except (ET.ParseError, OSError) as exc:
            print(f"[FILE_HANDLER] Skipping YOLO backfill for {xml_file}: {exc}")
            continue
        size = _parse_voc_size(root)
        if size is None:
            print(f"[FILE_HANDLER] Skipping YOLO backfill for {xml_file}: no <size> in XML")
            continue
        h, w = size
        boxes, polygons = _parse_voc_objects(root)
        with open(os.path.join(labels_dir, base + ".txt"), "w") as f:
            f.write("\n".join(_yolo_lines_from_annotations(boxes, polygons, classes, w, h)))
        generated += 1
        if progress_cb:
            progress_cb(generated, len(missing))
    if generated:
        print(f"[FILE_HANDLER] Backfilled {generated} missing YOLO label(s) from VOC XML")
    return generated
