"""
File handling functions for VOC and YOLO formats
Supports both bounding boxes and polygons
"""
import os
import xml.etree.ElementTree as ET
from xml.dom import minidom
from .config import vocdataset_folder, yolo_labels_folder, CLASSLIST, state

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


def save_pascal_voc(img_name, img_shape):
    """Save annotations in Pascal VOC format (bboxes and polygons)"""
    xml_path = os.path.join(vocdataset_folder, os.path.splitext(img_name)[0] + ".xml")
    ann = build_voc_xml(img_name, img_shape, state.bboxes, state.polygons)
    with open(xml_path, "w") as f:
        f.write(prettify_xml(ann))
    print(f"[INFO] Saved VOC: {xml_path}")

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


def save_yolo_label(img_name, orig_img, classList):
    """
    Save YOLO format labels for the currently annotated image (state.bboxes /
    state.polygons). Images are never copied - YOLO training/export read them
    straight out of datasetsInput, keyed by filename against these labels.
    """
    base = os.path.splitext(img_name)[0]
    label_path = os.path.join(yolo_labels_folder, base + ".txt")
    h, w, c = orig_img

    lines = _yolo_lines_from_annotations(state.bboxes, state.polygons, classList, w, h)

    with open(label_path, "w") as f:
        f.write("\n".join(lines))

    print(f"[INFO] Saved YOLO label: {label_path} ({len(lines)} annotations)")


def save_annotations(img_name, img_shape, bboxes, polygons, classList):
    """Write one image's VOC XML and YOLO label from explicit lists, without touching the GUI state.
    Used by Auto-annotate all, which labels images other than the one on screen."""
    base = os.path.splitext(img_name)[0]
    with open(os.path.join(vocdataset_folder, base + ".xml"), "w") as f:
        f.write(prettify_xml(build_voc_xml(img_name, img_shape, bboxes, polygons)))
    h, w = img_shape[:2]
    lines = _yolo_lines_from_annotations(bboxes, polygons, classList, w, h)
    with open(os.path.join(yolo_labels_folder, base + ".txt"), "w") as f:
        f.write("\n".join(lines))


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


def sync_yolo_labels(classList, progress_cb=None):
    """
    Ensure every VOC XML annotation has a matching YOLO label in
    yolo_labels_folder, generating any that are missing (e.g. after
    migrating a workspace, or restoring vocdataset/ from backup).

    Cheap to call on every workspace open: only XML files that don't already
    have a .txt counterpart get parsed - each parsed exactly once.

    progress_cb(done, total), if given, is called after each missing file is
    processed so a caller can show progress on a first-time bulk backfill.

    Returns the number of YOLO labels generated.
    """
    if not os.path.isdir(vocdataset_folder):
        return 0

    xml_files = [f for f in os.listdir(vocdataset_folder) if f.lower().endswith('.xml')]
    if not xml_files:
        return 0

    os.makedirs(yolo_labels_folder, exist_ok=True)
    existing_labels = set(os.listdir(yolo_labels_folder))

    missing = [f for f in xml_files if os.path.splitext(f)[0] + ".txt" not in existing_labels]

    generated = 0
    for xml_file in missing:
        base = os.path.splitext(xml_file)[0]
        xml_path = os.path.join(vocdataset_folder, xml_file)

        try:
            root = ET.parse(xml_path).getroot()
        except (ET.ParseError, OSError) as exc:
            print(f"[FILE_HANDLER] Skipping YOLO backfill for {xml_file}: {exc}")
            continue

        size = _parse_voc_size(root)
        if size is None:
            print(f"[FILE_HANDLER] Skipping YOLO backfill for {xml_file}: no <size> in XML")
            continue
        h, w = size

        boxes, polygons = _parse_voc_objects(root)
        lines = _yolo_lines_from_annotations(boxes, polygons, classList, w, h)

        with open(os.path.join(yolo_labels_folder, base + ".txt"), "w") as f:
            f.write("\n".join(lines))
        generated += 1

        if progress_cb:
            progress_cb(generated, len(missing))

    if generated:
        print(f"[FILE_HANDLER] Backfilled {generated} missing YOLO label(s) from VOC XML")

    return generated


def load_annotation_local(img_name_local):
    """Load annotations from VOC XML file (both bboxes and polygons)
    Supports multiple formats:
    - Our format: <type>polygon</type> with <point><x/><y/></point>
    - Roboflow format: <polygon><x1/><y1/><x2/><y2/>... (no <type> element)
    """
    xml_path = os.path.join(vocdataset_folder, os.path.splitext(img_name_local)[0] + ".xml")
    if not os.path.exists(xml_path):
        return [], []  # Return empty bboxes and polygons
    
    tree = ET.parse(xml_path)
    root = tree.getroot()
    boxes = []
    polygons = []
    
    for obj in root.findall("object"):
        cls = obj.find("name").text
        
        # ===== CHECK FOR POLYGON FIRST (both formats) =====
        poly_elem = obj.find("polygon")
        is_polygon = False
        points = []
        
        if poly_elem is not None:
            # Try our format first: <point><x/><y/></point>
            point_elems = poly_elem.findall("point")
            if point_elems:
                for point_elem in point_elems:
                    x = float(point_elem.find("x").text)
                    y = float(point_elem.find("y").text)
                    # Store in ORIGINAL coordinates as float (no scale conversion, no rounding)
                    points.append((x, y))
                is_polygon = len(points) >= 3
                if is_polygon:
                    print(f"[FILE_HANDLER] Loaded polygon (our format): {cls} with {len(points)} points")
            
            # Try Roboflow format: <x1/><y1/><x2/><y2/>... <xN/><yN/>
            if not is_polygon:
                roboflow_points = []
                i = 1
                while True:
                    xi_elem = poly_elem.find(f"x{i}")
                    yi_elem = poly_elem.find(f"y{i}")
                    if xi_elem is None or yi_elem is None:
                        break
                    x = float(xi_elem.text)
                    y = float(yi_elem.text)
                    # Store in ORIGINAL coordinates as float (no scale conversion)
                    roboflow_points.append((x, y))
                    i += 1
                
                if len(roboflow_points) >= 3:
                    points = roboflow_points
                    is_polygon = True
                    print(f"[FILE_HANDLER] Loaded polygon (Roboflow format): {cls} with {len(points)} points")
        
        # Add as polygon if we found valid points
        if is_polygon:
            polygons.append([points, cls])  # [points_list, class_name]
        else:
            # ===== ELSE LOAD AS BBOX =====
            bb = obj.find("bndbox")
            if bb is not None:
                x1 = int(float(bb.find("xmin").text))
                y1 = int(float(bb.find("ymin").text))
                x2 = int(float(bb.find("xmax").text))
                y2 = int(float(bb.find("ymax").text))
                # Store as ORIGINAL coordinates (no scaling)
                boxes.append([
                    int(round(x1)),
                    int(round(y1)),
                    int(round(x2)),
                    int(round(y2)),
                    cls
                ])
                print(f"[FILE_HANDLER] Loaded bbox: {cls}")
    
    print(f"[FILE_HANDLER] Loaded {len(boxes)} bboxes, {len(polygons)} polygons from {img_name_local}")
    return boxes, polygons


def detect_dataset_has_polygons():
    """
    Check if dataset has any polygon annotations
    Returns: True if any polygon found, False if only bboxes
    """
    if not os.path.exists(vocdataset_folder):
        return False
    
    xml_files = [f for f in os.listdir(vocdataset_folder) if f.endswith('.xml')]
    
    if not xml_files:
        return False
    
    # Check first 10 XML files to determine annotation type
    for xml_file in xml_files[:10]:
        xml_path = os.path.join(vocdataset_folder, xml_file)
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            for obj in root.findall("object"):
                # Check if has <type>polygon</type>
                obj_type = obj.find("type")
                if obj_type is not None and obj_type.text == "polygon":
                    print(f"[TRAINING] Detected polygon annotations in dataset")
                    return True
                
                # Check if has Roboflow format polygon
                poly_elem = obj.find("polygon")
                if poly_elem is not None:
                    # Check for our format
                    if poly_elem.findall("point"):
                        print(f"[TRAINING] Detected polygon annotations (our format) in dataset")
                        return True
                    
                    # Check for Roboflow format
                    x1_elem = poly_elem.find("x1")
                    if x1_elem is not None:
                        print(f"[TRAINING] Detected polygon annotations (Roboflow format) in dataset")
                        return True
        except Exception as e:
            print(f"[WARN] Error reading {xml_file}: {e}")
            continue
    
    print(f"[TRAINING] No polygon annotations found - dataset has only bboxes")
    return False
