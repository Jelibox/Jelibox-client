"""
File handling functions for VOC and YOLO formats
Supports both bounding boxes and polygons
"""
import os
import xml.etree.ElementTree as ET
from xml.dom import minidom
from .config import vocdataset_folder, yolo_labels_folder, CLASSLIST, state
from .annotation_io import (prettify_xml, build_voc_xml, _yolo_lines_from_annotations, _read_voc_size,
                            _parse_voc_size, _parse_voc_objects, write_annotations,
                            sync_yolo_labels as _sync_yolo_labels)  # noqa: F401 (re-exported)

def save_pascal_voc(img_name, img_shape):
    """Save annotations in Pascal VOC format (bboxes and polygons)"""
    xml_path = os.path.join(vocdataset_folder, os.path.splitext(img_name)[0] + ".xml")
    ann = build_voc_xml(img_name, img_shape, state.bboxes, state.polygons)
    with open(xml_path, "w") as f:
        f.write(prettify_xml(ann))
    print(f"[INFO] Saved VOC: {xml_path}")

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
    write_annotations(vocdataset_folder, yolo_labels_folder, img_name, img_shape, bboxes, polygons, classList)


def sync_yolo_labels(classList, progress_cb=None):
    """Ensure every VOC XML of this workspace has a YOLO label, generating the missing ones (see annotation_io).
    Returns the number of YOLO labels generated."""
    return _sync_yolo_labels(vocdataset_folder, yolo_labels_folder, classList, progress_cb)


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
