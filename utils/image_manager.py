"""
Image navigation and management
"""
import os
import copy
from .config import state, vocdataset_folder, yolo_labels_folder, input_folder
import cv2
import numpy as np
def repeat_last_annotations(images, current_index, classList):
    """Repeat annotations from previous image (both bboxes and polygons)"""
    from .file_handler import save_pascal_voc, save_yolo_label

    if not state.prev_bboxes and not state.prev_polygons:
        print("[INFO] No previous annotations to repeat.")
        return

    print("[INFO] Reapplying previous annotations...")
    state.bboxes = [b.copy() for b in state.prev_bboxes]
    state.polygons = [p.copy() for p in state.prev_polygons]
    save_pascal_voc(images[current_index], state.frame.shape)
    save_yolo_label(images[current_index], state.frame.shape, classList)

def delete_current_image(images, current_index):
    """Delete current image and all associated files"""
    
    if not images:
        print("[WARN] No images left.")
        return current_index, images

    img_name = images[current_index]
    base_name = os.path.splitext(img_name)[0]

    # Paths to all related files
    input_path = os.path.join(input_folder, img_name)
    xml_path = os.path.join(vocdataset_folder, base_name + ".xml")
    yolo_label_path = os.path.join(yolo_labels_folder, base_name + ".txt")

    # Delete files if they exist
    for f in [input_path, xml_path, yolo_label_path]:
        if os.path.exists(f):
            os.remove(f)
            print(f"[INFO] Deleted: {f}")

    # Remove from list
    del images[current_index]

    # Clear annotations
    state.bboxes.clear()
    state.polygons.clear()

    # Determine new index
    if current_index >= len(images):
        current_index = max(0, len(images) - 1)

    # If no images left, show notification
    if not images:
        print("[INFO] All images deleted.")
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        cv2.putText(frame, "No images left", (400, 360),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Annotator", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return None, None

    return current_index, images

def save_and_backup_bboxes(img_name, orig_shape, ClassList):
    """Save annotations and backup current bboxes/polygons"""
    from .file_handler import save_pascal_voc, save_yolo_label
    save_pascal_voc(img_name, orig_shape)
    save_yolo_label(img_name, orig_shape, ClassList)
    state.prev_bboxes = copy.deepcopy(state.bboxes)
    state.prev_polygons = copy.deepcopy(state.polygons)
