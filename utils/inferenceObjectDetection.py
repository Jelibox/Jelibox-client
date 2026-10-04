"""
Label Assistant inference (the G shortcut).

Two providers, picked per workspace in the Label Assistant window and stored in
configs/<workspace>.json (see workspace_config.py):
  - custom_model : the workspace's own trained model (models/<ws>/modelAssistant.pt)
  - yolo_world   : zero-shot YOLO-World, prompts translated to workspace classes

New predictions are MERGED into the image's current annotations. Anything
already on the image counts as confidence 1.0, so a prediction that overlaps
an existing annotation (IoU >= MERGE_IOU) is dropped and the existing one wins.
"""
import os
import gc

import cv2
import torch

from . import workspace_config as wcfg
from .annotation_merge import merge as _merge, poly_rect as _poly_rect
from .config import model_path, CLASSLIST, state, input_folder, workspaceName, BASE_DIR

try:
    from ultralytics import YOLO, YOLOWorld
except Exception as e:
    YOLO = None
    YOLOWorld = None
    print("[INFO] ultralytics not installed. Training won't work.", e)

YOLO_WORLD_DIR = os.path.join(BASE_DIR, "models", "_yolo_world")

# (weights name, prompts) -> loaded model, so G doesn't reload CLIP + weights every press
_world_cache = {}


def custom_model_available():
    return os.path.exists(model_path)


def _collect(results, class_name_for):
    """Turn ultralytics results into [{'rect','conf','cls', 'poly'?}], plus
    whether the output is segmentation."""
    preds, is_polygon = [], False
    for r in results:
        if getattr(r, "masks", None) is not None:
            is_polygon = True
            for i, poly in enumerate(r.masks.xy):
                pts = poly.tolist()
                if len(pts) < 3:
                    continue
                preds.append({
                    "rect": _poly_rect(pts),
                    "conf": float(r.boxes.conf[i].item()),
                    "cls": class_name_for(int(r.boxes.cls[i].item())),
                    "poly": pts,
                })
        elif getattr(r, "boxes", None) is not None:
            for box in r.boxes:
                x1, y1, x2, y2 = (int(round(v)) for v in box.xyxy[0].tolist())
                preds.append({
                    "rect": (x1, y1, x2, y2),
                    "conf": float(box.conf[0].item()),
                    "cls": class_name_for(int(box.cls[0].item())),
                })
    return preds, is_polygon


def _load_world_model(weights_name, prompts):
    key = (weights_name, tuple(prompts))
    if key not in _world_cache:
        _world_cache.clear()  # keep at most one model in memory
        os.makedirs(YOLO_WORLD_DIR, exist_ok=True)
        model = YOLOWorld(os.path.join(YOLO_WORLD_DIR, weights_name + ".pt"))
        model.set_classes(prompts)
        _world_cache[key] = model
    return _world_cache[key]


def inference_current(images, current_index, conf=None):
    """Run the configured Label Assistant on the current image and merge the
    result into state. Returns (ok, message); message is user-facing."""
    if YOLO is None:
        return False, "ultralytics is not installed."

    cfg = wcfg.get_assistant(workspaceName)
    if conf is None:
        conf = cfg["confidence"]

    img_path = os.path.join(input_folder, images[current_index])
    orig_img = cv2.imread(img_path)
    if orig_img is None:
        return False, f"Could not read image: {images[current_index]}"

    if cfg["provider"] == wcfg.PROVIDER_CUSTOM:
        if not custom_model_available():
            return False, "Your trained model was not found in this workspace."
        model = YOLO(model_path)
        class_name_for = lambda idx: CLASSLIST[idx] if idx < len(CLASSLIST) else str(idx)
    else:
        yw = cfg["yolo_world"]
        targets = [t for t in yw["target_classes"] if t["prompt"].strip() and t["map_to"]]
        if not targets:
            return False, ("YOLO-World can't work without a target class. "
                           "Open Label Assistant and add at least one target class.")
        prompts = [t["prompt"].strip() for t in targets]
        try:
            model = _load_world_model(yw["model"], prompts)
        except Exception as exc:
            print(f"[Assistant] YOLO-World failed to load: {exc}")
            return False, f"YOLO-World could not be loaded:\n{exc}"
        class_name_for = lambda idx: targets[idx]["map_to"]

    with torch.no_grad():
        results = model.predict(orig_img, conf=conf, iou=0.3, verbose=False)
    preds, is_polygon = _collect(results, class_name_for)

    if is_polygon:
        kept = _merge(preds, [_poly_rect(p[0]) for p in state.polygons])
        state.polygons.extend([p["poly"], p["cls"]] for p in kept)
    else:
        kept = _merge(preds, [tuple(b[:4]) for b in state.bboxes])
        state.bboxes.extend([*p["rect"], p["cls"]] for p in kept)

    skipped = len(preds) - len(kept)
    kind = "polygons" if is_polygon else "bboxes"
    msg = f"Added {len(kept)} {kind}" + (f" ({skipped} skipped: overlap with existing)" if skipped else "")
    print(f"[Assistant] {msg}")

    del results
    torch.cuda.empty_cache()
    gc.collect()
    return True, msg
