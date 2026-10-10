"""
Label Assistant inference (the G shortcut and Auto-annotate all).

The front menu picks one of three modes (see assistant_modes.py); the workspace's settings (configs/<ws>.json,
see workspace_config.py) say how that mode works. Providers:
  - yolo_world      : zero-shot YOLO-World, prompts translated to workspace classes   (YOLO-World mode)
  - custom_model    : the workspace's own trained model (models/<ws>/modelAssistant.pt) (YOLO-World mode)
  - locate_anything : LocateAnything-3B, categories translated to workspace classes   (LocateAnything mode)
  - custom_head     : frozen YOLO detector + trained head, head classes mapped         (Custom Model mode)
  - sam2_dynamic    : SAM 2 learning from the annotated images, no training             (SAM 2 Dynamic mode)

New predictions are MERGED into the image's current annotations. Anything already on the image counts as
confidence 1.0, so a prediction that overlaps an existing annotation (IoU >= MERGE_IOU) is dropped and the
existing one wins.
"""
import os
import gc

import cv2
import torch

from . import workspace_config as wcfg
from . import assistant_modes
from .assistant_providers import (AssistantError, locate_predictor, head_predictor, sam2_predictor,
                                  map_head_classes as map_classes)
from .annotation_merge import merge as _merge, poly_rect as _poly_rect
from .config import model_path, class_manager, state, input_folder, workspaceName, BASE_DIR

try:
    from ultralytics import YOLO, YOLOWorld
except Exception as e:
    YOLO = None
    YOLOWorld = None
    print("[INFO] ultralytics not installed. Training won't work.", e)

YOLO_WORLD_DIR = os.path.join(BASE_DIR, "models", "_yolo_world")

# (weights name, prompts) -> loaded model, so G doesn't reload CLIP + weights every press
_world_cache = {}


def current_classes():
    """The workspace's classes right now. config.CLASSLIST is a snapshot from when the window opened, so a class
    added afterwards in the class manager is missing from it."""
    return class_manager.get_classes()


def custom_model_file(cfg=None):
    """The custom model to use: the file chosen with "Browse custom model" when there is one, otherwise the
    workspace's own trained model (models/<ws>/modelAssistant.pt). None when neither exists."""
    cfg = cfg or wcfg.get_assistant(workspaceName)
    chosen = cfg["custom_model"]["path"]
    if chosen:
        return chosen if os.path.isfile(chosen) else None
    return model_path if os.path.exists(model_path) else None


def custom_model_available():
    return custom_model_file() is not None


def _collect(results, class_name_for):
    """Turn ultralytics results into [{'rect','conf','cls', 'poly'?}], plus
    whether the output is segmentation."""
    preds, is_polygon = [], False
    for r in results:
        if getattr(r, "masks", None) is not None:
            is_polygon = True
            for i, poly in enumerate(r.masks.xy):
                pts = poly.tolist()
                cls = class_name_for(int(r.boxes.cls[i].item()))
                if len(pts) < 3 or cls is None:
                    continue
                preds.append({
                    "rect": _poly_rect(pts),
                    "conf": float(r.boxes.conf[i].item()),
                    "cls": cls,
                    "poly": pts,
                })
        elif getattr(r, "boxes", None) is not None:
            for box in r.boxes:
                x1, y1, x2, y2 = (int(round(v)) for v in box.xyxy[0].tolist())
                cls = class_name_for(int(box.cls[0].item()))
                if cls is None:                      # a class of a browsed model that is not mapped
                    continue
                preds.append({
                    "rect": (x1, y1, x2, y2),
                    "conf": float(box.conf[0].item()),
                    "cls": cls,
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


# ------------------------------------------------------------------ providers
def current_provider(mode=None):
    """The provider that answers in `mode` (default: the mode chosen on the front menu)."""
    cfg = wcfg.get_assistant(workspaceName)
    return assistant_modes.provider_for(mode or assistant_modes.get_mode(), cfg["provider"])


def is_heavy(mode=None):
    """True for providers whose first use loads a big model (minutes, a download): the GUI shows a progress window."""
    return current_provider(mode) in (wcfg.PROVIDER_LOCATE, wcfg.PROVIDER_HEAD, wcfg.PROVIDER_SAM2)


def build_predictor(mode=None, status=None, conf=None, exclude=None):
    """predict(bgr_image) -> (predictions, is_polygon) for the configured provider. Loads what is needed (and
    caches it). `conf` overrides the saved confidence threshold. `exclude` is the path of the image about to be
    annotated (SAM 2 Dynamic never uses it as its own reference). Raises AssistantError with a message the user
    can act on."""
    cfg = wcfg.get_assistant(workspaceName)
    provider = assistant_modes.provider_for(mode or assistant_modes.get_mode(), cfg["provider"])
    conf = cfg["confidence"] if conf is None else conf

    if provider == wcfg.PROVIDER_LOCATE:
        return locate_predictor(cfg["locate_anything"], status)

    if provider == wcfg.PROVIDER_HEAD:
        return head_predictor(cfg["custom_head"], conf, current_classes(), BASE_DIR)

    if provider == wcfg.PROVIDER_SAM2:
        return sam2_predictor(cfg["sam2_dynamic"], current_classes(), BASE_DIR,
                              polygon=state.annotation_mode == "polygon", exclude=exclude, status=status)

    if YOLO is None:
        raise AssistantError("ultralytics is not installed.")

    if provider == wcfg.PROVIDER_CUSTOM:
        chosen = cfg["custom_model"]
        file = custom_model_file(cfg)
        if not file:
            raise AssistantError(f"The custom model was not found:\n{chosen['path']}" if chosen["path"]
                                 else "Your trained model was not found in this workspace.")
        model = YOLO(file)
        if chosen["path"]:
            # a model chosen with Browse: its class NAMES are mapped to workspace classes
            names = {int(k): str(v) for k, v in dict(model.names).items()}
            mapping = map_classes(list(names.values()), chosen["class_map"], current_classes())
            if not any(mapping.values()):
                raise AssistantError("None of the model's classes (" + ", ".join(names.values()) + ") is mapped to a "
                                     "workspace class. Open Label Assistant and map them.")
            class_name_for = lambda idx: mapping.get(names.get(idx))
        else:
            # the workspace's own trained model: class index = position in the workspace's class list
            classes = current_classes()
            class_name_for = lambda idx: classes[idx] if idx < len(classes) else str(idx)
    else:
        yw = cfg["yolo_world"]
        targets = [t for t in yw["target_classes"] if t["prompt"].strip() and t["map_to"]]
        if not targets:
            raise AssistantError("YOLO-World can't work without a target class. "
                                 "Open Label Assistant and add at least one target class.")
        prompts = [t["prompt"].strip() for t in targets]
        try:
            model = _load_world_model(yw["model"], prompts)
        except Exception as exc:
            print(f"[Assistant] YOLO-World failed to load: {exc}")
            raise AssistantError(f"YOLO-World could not be loaded:\n{exc}") from exc
        class_name_for = lambda idx: targets[idx]["map_to"]

    def predict(bgr):
        with torch.no_grad():
            results = model.predict(bgr, conf=conf, iou=0.3, verbose=False)
        preds, is_polygon = _collect(results, class_name_for)
        del results
        return preds, is_polygon

    return predict


def read_image(path):
    """BGR image or None. Unlike cv2.imread this also works for paths with non-ASCII characters on Windows."""
    try:
        import numpy as np
        data = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
    except OSError:
        return None


def apply_predictions(preds, is_polygon):
    """Merge predictions into state.bboxes / state.polygons. Returns the user-facing message."""
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
    return msg


def predict_current(images, current_index, mode=None, status=None, conf=None):
    """Run the configured provider on one image WITHOUT touching the GUI state (safe on a worker thread).
    Returns (preds, is_polygon). Raises AssistantError."""
    img_path = os.path.join(input_folder, images[current_index])
    orig_img = cv2.imread(img_path)
    if orig_img is None:
        orig_img = read_image(img_path)
    if orig_img is None:
        raise AssistantError(f"Could not read image: {images[current_index]}")
    predict = build_predictor(mode, status, conf, exclude=img_path)
    try:
        return predict(orig_img)
    finally:
        torch.cuda.empty_cache()
        gc.collect()


def inference_current(images, current_index, conf=None, mode=None):
    """Run the configured Label Assistant on the current image and merge the
    result into state. Returns (ok, message); message is user-facing."""
    try:
        preds, is_polygon = predict_current(images, current_index, mode, conf=conf)
    except AssistantError as exc:
        return False, str(exc)
    return True, apply_predictions(preds, is_polygon)
