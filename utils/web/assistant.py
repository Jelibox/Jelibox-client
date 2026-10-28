"""
The Label Assistant for the web server: the same four modes and providers as the desktop window (assistant_modes,
assistant_providers, auto_annotate), but every function takes the workspace it works for instead of reading the
desktop's globals, so the server can serve many workspaces. Nothing here imports tkinter; torch, ultralytics and the
models are imported only when a predictor is built.

One model job at a time, server-wide (`Busy` tells who has it): the models are single instances (SAM 2's memory
bank, LocateAnything's weights) and a GPU does not share well. Single-image Infer and Auto-annotate all both take it.

What the browser sends and gets:
  infer      the annotations now on screen (they may not be saved) -> only the NEW annotations to add (the same merge
             as the desktop's G key: an existing annotation always wins over an overlapping prediction)
  autoannotate  runs over the dataset on the server and saves each picture as it goes
"""
import gc
import os
import threading

from .. import annotation_io as aio
from .. import assistant_modes, assistant_providers as ap
from .. import workspace_config as wcfg
from ..annotation_merge import merge, poly_rect
from ..assistant_providers import AssistantError
from ..auto_annotate import run_batch
from ..detector_classes import COCO_NAMES, parse_class_ids
from .editing import EditError

YOLO_WORLD_DIR_NAME = os.path.join("models", "_yolo_world")

# ------------------------------------------------------------------ one model job at a time
_busy = threading.Lock()
_busy_info = {}


class Busy(EditError):
    """Another model job is running."""


def acquire(user, what):
    if not _busy.acquire(blocking=False):
        who = _busy_info.get("user", "someone")
        raise Busy(f"The Label Assistant is busy: {who} is running {_busy_info.get('what', 'a job')}. "
                   f"Try again when it has finished.")
    _busy_info.update(user=user, what=what)


def release():
    _busy_info.clear()
    _busy.release()


# ------------------------------------------------------------------ SAM 2 references, one store per workspace
_stores = {}
_stores_lock = threading.Lock()


def sam_store(workspace):
    from .. import sam2_dynamic as sd
    with _stores_lock:
        return _stores.setdefault(workspace, sd.ReferenceStore())


def forget_references(workspace, path=None):
    with _stores_lock:
        store = _stores.get(workspace)
    if store is None:
        return
    if path is None:
        store.clear()
    else:
        store.forget(path)


# ------------------------------------------------------------------ what a run needs to know
class Ctx:
    """The workspace, its saved Label Assistant settings and the mode, for one run."""

    def __init__(self, ws, mode=None):
        self.ws = ws
        self.base_dir = ws.base_dir
        self.cfg = wcfg.get_assistant(ws.name)
        self.mode = mode or assistant_modes.get_mode()
        self.provider = assistant_modes.provider_for(self.mode, self.cfg["provider"])
        self.classes = ws.classes.get_classes()
        self.own_model = os.path.join(ws.base_dir, "models", ws.name, "modelAssistant.pt")

    def custom_model_file(self):
        """The browsed model when one is chosen (None if it is gone), else the workspace's own trained model."""
        chosen = self.cfg["custom_model"]["path"]
        if chosen:
            return chosen if os.path.isfile(chosen) else None
        return self.own_model if os.path.exists(self.own_model) else None

    def world_weights(self):
        return os.path.join(self.base_dir, YOLO_WORLD_DIR_NAME, self.cfg["yolo_world"]["model"] + ".pt")


def needs_download(ctx):
    """None, or {"title", "message"} for what would leave the machine, to ask before it happens."""
    if ctx.provider == wcfg.PROVIDER_YOLO_WORLD:
        if os.path.isfile(ctx.world_weights()):
            return None
        return {"title": "YOLO-World - first use",
                "message": f"The YOLO-World model ({ctx.cfg['yolo_world']['model']}) is not on the server yet.\n\n"
                           f"Jelibox will download it once from github.com (Ultralytics' model files) into\n"
                           f"{os.path.dirname(ctx.world_weights())}\n\n"
                           f"The CLIP text model it needs may be fetched too, if it is missing.\n\nDownload now?"}
    if ctx.provider == wcfg.PROVIDER_LOCATE:
        from .. import locate_anything as la
        problem = la.check_requirements()
        if problem:
            raise AssistantError(problem)
        if la.is_cached():
            return None
        return {"title": "LocateAnything - first use",
                "message": f"LocateAnything-3B is not on the server yet.\n\nJelibox will download it from huggingface.co "
                           f"(about {la.DOWNLOAD_GB} GB) into\n{la.HF_HOME}\n\nand run the model's own code from that "
                           f"repository (pinned to a fixed version). This happens once. Download now?"}
    if ctx.provider == wcfg.PROVIDER_SAM2:
        from .. import sam2_dynamic as sd
        model = ctx.cfg["sam2_dynamic"]["model"]
        weights = sd.weights_path(ctx.base_dir, model)
        if os.path.isfile(weights):
            return None
        return {"title": "SAM 2 Dynamic - first use",
                "message": f"The SAM 2 model ({model}, about {sd.model_size_mb(model)} MB) is not on the server yet.\n\n"
                           f"Jelibox will download it once from github.com (Ultralytics' model files) into\n"
                           f"{os.path.dirname(weights)}\n\nDownload now?"}
    return None


# ------------------------------------------------------------------ predictors
_world_cache = {}              # (weights name, prompts) -> loaded YOLO-World model: one is kept in memory


def collect(results, class_name_for):
    """Ultralytics results -> ([{'rect', 'conf', 'cls', 'poly'?}], is_polygon)."""
    preds, is_polygon = [], False
    for r in results:
        if getattr(r, "masks", None) is not None:
            is_polygon = True
            for i, poly in enumerate(r.masks.xy):
                pts = poly.tolist()
                cls = class_name_for(int(r.boxes.cls[i].item()))
                if len(pts) < 3 or cls is None:
                    continue
                preds.append({"rect": poly_rect(pts), "conf": float(r.boxes.conf[i].item()), "cls": cls, "poly": pts})
        elif getattr(r, "boxes", None) is not None:
            for box in r.boxes:
                x1, y1, x2, y2 = (int(round(v)) for v in box.xyxy[0].tolist())
                cls = class_name_for(int(box.cls[0].item()))
                if cls is None:
                    continue
                preds.append({"rect": (x1, y1, x2, y2), "conf": float(box.conf[0].item()), "cls": cls})
    return preds, is_polygon


def _world_model(ctx, prompts, status):
    from ultralytics import YOLOWorld
    name = ctx.cfg["yolo_world"]["model"]
    key = (name, tuple(prompts))
    if key not in _world_cache:
        _world_cache.clear()
        os.makedirs(os.path.dirname(ctx.world_weights()), exist_ok=True)
        if status:
            status("Downloading YOLO-World (first use) ..." if not os.path.isfile(ctx.world_weights())
                   else "Loading YOLO-World ...")
        model = YOLOWorld(ctx.world_weights())
        if status:
            status("Preparing the text prompts (CLIP) ...")
        model.set_classes(prompts)
        _world_cache[key] = model
    return _world_cache[key]


def build_predictor(ctx, status=None, conf=None, exclude=None, polygon=False):
    """predict(bgr) -> (predictions, is_polygon) for the mode's provider. Raises AssistantError."""
    cfg = ctx.cfg
    conf = cfg["confidence"] if conf is None else conf
    if ctx.provider == wcfg.PROVIDER_LOCATE:
        return ap.locate_predictor(cfg["locate_anything"], status)
    if ctx.provider == wcfg.PROVIDER_HEAD:
        return ap.head_predictor(cfg["custom_head"], conf, ctx.classes, ctx.base_dir)
    if ctx.provider == wcfg.PROVIDER_SAM2:
        return ap.sam2_predictor(cfg["sam2_dynamic"], ctx.classes, ctx.base_dir, polygon=polygon, exclude=exclude,
                                 status=status, store=sam_store(ctx.ws.name))
    try:
        from ultralytics import YOLO
    except Exception as exc:
        raise AssistantError(f"ultralytics is not installed ({exc}).") from exc

    if ctx.provider == wcfg.PROVIDER_CUSTOM:
        chosen = cfg["custom_model"]
        file = ctx.custom_model_file()
        if not file:
            raise AssistantError(f"The custom model was not found:\n{chosen['path']}" if chosen["path"]
                                 else "Your trained model was not found in this workspace.")
        model = YOLO(file)
        if chosen["path"]:
            names = {int(k): str(v) for k, v in dict(model.names).items()}
            mapping = ap.map_head_classes(list(names.values()), chosen["class_map"], ctx.classes)
            if not any(mapping.values()):
                raise AssistantError("None of the model's classes (" + ", ".join(names.values()) + ") is mapped to a "
                                     "workspace class. Open Label Assistant and map them.")
            class_name_for = lambda idx: mapping.get(names.get(idx))
        else:
            class_name_for = lambda idx: ctx.classes[idx] if idx < len(ctx.classes) else str(idx)
    else:
        targets = [t for t in cfg["yolo_world"]["target_classes"] if t["prompt"].strip() and t["map_to"]]
        if not targets:
            raise AssistantError("YOLO-World can't work without a target class. "
                                 "Open Label Assistant and add at least one target class.")
        try:
            model = _world_model(ctx, [t["prompt"].strip() for t in targets], status)
        except Exception as exc:
            raise AssistantError(f"YOLO-World could not be loaded:\n{exc}") from exc
        class_name_for = lambda idx: targets[idx]["map_to"]

    def predict(bgr):
        import torch
        with torch.no_grad():
            results = model.predict(bgr, conf=conf, iou=0.3, verbose=False)
        return collect(results, class_name_for)

    return predict


def _cleanup():
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()


def read_bgr(path):
    import cv2
    import numpy as np
    try:
        data = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
    except OSError:
        return None


# ------------------------------------------------------------------ Infer on one picture
def infer_one(ctx, name, bboxes, polygons, polygon=False, status=None):
    """Predict on one picture and return what to ADD to what is on screen (`bboxes`, `polygons`)."""
    path = ctx.ws.path(name)
    image = read_bgr(path)
    if image is None:
        raise AssistantError(f"Could not read the picture {name}.")
    predict = build_predictor(ctx, status, exclude=path, polygon=polygon)
    if status:
        status("Looking at the picture ...")
    try:
        preds, is_polygon = predict(image)
    finally:
        _cleanup()
    allowed = set(ctx.classes)
    preds = [p for p in preds if p["cls"] in allowed]
    new_boxes, new_polygons = [], []
    if is_polygon:
        kept = merge(preds, [poly_rect(p[0]) for p in polygons])
        new_polygons = [[[list(pt) for pt in p["poly"]], p["cls"]] for p in kept]
    else:
        kept = merge(preds, [tuple(b[:4]) for b in bboxes])
        new_boxes = [[*p["rect"], p["cls"]] for p in kept]
    skipped = len(preds) - len(kept)
    kind = "polygons" if is_polygon else "bboxes"
    message = f"Added {len(kept)} {kind}" + (f" ({skipped} skipped: overlap with existing)" if skipped else "")
    return {"bboxes": new_boxes, "polygons": new_polygons, "polygon": is_polygon, "found": len(preds),
            "skipped": skipped, "added": len(kept), "message": message}


# ------------------------------------------------------------------ Auto-annotate all
def autoannotate(ctx, job, only_unlabeled=True, polygon=False):
    """Label every picture of the dataset on the server (the desktop's "Auto-annotate all"). Saves each picture as it
    goes; `job.cancelled()` stops it before the next one."""
    ws = ctx.ws
    classes = ctx.classes

    def load(name):
        found = ws.load(name)
        return found["bboxes"], found["polygons"]

    def save(name, shape, bboxes, polygons):
        with ws.lock:
            aio.write_annotations(ws.voc, ws.labels, name, shape, bboxes, polygons, classes)

    def work():
        predict = build_predictor(ctx, lambda text: job.progress(text=text), polygon=polygon)
        job.progress(text="Annotating ...")
        return run_batch(
            ws.images(), read_image=lambda n: read_bgr(ws.path(n)), predict=predict, load_annotations=load,
            save_annotations=save, classes=classes, only_unlabeled=only_unlabeled,
            progress=lambda done, total, name, res: job.progress(
                done, total, (f"{name}   ({res.annotated} annotated, {res.added} new)" if name else "Annotating ...")),
            should_cancel=job.cancelled)

    try:
        res = work()
    finally:
        _cleanup()
    return {"total": res.total, "processed": res.processed, "annotated": res.annotated, "added": res.added,
            "skipped_labeled": res.skipped_labeled, "no_detection": res.no_detection,
            "failed": [[n, m] for n, m in res.failed[:50]], "failed_count": len(res.failed),
            "cancelled": res.cancelled, "seconds": round(res.seconds, 1), "summary": res.summary()}


# ------------------------------------------------------------------ settings
def settings_payload(ws):
    """Everything the settings window shows, for the mode chosen on the front menu."""
    ctx = Ctx(ws)
    from .. import sam2_dynamic as sd
    cfg = ctx.cfg
    out = {
        "mode": ctx.mode, "title": assistant_modes.title(ctx.mode), "provider": ctx.provider, "classes": ctx.classes,
        "confidence": cfg["confidence"], "yolo_world": cfg["yolo_world"], "custom_model": cfg["custom_model"],
        "locate_anything": cfg["locate_anything"], "custom_head": cfg["custom_head"], "sam2_dynamic": cfg["sam2_dynamic"],
        "batch": cfg["batch"],
        "options": {
            "yolo_world_models": wcfg.YOLO_WORLD_MODELS, "locate_modes": wcfg.LOCATE_GENERATION_MODES,
            "locate_devices": wcfg.LOCATE_DEVICES, "sam2_sizes": wcfg.SAM2_IMAGE_SIZES,
            "sam2_models": [{"id": m, "size": sd.MODEL_INFO[m][0], "note": sd.MODEL_INFO[m][1]} for m in sd.MODELS],
            "coco_names": COCO_NAMES,
        },
        "own_model": os.path.exists(ctx.own_model),
        "references": len(sam_store(ws.name)),
    }
    if ctx.mode == assistant_modes.MODE_LOCATE:
        out["status"] = locate_status()
    return out


def locate_status():
    from .. import locate_anything as la
    problem = la.check_requirements()
    lines = ["Packages: ready" if not problem else "Packages: missing - " + problem.split("\n")[0],
             "Model: " + ("downloaded" if not problem and la.is_cached()
                          else f"not downloaded yet (about {la.DOWNLOAD_GB} GB, asked before it happens)")]
    try:
        import torch
        lines.append("Device: " + (f"NVIDIA GPU - {torch.cuda.get_device_name(0)}" if torch.cuda.is_available()
                                   else "CPU only (slow, needs ~9 GB free RAM)"))
    except Exception:
        lines.append("Device: unknown")
    return lines


def _conf(value, default):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return round(min(max(number, 0.05), 0.95), 2)


def _targets(data, classes):
    targets, error = wcfg.validate_targets((t.get("prompt"), t.get("map_to")) for t in (data or []) if isinstance(t, dict))
    if error:
        raise EditError(error)
    for t in targets:
        if t["map_to"] not in classes:
            raise EditError(f"“{t['map_to']}” is not a class of this workspace.")
        if not wcfg.is_valid_prompt_text(t["prompt"]):
            raise EditError(f"“{t['prompt']}”: use letters and spaces only, up to {wcfg.MAX_PROMPT_LEN} characters.")
    return targets


def _class_map(raw, classes):
    if not isinstance(raw, dict):
        return {}
    return {str(k): str(v) for k, v in raw.items() if str(v) in classes and str(v)}


def _resolve(base_dir, path):
    return ap.resolve_head_path(path, base_dir)


def save_settings(ws, data):
    """Validate and store one section of the settings (the desktop dialogs' Save). Returns warnings (a list of text)."""
    section = data.get("section")
    ctx = Ctx(ws)
    cfg, classes = ctx.cfg, ctx.classes
    warnings = []
    if section == "yolo_world":
        provider = data.get("provider")
        if provider not in (wcfg.PROVIDER_YOLO_WORLD, wcfg.PROVIDER_CUSTOM):
            raise EditError("Choose YOLO-World or your own model.")
        model = data.get("model")
        if model not in wcfg.YOLO_WORLD_MODELS:
            raise EditError("Unknown YOLO-World model.")
        try:
            targets = _targets(data.get("targets"), classes)
        except EditError:
            if provider == wcfg.PROVIDER_YOLO_WORLD:
                raise
            targets = cfg["yolo_world"]["target_classes"]        # the own model is chosen: keep what was saved
        if provider == wcfg.PROVIDER_YOLO_WORLD and not targets:
            warnings.append("YOLO-World can't work without a target class: G will not annotate anything until you add one.")
        path = str(data.get("model_path") or "").strip()
        class_map = _class_map(data.get("model_class_map"), classes)
        if provider == wcfg.PROVIDER_CUSTOM and path:
            if not os.path.isfile(path):
                raise EditError(f"The chosen model was not found:\n{path}")
            if not class_map:
                raise EditError("Map at least one model class to a workspace class.")
        cfg["custom_model"] = {"path": path, "class_map": class_map if path else {}}
        cfg["provider"] = provider
        cfg["confidence"] = _conf(data.get("confidence"), cfg["confidence"])
        cfg["yolo_world"]["model"] = model
        cfg["yolo_world"]["target_classes"] = targets
    elif section == "locate":
        targets = _targets(data.get("targets"), classes)
        if not targets:
            warnings.append("LocateAnything can't work without a category: G and Auto-annotate will not annotate anything "
                            "until you add one.")
        la = cfg["locate_anything"]
        mode, device = data.get("generation_mode"), data.get("device")
        if mode not in wcfg.LOCATE_GENERATION_MODES or device not in wcfg.LOCATE_DEVICES:
            raise EditError("Choose a decoding mode and a device from the lists.")
        try:
            side, passes = int(data.get("short_side")), int(data.get("passes"))
            temperature = round(float(data.get("temperature", 0.0)), 2)
        except (TypeError, ValueError):
            raise EditError("Image short side and passes must be whole numbers, the temperature a number.")
        la.update(target_classes=targets, generation_mode=mode, device=device, short_side=side, passes=passes,
                  temperature=temperature, min_votes=min(la["min_votes"], passes))
        cfg["provider"] = wcfg.PROVIDER_LOCATE
    elif section == "head":
        path = str(data.get("head_path") or "").strip()
        if not path:
            raise EditError("Choose the head checkpoint (head_best.pt) first.")
        if not os.path.isfile(_resolve(ws.base_dir, path)):
            raise EditError(f"The head checkpoint was not found:\n{path}")
        weights = str(data.get("detector_weights") or "").strip()
        names = list(COCO_NAMES)
        if weights and (os.path.isabs(weights) or "/" in weights or "\\" in weights):
            if not os.path.isfile(_resolve(ws.base_dir, weights)):
                raise EditError(f"The detector model was not found:\n{weights}")
        if weights and os.path.isfile(_resolve(ws.base_dir, weights)):
            names = ap.model_class_names(_resolve(ws.base_dir, weights))
        try:
            ids = parse_class_ids(str(data.get("detector_classes") or ""), names)
        except ValueError as exc:
            raise EditError(str(exc))
        class_map = _class_map(data.get("class_map"), classes)
        if data.get("has_head_names") and not class_map:
            raise EditError("Map at least one head class to a workspace class.")
        try:
            iou, min_head = round(float(data.get("iou")), 2), round(float(data.get("min_head_conf")), 2)
        except (TypeError, ValueError):
            raise EditError("The sliders need numbers.")
        cfg["custom_head"].update(head_path=path, detector_weights=weights, detector_classes=ids, iou=iou,
                                  min_head_conf=min_head, class_map=class_map)
        cfg["confidence"] = _conf(data.get("confidence"), cfg["confidence"])
        cfg["provider"] = wcfg.PROVIDER_HEAD
    elif section == "sam2":
        from .. import sam2_dynamic as sd
        if data.get("model") not in sd.MODELS:
            raise EditError("Unknown SAM 2 model.")
        try:
            size, refs, score = int(data.get("imgsz")), int(data.get("max_references")), round(float(data.get("min_score")), 2)
        except (TypeError, ValueError):
            raise EditError("Image size and reference images must be whole numbers, the score a number.")
        if size not in wcfg.SAM2_IMAGE_SIZES:
            raise EditError("Choose an image size from the list.")
        cfg["sam2_dynamic"].update(model=data["model"], imgsz=size, max_references=refs, min_score=score)
        cfg["provider"] = wcfg.PROVIDER_SAM2
    elif section == "batch":
        cfg["batch"]["only_unlabeled"] = bool(data.get("only_unlabeled"))
    else:
        raise EditError("Unknown settings section.")
    wcfg.set_assistant(ws.name, cfg)
    return warnings


# ------------------------------------------------------------------ looking into model files (for the mapping tables)
def inspect_file(ws, kind, path):
    """Class names of a model the user picked: kind 'model' (any Ultralytics .pt), 'detector' (same) or 'head'."""
    full = _resolve(ws.base_dir, str(path or "").strip())
    if kind in ("model", "detector"):
        names = ap.model_class_names(full)
        return {"names": [n for n in names if n], "all_names": names}
    if kind == "head":
        if not os.path.isfile(full):
            raise AssistantError(f"The head checkpoint was not found:\n{full}")
        try:
            from .. import custom_head as ch
            ck = ch.read_checkpoint(full)
        except Exception as exc:
            raise AssistantError(str(exc).split("\n")[0]) from exc
        cfg = ck["cfg"]
        return {"names": list(ck["names"]),
                "info": f"{len(ck['names'])} classes: {', '.join(ck['names'])}   |   detector {cfg['weights']}, "
                        f"{cfg['imgsz'][0]}x{cfg['imgsz'][1]}"}
    raise EditError("Unknown kind.")


def load_labeled(ws):
    """Fill the workspace's SAM 2 references from the pictures already annotated on disk (the newest ones)."""
    from .. import sam2_dynamic as sd
    from .. import workspace_manager as wm
    limit = wcfg.get_assistant(ws.name)["sam2_dynamic"]["max_references"]
    records = sd.workspace_images(wm.DATASETS_ROOT, ws.name)
    return sam_store(ws.name).load_labeled(records, ws.voc, limit)
