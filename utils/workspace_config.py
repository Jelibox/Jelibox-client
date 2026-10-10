"""
Per-workspace configuration file: configs/<workspace>.json

Single source of truth for everything Jelibox remembers about a workspace
(shared by all of its instances, e.g. weapon-1, weapon-2 -> configs/weapon.json).

Layout:

    {
      "version": 1,
      "classes": ["horse", "rider"],
      "label_assistant": {
        "provider": "yolo_world",          # "yolo_world" | "custom_model"
        "confidence": 0.3,
        "yolo_world": {
          "model": "yolov8s-world",        # one of YOLO_WORLD_MODELS
          "target_classes": [              # prompt -> workspace class
            {"prompt": "white horse", "map_to": "horse"}
          ]
        },
        "locate_anything": {               # LocateAnything-3B (see LOCATE_DEFAULTS)
          "target_classes": [{"prompt": "person", "map_to": "rider"}],
          "generation_mode": "hybrid", "device": "auto", ...
        },
        "custom_head": {                   # frozen YOLO detector + trained head (see HEAD_DEFAULTS)
          "head_path": "models/horse/heads/head_best.pt",
          "detector_classes": [0],         # detector class ids whose boxes the head relabels
          "class_map": {"sleeping": "rider"}
        },
        "sam2_dynamic": {                  # SAM 2 that learns from annotated images (see SAM2_DEFAULTS)
          "model": "sam2.1_b.pt", "imgsz": 1024, "max_references": 6, "min_score": 0.05
        },
        "batch": {"only_unlabeled": true}  # Auto-annotate all images
      }
    }

Legacy configs/<workspace>.txt files (one class per line) are migrated to
this format the first time they are read.

No tkinter / utils.config imports here on purpose - workspace_manager,
dataset_import and class_manager all use this module.
"""
import copy
import json
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONFIGS_ROOT = os.path.join(BASE_DIR, "configs")

CONFIG_VERSION = 1

PROVIDER_YOLO_WORLD = "yolo_world"
PROVIDER_CUSTOM = "custom_model"
PROVIDER_LOCATE = "locate_anything"
PROVIDER_HEAD = "custom_head"
PROVIDER_SAM2 = "sam2_dynamic"
PROVIDERS = (PROVIDER_YOLO_WORLD, PROVIDER_CUSTOM, PROVIDER_LOCATE, PROVIDER_HEAD, PROVIDER_SAM2)

YOLO_WORLD_MODELS = [
    "yolov8s-world", "yolov8s-worldv2",
    "yolov8m-world", "yolov8m-worldv2",
    "yolov8l-world", "yolov8l-worldv2",
    "yolov8x-world", "yolov8x-worldv2",
]

LOCATE_GENERATION_MODES = ["hybrid", "fast", "slow"]
LOCATE_DEVICES = ["auto", "cuda", "cpu"]

LOCATE_DEFAULTS = {
    "target_classes": [],
    "generation_mode": "hybrid",    # MTP with AR fallback: best recall
    "device": "auto",               # auto = CUDA when available, else CPU
    "short_side": 1024,             # larger images are shrunk to this short side before inference
    "max_new_tokens": 4096,
    "passes": 1,                    # >1 = several sampled decodes merged by agreement (needs temperature > 0)
    "min_votes": 1,
    "temperature": 0.0,
    "iou_dedup": 0.9,
}

CUSTOM_MODEL_DEFAULTS = {
    "path": "",                     # a model chosen with "Browse custom model"; "" = models/<workspace>/modelAssistant.pt
    "class_map": {},                # model class name -> workspace class (only used with a browsed model)
}

HEAD_DEFAULTS = {
    "head_path": "",                # trained head checkpoint (head_best.pt)
    "detector_weights": "",         # "" = the detector the head was trained on
    "detector_classes": [0],        # detector class ids whose boxes get relabelled (COCO: 0 person, 39 bottle, ...)
    "iou": 0.6,                     # NMS IoU of the detector
    "min_head_conf": 0.0,           # drop boxes whose head probability is below this
    "class_map": {},                # head class name -> workspace class
}

SAM2_MODELS = ["sam2.1_t.pt", "sam2.1_s.pt", "sam2.1_b.pt", "sam2.1_l.pt"]
SAM2_IMAGE_SIZES = [512, 768, 1024]

SAM2_DEFAULTS = {
    "model": "sam2.1_b.pt",         # downloaded once into models/_sam2/
    "imgsz": 1024,                  # SAM 2 is trained at 1024; smaller is faster on a CPU and a bit less exact
    "max_references": 6,            # how many of the newest annotated images SAM 2 learns from at once
    "min_score": 0.05,              # drop objects SAM 2 is less sure about (its scores run low: 0.2 is a good find)
}

DEFAULT_ASSISTANT = {
    "provider": PROVIDER_YOLO_WORLD,
    "confidence": 0.3,
    "yolo_world": {
        "model": YOLO_WORLD_MODELS[0],
        "target_classes": [],
    },
    "locate_anything": copy.deepcopy(LOCATE_DEFAULTS),
    "custom_model": copy.deepcopy(CUSTOM_MODEL_DEFAULTS),
    "custom_head": copy.deepcopy(HEAD_DEFAULTS),
    "sam2_dynamic": copy.deepcopy(SAM2_DEFAULTS),
    "batch": {"only_unlabeled": True},
}


def config_path(workspace_name):
    return os.path.join(CONFIGS_ROOT, f"{workspace_name}.json")


def _legacy_path(workspace_name):
    return os.path.join(CONFIGS_ROOT, f"{workspace_name}.txt")


def exists(workspace_name):
    return os.path.exists(config_path(workspace_name)) or os.path.exists(_legacy_path(workspace_name))


def _clean_targets(raw):
    targets = []
    for t in raw or []:
        if isinstance(t, dict) and isinstance(t.get("prompt"), str):
            targets.append({"prompt": t["prompt"],
                            "map_to": t.get("map_to") if isinstance(t.get("map_to"), str) else ""})
    return targets


def _num(value, default, low, high, kind=float):
    """value as `kind` when it is a number inside [low, high], otherwise the default."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not (low <= value <= high):
        return default
    return kind(value)


def _normalize_locate(raw):
    out = copy.deepcopy(LOCATE_DEFAULTS)
    if not isinstance(raw, dict):
        return out
    out["target_classes"] = _clean_targets(raw.get("target_classes"))
    if raw.get("generation_mode") in LOCATE_GENERATION_MODES:
        out["generation_mode"] = raw["generation_mode"]
    if raw.get("device") in LOCATE_DEVICES:
        out["device"] = raw["device"]
    out["short_side"] = _num(raw.get("short_side"), out["short_side"], 128, 4096, int)
    out["max_new_tokens"] = _num(raw.get("max_new_tokens"), out["max_new_tokens"], 64, 16384, int)
    out["passes"] = _num(raw.get("passes"), out["passes"], 1, 16, int)
    out["min_votes"] = _num(raw.get("min_votes"), out["min_votes"], 1, 16, int)
    out["temperature"] = _num(raw.get("temperature"), out["temperature"], 0.0, 2.0)
    out["iou_dedup"] = _num(raw.get("iou_dedup"), out["iou_dedup"], 0.1, 1.0)
    return out


def _clean_class_map(raw):
    return {k: v for k, v in raw.items() if isinstance(k, str) and isinstance(v, str)} if isinstance(raw, dict) else {}


def _normalize_custom_model(raw):
    out = copy.deepcopy(CUSTOM_MODEL_DEFAULTS)
    if isinstance(raw, dict):
        if isinstance(raw.get("path"), str):
            out["path"] = raw["path"].strip()
        out["class_map"] = _clean_class_map(raw.get("class_map"))
    return out


def _normalize_head(raw):
    out = copy.deepcopy(HEAD_DEFAULTS)
    if not isinstance(raw, dict):
        return out
    for key in ("head_path", "detector_weights"):
        if isinstance(raw.get(key), str):
            out[key] = raw[key].strip()
    ids = raw.get("detector_classes")
    if isinstance(ids, list):
        clean = [i for i in ids if isinstance(i, int) and not isinstance(i, bool) and i >= 0]
        if clean:
            out["detector_classes"] = sorted(set(clean))
    out["iou"] = _num(raw.get("iou"), out["iou"], 0.05, 0.95)
    out["min_head_conf"] = _num(raw.get("min_head_conf"), out["min_head_conf"], 0.0, 1.0)
    out["class_map"] = _clean_class_map(raw.get("class_map"))
    return out


def _normalize_sam2(raw):
    out = copy.deepcopy(SAM2_DEFAULTS)
    if not isinstance(raw, dict):
        return out
    if raw.get("model") in SAM2_MODELS:
        out["model"] = raw["model"]
    if raw.get("imgsz") in SAM2_IMAGE_SIZES:
        out["imgsz"] = raw["imgsz"]
    out["max_references"] = _num(raw.get("max_references"), out["max_references"], 1, 20, int)
    out["min_score"] = _num(raw.get("min_score"), out["min_score"], 0.0, 0.5)
    return out


def _normalize(data):
    """Fill in anything missing / malformed so callers can rely on the schema."""
    assistant = copy.deepcopy(DEFAULT_ASSISTANT)
    raw_assistant = data.get("label_assistant")
    if isinstance(raw_assistant, dict):
        if raw_assistant.get("provider") in PROVIDERS:
            assistant["provider"] = raw_assistant["provider"]
        assistant["locate_anything"] = _normalize_locate(raw_assistant.get("locate_anything"))
        assistant["custom_model"] = _normalize_custom_model(raw_assistant.get("custom_model"))
        assistant["custom_head"] = _normalize_head(raw_assistant.get("custom_head"))
        assistant["sam2_dynamic"] = _normalize_sam2(raw_assistant.get("sam2_dynamic"))
        raw_batch = raw_assistant.get("batch")
        if isinstance(raw_batch, dict) and isinstance(raw_batch.get("only_unlabeled"), bool):
            assistant["batch"]["only_unlabeled"] = raw_batch["only_unlabeled"]
        conf = raw_assistant.get("confidence")
        if isinstance(conf, (int, float)) and 0.0 < conf <= 1.0:
            assistant["confidence"] = float(conf)
        raw_yw = raw_assistant.get("yolo_world")
        if isinstance(raw_yw, dict):
            if raw_yw.get("model") in YOLO_WORLD_MODELS:
                assistant["yolo_world"]["model"] = raw_yw["model"]
            assistant["yolo_world"]["target_classes"] = _clean_targets(raw_yw.get("target_classes"))

    classes = data.get("classes")
    classes = [c for c in classes if isinstance(c, str)] if isinstance(classes, list) else []
    return {"version": CONFIG_VERSION, "classes": classes, "label_assistant": assistant}


def load(workspace_name):
    """Return the normalized config dict, or None if the workspace has none yet.
    A legacy .txt class list is migrated to JSON on the way."""
    path = config_path(workspace_name)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return _normalize(data)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[WorkspaceConfig] Could not read {path}: {exc}")
        return None

    legacy = _legacy_path(workspace_name)
    if os.path.exists(legacy):
        with open(legacy, "r", encoding="utf-8") as f:
            classes = [line.strip() for line in f if line.strip()]
        data = _normalize({"classes": classes})
        save(workspace_name, data)
        os.remove(legacy)
        print(f"[WorkspaceConfig] Migrated {legacy} -> {path}")
        return data
    return None


def save(workspace_name, data):
    os.makedirs(CONFIGS_ROOT, exist_ok=True)
    path = config_path(workspace_name)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(_normalize(data), f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def delete(workspace_name):
    for p in (config_path(workspace_name), _legacy_path(workspace_name)):
        if os.path.exists(p):
            os.remove(p)


# ---------------------------------------------------------------- classes
def get_classes(workspace_name):
    """List of classes, or None if the workspace has no config yet."""
    data = load(workspace_name)
    return None if data is None else list(data["classes"])


def set_classes(workspace_name, classes):
    """Replace the class list, keeping the assistant settings. YOLO-World and
    LocateAnything target classes (and custom-head class mappings) that point
    at a class which no longer exists are removed.
    Returns the list of removed target prompts."""
    data = load(workspace_name) or _normalize({})
    data["classes"] = list(classes)
    assistant = data["label_assistant"]
    removed = []
    for holder in (assistant["yolo_world"], assistant["locate_anything"]):
        removed += [t["prompt"] for t in holder["target_classes"] if t["map_to"] not in data["classes"]]
        holder["target_classes"] = [t for t in holder["target_classes"] if t["map_to"] in data["classes"]]
    for holder in (assistant["custom_model"], assistant["custom_head"]):
        holder["class_map"] = {k: v for k, v in holder["class_map"].items() if v in data["classes"]}
    save(workspace_name, data)
    return removed


# -------------------------------------------------------------- assistant
def get_assistant(workspace_name):
    data = load(workspace_name) or _normalize({})
    return data["label_assistant"]


def set_assistant(workspace_name, assistant):
    data = load(workspace_name) or _normalize({})
    data["label_assistant"] = assistant
    save(workspace_name, data)


# ------------------------------------------------- YOLO-World target validation
MAX_PROMPT_LEN = 40


def is_valid_prompt_text(text):
    """Prompt text may hold letters and spaces only - no digits, no symbols -
    and at most MAX_PROMPT_LEN characters."""
    return len(text) <= MAX_PROMPT_LEN and all(ch.isalpha() or ch == " " for ch in text)


def validate_targets(rows):
    """rows: iterable of (prompt, workspace_class). Fully empty rows are ignored.
    Returns (targets, error) - targets is None when error is set."""
    targets, seen = [], set()
    for prompt, cls in rows:
        prompt = " ".join((prompt or "").split())
        cls = cls or ""
        if not prompt and not cls:
            continue
        if not prompt:
            return None, "A target class has a workspace class but nothing to search for."
        if not cls:
            return None, f"Choose a workspace class for “{prompt}”."
        if prompt.lower() in seen:
            return None, f"“{prompt}” is listed more than once."
        seen.add(prompt.lower())
        targets.append({"prompt": prompt, "map_to": cls})
    return targets, None
