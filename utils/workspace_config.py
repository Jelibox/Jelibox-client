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
        }
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

YOLO_WORLD_MODELS = [
    "yolov8s-world", "yolov8s-worldv2",
    "yolov8m-world", "yolov8m-worldv2",
    "yolov8l-world", "yolov8l-worldv2",
    "yolov8x-world", "yolov8x-worldv2",
]

DEFAULT_ASSISTANT = {
    "provider": PROVIDER_YOLO_WORLD,
    "confidence": 0.3,
    "yolo_world": {
        "model": YOLO_WORLD_MODELS[0],
        "target_classes": [],
    },
}


def config_path(workspace_name):
    return os.path.join(CONFIGS_ROOT, f"{workspace_name}.json")


def _legacy_path(workspace_name):
    return os.path.join(CONFIGS_ROOT, f"{workspace_name}.txt")


def exists(workspace_name):
    return os.path.exists(config_path(workspace_name)) or os.path.exists(_legacy_path(workspace_name))


def _normalize(data):
    """Fill in anything missing / malformed so callers can rely on the schema."""
    assistant = copy.deepcopy(DEFAULT_ASSISTANT)
    raw_assistant = data.get("label_assistant")
    if isinstance(raw_assistant, dict):
        if raw_assistant.get("provider") in (PROVIDER_YOLO_WORLD, PROVIDER_CUSTOM):
            assistant["provider"] = raw_assistant["provider"]
        conf = raw_assistant.get("confidence")
        if isinstance(conf, (int, float)) and 0.0 < conf <= 1.0:
            assistant["confidence"] = float(conf)
        raw_yw = raw_assistant.get("yolo_world")
        if isinstance(raw_yw, dict):
            if raw_yw.get("model") in YOLO_WORLD_MODELS:
                assistant["yolo_world"]["model"] = raw_yw["model"]
            targets = []
            for t in raw_yw.get("target_classes") or []:
                if isinstance(t, dict) and isinstance(t.get("prompt"), str):
                    targets.append({"prompt": t["prompt"],
                                    "map_to": t.get("map_to") if isinstance(t.get("map_to"), str) else ""})
            assistant["yolo_world"]["target_classes"] = targets

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
    """Replace the class list, keeping the assistant settings. YOLO-World
    target classes that map to a class which no longer exists are removed.
    Returns the list of removed target prompts."""
    data = load(workspace_name) or _normalize({})
    data["classes"] = list(classes)
    yw = data["label_assistant"]["yolo_world"]
    removed = [t["prompt"] for t in yw["target_classes"] if t["map_to"] not in data["classes"]]
    yw["target_classes"] = [t for t in yw["target_classes"] if t["map_to"] in data["classes"]]
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
