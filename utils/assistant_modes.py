"""
The four ways Jelibox can annotate for you, picked on the front menu of the workspace picker.

  - YOLO-World       : the original Jelibox - zero-shot text prompts, or the model you trained here
  - LocateAnything   : nvidia/LocateAnything-3B (HF Transformers), a vision-language model
  - Custom Model     : a frozen YOLO detector (e.g. COCO "person" or "bottle" boxes) plus a head you trained
                       that gives every box your own class (behavior, brand, ...). Called "custom head" in the code.
  - SAM 2 Dynamic    : Ultralytics' SAM2DynamicInteractivePredictor, a training-free SAM 2 that learns from the
                       images you annotate (utils/sam2_dynamic.py)

The chosen mode is app-wide (configs/_app.json); the settings of each mode are per workspace
(configs/<workspace>.json, see workspace_config.py). Dependency-free on purpose: the picker imports it
before any workspace is loaded.
"""
from . import app_settings
from . import workspace_config as wcfg

MODE_YOLO_WORLD = "yolo_world"
MODE_LOCATE = "locate_anything"
MODE_HEAD = "custom_head"
MODE_SAM2 = "sam2_dynamic"

DEFAULT_MODE = MODE_YOLO_WORLD

# (id, title, one-line summary, what it needs)
MODES = [
    (MODE_YOLO_WORLD, "YOLO-World",
     "The original Jelibox. Describe what to look for in words, or use the model you trained here.",
     "Works out of the box. Light enough for CPU."),
    (MODE_LOCATE, "LocateAnything",
     "NVIDIA LocateAnything-3B. Labels a whole folder from a list of categories, in English or Chinese.",
     "About 7 GB download on first use. Needs a GPU with 8 GB+ or about 10 GB of free RAM."),
    (MODE_HEAD, "Custom Model",
     "Teach Jelibox your own labels. It finds the objects, then a small model that you train names each one "
     "(for example sleeping / working, or Coca-Cola / Fanta / Sprite).",
     "Train it once with the Train button, using images you have already labeled."),
    (MODE_SAM2, "SAM 2 Dynamic",
     "No training. Annotate a few images and SAM 2 learns from every one of them, so it finds the same "
     "objects in the next image more and more accurately.",
     "Downloads a SAM 2 model once (75-224 MB). A GPU makes it fast; a CPU works, a few seconds per image."),
]

_TITLES = {m[0]: m[1] for m in MODES}


def title(mode):
    return _TITLES.get(mode, _TITLES[DEFAULT_MODE])


def get_mode():
    mode = app_settings.get("mode", DEFAULT_MODE)
    return mode if mode in _TITLES else DEFAULT_MODE


def set_mode(mode):
    if mode not in _TITLES:
        raise ValueError(f"unknown mode: {mode!r}")
    app_settings.set("mode", mode)


def provider_for(mode, saved_provider):
    """Which provider answers G / Auto-annotate in `mode`. YOLO-World mode honours the workspace's own choice
    between YOLO-World and 'my trained model'; the other modes have exactly one provider."""
    if mode == MODE_LOCATE:
        return wcfg.PROVIDER_LOCATE
    if mode == MODE_HEAD:
        return wcfg.PROVIDER_HEAD
    if mode == MODE_SAM2:
        return wcfg.PROVIDER_SAM2
    return saved_provider if saved_provider in (wcfg.PROVIDER_YOLO_WORLD, wcfg.PROVIDER_CUSTOM) \
        else wcfg.PROVIDER_YOLO_WORLD
