"""
LocateAnything-3B (nvidia/LocateAnything-3B) inference through HF Transformers.

Ported from the LocateAnything AutoLabeller (TOBEADDED/locateanything/backend/engine.py). Differences:
  - runs inside Jelibox (no web server); the model is loaded on first use, never at import time
  - the model files live in <Jelibox>/models/_huggingface, not in the user's global Hugging Face cache
    (set HF_HOME yourself to use another cache)
  - the model code downloaded with `trust_remote_code` is pinned to a reviewed commit (MODEL_REVISION)
  - checks free memory before loading, so a machine that is too small gets a message instead of freezing

Importing this module is cheap: torch and transformers are only imported when a model is actually used.
"""
import importlib.metadata as _md
import logging
import os
import platform
import sys
import threading
import time

from . import locate_parsing as lp

log = logging.getLogger(__name__)

MODEL_ID = "nvidia/LocateAnything-3B"
# Commit of MODEL_ID whose remote code was checked. trust_remote_code runs Python from the model repo, so it is
# pinned instead of following "main". Change it deliberately, after reading what changed.
MODEL_REVISION = "c32291ca5e996f5a7a485845b4f57a233936bba0"
DOWNLOAD_GB = 7.2
ATTN_IMPL = "sdpa"                 # the checkpoint default is "magi", which needs kernels most machines lack
MIN_FREE_RAM_GB = 9.0              # the weights pass through system RAM while loading, on GPU and CPU alike
MIN_FREE_VRAM_GB = 8.0

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
HF_HOME = os.path.join(BASE_DIR, "models", "_huggingface")

# Keep the download inside Jelibox unless the user already chose a cache. Must happen before huggingface_hub or
# transformers are imported, which is why this runs at import time and those imports are lazy.
if not (os.environ.get("HF_HOME") or os.environ.get("HF_HUB_CACHE") or os.environ.get("TRANSFORMERS_CACHE")):
    os.environ["HF_HOME"] = HF_HOME
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

_SHARDS = ("config.json", "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors")


def _version(dist):
    try:
        return _md.version(dist)
    except _md.PackageNotFoundError:
        return None


def check_requirements():
    """None when every package LocateAnything needs is installed in this environment, otherwise a message that
    says what is wrong and how to fix it. Looks at package metadata only - nothing heavy is imported."""
    fix = ('Run the Jelibox installer again to add them, or install them yourself:\n'
           f'"{sys.executable}" -m pip install -r requirements.txt')
    missing = []
    for dist in ("torch", "torchvision", "transformers", "huggingface_hub", "peft", "lmdb", "accelerate",
                 "safetensors"):
        if _version(dist) is None:
            missing.append(dist)
    # decord (imported by the model's processor) only has wheels for 64-bit Windows and x86-64 Linux
    if (sys.platform == "win32" or sys.platform.startswith("linux")) \
            and platform.machine().lower() in ("amd64", "x86_64") and _version("decord") is None:
        missing.append("decord")
    if missing:
        return "LocateAnything needs packages this Jelibox does not have yet: " + ", ".join(missing) + ".\n\n" + fix
    tf = _version("transformers")
    if int(tf.split(".")[0]) >= 5:
        return (f"LocateAnything needs transformers below 5 (found {tf}); the model was published against "
                f"4.57.6.\n\n{fix}")
    hub = _version("huggingface_hub")
    if int(hub.split(".")[0]) >= 1:
        return (f"transformers {tf} needs huggingface_hub below 1.0 (found {hub}).\n\n{fix}")
    return None


def cache_dir():
    """Where the model is cached: None (= Hugging Face's own default/HF_HOME) when the user chose a cache."""
    if os.environ.get("HF_HOME") == HF_HOME:
        return os.path.join(HF_HOME, "hub")
    return None


def is_cached():
    """True when the model files are already on disk, so using it needs no download."""
    try:
        from huggingface_hub import try_to_load_from_cache
    except Exception:
        return False
    for name in _SHARDS:
        path = try_to_load_from_cache(MODEL_ID, name, cache_dir=cache_dir(), revision=MODEL_REVISION)
        if not isinstance(path, str):
            return False
    return True


def resolve_device(preference="auto"):
    import torch
    if preference == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was selected but no NVIDIA GPU is available to PyTorch.")
    if preference in ("cuda", "cpu"):
        return preference
    return "cuda" if torch.cuda.is_available() else "cpu"


def memory_problem(device):
    """None when there is enough free memory to load the model, otherwise a message."""
    try:
        import psutil
        free_ram = psutil.virtual_memory().available / 1024 ** 3
    except Exception:
        free_ram = None
    if free_ram is not None and free_ram < MIN_FREE_RAM_GB:
        return (f"Only {free_ram:.1f} GB of RAM is free, LocateAnything needs about {MIN_FREE_RAM_GB:.0f} GB "
                f"(the 3B model is {DOWNLOAD_GB} GB). Close other programs and try again.")
    if device == "cuda":
        import torch
        free_vram = torch.cuda.mem_get_info()[0] / 1024 ** 3
        if free_vram < MIN_FREE_VRAM_GB:
            return (f"Only {free_vram:.1f} GB of GPU memory is free, LocateAnything needs about "
                    f"{MIN_FREE_VRAM_GB:.0f} GB. Close other GPU programs, or choose the CPU in the settings.")
    return None


def _resize_short_side(img, short_side):
    w, h = img.size
    if min(w, h) <= short_side:
        return img
    from PIL import Image
    if w <= h:
        new_w, new_h = short_side, max(1, round(h * short_side / w))
    else:
        new_h, new_w = short_side, max(1, round(w * short_side / h))
    return img.resize((new_w, new_h), Image.BILINEAR)


class LocateAnythingEngine:
    """One model, one image at a time (the model only supports batch size 1 and is not re-entrant)."""

    def __init__(self):
        self._lock = threading.Lock()
        self.model = self.processor = self.tokenizer = None
        self.device = None

    @property
    def loaded(self):
        return self.model is not None

    def load(self, device_pref="auto", status=None):
        """Download (first time only) and load the model. `status(text)` gets human-readable progress lines."""
        say = status or (lambda text: None)
        with self._lock:
            if self.loaded:
                return
            problem = check_requirements()
            if problem:
                raise RuntimeError(problem)
            device = resolve_device(device_pref)
            problem = memory_problem(device)
            if problem:
                raise RuntimeError(problem)

            import torch
            from transformers import AutoModel, AutoProcessor, AutoTokenizer

            kw = dict(trust_remote_code=True, revision=MODEL_REVISION, cache_dir=cache_dir())
            say("Downloading LocateAnything-3B (first use only) ..." if not is_cached()
                else "Loading LocateAnything-3B ...")
            t0 = time.time()
            tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, **kw)
            processor = AutoProcessor.from_pretrained(MODEL_ID, **kw)
            model = AutoModel.from_pretrained(MODEL_ID, dtype=torch.bfloat16, _attn_implementation=ATTN_IMPL, **kw)
            say(f"Moving the model to {device} ...")
            model = model.to(device).eval()
            self.tokenizer, self.processor, self.model, self.device = tokenizer, processor, model, device
            log.info("LocateAnything ready on %s in %.1fs", device, time.time() - t0)
            say("LocateAnything is ready.")

    def unload(self):
        with self._lock:
            self.model = self.processor = self.tokenizer = None
            self.device = None
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _infer_once(self, image, categories, s):
        import torch
        prompt = lp.build_prompt(categories)
        orig_w, orig_h = image.size
        model_img = _resize_short_side(image.convert("RGB"), s["short_side"])
        messages = [{"role": "user", "content": [{"type": "image", "image": model_img},
                                                  {"type": "text", "text": prompt}]}]
        with self._lock, torch.no_grad():
            text = self.processor.py_apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            images, videos = self.processor.process_vision_info(messages)
            inputs = self.processor(text=[text], images=images, videos=videos, return_tensors="pt").to(self.device)
            temperature = float(s["temperature"])
            result = self.model.generate(
                pixel_values=inputs["pixel_values"].to(torch.bfloat16),
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                image_grid_hws=inputs.get("image_grid_hws"),
                tokenizer=self.tokenizer,
                max_new_tokens=s["max_new_tokens"],
                use_cache=True,
                generation_mode=s["generation_mode"],
                temperature=temperature,
                do_sample=temperature > 0,
                top_p=0.9,
                top_k=20,
                repetition_penalty=1.1,
                verbose=False,
            )
        raw = result[0] if isinstance(result, tuple) else result
        raw = raw if isinstance(raw, str) else str(raw)
        parsed = lp.parse_output(raw, fallback_label=categories[0] if categories else "object")
        return raw, lp.to_pixel_boxes(parsed, orig_w, orig_h, iou_dedup=s["iou_dedup"])

    def detect(self, image, categories, settings):
        """image: PIL.Image. Returns (boxes, raw_output). boxes: [{'label','x1','y1','x2','y2',('votes','score')}]
        in the ORIGINAL image's pixels. `settings` is the 'locate_anything' block of the workspace config."""
        if not self.loaded:
            raise RuntimeError("The LocateAnything model is not loaded.")
        passes = max(1, int(settings.get("passes", 1)))
        if passes == 1:
            raw, boxes = self._infer_once(image, categories, settings)
            return boxes, raw
        raws, per_pass = [], []
        for _ in range(passes):
            raw, boxes = self._infer_once(image, categories, settings)
            raws.append(raw)
            per_pass.append(boxes)
        return lp.merge_passes(per_pass, min_votes=int(settings.get("min_votes", 1))), "\n---\n".join(raws)


engine = LocateAnythingEngine()
