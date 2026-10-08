"""
Pure helpers for LocateAnything-3B: prompt building, parsing the model's text output into boxes, and merging
several decoding passes. Ported from the LocateAnything AutoLabeller (TOBEADDED/locateanything/backend).

No torch / transformers imports on purpose - cheap to import and unit-testable.
"""
import re
from typing import Any

COORD_SCALE = 1000.0   # the model emits coordinates on a 0..1000 grid, relative to the image size

_REF_OR_BOX = re.compile(r"(<ref>.*?</ref>)|(<box>.*?</box>)", re.IGNORECASE | re.DOTALL)
_NUM = re.compile(r"<\s*([0-9]+(?:\.[0-9]+)?)\s*>")
_STRIP_REF = re.compile(r"</?ref>", re.IGNORECASE)
_STRIP_BOX = re.compile(r"</?box>", re.IGNORECASE)


# ------------------------------------------------------------------ prompts
def build_prompt(categories):
    """The detection prompt the model was trained on. Categories are joined with </c>."""
    cats = "</c>".join(c.strip() for c in categories if c and c.strip()) or "objects"
    return f"Locate all the instances that matches the following description: {cats}."


# ------------------------------------------------------------------ parsing
def parse_output(text, fallback_label="object"):
    """Extract <ref>label</ref><box><x1><y1><x2><y2></box> groups. A <ref> applies to every following <box>
    until the next <ref>. Coordinates stay in the model's 0..1000 space."""
    out = []
    current = None
    for m in _REF_OR_BOX.finditer(text or ""):
        token = m.group(0)
        if token.lower().startswith("<ref>"):
            label = _STRIP_REF.sub("", token).strip()
            if label:
                current = label
            continue
        coords = [float(n) for n in _NUM.findall(_STRIP_BOX.sub("", token))]
        label = current or fallback_label
        if len(coords) >= 4:
            out.append({"kind": "box", "coords": coords[:4], "label": label})
        elif len(coords) == 2:
            out.append({"kind": "point", "coords": coords, "label": label})
    return out


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def to_pixel_boxes(parsed, width, height, *, iou_dedup=0.9, min_size_px=2.0, max_boxes=300):
    """Map 0..1000 coordinates onto pixels, drop degenerate boxes and collapse duplicates. The fast/hybrid
    decoding paths can stutter and repeat the same box many times; per-label IoU dedup keeps one of each.
    Returns [{'label', 'x1', 'y1', 'x2', 'y2'}] in pixels."""
    kept = []
    per_label = {}
    for det in parsed:
        if det["kind"] != "box":
            continue
        x1, y1, x2, y2 = det["coords"]
        x1, x2 = sorted((x1 * width / COORD_SCALE, x2 * width / COORD_SCALE))
        y1, y2 = sorted((y1 * height / COORD_SCALE, y2 * height / COORD_SCALE))
        x1, x2 = max(0.0, min(float(width), x1)), max(0.0, min(float(width), x2))
        y1, y2 = max(0.0, min(float(height), y1)), max(0.0, min(float(height), y2))
        if (x2 - x1) < min_size_px or (y2 - y1) < min_size_px:
            continue
        box = (x1, y1, x2, y2)
        label = det["label"]
        if any(iou(box, prev) >= iou_dedup for prev in per_label.get(label, ())):
            continue
        per_label.setdefault(label, []).append(box)
        kept.append({"label": label, "x1": x1, "y1": y1, "x2": x2, "y2": y2})
        if len(kept) >= max_boxes:
            break
    return kept


def merge_passes(passes_boxes, merge_iou=0.6, min_votes=1):
    """Cluster the boxes of several sampled decoding passes by IoU (same label only). Coordinates of a cluster
    are averaged, `votes` is how many passes found it. Returns [{'label','x1','y1','x2','y2','votes','score'}]."""
    n_passes = max(1, len(passes_boxes))
    clusters = []
    for boxes in passes_boxes:
        for b in boxes:
            box = (b["x1"], b["y1"], b["x2"], b["y2"])
            best, best_iou = None, 0.0
            for c in clusters:
                if c["label"] != b["label"]:
                    continue
                value = iou(box, c["mean"])
                if value > best_iou:
                    best, best_iou = c, value
            if best is not None and best_iou >= merge_iou:
                best["members"].append(box)
                count = len(best["members"])
                best["mean"] = tuple(sum(m[i] for m in best["members"]) / count for i in range(4))
            else:
                clusters.append({"label": b["label"], "members": [box], "mean": box})
    out = []
    for c in clusters:
        votes = len(c["members"])
        if votes < min_votes:
            continue
        x1, y1, x2, y2 = c["mean"]
        out.append({"label": c["label"], "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "votes": votes, "score": round(min(1.0, votes / n_passes), 3)})
    out.sort(key=lambda b: (-b["votes"], b["x1"]))
    return out


# ------------------------------------------------------------------ labels -> workspace classes
def map_to_workspace(boxes, targets, fallback_when_single=True):
    """Turn model labels into workspace classes using the target list [{'prompt', 'map_to'}].
    The model repeats a prompt back as the label (matched ignoring case and extra spaces). When the model
    answers with a label that is not in the list and there is exactly one target, that target is used.
    Boxes that cannot be matched are dropped. Returns [{'rect', 'conf', 'cls'}] with integer pixel rects."""
    norm = lambda s: " ".join((s or "").lower().split())
    by_prompt = {norm(t["prompt"]): t["map_to"] for t in targets if t.get("prompt", "").strip() and t.get("map_to")}
    only = next(iter(by_prompt.values())) if len(by_prompt) == 1 and fallback_when_single else None
    preds = []
    for b in boxes:
        cls = by_prompt.get(norm(b["label"]), only)
        if not cls:
            continue
        preds.append({"rect": (int(round(b["x1"])), int(round(b["y1"])), int(round(b["x2"])), int(round(b["y2"]))),
                      "conf": float(b.get("score", 1.0)), "cls": cls})
    return preds
