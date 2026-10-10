"""
Custom-trained dual-head YOLO models: a frozen Ultralytics detector plus a small RoI head you trained.

The detector is the stock pretrained model and is never updated. It finds boxes of the class(es) you choose
(COCO: person = 0, bottle = 39, ...). The head looks at the SAME neck features inside each box and gives the box
your own class - a behavior ("sleeping", "working") or a product ("CocaCola", "Fanta", "Sprite").

Ported from TOBEADDED/custom_dual_heads (model.py, dataset.py letterbox, infer.py). Differences: the detector
classes are a setting instead of always being person, results come back in the original image's pixels, end-to-end
(NMS-free) detectors are handled, and the checkpoint is read with torch.load(weights_only=True).

Needs torch + ultralytics >= 8.4.68. Import this module only when a head is actually used.
"""
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision.ops import roi_align

MIN_ULTRALYTICS = (8, 4, 68)

try:  # newer ultralytics versions
    from ultralytics.utils.nms import non_max_suppression
except ImportError:  # older versions
    from ultralytics.utils.ops import non_max_suppression

from .detector_classes import COCO_NAMES, parse_class_ids, format_class_ids  # noqa: E402,F401


def ultralytics_problem():
    """None when the installed ultralytics is new enough for dual-head models, otherwise a message."""
    try:
        import ultralytics
    except Exception as exc:
        return f"ultralytics is not installed ({exc})."
    try:
        found = tuple(int(p) for p in ultralytics.__version__.split(".")[:3])
    except ValueError:
        return None
    if found < MIN_ULTRALYTICS:
        return (f"The Custom Model mode needs ultralytics {'.'.join(map(str, MIN_ULTRALYTICS))} or newer "
                f"(found {ultralytics.__version__}). Run the Jelibox installer again to update it.")
    return None


# ------------------------------------------------------------------ geometry
def letterbox(img, size_hw, color=114):
    """Aspect-preserving resize + pad to (H, W). Returns img, ratio, left, top."""
    H, W = size_hw
    h, w = img.shape[:2]
    r = min(H / h, W / w)
    nh, nw = round(h * r), round(w * r)
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, left = (H - nh) // 2, (W - nw) // 2
    out = np.full((H, W, 3), color, np.uint8)
    out[top:top + nh, left:left + nw] = img
    return out, r, left, top


def unletterbox(box, r, left, top, width, height):
    """Box in letterboxed pixels -> integer (x1, y1, x2, y2) in the original image, clamped to it."""
    x1, y1, x2, y2 = (box[0] - left) / r, (box[1] - top) / r, (box[2] - left) / r, (box[3] - top) / r
    clamp = lambda v, hi: int(round(max(0.0, min(float(hi), v))))
    return clamp(x1, width), clamp(y1, height), clamp(x2, width), clamp(y2, height)


def scale_boxes(b, s, hw):
    """Scale xyxy boxes around their center by s and clip to image (h, w)."""
    h, w = hw
    cx, cy = (b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2
    bw, bh = (b[:, 2] - b[:, 0]) * s, (b[:, 3] - b[:, 1]) * s
    out = torch.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1)
    out[:, [0, 2]] = out[:, [0, 2]].clamp(0, w)
    out[:, [1, 3]] = out[:, [1, 3]].clamp(0, h)
    return out


# ------------------------------------------------------------------ model
class FrozenYOLO(nn.Module):
    """Wraps a pretrained Ultralytics DetectionModel. Fully frozen, always eval().
    Returns (detector_output, [P3, P4, P5]) from ONE forward pass via forward hooks."""

    def __init__(self, weights="yolov9c.pt", taps=(15, 18, 21)):
        super().__init__()
        from ultralytics import YOLO
        yolo = YOLO(weights)
        self.det = yolo.model.float()
        self.det_names = dict(getattr(yolo, "names", None) or {})
        self.taps = tuple(taps)
        for p in self.det.parameters():
            p.requires_grad_(False)
        self.det.eval()
        self._feats = {}
        for i in self.taps:
            self.det.model[i].register_forward_hook(self._hook(i))
        with torch.no_grad():  # discover channel widths
            self.det(torch.zeros(1, 3, 64, 64))
        self.channels = [self._feats[i].shape[1] for i in self.taps]
        self.sizes = [self._feats[i].shape[-1] for i in self.taps]     # 8, 4, 2 for a 64 px input = strides 8 / 16 / 32

    def _hook(self, i):
        def fn(_, __, out):
            self._feats[i] = out
        return fn

    def train(self, mode=True):  # never let BatchNorm stats update
        super().train(mode)
        self.det.eval()
        return self

    @torch.no_grad()
    def forward(self, x):
        self._feats.clear()
        out = self.det(x)
        return out, [self._feats[i] for i in self.taps]

    def strides_ok(self):
        """True when the tapped layers sit at strides 8 / 16 / 32, which is what BehaviorHead assumes."""
        return self.sizes == [8, 4, 2]

    def fingerprint(self):
        """Sum of every float parameter and buffer of the detector. Must be identical before and after training."""
        total = 0.0
        for t in self.det.state_dict().values():
            if t.is_floating_point():
                total += t.double().sum().item()
        return total


class BehaviorHead(nn.Module):
    """RoI head: pools the neck features inside each box (tight + context crop) and classifies it."""

    def __init__(self, in_chs, num_classes, strides=(8, 16, 32), c=256, out_size=7,
                 ctx_scale=1.6, canonical=224.0, dropout=0.3):
        super().__init__()
        self.strides, self.S, self.ctx_scale, self.canonical = strides, out_size, ctx_scale, canonical
        # per-level adapters (applied AFTER roi_align -> cheap, only K rois)
        self.proj = nn.ModuleList([
            nn.Sequential(nn.Conv2d(ch, c, 1, bias=False), nn.GroupNorm(32, c), nn.SiLU())
            for ch in in_chs])
        self.glob = nn.Sequential(nn.Linear(in_chs[-1], c), nn.SiLU())  # scene-level context
        self.tower = nn.Sequential(  # tight + context crops stacked on channels
            nn.Conv2d(2 * c, c, 3, padding=1, bias=False), nn.GroupNorm(32, c), nn.SiLU(),
            nn.Conv2d(c, 128, 3, padding=1, bias=False), nn.GroupNorm(32, 128), nn.SiLU())
        self.fc = nn.Sequential(  # flatten keeps spatial layout
            nn.Dropout(dropout), nn.Linear(128 * out_size * out_size + c, 512), nn.SiLU(),
            nn.Dropout(dropout), nn.Linear(512, num_classes))

    def _levels(self, boxes):
        wh = ((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])).clamp(min=1).sqrt()
        lvl = torch.floor(4 + torch.log2(wh / self.canonical + 1e-6))  # 224px -> P4
        return (lvl.clamp(3, 5) - 3).long()

    def _pool(self, feats, bidx, boxes, lvl):
        parts, order = [], []
        for l in range(len(feats)):
            idx = (lvl == l).nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                continue
            rois = torch.cat([bidx[idx, None].to(boxes.dtype), boxes[idx]], 1)
            p = roi_align(feats[l], rois, self.S, 1.0 / self.strides[l], 2, True)
            parts.append(self.proj[l](p))
            order.append(idx)
        pooled, order = torch.cat(parts), torch.cat(order)
        inv = torch.empty_like(order)
        inv[order] = torch.arange(order.numel(), device=order.device)
        return pooled[inv]

    def forward(self, feats, bidx, boxes, hw):
        boxes = boxes.float()
        tight = scale_boxes(boxes, 1.0, hw)
        ctx = scale_boxes(boxes, self.ctx_scale, hw)
        lvl = self._levels(tight)
        x = torch.cat([self._pool(feats, bidx, tight, lvl),
                       self._pool(feats, bidx, ctx, lvl)], 1)
        x = self.tower(x).flatten(1)
        g = self.glob(feats[-1].mean((2, 3)))[bidx]
        return self.fc(torch.cat([x, g.to(x.dtype)], 1))


class BehaviorModel(nn.Module):
    def __init__(self, weights, num_classes, taps=(15, 18, 21), strides=(8, 16, 32), **head_kw):
        super().__init__()
        self.backbone = FrozenYOLO(weights, taps)
        self.head = BehaviorHead(self.backbone.channels, num_classes, strides, **head_kw)

    def forward(self, imgs, bidx, boxes):
        _, feats = self.backbone(imgs)
        return self.head(feats, bidx, boxes, imgs.shape[-2:])


# ------------------------------------------------------------------ checkpoint
def read_checkpoint(path):
    """Load and validate a head checkpoint written by train.py. Returns the dict
    {'head': state_dict, 'names': [...], 'cfg': {'weights', 'taps', 'imgsz', 'nc'}}."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Head checkpoint not found: {path}")
    try:
        ck = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError(f"{os.path.basename(path)} is not a head checkpoint Jelibox can read: {exc}") from exc
    cfg = ck.get("cfg") if isinstance(ck, dict) else None
    ok = (isinstance(ck, dict) and "head" in ck and isinstance(ck.get("names"), (list, tuple)) and isinstance(cfg, dict)
          and all(k in cfg for k in ("weights", "taps", "imgsz", "nc")))
    if not ok:
        raise ValueError(f"{os.path.basename(path)} is missing head / names / cfg (weights, taps, imgsz, nc) - "
                         f"was it saved by train.py?")
    ck["names"] = [str(n) for n in ck["names"]]
    return ck


class HeadDetector:
    """Frozen detector + trained head, ready to annotate images."""

    def __init__(self, head_path, detector_weights="", detectors_dir=None, device="auto"):
        problem = ultralytics_problem()
        if problem:
            raise RuntimeError(problem)
        ck = read_checkpoint(head_path)
        cfg = ck["cfg"]
        self.names = ck["names"]
        self.imgsz = tuple(int(v) for v in cfg["imgsz"])
        self.device = torch.device("cuda" if (device == "cuda" or (device == "auto" and torch.cuda.is_available()))
                                   else "cpu")
        weights = detector_weights or cfg["weights"]
        if detectors_dir and not os.path.isabs(weights) and not os.path.exists(weights):
            os.makedirs(detectors_dir, exist_ok=True)       # ultralytics downloads a missing file to this path
            weights = os.path.join(detectors_dir, os.path.basename(weights))
        self.model = BehaviorModel(weights, int(cfg["nc"]), taps=tuple(cfg["taps"])).to(self.device)
        self.model.head.load_state_dict(ck["head"])
        self.model.eval()
        self.detector_names = self.model.backbone.det_names

    def _detections(self, out, conf, iou, classes):
        out0 = out[0] if isinstance(out, (list, tuple)) else out
        if isinstance(out0, dict):                                 # some end-to-end exports
            out0 = out0.get("one2one", next(iter(out0.values())))
        if out0.ndim == 3 and out0.shape[-1] == 6:                # end-to-end (NMS-free): already (x1,y1,x2,y2,conf,cls)
            det = out0[0]
            keep = det[:, 4] >= conf
            if classes:
                keep &= torch.isin(det[:, 5].long(), torch.tensor(classes, device=det.device))
            return det[keep]
        return non_max_suppression(out0, conf, iou, classes=classes, max_det=300)[0]

    @torch.no_grad()
    def predict(self, frame_bgr, conf=0.3, iou=0.6, classes=(0,), min_head_conf=0.0):
        """frame_bgr: HxWx3 uint8. Returns [{'rect': (x1,y1,x2,y2) ints in the original image, 'det_conf',
        'head': class name, 'head_conf'}], most confident box first."""
        H, W = self.imgsz
        h0, w0 = frame_bgr.shape[:2]
        img, r, left, top = letterbox(frame_bgr, (H, W))
        x = torch.from_numpy(img[:, :, ::-1].copy()).permute(2, 0, 1)[None].to(self.device).float() / 255
        with torch.autocast(self.device.type, enabled=self.device.type == "cuda"):
            out, feats = self.model.backbone(x)
            det = self._detections(out, conf, iou, list(classes) if classes else None)
            if not len(det):
                return []
            boxes = det[:, :4].float()
            logits = self.model.head(feats, torch.zeros(len(boxes), dtype=torch.long, device=self.device),
                                     boxes, (H, W))
        probs = logits.float().softmax(1).cpu()
        results = []
        for box, det_conf, p in zip(boxes.cpu().tolist(), det[:, 4].cpu().tolist(), probs):
            k = int(p.argmax())
            if float(p[k]) < min_head_conf:
                continue
            rect = unletterbox(box, r, left, top, w0, h0)
            if rect[2] - rect[0] < 2 or rect[3] - rect[1] < 2:
                continue
            results.append({"rect": rect, "det_conf": float(det_conf), "head": self.names[k],
                            "head_conf": float(p[k])})
        results.sort(key=lambda d: d["det_conf"], reverse=True)
        return results
