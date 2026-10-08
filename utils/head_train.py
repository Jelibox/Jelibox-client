"""
Train a custom head on a frozen YOLO detector - the training half of the Custom head mode.

Ported from TOBEADDED/custom_dual_heads (train.py + dataset.py). It runs as its own process so a long run never
freezes the annotation window:

    python -m utils.head_train <job.json>          (run from the Jelibox folder; the GUI does this)

The job file says which images / labels / classes / detector to use (see HeadTrainDialog). Progress goes to stdout as
lines `@@ {json}` (events: status, start, epoch, done, error) that the GUI turns into a progress window; anything
else printed is plain log text. The detector is never changed: its weights are fingerprinted before and after.

Needs torch + ultralytics >= 8.4.68. Import this module only when a head is actually trained.
"""
import json
import math
import os
import random
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from . import head_data
from .custom_head import BehaviorModel, letterbox, ultralytics_problem

PREFIX = "@@ "


# ------------------------------------------------------------------ data
class HeadDataset(Dataset):
    """Items are (image CHW RGB uint8, boxes xyxy in letterboxed pixels, class ids)."""

    def __init__(self, pairs, imgsz, n_classes, train=True, hflip=True):
        self.pairs, self.imgsz, self.nc, self.train, self.hflip = list(pairs), tuple(imgsz), n_classes, train, hflip

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        img_path, label_path = self.pairs[i]
        data = np.fromfile(img_path, dtype=np.uint8)                       # also works for non-ASCII paths
        img = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
        if img is None:
            raise OSError(f"could not read {img_path}")
        h0, w0 = img.shape[:2]
        rows = [r for r in head_data.read_label_rows(label_path) if 0 <= r[0] < self.nc]
        lab = np.array(rows, np.float32).reshape(-1, 5)
        cls = lab[:, 0].astype(np.int64)
        cx, cy, bw, bh = lab[:, 1] * w0, lab[:, 2] * h0, lab[:, 3] * w0, lab[:, 4] * h0
        boxes = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1).astype(np.float32)
        img, r, left, top = letterbox(img, self.imgsz)
        boxes = boxes * r + np.array([left, top, left, top], np.float32)
        if self.train:
            img, boxes, cls = self._augment(img, boxes, cls)
        img = np.ascontiguousarray(img[:, :, ::-1].transpose(2, 0, 1))      # BGR -> RGB, CHW
        return torch.from_numpy(img), torch.from_numpy(boxes), torch.from_numpy(cls)

    def _augment(self, img, boxes, cls):
        H, W = img.shape[:2]
        s = random.uniform(0.75, 1.25)
        tx, ty = random.uniform(-0.1, 0.1) * W, random.uniform(-0.1, 0.1) * H
        M = np.array([[s, 0, (1 - s) * W / 2 + tx], [0, s, (1 - s) * H / 2 + ty]], np.float32)
        img = cv2.warpAffine(img, M, (W, H), borderValue=(114, 114, 114))
        if len(boxes):
            b = boxes.copy()
            b[:, [0, 2]] = b[:, [0, 2]] * s + M[0, 2]
            b[:, [1, 3]] = b[:, [1, 3]] * s + M[1, 2]
            a0 = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
            b[:, [0, 2]] = b[:, [0, 2]].clip(0, W)
            b[:, [1, 3]] = b[:, [1, 3]].clip(0, H)
            bw, bh = b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]
            keep = (bw * bh / np.maximum(a0, 1e-6) > 0.5) & (bw > 8) & (bh > 8)
            boxes, cls = b[keep], cls[keep]
        if self.hflip and random.random() < 0.5:
            img = img[:, ::-1]
            if len(boxes):
                boxes = boxes.copy()
                boxes[:, [0, 2]] = W - boxes[:, [2, 0]]
        hsv = cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 1] *= random.uniform(0.6, 1.4)
        hsv[..., 2] *= random.uniform(0.6, 1.4)
        img = cv2.cvtColor(hsv.clip(0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
        return img, boxes, cls


def collate(batch):
    imgs, rois, labels = [], [], []
    for i, (im, b, c) in enumerate(batch):
        imgs.append(im)
        if len(b):
            rois.append(torch.cat([torch.full((len(b), 1), float(i)), b], 1))
            labels.append(c)
    imgs = torch.stack(imgs)
    rois = torch.cat(rois) if rois else torch.zeros(0, 5)
    labels = torch.cat(labels) if labels else torch.zeros(0, dtype=torch.long)
    return imgs, rois, labels


def jitter(boxes, hw, amt=0.1):
    """Simulate detector box noise so the head doesn't overfit to perfect labelled boxes."""
    h, w = hw
    bw, bh = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
    d = (torch.rand(len(boxes), 4, device=boxes.device) * 2 - 1) * amt
    out = boxes + d * torch.stack([bw, bh, bw, bh], 1)
    out[:, [0, 2]] = out[:, [0, 2]].clamp(0, w)
    out[:, [1, 3]] = out[:, [1, 3]].clamp(0, h)
    ok = ((out[:, 2] - out[:, 0]) > 4) & ((out[:, 3] - out[:, 1]) > 4)
    return torch.where(ok[:, None], out, boxes)


@torch.no_grad()
def evaluate(model, dl, nc, dev, amp):
    model.eval()
    cm = torch.zeros(nc, nc, dtype=torch.long)
    for imgs, rois, labels in dl:
        if len(rois) == 0:
            continue
        imgs = imgs.to(dev).float() / 255
        with torch.autocast(dev.type, enabled=amp):
            logits = model(imgs, rois[:, 0].long().to(dev), rois[:, 1:].to(dev))
        pred = logits.argmax(1).cpu()
        for t, p in zip(labels, pred):
            cm[t, p] += 1
    tp = cm.diag().float()
    rec = tp / cm.sum(1).clamp(min=1)
    prec = tp / cm.sum(0).clamp(min=1)
    f1 = 2 * prec * rec / (prec + rec).clamp(min=1e-9)
    present = cm.sum(1) > 0                                   # a class with no validation boxes says nothing
    return dict(acc=(tp.sum() / cm.sum().clamp(min=1)).item(),
                f1=(f1[present].mean().item() if present.any() else 0.0),
                recall=rec.tolist(), support=cm.sum(1).tolist(), cm=cm)


# ------------------------------------------------------------------ training
def emit(event, **fields):
    print(PREFIX + json.dumps({"event": event, **fields}), flush=True)


def resolve_weights(weights, detectors_dir):
    """A stock name (yolov8m.pt) that is not on disk is fetched into the detectors folder, like at annotation time."""
    if detectors_dir and not os.path.isabs(weights) and not os.path.exists(weights):
        os.makedirs(detectors_dir, exist_ok=True)
        return os.path.join(detectors_dir, os.path.basename(weights))
    return weights


def train(job, emit=emit, should_stop=lambda: False):
    """Run one training job (a dict, see the module doc). Returns the summary dict sent with the `done` event."""
    problem = ultralytics_problem()
    if problem:
        raise RuntimeError(problem)
    classes = [str(c) for c in job["classes"]]
    nc = len(classes)
    imgsz = tuple(int(v) for v in job["imgsz"])
    if len(imgsz) != 2 or any(v < 32 or v % 32 for v in imgsz):
        raise ValueError(f"Image size {imgsz[0]}x{imgsz[1]} must be two multiples of 32 (height, width).")
    out = job["out_dir"]
    os.makedirs(out, exist_ok=True)

    pairs = head_data.labelled_pairs(job["images_folders"], job["labels_folder"])
    if len(pairs) < head_data.MIN_LABELLED:
        raise ValueError(f"Only {len(pairs)} labelled images; at least {head_data.MIN_LABELLED} are needed.")
    train_pairs, val_pairs = head_data.split_contiguous(pairs, float(job.get("val_fraction", 0.2)))

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = dev.type == "cuda"
    batch = max(1, min(int(job["batch"]), len(train_pairs)))
    workers = int(job.get("workers", 2))
    tr = HeadDataset(train_pairs, imgsz, nc, True, bool(job.get("hflip", True)))
    va = HeadDataset(val_pairs, imgsz, nc, False)
    tdl = DataLoader(tr, batch, shuffle=True, num_workers=workers, collate_fn=collate,
                     drop_last=len(tr) > batch, pin_memory=amp)
    vdl = DataLoader(va, batch, shuffle=False, num_workers=workers, collate_fn=collate)

    emit("status", text="Loading the detector ...")
    weights = resolve_weights(job["weights"], job.get("detectors_dir"))
    taps = tuple(int(t) for t in job["taps"])
    model = BehaviorModel(weights, nc, taps=taps).to(dev)
    if not model.backbone.strides_ok():
        raise ValueError(f"Layers {list(taps)} of {os.path.basename(weights)} are not at strides 8 / 16 / 32. "
                         f"Pick the neck layers that match this detector (see Taps).")
    fingerprint = model.backbone.fingerprint()

    counts = head_data.class_counts(train_pairs, nc)
    if not any(counts):
        raise ValueError("The training labels use none of the workspace classes.")
    weight = None
    if job.get("class_weights", True):
        c = torch.tensor(counts, dtype=torch.float)
        present = c > 0                                        # a class with no examples must not soak up weight
        weight = torch.zeros(nc)
        weight[present] = (1 / c[present]).sqrt()
        weight = (weight / weight[present].mean()).to(dev)
    crit = nn.CrossEntropyLoss(weight=weight, label_smoothing=0.1)

    epochs = int(job["epochs"])
    opt = torch.optim.AdamW(model.head.parameters(), lr=float(job["lr"]), weight_decay=0.01)
    total, warm = epochs * len(tdl), min(300, len(tdl))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=amp)
    except AttributeError:                                      # torch older than 2.3
        scaler = torch.cuda.amp.GradScaler(enabled=amp)

    emit("start", train_images=len(train_pairs), val_images=len(val_pairs), classes=classes, counts=counts,
         epochs=epochs, device=dev.type, batch=batch, imgsz=list(imgsz),
         head_params_m=round(sum(p.numel() for p in model.head.parameters()) / 1e6, 2))

    best, best_epoch, stopped, last_metrics = -1.0, 0, False, None
    for ep in range(epochs):
        if should_stop():
            stopped = True
            break
        model.train()                                          # the detector inside stays in eval() by design
        t0, run, n = time.time(), 0.0, 0
        for imgs, rois, labels in tdl:
            if len(rois) == 0:
                continue
            imgs = imgs.to(dev, non_blocking=True).float() / 255
            rois, labels = rois.to(dev), labels.to(dev)
            boxes = jitter(rois[:, 1:], imgs.shape[-2:])
            with torch.autocast(dev.type, enabled=amp):
                logits = model(imgs, rois[:, 0].long(), boxes)
            loss = crit(logits.float(), labels)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            run += loss.item()
            n += 1
        m = evaluate(model, vdl, nc, dev, amp)
        last_metrics = m
        ck = dict(head=model.head.state_dict(), names=classes,
                  cfg=dict(weights=os.path.basename(job["weights"]), taps=list(taps), imgsz=list(imgsz), nc=nc))
        torch.save(ck, os.path.join(out, "head_last.pt"))
        is_best = m["f1"] > best
        if is_best:
            best, best_epoch = m["f1"], ep + 1
            torch.save(ck, os.path.join(out, "head_best.pt"))
        emit("epoch", epoch=ep + 1, epochs=epochs, loss=run / max(n, 1), acc=m["acc"], f1=m["f1"], best=is_best,
             recall=m["recall"], support=m["support"], seconds=time.time() - t0)

    if model.backbone.fingerprint() != fingerprint:
        raise RuntimeError("The detector's weights changed during training - this must never happen.")
    summary = dict(best_f1=max(best, 0.0), best_epoch=best_epoch, epochs_run=ep + (0 if stopped else 1),
                   stopped=stopped, out_dir=out, has_head=os.path.isfile(os.path.join(out, "head_best.pt")),
                   classes=classes, last_recall=(last_metrics or {}).get("recall"),
                   last_support=(last_metrics or {}).get("support"))
    emit("done", **summary)
    return summary


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print("usage: python -m utils.head_train <job.json>")
        return 2
    with open(argv[0], "r", encoding="utf-8") as f:
        job = json.load(f)
    try:
        train(job)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        emit("error", message=str(exc) or exc.__class__.__name__)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
