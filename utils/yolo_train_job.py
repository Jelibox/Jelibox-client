"""
Full YOLO training (every layer, Ultralytics `YOLO.train`) for the Train window of the Custom Model mode.

It runs as its own process, like utils/head_train.py, so a long run never freezes the annotation window:

    python -m utils.yolo_train_job <job.json>          (run from the Jelibox folder; the GUI does this)

Progress goes to stdout as lines `@@ {json}` (events: status, start, epoch, done, error) that the window turns into
a per-epoch table; anything else printed is plain log text. Each `epoch` event carries what Ultralytics measured
after that epoch: the three training losses, precision, recall, mAP50 and mAP50-95.

The images are read from datasetsInput and copied only into a scratch train/val tree inside the run folder (what
Ultralytics needs: images/ and labels/ side by side), which is removed at the end. The split is by position, like
the head trainer: frames of a video are near-duplicates and a random split would flatter the scores.

Needs torch + ultralytics. Import this module only when a model is actually trained.
"""
import json
import os
import shutil
import sys
import time

from . import head_data

PREFIX = "@@ "
RUN_NAME = "train_run"                  # <out_dir>/train_run/weights/best.pt, as in the plain Train button


def emit(event, **fields):
    print(PREFIX + json.dumps({"event": event, **fields}), flush=True)


def pick_device():
    """0 (the first NVIDIA GPU) when PyTorch can use one, otherwise 'cpu'."""
    try:
        import torch
        return 0 if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def load_yolo(weights):
    from ultralytics import YOLO
    return YOLO(weights)


def prepare_dataset(job):
    """Copy the labelled images + their labels into <scratch>/{train,val}/{images,labels} and write data.yaml.
    Returns (yaml path, scratch folder, train count, val count, boxes per class in the training part)."""
    pairs = head_data.labelled_pairs(job["images_folders"], job["labels_folder"])
    if len(pairs) < head_data.MIN_LABELLED:
        raise ValueError(f"Only {len(pairs)} labelled images; at least {head_data.MIN_LABELLED} are needed.")
    train_pairs, val_pairs = head_data.split_contiguous(pairs, float(job.get("val_fraction", 0.2)))
    classes = [str(c) for c in job["classes"]]
    scratch = job["scratch_dir"]
    shutil.rmtree(scratch, ignore_errors=True)                 # a stopped run leaves its copy behind
    for split, items in (("train", train_pairs), ("val", val_pairs)):
        images = os.path.join(scratch, split, "images")
        labels = os.path.join(scratch, split, "labels")
        os.makedirs(images)
        os.makedirs(labels)
        for image, label in items:
            shutil.copy2(image, os.path.join(images, os.path.basename(image)))
            shutil.copy2(label, os.path.join(labels, os.path.splitext(os.path.basename(image))[0] + ".txt"))
    yaml_path = os.path.join(scratch, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(f"train: {json.dumps(os.path.abspath(os.path.join(scratch, 'train', 'images')))}\n")
        f.write(f"val: {json.dumps(os.path.abspath(os.path.join(scratch, 'val', 'images')))}\n")
        f.write(f"nc: {len(classes)}\n")
        f.write("names:\n" + "".join(f"  {i}: {json.dumps(name)}\n" for i, name in enumerate(classes)))
    return yaml_path, scratch, len(train_pairs), len(val_pairs), head_data.class_counts(train_pairs, len(classes))


def _metric(metrics, *names):
    for name in names:
        if name in metrics and metrics[name] is not None:
            return float(metrics[name])
    return 0.0


def train(job, emit=emit):
    """Run one training job (a dict, see HeadTrainDialog.build_yolo_job). Returns the summary sent with `done`."""
    out = job["out_dir"]
    os.makedirs(out, exist_ok=True)
    classes = [str(c) for c in job["classes"]]
    imgsz = int(job["imgsz"])
    if imgsz < 32 or imgsz % 32:
        raise ValueError(f"Image size {imgsz} must be a multiple of 32.")
    device = job.get("device")
    if device is None:                                          # 0 is a real choice (the first GPU), so not `or`
        device = pick_device()

    emit("status", text="Preparing the training images ...")
    yaml_path, scratch, n_train, n_val, counts = prepare_dataset(job)
    try:
        emit("status", text="Loading the model ...")
        model = load_yolo(job["init_weights"])
        epochs = int(job["epochs"])
        started = {"t": time.time()}
        best = {"fitness": -1.0, "epoch": 0, "last": {}}

        def on_train_start(trainer):
            emit("start", kind="yolo", train_images=n_train, val_images=n_val, classes=classes, counts=counts,
                 epochs=epochs, device="cuda" if device == 0 else str(device), batch=int(job["batch"]), imgsz=[imgsz, imgsz],
                 model=os.path.basename(str(job["init_weights"])), freeze=int(job.get("freeze", 0)))

        def on_epoch_end(trainer):
            metrics = dict(getattr(trainer, "metrics", None) or {})
            try:
                losses = trainer.label_loss_items(trainer.tloss, prefix="train")
            except Exception:
                losses = {}
            fitness = float(getattr(trainer, "fitness", 0.0) or 0.0)
            top = float(getattr(trainer, "best_fitness", fitness) or fitness)
            is_best = fitness >= top
            epoch = int(trainer.epoch) + 1
            if is_best:
                best["fitness"], best["epoch"] = fitness, epoch
            now = time.time()
            row = dict(epoch=epoch, epochs=int(getattr(trainer, "epochs", epochs)),
                       box_loss=_metric(losses, "train/box_loss"), cls_loss=_metric(losses, "train/cls_loss"),
                       dfl_loss=_metric(losses, "train/dfl_loss"),
                       precision=_metric(metrics, "metrics/precision(B)"), recall=_metric(metrics, "metrics/recall(B)"),
                       map50=_metric(metrics, "metrics/mAP50(B)"), map5095=_metric(metrics, "metrics/mAP50-95(B)"),
                       fitness=fitness, best=is_best, seconds=now - started["t"])
            started["t"] = now
            best["last"] = row
            emit("epoch", **row)

        model.add_callback("on_train_start", on_train_start)
        model.add_callback("on_fit_epoch_end", on_epoch_end)
        model.train(data=yaml_path, epochs=epochs, imgsz=imgsz, batch=int(job["batch"]), optimizer="AdamW",
                    lr0=float(job["lr"]), weight_decay=float(job.get("weight_decay", 0.0005)),
                    freeze=int(job.get("freeze") or 0) or None, project=out, name=RUN_NAME,
                    exist_ok=True, device=device, amp=False, workers=int(job.get("workers", 1)), plots=False)
        save_dir = str(getattr(getattr(model, "trainer", None), "save_dir", "") or os.path.join(out, RUN_NAME))
        best_path = os.path.join(save_dir, "weights", "best.pt")
        last = best["last"]
        summary = dict(best_epoch=best["epoch"], best_fitness=max(best["fitness"], 0.0), epochs_run=last.get("epoch", 0),
                       stopped=False, out_dir=out, weights=best_path, has_model=os.path.isfile(best_path),
                       classes=classes, map50=last.get("map50"), map5095=last.get("map5095"))
        emit("done", **summary)
        return summary
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print("usage: python -m utils.yolo_train_job <job.json>")
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
