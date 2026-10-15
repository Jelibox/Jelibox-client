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

"Finish early": the window creates the file named by `job["stop_file"]`. The next batch sees it and tells Ultralytics
to stop: it leaves the loop, validates, saves, and then runs its usual final evaluation of best.pt, so the run ends
properly instead of being killed.

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


def prepare_dataset(job, progress=None):
    """The legacy 5-tuple of `prepare` (see there)."""
    r = prepare(job, progress)
    return r["yaml"], r["scratch"], r["n_train"], r["n_val"], r["counts"]


def prepare(job, progress=None):
    """Copy the labelled images + their labels into <scratch>/{train,val}/{images,labels} and write data.yaml.
    With job["augment"] (an AugmentConfig as a dict) the training part also gets its augmented copies, made here with
    the same engine as Export Dataset; the validation and test parts only when the config's `splits` names them.
    job["test_fraction"] (default 0) holds a test part back that training never sees; job["shuffle"] mixes the
    pictures before the split (by default the split is by position). Returns a dict: yaml, scratch, n_train (augmented
    copies included), n_val, n_test, counts (boxes per class in the training part). `progress(done, total, text)` is called after every picture."""
    say = progress or (lambda done, total, text: None)
    say(0, 0, "Step 1/3  Reading the labels ...")
    pairs = head_data.labelled_pairs(job["images_folders"], job["labels_folder"])
    if len(pairs) < head_data.MIN_LABELLED:
        raise ValueError(f"Only {len(pairs)} labelled images; at least {head_data.MIN_LABELLED} are needed.")
    train_pairs, val_pairs, test_pairs = head_data.split_three(
        pairs, float(job.get("val_fraction", 0.2)), float(job.get("test_fraction", 0.0)), bool(job.get("shuffle")),
        job.get("split_seed"))
    classes = [str(c) for c in job["classes"]]
    scratch = job["scratch_dir"]
    shutil.rmtree(scratch, ignore_errors=True)                 # a stopped run leaves its copy behind
    augmenter = _augmenter(job)
    written = {"train": 0, "val": 0, "test": 0}
    total, done = len(pairs), 0
    how = "copying and making augmented copies" if augmenter is not None else "copying"
    held = f", {len(test_pairs)} to test" if test_pairs else ""
    say(0, total, f"Step 2/3  Splitting: {len(train_pairs)} to train, {len(val_pairs)} to validate{held}")
    parts = [("train", train_pairs), ("val", val_pairs)] + ([("test", test_pairs)] if test_pairs else [])
    for split, items in parts:
        images = os.path.join(scratch, split, "images")
        labels = os.path.join(scratch, split, "labels")
        os.makedirs(images)
        os.makedirs(labels)
        for image, label in items:
            name = os.path.basename(image)
            stem = os.path.splitext(name)[0]
            if augmenter is None or augmenter.keeps_original(split):
                shutil.copy2(image, os.path.join(images, name))
                shutil.copy2(label, os.path.join(labels, stem + ".txt"))
                written[split] += 1
            if augmenter is not None:
                written[split] += _write_copies(augmenter, image, label, images, labels, split)
            done += 1
            if done == total or done % 5 == 0:
                part = {"train": "training", "val": "validation", "test": "test"}[split]
                say(done, total, f"Step 3/3  Preparing the pictures, {how}:  {done} / {total}  ({part} part)  -  "
                                 f"{written['train'] + written['val']} files written")
    yaml_path = os.path.join(scratch, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(f"train: {json.dumps(os.path.abspath(os.path.join(scratch, 'train', 'images')))}\n")
        f.write(f"val: {json.dumps(os.path.abspath(os.path.join(scratch, 'val', 'images')))}\n")
        if test_pairs:
            f.write(f"test: {json.dumps(os.path.abspath(os.path.join(scratch, 'test', 'images')))}" + chr(10))
        f.write(f"nc: {len(classes)}\n")
        f.write("names:\n" + "".join(f"  {i}: {json.dumps(name)}\n" for i, name in enumerate(classes)))
    return {"yaml": yaml_path, "scratch": scratch, "n_train": written["train"], "n_val": written["val"],
            "n_test": written["test"], "counts": head_data.class_counts(train_pairs, len(classes))}


def _augmenter(job):
    """The Augmenter for job["augment"], or None when the job has none (or it makes no copies)."""
    spec = job.get("augment")
    if not spec:
        return None
    from .augmentation import Augmenter, config_from_dict
    config = config_from_dict(spec)
    config.splits = tuple(dict.fromkeys(("train",) + tuple(config.splits)))        # training is always augmented
    return Augmenter(config) if config.enabled else None


def _write_copies(augmenter, image, label, images_dir, labels_dir, split="train"):
    """Write the augmented copies of one training picture (labels moved with flips / turns); returns how many."""
    from .augment_geometry import yolo_label_text
    from .augmentation import write_image
    stem, ext = os.path.splitext(os.path.basename(image))
    with open(label, encoding="utf-8") as f:
        text = f.read()
    made = 0
    for suffix, picture, tf in augmenter.copies(image, split):
        try:
            write_image(os.path.join(images_dir, stem + suffix + ext), picture)
            with open(os.path.join(labels_dir, stem + suffix + ".txt"), "w", encoding="utf-8") as f:
                f.write(text if tf is None else yolo_label_text(text, tf))
            made += 1
        except (OSError, ValueError) as exc:
            print(f"Could not write the augmented copy of {stem}: {exc}", flush=True)
    return made


def _evaluate_test(job, yaml_path, best_path, n_test, device, emit):
    """Score the best model on the held-back test pictures; {} (and a log line) when that cannot be done."""
    if not os.path.isfile(best_path):
        return {}
    emit("status", text=f"Evaluating the best model on the {n_test} test pictures it has never seen ...")
    try:
        result = load_yolo(best_path).val(data=yaml_path, split="test", imgsz=int(job["imgsz"]), device=device,
                                          batch=int(job["batch"]), plots=False, workers=int(job.get("workers", 1)))
        box = result.box
        return {"test_images": n_test, "test_map50": float(box.map50), "test_map5095": float(box.map),
                "test_precision": float(box.mp), "test_recall": float(box.mr)}
    except Exception as exc:
        print(f"The test evaluation failed: {exc}", flush=True)
        return {}


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
    prepared = prepare(job, lambda done, total, text: emit("prepare", done=done, total=total, text=text))
    yaml_path, scratch, counts = prepared["yaml"], prepared["scratch"], prepared["counts"]
    n_train, n_val, n_test = prepared["n_train"], prepared["n_val"], prepared["n_test"]
    try:
        emit("status", text="Loading the model ...")
        model = load_yolo(job["init_weights"])
        epochs = int(job["epochs"])
        started = {"t": time.time()}
        best = {"fitness": -1.0, "epoch": 0, "last": {}}

        def on_train_start(trainer):
            emit("start", kind="yolo", train_images=n_train, val_images=n_val, test_images=n_test, classes=classes, counts=counts,
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

        stop_file = job.get("stop_file")
        asked = {"early": False}

        def finish_early(trainer):
            """The user pressed Finish early: stop after this batch, validate, save, final evaluation."""
            if not asked["early"] and stop_file and os.path.exists(stop_file):
                asked["early"] = True
                trainer.stop = True
                emit("status", text="Finishing early: validating this epoch, then the final evaluation ...")

        model.add_callback("on_train_start", on_train_start)
        model.add_callback("on_train_batch_end", finish_early)
        model.add_callback("on_fit_epoch_end", on_epoch_end)
        model.add_callback("on_fit_epoch_end", finish_early)
        emit("status", text="Starting Ultralytics: it checks every picture and label first, which can take a minute "
                            "before the first epoch shows up ...")
        model.train(data=yaml_path, epochs=epochs, imgsz=imgsz, batch=int(job["batch"]), optimizer="AdamW",
                    lr0=float(job["lr"]), weight_decay=float(job.get("weight_decay", 0.0005)),
                    freeze=int(job.get("freeze") or 0) or None, project=out, name=RUN_NAME,
                    exist_ok=True, device=device, amp=False, workers=int(job.get("workers", 1)), plots=False)
        save_dir = str(getattr(getattr(model, "trainer", None), "save_dir", "") or os.path.join(out, RUN_NAME))
        best_path = os.path.join(save_dir, "weights", "best.pt")
        last = best["last"]
        test = _evaluate_test(job, yaml_path, best_path, n_test, device, emit) if n_test else {}
        summary = dict(**test, best_epoch=best["epoch"], best_fitness=max(best["fitness"], 0.0), epochs_run=last.get("epoch", 0),
                       stopped=asked["early"], out_dir=out, weights=best_path, has_model=os.path.isfile(best_path),
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
