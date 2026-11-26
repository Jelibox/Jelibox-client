"""
What happened to a full YOLO training run, and the file that lets it go on later.

Every full training writes `<run folder>/run_state.json` (the run folder is models/<workspace>/yolo_run). Its `status` tells
Jelibox whether the run was ended ON PURPOSE or may be continued:

  running    the process is alive. A file that still says `running` while no process is alive means the run was cut off
             (the server stopped, a Kaggle session ended, the power went): it can be continued.
  paused     the person (or the session time limit) asked for a pause: the epoch in progress was finished and saved so the
             run goes on from there later.
  failed     the process crashed (an error, out of memory): it can be continued from the last saved epoch.
  finished   all the epochs ran.
  stopped    ended on purpose with Finish early or Stop now: nothing to continue.

Ultralytics rewrites last.pt every epoch but strips the optimizer from it when a run ends, which makes it useless for
continuing; so after every saved epoch the job also copies last.pt to `<run folder>/resume_point.pt`
(the file Resume reads). The state file is the metadata of the run and lives next to resume_point.pt: it names the validation and
test pictures (`split`, every other picture trains), the number of labelled pictures and annotations (`dataset`, the quick
check that the dataset is still the same one) and the settings. A run that goes on later (maybe on another computer, with
the dataset uploaded again) gets the same validation and test pictures, or is refused when the dataset is not the same.

No torch / tkinter / web in here: the job process, the web server and the tests all use it.
"""
import csv
import json
import os
import time

STATE_FILE = "run_state.json"
RESUME_FILE = "resume_point.pt"
PAUSE_FLAG = "pause.flag"
SCHEMA = 1
STATUSES = ("running", "paused", "failed", "finished", "stopped")
CONTINUABLE = ("running", "paused", "failed")


def state_path(run_dir):
    return os.path.join(run_dir, STATE_FILE)


def resume_path(run_dir):
    return os.path.join(run_dir, RESUME_FILE)


def pause_file_for(run_dir):
    return os.path.join(run_dir, PAUSE_FLAG)


def read(run_dir):
    """The state of the run in `run_dir`, or None (no file, or one that cannot be read)."""
    try:
        with open(state_path(run_dir), "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) and data.get("kind") == "yolo" else None


def write(run_dir, **fields):
    """Merge `fields` into the state file (written whole and swapped in, so a reader never sees half of it)."""
    os.makedirs(run_dir, exist_ok=True)
    data = read(run_dir) or {"schema": SCHEMA, "kind": "yolo"}
    data.update(fields)
    data["updated"] = time.time()
    tmp = state_path(run_dir) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, state_path(run_dir))
    return data


def mark(run_dir, status, reason=""):
    """Set the status of an existing run (nothing happens when there is no state file: it never was a full YOLO run)."""
    if status not in STATUSES:
        raise ValueError(f"Unknown status {status!r}")
    if read(run_dir) is None:
        return None
    return write(run_dir, status=status, reason=reason)


def forget(run_dir):
    """Delete the state and the resume file (a new training replaces the old run). The pause flag is the caller's."""
    for path in (state_path(run_dir), resume_path(run_dir)):
        try:
            os.remove(path)
        except OSError:
            pass


def rows_from_csv(save_dir, upto=None, epochs=None):
    """The epoch table of a run so far, read from Ultralytics' results.csv, as the rows the Train window shows (the same
    fields as the job's `epoch` events). `upto` keeps the epochs up to that number; `epochs` is the run's total."""
    rows = []
    try:
        with open(os.path.join(save_dir, "results.csv"), newline="", encoding="utf-8") as f:
            for raw in csv.DictReader(f):
                row = {str(k).strip(): v for k, v in raw.items()}

                def num(*names):
                    for name in names:
                        try:
                            return float(row[name])
                        except (KeyError, TypeError, ValueError):
                            continue
                    return 0.0
                epoch = int(num("epoch"))
                if upto is not None and epoch > upto:
                    continue
                m50, m5095 = num("metrics/mAP50(B)"), num("metrics/mAP50-95(B)")
                rows.append({"epoch": epoch, "epochs": epochs or epoch, "box_loss": num("train/box_loss"), "cls_loss": num("train/cls_loss"),
                             "dfl_loss": num("train/dfl_loss"), "precision": num("metrics/precision(B)"),
                             "recall": num("metrics/recall(B)"), "map50": m50, "map5095": m5095,
                             "fitness": 0.1 * m50 + 0.9 * m5095, "seconds": 0.0, "best": False})
    except OSError:
        return []
    top = -1.0
    for row in rows:
        if row["fitness"] >= top:
            top, row["best"] = row["fitness"], True
    return rows


def describe(run_dir, alive=False):
    """What the Train window offers for this workspace: None when nothing can be continued, else a dict with the
    state's facts and a sentence for the person. `alive` = a training process is running now (then nothing is offered)."""
    state = read(run_dir)
    if not state or alive or not os.path.isfile(resume_path(run_dir)):
        return None
    status = state.get("status")
    epoch, epochs = int(state.get("epoch") or 0), int(state.get("epochs") or 0)
    if status not in CONTINUABLE or not 0 < epoch < epochs:
        return None
    how = {"paused": "paused on purpose",
           "failed": "ended by an error",
           "running": "cut off (the server stopped, the session ended or the power went)"}[status]
    return {"status": "interrupted" if status == "running" else status, "epoch": epoch, "epochs": epochs,
            "classes": list(state.get("classes") or []), "imgsz": state.get("imgsz"), "batch": state.get("batch"),
            "text": f"A training of this workspace was {how} after epoch {epoch} of {epochs}. It can go on from there.",
            "updated": state.get("updated")}

