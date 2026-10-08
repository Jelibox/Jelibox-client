"""
Class names of the stock COCO detectors and helpers to turn "person, bottle" into class ids (and back).
Pure Python on purpose: the settings window uses it without importing torch.
"""

COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush",
]


def parse_class_ids(text, names=COCO_NAMES):
    """'person, 39, Bottle' -> [0, 39]  (names are looked up in `names`, case-insensitive; numbers are ids).
    Raises ValueError with a readable message for anything it cannot place."""
    lookup = {n.lower(): i for i, n in enumerate(names)}
    ids = []
    for part in [p.strip() for p in (text or "").replace(";", ",").split(",") if p.strip()]:
        if part.isdigit():
            ids.append(int(part))
        elif part.lower() in lookup:
            ids.append(lookup[part.lower()])
        else:
            raise ValueError(f"'{part}' is not a class name or number of the detector.")
    if not ids:
        raise ValueError("Enter at least one detector class, for example: person")
    return sorted(set(ids))


def format_class_ids(ids, names=COCO_NAMES):
    return ", ".join(names[i] if 0 <= i < len(names) else str(i) for i in ids)
