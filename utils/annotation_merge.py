"""
Pure geometry helpers for merging model predictions into existing annotations.

Kept free of torch / ultralytics / config imports so they are cheap to import
and trivial to unit test.
"""

MERGE_IOU = 0.5


def iou(a, b):
    """IoU of two (x1, y1, x2, y2) rectangles; 0.0 when they don't overlap."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return inter / float(area_a + area_b - inter)


def poly_rect(points):
    """Bounding rectangle (x1, y1, x2, y2) of a polygon's points."""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def merge(predictions, existing_rects, threshold=MERGE_IOU):
    """Keep predictions, highest confidence first, that don't overlap an
    existing annotation (treated as confidence 1.0) or an already-kept
    prediction by `threshold` IoU or more. Each prediction is a dict with at
    least 'rect' and 'conf'. Returns the kept list."""
    kept, kept_rects = [], list(existing_rects)
    for p in sorted(predictions, key=lambda p: p["conf"], reverse=True):
        if any(iou(p["rect"], r) >= threshold for r in kept_rects):
            continue
        kept.append(p)
        kept_rects.append(p["rect"])
    return kept
