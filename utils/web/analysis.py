"""
Analyze Dataset for the web: the numbers of `utils/dataset_analysis.py` shaped for the page's charts. The page draws
everything; nothing here needs matplotlib or tkinter.

An analysis (reading every XML file) takes a while, so it runs as a job and is kept in memory under a token; changing the
simulated size, the resize mode or the small-object limit only recalculates from it (`payload`).
"""
import secrets
import threading
import time
from collections import Counter, OrderedDict

from .. import dataset_analysis as da

KEEP = 8                            # analyses kept in memory
SIZE_RANGE = (16, 8192)
LIMIT_RANGE = (1, 1024)
SCATTER_POINTS = 3000
MODES = (da.LETTERBOX, da.STRETCH)


class Kept:
    def __init__(self, workspace, user, data):
        self.workspace, self.user, self.data = workspace, user, data
        self.used = time.time()


class Cache:
    """The analyses that are open in a browser: token -> Kept, the oldest ones are dropped."""

    def __init__(self):
        self._items = OrderedDict()
        self._lock = threading.Lock()

    def add(self, workspace, user, data):
        token = secrets.token_hex(8)
        with self._lock:
            self._items[token] = Kept(workspace, user, data)
            while len(self._items) > KEEP:
                self._items.popitem(last=False)
        return token

    def get(self, token):
        with self._lock:
            item = self._items.get(token)
            if item:
                item.used = time.time()
                self._items.move_to_end(token)
            return item

    def replace(self, token, data):
        with self._lock:
            if token in self._items:
                self._items[token].data = data


def check_view(body, default_limit):
    """(size or None, mode, limit) from what the page sent; ValueError with a sentence the page can show."""
    size = body.get("size")
    if size in (None, "", 0):
        size = None
    else:
        try:
            size = int(float(size))
        except (TypeError, ValueError):
            raise ValueError("Enter the input size as a number of pixels, e.g. 640.")
        if not SIZE_RANGE[0] <= size <= SIZE_RANGE[1]:
            raise ValueError(f"The input size must be between {SIZE_RANGE[0]} and {SIZE_RANGE[1]} px.")
    mode = body.get("mode") or da.LETTERBOX
    if mode not in MODES:
        raise ValueError("The resize mode is letterbox or stretch.")
    raw = body.get("limit", default_limit)
    try:
        limit = int(raw)
    except (TypeError, ValueError):
        raise ValueError(f"Enter a whole number of pixels between {LIMIT_RANGE[0]} and {LIMIT_RANGE[1]}.")
    if not LIMIT_RANGE[0] <= limit <= LIMIT_RANGE[1]:
        raise ValueError(f"Enter a whole number of pixels between {LIMIT_RANGE[0]} and {LIMIT_RANGE[1]}.")
    return size, mode, limit


def _pair(p):
    return None if p is None else [round(p[0], 1), round(p[1], 1)]


def _box(values):
    """min / quartiles / max with matplotlib's whiskers (the last value within 1.5 IQR), outliers left out."""
    import numpy as np
    a = np.sort(np.asarray(values, float))
    q1, med, q3 = (float(x) for x in np.percentile(a, [25, 50, 75]))
    iqr = q3 - q1
    inside = a[(a >= q1 - 1.5 * iqr) & (a <= q3 + 1.5 * iqr)]
    return {"low": float(inside.min()), "q1": q1, "median": med, "q3": q3, "high": float(inside.max())}


def _hist(values, bins):
    import numpy as np
    counts, edges = np.histogram(values, bins=bins)
    return [int(c) for c in counts], [round(float(e), 1) for e in edges]


def payload(data, size, mode, limit):
    """Everything the window shows for one view (`size` None = the original image sizes)."""
    import numpy as np
    v = da.view(data, size, mode) if size else da.view(data)
    w, h = v["w"], v["h"]
    out = {
        "workspace": data.workspace, "size": size, "mode": mode, "limit": limit,
        "images": data.n_images, "labelled": data.n_labelled, "unlabelled": data.n_unlabelled,
        "objects": data.n_objects, "classes": len(data.classes), "datasets": len(data.instances),
        "xml": data.n_xml, "unmatched": data.n_unmatched_xml, "no_size": data.n_no_size,
        "smallest_original": _pair(da.smallest(da.view(data))),
        "smallest": _pair(da.smallest(v)),
        "not_simulated": data.n_objects - len(w) if size else 0,
        "median": None,
        "class_counts": [{"name": n, "count": c} for n, c in data.class_counts()],
        "instances": data.instances,
    }
    stats_w, stats_h = da.spread(w), da.spread(h)
    if stats_w:
        out["median"] = [round(stats_w["median"]), round(stats_h["median"])]
    info = da.small_summary(v, limit)
    out["small"] = None if info is None else {
        "count": info["count"], "percent": round(info["percent"], 1),
        "by_class": [{"name": n, "count": c} for n, c in info["by_class"]], "smallest": _pair(info["smallest"])}

    # object size charts
    names = sorted(set(v["cls"].tolist()))
    out["size_classes"] = names
    if len(w):
        hi = float(np.percentile(np.concatenate([w, h]), 99.5)) or 1.0
        bins = np.linspace(0, hi, 41)
        out["hist_w"], out["hist_edges"] = _hist(np.clip(w, 0, hi), bins)
        out["hist_h"], _ = _hist(np.clip(h, 0, hi), bins)
        pick = np.arange(len(w)) if len(w) <= SCATTER_POINTS else \
            np.sort(np.random.default_rng(0).choice(len(w), SCATTER_POINTS, replace=False))
        index = {c: i for i, c in enumerate(names)}
        out["scatter"] = [[round(float(w[i]), 1), round(float(h[i]), 1), index[v["cls"][i]]] for i in pick]
        out["scatter_total"] = int(len(w))
        short = np.minimum(w, h)
        out["boxes"] = [{"name": c, **_box(short[v["cls"] == c])} for c in names]
    else:
        out.update(hist_w=[], hist_h=[], hist_edges=[], scatter=[], scatter_total=0, boxes=[])

    # images
    top = data.image_sizes.most_common(10)
    out["resolutions"] = [{"label": f"{a}x{b}", "count": c} for (a, b), c in top]
    out["resolutions_more"] = sum(data.image_sizes.values()) - sum(c for _, c in top)
    if data.per_image:
        capped = Counter(min(c, 40) for c in data.per_image)
        out["per_image"] = [{"objects": k, "images": capped.get(k, 0)} for k in range(1, min(max(data.per_image), 40) + 1)]
    else:
        out["per_image"] = []

    # the resize table is always calculated from the original sizes, in the chosen mode
    out["table"] = [{"size": r["size"], "objects": r["objects"], "smallest": _pair(r["smallest"]),
                     "median_side": None if r["median_side"] is None else round(r["median_side"], 1),
                     "small": r["small"], "small_percent": round(r["small_percent"], 1)}
                    for r in da.compare(data, mode=mode, limit=limit)]
    return out
