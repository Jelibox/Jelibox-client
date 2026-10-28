"""
One dataset folder as the annotation editor sees it, for the web server. No tkinter, no `utils.config` globals: a
`Workspace` is built per request from the Jelibox folder and an instance name, so any number of workspaces and people
can be served at once. The files are the same ones the desktop window uses:

    datasetsInput/<instance>/       the pictures (the editor lists .jpg .jpeg .png, sorted by name, like the desktop)
    vocdataset/<workspace>/         one Pascal VOC XML per picture (what the editor reads and writes)
    YOLOdataset/<workspace>/labels  one YOLO text per picture (written with every save)

Annotations travel as plain lists in ORIGINAL picture pixels: bboxes [x1, y1, x2, y2, class], polygons
[[[x, y], ...], class]. Saving checks them (finite numbers, known classes, inside the picture) before anything is written.

Two people on the same picture: every annotation read returns a `version` (the XML's date and size); a save that
carries an older one is refused with Conflict unless it is forced, so nobody silently overwrites someone else's work.
"""
import math
import os
import threading

from .. import annotation_io as aio
from .. import workspace_config, workspace_manager as wm
from ..class_manager import ClassManager

IMAGE_TYPES = (".jpg", ".jpeg", ".png")
MAX_OBJECTS = 5000
MAX_POINTS = 5000
MAX_MASKS = 200

_locks = {}
_locks_guard = threading.Lock()
_managers = {}                  # (jelibox folder, workspace) -> (stamp of its class file, ClassManager)


class EditError(ValueError):
    """A message that is safe to show to the person."""


class Conflict(EditError):
    """The picture's annotations changed since they were read."""


def lock_for(workspace):
    with _locks_guard:
        return _locks.setdefault(workspace, threading.RLock())


def _stamp(workspace):
    try:
        return os.stat(workspace_config.config_path(workspace)).st_mtime_ns
    except OSError:
        return None


def class_manager(base_dir, workspace):
    """The ClassManager of a workspace, kept between requests and rebuilt when its class file changed on disk (the
    desktop window or another call edited it)."""
    key = (base_dir, workspace)
    with _locks_guard:
        hit = _managers.get(key)
    if hit and hit[0] is not None and hit[0] == _stamp(workspace):
        return hit[1]
    manager = ClassManager(workspace, base_dir)
    with _locks_guard:
        _managers[key] = (_stamp(workspace), manager)
    return manager


def _remember(base_dir, workspace, manager):
    with _locks_guard:
        _managers[(base_dir, workspace)] = (_stamp(workspace), manager)


def image_size(path):
    """(width, height) of a picture as the editor shows it: EXIF rotation is applied, like cv2.imread and browsers do."""
    try:
        from PIL import Image
        with Image.open(path) as im:
            width, height = im.size
            try:
                orientation = im.getexif().get(274)
            except Exception:
                orientation = None
        if orientation in (5, 6, 7, 8):
            width, height = height, width
        return int(width), int(height)
    except Exception:
        img = _read_bgr(path)
        if img is None:
            raise EditError("This picture cannot be read.")
        return int(img.shape[1]), int(img.shape[0])


def _read_bgr(path):
    import cv2
    import numpy as np
    try:
        data = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
    except OSError:
        return None


def _number(value, what):
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise EditError(f"{what} is not a number.")
    if not math.isfinite(number):
        raise EditError(f"{what} is not a finite number.")
    return number


def _point(point, width, height, what):
    if not isinstance(point, (list, tuple)) or len(point) != 2:
        raise EditError(f"{what} is not a point.")
    x, y = _number(point[0], what), _number(point[1], what)
    return [min(max(x, 0.0), float(width)), min(max(y, 0.0), float(height))]


class Workspace:
    def __init__(self, base_dir, instance):
        name = wm.workspace_name_for(str(instance))
        if str(instance) not in wm.list_workspaces().get(name, []):
            raise EditError("That dataset does not exist.")
        self.base_dir, self.instance, self.name = base_dir, str(instance), name
        self.input = wm.instance_path(self.instance)
        self.voc = os.path.join(base_dir, "vocdataset", name)
        self.labels = os.path.join(base_dir, "YOLOdataset", name, "labels")
        os.makedirs(self.voc, exist_ok=True)
        os.makedirs(self.labels, exist_ok=True)
        self.classes = class_manager(base_dir, name)
        if not self.classes.get_classes():                      # the desktop does the same: a workspace always has one
            self.classes.add_class("Object")
            _remember(base_dir, name, self.classes)
        self.lock = lock_for(name)

    # ---------------------------------------------------------------- pictures
    def images(self):
        return sorted(f for f in os.listdir(self.input) if f.lower().endswith(IMAGE_TYPES))

    def path(self, name):
        """The file of a picture of this dataset. Anything that is not in the folder listing is refused, so no path
        from the browser can point elsewhere."""
        if not name or name != os.path.basename(name) or name not in set(self.images()):
            raise EditError("That picture does not exist.")
        return os.path.join(self.input, name)

    def xml(self, name):
        return os.path.join(self.voc, os.path.splitext(name)[0] + ".xml")

    def annotated(self):
        """Names of the pictures that have been saved (they have an XML, also an empty one)."""
        try:
            done = {os.path.splitext(f)[0] for f in os.listdir(self.voc) if f.lower().endswith(".xml")}
        except OSError:
            return []
        return [n for n in self.images() if os.path.splitext(n)[0] in done]

    def version(self, name):
        try:
            stat = os.stat(self.xml(name))
        except OSError:
            return "0"
        return f"{stat.st_mtime_ns}-{stat.st_size}"

    def open_info(self, progress_cb=None):
        aio.sync_yolo_labels(self.voc, self.labels, self.classes.get_classes(), progress_cb)
        return {"instance": self.instance, "workspace": self.name, "images": self.images(),
                "annotated": self.annotated(), "classes": self.class_info()}

    # ---------------------------------------------------------------- annotations
    def load(self, name):
        path = self.path(name)
        width, height = image_size(path)
        xml = self.xml(name)
        boxes, polygons = [], []
        if os.path.exists(xml):
            try:
                import xml.etree.ElementTree as ET
                boxes, polygons = aio._parse_voc_objects(ET.parse(xml).getroot())
            except Exception:
                raise EditError("The annotation file of this picture is damaged.")
        return {"name": name, "width": width, "height": height, "version": self.version(name),
                "bboxes": [list(b) for b in boxes],
                "polygons": [[[list(p) for p in pts], cls] for pts, cls in polygons]}

    def clean(self, bboxes, polygons, width, height):
        known = set(self.classes.get_classes())
        if not isinstance(bboxes, list) or not isinstance(polygons, list):
            raise EditError("The annotations must be two lists.")
        if len(bboxes) + len(polygons) > MAX_OBJECTS:
            raise EditError(f"A picture can hold at most {MAX_OBJECTS} annotations.")

        def cls_of(value):
            if not isinstance(value, str) or value not in known:
                raise EditError(f"Unknown class: {str(value)[:40]!r}. Reload the page: the classes changed.")
            return value

        out_boxes = []
        for box in bboxes:
            if not isinstance(box, (list, tuple)) or len(box) != 5:
                raise EditError("A box needs x1, y1, x2, y2 and a class.")
            x1, y1 = _point(box[0:2], width, height, "A box corner")
            x2, y2 = _point(box[2:4], width, height, "A box corner")
            x1, x2 = sorted((x1, x2))
            y1, y2 = sorted((y1, y2))
            cls = cls_of(box[4])
            if x2 - x1 >= 1 and y2 - y1 >= 1:                  # nothing left of a box squeezed to a line
                out_boxes.append([x1, y1, x2, y2, cls])
        out_polygons = []
        for polygon in polygons:
            if not isinstance(polygon, (list, tuple)) or len(polygon) != 2 or not isinstance(polygon[0], (list, tuple)):
                raise EditError("A polygon needs its points and a class.")
            if not 3 <= len(polygon[0]) <= MAX_POINTS:
                raise EditError(f"A polygon needs 3 to {MAX_POINTS} points.")
            out_polygons.append([[_point(p, width, height, "A polygon point") for p in polygon[0]], cls_of(polygon[1])])
        return out_boxes, out_polygons

    def clean_masks(self, masks, width, height):
        if not masks:
            return []
        if not isinstance(masks, list) or len(masks) > MAX_MASKS:
            raise EditError("Too many ZeroFill regions.")
        cleaned = []
        for region in masks:
            if not isinstance(region, (list, tuple)) or not 3 <= len(region) <= MAX_POINTS:
                raise EditError("A ZeroFill region needs 3 or more points.")
            cleaned.append([_point(p, width, height, "A ZeroFill point") for p in region])
        return cleaned

    def burn_masks(self, name, regions):
        """Fill the regions with black in the picture FILE (what ZeroFill does when you leave the picture)."""
        import cv2
        import numpy as np
        path = self.path(name)
        image = _read_bgr(path)
        if image is None:
            raise EditError("This picture cannot be read.")
        for region in regions:
            cv2.fillPoly(image, [np.array(region, dtype=np.int32)], (0, 0, 0))
        ext = os.path.splitext(path)[1].lower()
        ok, buf = cv2.imencode(ext, image)
        if not ok:
            raise EditError("The picture could not be written.")
        tmp = path + ".tmp"
        buf.tofile(tmp)
        os.replace(tmp, path)

    def save(self, name, bboxes, polygons, masks=None, version=None, force=False, on_saved=None):
        """Write the picture's VOC XML and YOLO label (and burn ZeroFill regions into the picture). Raises Conflict when
        `version` is not the current one and `force` is off. `on_saved(path, bboxes, polygons)` gets the cleaned lists
        once they are written (SAM 2 Dynamic learns from them)."""
        path = self.path(name)
        with self.lock:
            if version is not None and not force and str(version) != self.version(name):
                raise Conflict("This picture was changed by someone else since you opened it.")
            width, height = image_size(path)
            boxes, polygons = self.clean(bboxes, polygons, width, height)
            regions = self.clean_masks(masks, width, height)
            if regions:
                self.burn_masks(name, regions)
            aio.write_annotations(self.voc, self.labels, name, (height, width, 3), boxes, polygons,
                                  self.classes.get_classes())
            if on_saved:
                on_saved(path, boxes, polygons)
            return {"version": self.version(name), "masks": len(regions), "annotated": len(self.annotated()),
                    "total": len(self.images())}

    def delete_image(self, name):
        path = self.path(name)
        with self.lock:
            for target in (path, self.xml(name), os.path.join(self.labels, os.path.splitext(name)[0] + ".txt")):
                try:
                    os.remove(target)
                except FileNotFoundError:
                    pass

    # ---------------------------------------------------------------- classes
    def class_info(self):
        colors = self.classes.get_colors()
        return [{"name": name, "color": "#%02x%02x%02x" % (colors[i][2], colors[i][1], colors[i][0])}
                for i, name in enumerate(self.classes.get_classes()) if i < len(colors)]

    def add_class(self, name):
        with self.lock:
            ok, message = self.classes.add_class(str(name or "").strip())
            _remember(self.base_dir, self.name, self.classes)
        if not ok:
            raise EditError(message)
        return message

    def delete_class(self, name):
        with self.lock:
            if len(self.classes.get_classes()) <= 1 and name in self.classes.get_classes():
                raise EditError("A workspace needs at least one class.")
            ok, message = self.classes.delete_class(str(name))
            _remember(self.base_dir, self.name, self.classes)
        if not ok:
            raise EditError(message)
        return message
