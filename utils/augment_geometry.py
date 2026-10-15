"""Flip / rotate / shear for augmentation and Resize for the export's Preprocessing: one affine matrix moves the
pixels *and* the annotations, so image and labels can't drift apart.

Coordinates are image-edge coordinates (0..W, 0..H), the same convention the
annotations use (a box covering the whole image is 0,0,W,H). Nothing is
scaled or cropped: a rotation grows the canvas just enough to keep every pixel
of the original, and the new corners are filled with black.

A bounding box that gets rotated can only be stored as an upright box, so it
becomes the tightest upright box around the rotated one (looser than the
object, an unavoidable limit of box labels). Polygons - and YOLO segmentation
labels, where boxes are already stored as 4-point polygons - rotate exactly.
"""
import copy
import math
import xml.etree.ElementTree as ET

import cv2
import numpy as np


class _Affine:
    """What the label writers need from a transform: its size before and after, and a way to move points / boxes.
    Subclasses set src_w / src_h / dst_w / dst_h and `matrix` (3x3, image-edge coordinates)."""

    def points(self, pts):
        """[(x, y), ...] in source coordinates -> destination coordinates."""
        if not len(pts):
            return []
        arr = np.array([[x, y, 1.0] for x, y in pts], dtype=float)
        out = arr @ self.matrix.T
        xs = np.clip(out[:, 0], 0, self.dst_w)
        ys = np.clip(out[:, 1], 0, self.dst_h)
        return list(zip(xs.tolist(), ys.tolist()))

    def box(self, x1, y1, x2, y2):
        """Upright box in source coordinates -> tightest upright box around its image."""
        moved = self.points([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
        xs = [p[0] for p in moved]
        ys = [p[1] for p in moved]
        return min(xs), min(ys), max(xs), max(ys)


class GeoTransform(_Affine):
    """Flip, then rotate (counter-clockwise on screen), then shear about the centre. The canvas grows to keep every
    pixel of the original (new corners are black); a shear and a 90 degree turn are just special cases."""

    def __init__(self, src_w, src_h, flip_h=False, flip_v=False, angle=0.0, shear_x=0.0, shear_y=0.0):
        self.src_w, self.src_h = int(src_w), int(src_h)
        self.flip_h, self.flip_v, self.angle = bool(flip_h), bool(flip_v), float(angle)
        self.shear_x, self.shear_y = float(shear_x), float(shear_y)

        flip = np.eye(3)
        if self.flip_h:
            flip[0] = [-1.0, 0.0, self.src_w]
        if self.flip_v:
            flip[1] = [0.0, -1.0, self.src_h]

        a = math.radians(self.angle)
        c, s = math.cos(a), math.sin(a)
        cx, cy = self.src_w / 2.0, self.src_h / 2.0
        rot = np.array([[c, s, cx - c * cx - s * cy],
                        [-s, c, cy + s * cx - c * cy],
                        [0.0, 0.0, 1.0]])
        shear = np.eye(3)
        if self.is_sheared:
            kx, ky = math.tan(math.radians(self.shear_x)), math.tan(math.radians(self.shear_y))
            shear = np.array([[1.0, kx, -kx * cy], [ky, 1.0, -ky * cx], [0.0, 0.0, 1.0]])

        before = shear @ rot @ flip
        corners = before @ np.array([[0, self.src_w, self.src_w, 0], [0, 0, self.src_h, self.src_h], [1, 1, 1, 1]], float)
        min_x, max_x = corners[0].min(), corners[0].max()
        min_y, max_y = corners[1].min(), corners[1].max()
        # 1e-6: cos(90deg) is 6e-17, which must not cost an extra pixel row
        self.dst_w = max(1, math.ceil(max_x - min_x - 1e-6))
        self.dst_h = max(1, math.ceil(max_y - min_y - 1e-6))
        move = np.eye(3)
        move[0, 2], move[1, 2] = -min_x, -min_y
        self.matrix = move @ before

    @property
    def is_rotated(self):
        return abs(self.angle) > 1e-9

    @property
    def is_sheared(self):
        return abs(self.shear_x) > 1e-9 or abs(self.shear_y) > 1e-9

    # ------------------------------------------------------------ pixels
    def warp(self, img):
        if not self.is_rotated and not self.is_sheared:
            if self.flip_h and self.flip_v:
                return cv2.flip(img, -1)
            if self.flip_h:
                return cv2.flip(img, 1)
            if self.flip_v:
                return cv2.flip(img, 0)
            return img
        # cv2 puts pixel centres on integers, annotations put pixel edges there
        half_in, half_out = np.eye(3), np.eye(3)
        half_in[0, 2] = half_in[1, 2] = 0.5
        half_out[0, 2] = half_out[1, 2] = -0.5
        m = half_out @ self.matrix @ half_in
        return cv2.warpAffine(img, m[:2], (self.dst_w, self.dst_h), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))


RESIZE_MODES = {
    "stretch": "Stretch to",                       # exactly W x H, the picture is squeezed or stretched
    "fit": "Fit within",                           # keeps the shape, the longer side reaches W or H
    "letterbox": "Fit (black edges) in",           # keeps the shape inside exactly W x H, black bars fill the rest
}


class ResizeTransform(_Affine):
    """The Resize of the export's Preprocessing step: the same matrix moves pixels and labels."""

    def __init__(self, src_w, src_h, mode, width, height):
        if mode not in RESIZE_MODES:
            raise ValueError(f"Unknown resize mode {mode!r}")
        self.src_w, self.src_h, self.mode = int(src_w), int(src_h), mode
        width, height = max(1, int(width)), max(1, int(height))
        if mode == "stretch":
            self.new_w, self.new_h = width, height
            self.dst_w, self.dst_h = width, height
            self.pad_x = self.pad_y = 0
        else:
            scale = min(width / self.src_w, height / self.src_h)
            self.new_w, self.new_h = max(1, round(self.src_w * scale)), max(1, round(self.src_h * scale))
            if mode == "fit":
                self.dst_w, self.dst_h = self.new_w, self.new_h
                self.pad_x = self.pad_y = 0
            else:
                self.dst_w, self.dst_h = width, height
                self.pad_x, self.pad_y = (width - self.new_w) // 2, (height - self.new_h) // 2
        self.matrix = np.array([[self.new_w / self.src_w, 0.0, float(self.pad_x)],
                                [0.0, self.new_h / self.src_h, float(self.pad_y)],
                                [0.0, 0.0, 1.0]])

    def warp(self, img):
        shrinking = self.new_w * self.new_h < self.src_w * self.src_h
        out = cv2.resize(img, (self.new_w, self.new_h),
                         interpolation=cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR)
        if self.pad_x or self.pad_y or (self.dst_w, self.dst_h) != (self.new_w, self.new_h):
            out = cv2.copyMakeBorder(out, self.pad_y, self.dst_h - self.new_h - self.pad_y, self.pad_x,
                                     self.dst_w - self.new_w - self.pad_x, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        return out


class ChainTransform(_Affine):
    """`first`, then `second` (an augmentation, then the export's Resize): one matrix for the labels, both steps
    for the pixels."""

    def __init__(self, first, second):
        self.first, self.second = first, second
        self.src_w, self.src_h = first.src_w, first.src_h
        self.dst_w, self.dst_h = second.dst_w, second.dst_h
        self.matrix = second.matrix @ first.matrix

    def warp(self, img):
        return self.second.warp(self.first.warp(img))


# ----------------------------------------------------------------------
#  YOLO
# ----------------------------------------------------------------------
def _unit(v):
    return max(0.0, min(1.0, v))


def yolo_label_text(text, tf):
    """Transform the contents of a YOLO label file (detection and/or segmentation lines)."""
    out = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls, vals = parts[0], [float(v) for v in parts[1:]]
        if len(vals) == 4:                                    # class cx cy w h
            cx, cy, w, h = vals
            x1, y1 = (cx - w / 2) * tf.src_w, (cy - h / 2) * tf.src_h
            x2, y2 = (cx + w / 2) * tf.src_w, (cy + h / 2) * tf.src_h
            nx1, ny1, nx2, ny2 = tf.box(x1, y1, x2, y2)
            out.append(f"{cls} {_unit((nx1 + nx2) / 2 / tf.dst_w):.6f} "
                       f"{_unit((ny1 + ny2) / 2 / tf.dst_h):.6f} "
                       f"{_unit((nx2 - nx1) / tf.dst_w):.6f} {_unit((ny2 - ny1) / tf.dst_h):.6f}")
        else:                                                 # class x1 y1 x2 y2 ...
            pts = [(vals[i] * tf.src_w, vals[i + 1] * tf.src_h) for i in range(0, len(vals) - 1, 2)]
            moved = tf.points(pts)
            out.append(cls + " " + " ".join(
                f"{_unit(x / tf.dst_w):.6f} {_unit(y / tf.dst_h):.6f}" for x, y in moved))
    return "\n".join(out)


# ----------------------------------------------------------------------
#  Pascal VOC
# ----------------------------------------------------------------------
def _rounded(v):
    return str(max(0, int(round(v))))


def voc_xml_write(src_xml, dest_xml, new_image_name, tf=None):
    """Copy a VOC annotation for an augmented image: new file name, and (when the
    geometry changed) new image size and transformed boxes/polygons."""
    tree = ET.parse(src_xml)
    root = tree.getroot()
    for tag in ("filename", "path"):
        elem = root.find(tag)
        if elem is not None:
            elem.text = new_image_name

    if tf is not None:
        size = root.find("size")
        if size is not None:
            if size.find("width") is not None:
                size.find("width").text = str(tf.dst_w)
            if size.find("height") is not None:
                size.find("height").text = str(tf.dst_h)

        for obj in root.findall("object"):
            poly = obj.find("polygon")
            if poly is not None and _move_voc_polygon(poly, tf):
                continue
            bnd = obj.find("bndbox")
            if bnd is not None:
                x1, y1, x2, y2 = (float(bnd.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
                nx1, ny1, nx2, ny2 = tf.box(x1, y1, x2, y2)
                for key, val in (("xmin", nx1), ("ymin", ny1), ("xmax", nx2), ("ymax", ny2)):
                    bnd.find(key).text = _rounded(val)
    tree.write(dest_xml, encoding="utf-8")


def _move_voc_polygon(poly, tf):
    """Rewrite a polygon in place (either <point><x/><y/></point> or x1/y1/x2/y2... style)."""
    point_elems = poly.findall("point")
    if point_elems:
        moved = tf.points([(float(p.findtext("x")), float(p.findtext("y"))) for p in point_elems])
        for elem, (x, y) in zip(point_elems, moved):
            elem.find("x").text = _rounded(x)
            elem.find("y").text = _rounded(y)
        return True
    count = 0
    while poly.find(f"x{count + 1}") is not None and poly.find(f"y{count + 1}") is not None:
        count += 1
    if count == 0:
        return False
    moved = tf.points([(float(poly.find(f"x{i}").text), float(poly.find(f"y{i}").text))
                       for i in range(1, count + 1)])
    for i, (x, y) in enumerate(moved, start=1):
        poly.find(f"x{i}").text = _rounded(x)
        poly.find(f"y{i}").text = _rounded(y)
    return True


# ----------------------------------------------------------------------
#  COCO
# ----------------------------------------------------------------------
def _shoelace(points):
    n = len(points)
    return 0.5 * abs(sum(points[i][0] * points[(i + 1) % n][1] - points[(i + 1) % n][0] * points[i][1]
                         for i in range(n)))


def coco_annotation(ann, tf):
    """Transform one COCO annotation dict (bbox, area, segmentation) into the new image."""
    new = copy.deepcopy(ann)
    rings = ann.get("segmentation") or []
    if rings:
        moved_rings = []
        for ring in rings:
            pts = [(ring[i], ring[i + 1]) for i in range(0, len(ring) - 1, 2)]
            moved_rings.append(tf.points(pts))
        new["segmentation"] = [[c for p in ring for c in p] for ring in moved_rings]
        xs = [p[0] for ring in moved_rings for p in ring]
        ys = [p[1] for ring in moved_rings for p in ring]
        new["bbox"] = [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]
        new["area"] = sum(_shoelace(r) for r in moved_rings)
    else:
        x, y, w, h = ann["bbox"]
        nx1, ny1, nx2, ny2 = tf.box(x, y, x + w, y + h)
        new["bbox"] = [nx1, ny1, nx2 - nx1, ny2 - ny1]
        new["area"] = (nx2 - nx1) * (ny2 - ny1)
    return new
