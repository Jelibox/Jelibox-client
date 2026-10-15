"""The export's Preprocessing step: Auto-Orient and Resize.

Applied to every exported picture (augmented copies included); the labels follow through the same matrix, so no box
ends up in the wrong place. No Tk here.
"""
from PIL import Image

from .augment_geometry import ChainTransform, ResizeTransform, RESIZE_MODES
from .augmentation import read_image

EXIF_ORIENTATION = 274
RESIZE_LABELS = {"stretch": "Stretch to", "fit": "Fit within", "letterbox": "Fit (black edges) in"}
RESIZE_HINTS = {
    "stretch": "Every picture becomes exactly W x H. The shape is squeezed or stretched to fit.",
    "fit": "The whole picture is kept in its own shape and made as large as possible inside W x H.",
    "letterbox": "Like Fit, but black bars are added so every picture is exactly W x H.",
}
SIZE_RANGE = (32, 8192)


def exif_orientation(path):
    """The EXIF orientation of a picture (1 = upright), 1 when there is none or it cannot be read."""
    try:
        with Image.open(path) as im:
            return int(im.getexif().get(EXIF_ORIENTATION, 1) or 1)
    except Exception:
        return 1


class Preprocess:
    """auto_orient: write the pictures upright and drop the EXIF rotation tag. resize: None or (mode, width, height)."""

    def __init__(self, auto_orient=True, resize=None):
        if resize is not None:
            mode, w, h = resize
            if mode not in RESIZE_MODES:
                raise ValueError(f"Unknown resize mode {mode!r}")
            lo, hi = SIZE_RANGE
            resize = (mode, max(lo, min(hi, int(w))), max(lo, min(hi, int(h))))
        self.auto_orient = bool(auto_orient)
        self.resize = resize

    @property
    def active(self):
        return self.resize is not None or self.auto_orient

    def describe(self):
        out = []
        if self.auto_orient:
            out.append("Auto-Orient")
        if self.resize:
            out.append(f"{RESIZE_LABELS[self.resize[0]]} {self.resize[1]}x{self.resize[2]}")
        return out

    def _resize_for(self, w, h):
        mode, rw, rh = self.resize
        return ResizeTransform(w, h, mode, rw, rh)

    def original(self, src_path):
        """(image, transform) the exporter must write instead of copying the file, or (None, None) when the file can be
        copied as it is. The transform is None when only the pixels change (Auto-Orient)."""
        if self.resize is not None:
            img = read_image(src_path)
            if img is None:
                return None, None
            tf = self._resize_for(img.shape[1], img.shape[0])
            return tf.warp(img), tf
        if self.auto_orient and exif_orientation(src_path) != 1:
            img = read_image(src_path)                    # OpenCV hands the pixels over already upright
            return (img, None) if img is not None else (None, None)
        return None, None

    def post(self, img, tf, src_hw):
        """For Augmenter(post=...): resize an augmented picture and chain its transform with the resize."""
        if self.resize is None:
            return img, tf
        rt = self._resize_for(img.shape[1], img.shape[0])
        out = rt.warp(img)
        if tf is None:
            rt.src_w, rt.src_h = src_hw[1], src_hw[0]            # labels are still in the source picture's pixels
            rt.matrix = rt.matrix @ _scale(src_hw, img.shape[:2])
            return out, rt
        return out, ChainTransform(tf, rt)


def _scale(src_hw, now_hw):
    """Matrix from source-pixel labels to the (equally sized or not) augmented picture: identity for photometric ops."""
    import numpy as np
    sy, sx = now_hw[0] / src_hw[0], now_hw[1] / src_hw[1]
    return np.array([[sx, 0.0, 0.0], [0.0, sy, 0.0], [0.0, 0.0, 1.0]])
