"""The picture the augmentation previews are made from.

By default it is the first picture of the workspace (the first image of its first dataset folder). The user can pick
another one; the choice is kept in the workspace's config (`preview.image`) and, when that file is gone, the first
picture is used again.
"""
import os

import cv2

from . import workspace_config
from . import workspace_manager as wm
from .augmentation import read_image

PREVIEW_SIDE = 360          # the longest side of the picture the previews are made from


def workspace_images(workspace):
    """The workspace's pictures in the order the exporter and the annotation window use: its dataset folders by
    name, the files of each by name."""
    root = wm.DATASETS_ROOT
    if not os.path.isdir(root):
        return []
    found = []
    for folder in sorted(f for f in os.listdir(root)
                         if f.startswith(workspace + "-") and os.path.isdir(os.path.join(root, f))):
        path = os.path.join(root, folder)
        found += [os.path.join(path, n) for n in sorted(os.listdir(path)) if n.lower().endswith(wm.IMAGE_EXTENSIONS)]
    return found


def resolve(path):
    """A stored path as a real one (relative paths are relative to the Jelibox folder, so Move Jelibox keeps them)."""
    return os.path.normpath(path if os.path.isabs(path) else os.path.join(wm.BASE_DIR, path))


def pick_sample(workspace, images=None):
    """(path, is_default): the picture the user chose when it still exists, otherwise the first picture of the
    workspace; (None, True) when the workspace has no pictures."""
    saved = workspace_config.get_preview_image(workspace)
    if saved:
        path = resolve(saved)
        if os.path.isfile(path) and path.lower().endswith(wm.IMAGE_EXTENSIONS):
            return path, False
    images = workspace_images(workspace) if images is None else images
    return (images[0] if images else None), True


def remember_sample(workspace, path):
    """Keep `path` as the preview picture of the workspace (relative to the Jelibox folder when it is inside it)."""
    try:
        stored = os.path.relpath(path, wm.BASE_DIR)
    except ValueError:                               # another drive
        stored = path
    if stored.startswith(".."):
        stored = path
    workspace_config.set_preview_image(workspace, stored.replace("\\", "/"))


def forget_sample(workspace):
    workspace_config.set_preview_image(workspace, "")


def load_sample(path):
    """The picture as a BGR array with its longest side at most PREVIEW_SIDE, or None when it cannot be read."""
    img = read_image(path) if path else None
    if img is None:
        return None
    h, w = img.shape[:2]
    scale = PREVIEW_SIDE / max(h, w)
    if scale < 1.0:
        img = cv2.resize(img, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
    return img
