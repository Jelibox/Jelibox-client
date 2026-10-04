"""
Workspace discovery for Jelibox.

A "workspace" is a group of dataset folders inside datasetsInput that share
a common name prefix - e.g. weapon-1 and weapon-2 both belong to the
"weapon" workspace. This mirrors the workspace-name derivation used in
utils/config.py (load_workspace) so a workspace's class config
(configs/<name>.json) stays shared across all of its instances.

No tkinter or config imports here on purpose: this module is used by the
workspace picker screen, which must be safe to import before any dataset
instance folder has been chosen.
"""

import os
import shutil

from . import workspace_config

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASETS_ROOT = os.path.join(BASE_DIR, "datasetsInput")
CONFIGS_ROOT = os.path.join(BASE_DIR, "configs")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")


def workspace_name_for(folder_name):
    """Derive the workspace group name for a dataset instance folder name."""
    if "-" in folder_name:
        base, suffix = folder_name.rsplit("-", 1)
        if suffix.isdigit():
            return base
    return folder_name


def _instance_sort_key(name):
    if "-" in name:
        base, suffix = name.rsplit("-", 1)
        if suffix.isdigit():
            return (base, int(suffix))
    return (name, -1)


def list_workspaces():
    """
    Scan datasetsInput and group instance folders by workspace name.

    Returns a dict, ordered by workspace name, of:
        {workspace_name: [instance_folder_name, ...]}
    with instances sorted naturally (weapon-1, weapon-2, ..., weapon-10).
    """
    os.makedirs(DATASETS_ROOT, exist_ok=True)

    groups = {}
    for entry in sorted(os.listdir(DATASETS_ROOT)):
        if entry.startswith("."):
            continue
        full_path = os.path.join(DATASETS_ROOT, entry)
        if not os.path.isdir(full_path):
            continue
        group = workspace_name_for(entry)
        groups.setdefault(group, []).append(entry)

    return {
        group: sorted(instances, key=_instance_sort_key)
        for group, instances in sorted(groups.items())
    }


def instance_path(instance_name):
    """Absolute path to a dataset instance folder given its folder name."""
    return os.path.join(DATASETS_ROOT, instance_name)


def count_images(instance_name):
    """Quick image count for a dataset instance, used as a hint in the UI."""
    return count_images_in_folder(instance_path(instance_name))


def count_images_in_folder(folder):
    """Count image files directly inside an arbitrary folder (not necessarily
    a datasetsInput instance) - used to validate a folder before importing it."""
    try:
        return sum(
            1 for f in os.listdir(folder)
            if f.lower().endswith(IMAGE_EXTENSIONS)
        )
    except OSError:
        return 0


def sanitize_workspace_name(name):
    """Normalize a user-typed workspace name: trim, and reject anything that
    would break the '<name>-<index>' folder convention or the filesystem."""
    name = (name or "").strip()
    if not name:
        raise ValueError("Enter a workspace name.")
    if not all(c.isalnum() or c in "_-" for c in name):
        raise ValueError("Workspace name can only contain letters, numbers, underscore (_), and dash (-).")
    if name[-1].isdigit() and "-" in name and name.rsplit("-", 1)[1].isdigit():
        raise ValueError("Workspace name can't end in \"-<number>\" - that suffix is added automatically per instance.")
    return name


def parse_classes_input(text):
    """
    Parse a '{cat, dog, rabbit}' style string into a validated list of class
    names. Raises ValueError with a user-facing message on invalid input.
    """
    text = (text or "").strip()
    if not (text.startswith("{") and text.endswith("}")):
        raise ValueError("Classes must be wrapped in curly braces, e.g. {cat, dog, rabbit}")

    inner = text[1:-1].strip()
    if not inner:
        raise ValueError("Enter at least one class name, e.g. {cat, dog, rabbit}")

    names = []
    seen = set()
    for raw in inner.split(","):
        name = raw.strip()
        if not name:
            raise ValueError("Found an empty class name - check for stray commas.")
        if len(name) > 16:
            raise ValueError(f"Class name '{name}' is too long (max 16 characters).")
        if not name.replace("_", "").replace("-", "").isalnum():
            raise ValueError(f"Class name '{name}' can only contain letters, numbers, underscore (_), and dash (-).")
        if name in seen:
            raise ValueError(f"Class '{name}' is listed more than once.")
        seen.add(name)
        names.append(name)

    return names


def existing_classes_for_workspace(workspace_name):
    """Class list from configs/<workspace_name>.json if it exists, else None."""
    return workspace_config.get_classes(workspace_name)


def next_instance_name(workspace_name):
    """The next free '<workspace_name>-<N>' instance folder name."""
    max_idx = 0
    for inst in list_workspaces().get(workspace_name, []):
        if "-" in inst:
            base, suffix = inst.rsplit("-", 1)
            if suffix.isdigit():
                max_idx = max(max_idx, int(suffix))
    return f"{workspace_name}-{max_idx + 1}"


def workspace_stats(workspace_name):
    """Summary used for delete confirmations: instances, image count, and
    how much shared annotation data (VOC XML / YOLO labels) exists."""
    instances = list_workspaces().get(workspace_name, [])
    image_count = sum(count_images(inst) for inst in instances)

    voc_dir = os.path.join(BASE_DIR, "vocdataset", workspace_name)
    xml_count = (len([f for f in os.listdir(voc_dir) if f.lower().endswith('.xml')])
                 if os.path.isdir(voc_dir) else 0)

    yolo_dir = os.path.join(BASE_DIR, "YOLOdataset", workspace_name, "labels")
    label_count = (len([f for f in os.listdir(yolo_dir) if f.lower().endswith('.txt')])
                   if os.path.isdir(yolo_dir) else 0)

    return {
        'instances': instances,
        'image_count': image_count,
        'xml_count': xml_count,
        'label_count': label_count,
        'has_config': existing_classes_for_workspace(workspace_name) is not None,
    }


def _image_basenames(folder):
    try:
        return {
            os.path.splitext(f)[0]
            for f in os.listdir(folder)
            if f.lower().endswith(IMAGE_EXTENSIONS)
        }
    except OSError:
        return set()


def delete_instance(instance_name, progress_cb=None):
    """
    Permanently delete a single datasetsInput instance folder, plus the VOC
    XML / YOLO label files belonging *only* to images in that instance (any
    basename still present in another instance of the same workspace is left
    alone, in case two instances happen to share a filename).

    progress_cb(done, total, label), if given, is called after each removed
    file/folder.
    """
    folder = instance_path(instance_name)
    if not os.path.isdir(folder):
        raise ValueError(f"'{instance_name}' doesn't exist.")

    workspace_name = workspace_name_for(instance_name)
    own_basenames = _image_basenames(folder)

    other_basenames = set()
    for sibling in list_workspaces().get(workspace_name, []):
        if sibling != instance_name:
            other_basenames |= _image_basenames(instance_path(sibling))
    orphaned = own_basenames - other_basenames

    voc_dir = os.path.join(BASE_DIR, "vocdataset", workspace_name)
    yolo_dir = os.path.join(BASE_DIR, "YOLOdataset", workspace_name, "labels")

    total = 1 + len(orphaned) * 2
    done = 0

    shutil.rmtree(folder)
    done += 1
    if progress_cb:
        progress_cb(done, total, f"Removed {instance_name}")

    for base in sorted(orphaned):
        xml_path = os.path.join(voc_dir, base + ".xml")
        if os.path.exists(xml_path):
            os.remove(xml_path)
        done += 1
        if progress_cb:
            progress_cb(done, total, "Removing annotations...")

        label_path = os.path.join(yolo_dir, base + ".txt")
        if os.path.exists(label_path):
            os.remove(label_path)
        done += 1
        if progress_cb:
            progress_cb(done, total, "Removing YOLO labels...")


def delete_workspace(workspace_name, progress_cb=None):
    """
    Permanently delete an entire workspace: every datasetsInput instance,
    all of its VOC XML annotations, all of its YOLO labels, and its class
    config. Trained models and exported datasets are left untouched.

    progress_cb(done, total, label), if given, is called after each step.
    """
    instances = list_workspaces().get(workspace_name, [])
    total = len(instances) + 3
    done = 0

    for inst in instances:
        folder = instance_path(inst)
        if os.path.isdir(folder):
            shutil.rmtree(folder)
        done += 1
        if progress_cb:
            progress_cb(done, total, f"Removed {inst}")

    voc_dir = os.path.join(BASE_DIR, "vocdataset", workspace_name)
    if os.path.isdir(voc_dir):
        shutil.rmtree(voc_dir)
    done += 1
    if progress_cb:
        progress_cb(done, total, "Removed VOC annotations")

    yolo_dir = os.path.join(BASE_DIR, "YOLOdataset", workspace_name)
    if os.path.isdir(yolo_dir):
        shutil.rmtree(yolo_dir)
    done += 1
    if progress_cb:
        progress_cb(done, total, "Removed YOLO labels")

    workspace_config.delete(workspace_name)
    done += 1
    if progress_cb:
        progress_cb(done, total, "Removed class config")


def create_workspace_instance(workspace_name, source_folder, classes=None, progress_cb=None):
    """
    Import an external folder of images as a new datasetsInput/<workspace>-<N>
    instance (images are copied in, one file at a time).

    If the workspace doesn't have a class config yet (configs/<workspace>.json),
    `classes` is written as its class list - a workspace's classes are shared
    across all of its instances, so an existing config is never overwritten
    here even if `classes` is given.

    progress_cb(done, total), if given, is called after each file is copied.

    Returns (instance_name, image_count). Raises ValueError on bad input.
    """
    if not source_folder or not os.path.isdir(source_folder):
        raise ValueError(f"'{source_folder}' is not a folder.")

    source_real = os.path.realpath(source_folder)
    root_real = os.path.realpath(DATASETS_ROOT)
    try:
        common = os.path.commonpath([source_real, root_real])
    except ValueError:
        common = None
    if common == root_real:
        raise ValueError(
            "This folder is already inside datasetsInput. "
            "Pick the original folder elsewhere on disk."
        )

    files = sorted(f for f in os.listdir(source_folder) if f.lower().endswith(IMAGE_EXTENSIONS))
    if not files:
        raise ValueError("No images found in that folder.")

    instance_name = next_instance_name(workspace_name)
    dest_folder = os.path.join(DATASETS_ROOT, instance_name)
    os.makedirs(dest_folder, exist_ok=True)

    for i, fname in enumerate(files, start=1):
        shutil.copy2(os.path.join(source_folder, fname), os.path.join(dest_folder, fname))
        if progress_cb:
            progress_cb(i, len(files))

    if classes and existing_classes_for_workspace(workspace_name) is None:
        workspace_config.set_classes(workspace_name, classes)

    return instance_name, len(files)
