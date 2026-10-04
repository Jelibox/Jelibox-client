"""
Shared test plumbing.

`isolated_workspace()` points EVERY path Jelibox uses (configs, datasetsInput,
vocdataset, YOLOdataset, models, app settings) at a throw-away temp directory
and loads a tiny real workspace into utils.config - so integration and GUI
tests exercise the real code without ever touching the user's data.

It must run before any module that does `from .config import <workspace
global>` is imported, and it is idempotent: one workspace per test process.
"""
import atexit
import os
import shutil
import sys
import tempfile

# Ultralytics would otherwise pip-install missing optional packages into the
# developer's venv the first time a test imports it.
os.environ.setdefault("YOLO_AUTOINSTALL", "False")

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

WORKSPACE = "testws"
INSTANCE = "testws-1"
CLASSES = ["cat", "dog"]

_ctx = None


def _cleanup(root):
    """Delete the temp workspace. Windows cannot remove the current directory, so leave it first."""
    os.chdir(REPO)
    shutil.rmtree(root, ignore_errors=True)


class Ctx:
    def __init__(self, root):
        self.root = root
        self.configs = os.path.join(root, "configs")
        self.datasets = os.path.join(root, "datasetsInput")
        self.instance_dir = os.path.join(self.datasets, INSTANCE)
        self.model_dir = os.path.join(root, "models", WORKSPACE)
        self.model_file = os.path.join(self.model_dir, "modelAssistant.pt")
        self.voc = os.path.join(root, "vocdataset", WORKSPACE)
        self.labels = os.path.join(root, "YOLOdataset", WORKSPACE, "labels")


def make_images(folder, count=3, size=(320, 240)):
    from PIL import Image
    os.makedirs(folder, exist_ok=True)
    names = []
    for i in range(count):
        name = f"img{i + 1}.png"
        Image.new("RGB", size, (60 + i * 40, 90, 140)).save(os.path.join(folder, name))
        names.append(name)
    return names


def isolated_workspace():
    """Create + load the temp workspace once; returns a Ctx."""
    global _ctx
    if _ctx is not None:
        return _ctx

    root = tempfile.mkdtemp(prefix="jelibox_test_")
    atexit.register(_cleanup, root)
    ctx = Ctx(root)
    os.makedirs(ctx.configs, exist_ok=True)
    os.chdir(root)                      # ClassManager still uses cwd-relative paths
    shutil.copytree(os.path.join(REPO, "assets"), os.path.join(root, "assets"))   # the GUI opens assets/... relative to cwd

    from utils import app_settings
    app_settings._PATH = os.path.join(ctx.configs, "_app.json")

    from utils import workspace_config, workspace_manager
    workspace_config.CONFIGS_ROOT = ctx.configs
    workspace_manager.BASE_DIR = root
    workspace_manager.DATASETS_ROOT = ctx.datasets
    workspace_manager.CONFIGS_ROOT = ctx.configs

    from utils import config
    config.BASE_DIR = root
    config.datasets_root = ctx.datasets

    make_images(ctx.instance_dir)
    workspace_config.set_classes(WORKSPACE, CLASSES)
    config.load_workspace(ctx.instance_dir)

    import utils.dataset_import as di      # binds paths at import time
    di.BASE_DIR = root
    di.DATASETS_ROOT = ctx.datasets
    di.CONFIGS_ROOT = ctx.configs

    _ctx = ctx
    return ctx
