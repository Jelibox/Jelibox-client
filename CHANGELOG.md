# Changelog

All notable changes to Jelibox are written here, newest first.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/): `MAJOR.MINOR.PATCH`.

Add a line under **[Unreleased]** in every pull request that a user would notice.
`python tools/release.py` turns that section into a version when you cut a release (see [RELEASING.md](RELEASING.md)).

## [Unreleased]

### Added
- **SAM 2 Dynamic mode** (the fourth mode): Ultralytics' `SAM2DynamicInteractivePredictor`, a training-free SAM 2 that learns from the images you annotate. Every image you annotate and leave becomes a reference; **Infer** (or `G`) then finds the same classes in the new image, as boxes or polygons following the Mode button. In polygon mode it learns from your polygons and answers with polygons, in box mode it learns from your boxes and answers with boxes, and it is shown every object of a class in a reference at once (boxes are first turned into the object's silhouette by SAM 2 itself, so loose boxes do not teach the background) (an unmarked object would be remembered as "not the object" and skipped later). Each image is remembered **once** (one record per image, replaced whole whenever you leave it again), the newest N are used (default 6), annotations the assistant made and you have not touched are never learned from, and the SAM 2 model (75-224 MB) is downloaded once into `models/_sam2/`. Its Label Assistant window picks the model, image size, number of references and minimum score, and can load your already-labeled images as references. **Train** explains that nothing needs training in this mode.
- **Undo / Redo** (`Ctrl+Z` / `Ctrl+Y`, and header buttons) for boxes, polygons, ZeroFill, deleting, class changes, **Repeat** and **Infer**. The record is per image and clears when you change image; while a polygon is half drawn, Undo takes back its last point.
- **Dataset augmentation** in **Export Dataset** (YOLO, Pascal VOC and COCO): brightness, contrast, saturation, hue, blur, noise, grayscale, JPEG compression, horizontal/vertical flip and rotation. Every effect has its own chance and strength limit, originals are always kept, Train is always augmented (Valid and Test are opt-in), and an optional seed makes an export repeatable. Flips and rotations move the boxes and polygons with the picture; rotation grows the canvas instead of cropping.
- A line now follows the cursor from the last point while you draw a polygon (segmentation and ZeroFill).
- **Open Folder** button in the workspace picker: opens the folder Jelibox is installed in.
- The one-line installers now say what they found: a fresh install, an update (`Found Jelibox v0.2.0 ... updating to v0.3.0`) or "already up to date" (`JELIBOX_FORCE=1` reinstalls anyway).
- **Four modes on a front menu** in the workspace picker (changeable with its Mode button): **YOLO-World** (the original Jelibox), **LocateAnything**, **Custom Model** and **SAM 2 Dynamic**. The chosen mode also drives the Label Assistant window, `G` and the new **Annotate all** button.
- **LocateAnything mode**: NVIDIA LocateAnything-3B through HF Transformers (`transformers==4.57.6`, `huggingface_hub<1.0`). List categories (English or Chinese), map them to workspace classes, choose decoding (hybrid / fast / slow), device, image size and passes. The model downloads on first use (about 7.2 GB, asked first) into `models/_huggingface`, its code is pinned to a reviewed commit, free memory is checked before loading, and it is never loaded in the other modes.
- **Custom Model mode**: a frozen YOLO detector (COCO person, bottle, ... - the classes are a setting) plus a head you trained that gives each box your own class (behavior, brand, ...). Needs `ultralytics>=8.4.68`; the checkpoint is read with `weights_only=True`.
- **Auto-annotate all images** (**Infer** → Full dataset, or `G`) for every mode, including YOLO-World: runs over the whole dataset folder with a progress bar, ETA and Stop, can skip images that already have annotations, and merges into existing annotations like `G` does.
- **Train a custom head**: in the Custom Model mode the Train button trains the head on the workspace's labels (detector frozen, never changed) behind a progress window, with a position-based train/validation split, 384×640 / 50 epochs / batch 16 as defaults, and installs the result as `models/<workspace>/head_best.pt`, already selected in Label Assistant.
- The Label Assistant windows scroll when they are taller than the screen (Save and Cancel stay visible).
- **Browse custom model** in the original mode: use any Ultralytics model (e.g. a fine-tuned YOLOv8m) from any folder, with its class names mapped to workspace classes; the workspace's own `modelAssistant.pt` keeps working as before. The Custom head mode got the same button for its detector weights (and the head button is now **Browse head**).
- `requirements.txt`: one list of packages for all four modes, installed by both installers; only PyTorch is chosen separately.

### Changed
- **ZeroFill** is now applied to the image file when you leave the image, not at once: it shows on screen immediately and can be undone until then. Closing the app discards masks that were not applied yet.
- **Infer** opens a window where you choose **Single image** (default) or **Full dataset**; `G` runs whichever is chosen (it used to be `G` = one image, `Shift+G` = all). The **Annotate all** button is gone, and **Infer** now sits with Label Assistant, Train and the export buttons.
- The annotation header now shrinks in steps (tighter buttons, then icons with tooltips) so every button stays reachable down to a 12-inch screen.
- Visibility panel: the class check boxes are lined up on the left.
- The **Custom head** mode is now called **Custom Model**, with a plainer description (the name `custom_head` stays inside the config files).
- Installers: Jelibox's Python is now **only** its own private copy. The system-wide Python fallback (python.org installer with `PrependPath`, apt/dnf/pacman Python packages, the deadsnakes PPA) and `JELIBOX_PYTHON=system` are gone, so installing Jelibox can no longer replace or shadow the Python you already use.
- Installers make sure **uv** is installed (using uv's official installer only when it is missing) instead of downloading a private copy into `.uv/`. Python is installed with `--no-bin` and not registered with Windows, and every environment command uses the environment's `python` by path (never `activate`), so `PATH` is not changed for Python.
- Windows: the installer is no longer restarted as administrator. Only the Visual C++ runtime installer is elevated, and only when the runtime is missing. Manual install no longer says "Run as administrator".
- Linux: `sudo` is only asked for when OpenCV's system libraries (libGL, GLib) are missing, and then the installer installs them itself.
- README and website describe these permissions accurately.

### Added
- The installers ask whether the computer has a **discrete NVIDIA GPU** (suggesting what they detected) and install the matching PyTorch: the CUDA build for yes, the CPU-only build for no. `JELIBOX_GPU=nvidia|cpu` skips the question. Before, any machine with `nvidia-smi` got CUDA, and on Linux the CPU path pulled PyPI's CUDA wheel with its several GB of NVIDIA libraries; now it uses PyTorch's CPU index.
- `JELIBOX_VERSION=main` makes the one-line installers install the latest development version instead of the newest release.
- **Move Jelibox** button in the workspace picker: moves the whole install (data included) to `<chosen folder>/Jelibox`, creates a new virtual environment with the same packages, and removes the old venv and folder.
- The one-line installers (`install.ps1`, `install.sh`) now ask where to install: Enter for the default (`Jelibox` in your user folder), `B` to pick a folder in a window, or type a path. `JELIBOX_HOME` still skips the question, and an existing install in the old default location is still updated in place.
- Groundwork for team collaboration (hidden unless `JELIBOX_COLLAB=1` or `"collab_enabled": true` in `configs/_app.json`): a per-installation device ID, a list of servers you can add, edit and remove from a ⚙ Server button in the workspace picker, and a separate private file for access keys. Jelibox still makes no network connections.

## [0.2.0] - 2026-10-04

### Added
- One-line install: `irm .../install.ps1 | iex` (Windows) and `curl .../install.sh | bash` (Linux), no git needed. Re-running updates in place and keeps your data.
- Windows installer installs the Visual C++ runtime automatically when it is missing.

### Fixed
- Windows installer works from folders whose path contains spaces.

## [0.1.0] - 2026-10-04

### Added
- **Label Assistant** (`G`): choose YOLO-World (8 models, text prompts mapped to your workspace classes) or your own trained model; predictions are merged into existing boxes and skip anything that overlaps.
- Per-workspace `configs/<workspace>.json` holding classes and assistant settings; the old `<workspace>.txt` class files are migrated automatically.
- Screen guard: Jelibox refuses non-desktop displays with a clear message and maximizes on desktops.
- Light (default) and dark themes with a toggle that restarts the window and resumes on the same image.
- Themed Windows title bar, Jelibox window/taskbar icon with hand-tuned small sizes.
- CLIP is installed by both installers (needed by YOLO-World).
- One-command test suite: `python tests/run.py`.
- Release tooling: `tools/release.py`, CI workflow, and `RELEASING.md`.

### Changed
- Boxify is now **Jelibox**: new name, icon, installers (`jelibox_*_installation`), launchers, and Linux virtual environment folder (`jelibox/`).
- New design system: warm cream neutrals, a single indigo accent, red only for danger; calm class colors.
- Header regrouped: quick actions next to Prev/Next, configuration buttons on the right; buttons follow a primary / secondary / danger hierarchy.
- "Toggle Text" is now "Hide label text" / "Show label text".
- The Force indicator moved to the status bar next to the mode pill.
- Redesigned the public site (`index.html`) and the Streamlit demo page.
- License changed from MIT to Apache 2.0.
- Deleting a class also removes the YOLO-World target classes that pointed at it.

### Removed
- The redundant top-right mode and "Text: ON" badges.
- The neon "cyber terminal" look.

[Unreleased]: https://github.com/Jelibox/Jelibox-client/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/Jelibox/Jelibox-client/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/Jelibox/Jelibox-client/releases/tag/v0.1.0
