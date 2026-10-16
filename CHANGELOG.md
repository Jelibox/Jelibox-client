# Changelog

All notable changes to Jelibox are written here, newest first.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/): `MAJOR.MINOR.PATCH`.

Add a line under **[Unreleased]** in every pull request that a user would notice.
`python tools/release.py` turns that section into a version when you cut a release (see [RELEASING.md](RELEASING.md)).

## [Unreleased]

### Fixed
- **Pictures with nothing in them are no longer thrown away when training.** A full YOLO training used only the pictures that have at least one box, so the "null" pictures you saved empty were not in the split at all and the model never learned what is NOT there (a workspace of 10,250 pictures trained on 7,821). They are now used as background pictures: a **Use the N pictures with nothing in them as background** box (on by default when there are some) in both Train windows, counted in the Train / Valid / Test split, split and augmented like the rest, and written with an empty label. Pictures that were never annotated (no label file) are still left out, because they may contain objects. The first time, the label cache is rebuilt once (it now also remembers which files are empty).

## [0.4.11] - 2026-10-15

### Fixed
- **The Train windows no longer freeze ("not responding") on big datasets.** Counting the labelled pictures meant opening every label file, which took about a minute for 10,000 pictures on a cold disk, and the plain Train window did it again at every click in the augmentation picker. The result is now remembered (a `.label_cache.json` next to the labels folder, checked by file size and date), counted once per window, and warmed up in the background a moment after the annotation window opens. The window itself also opens in about half a second instead of eight: the boxes-per-class numbers come from the same cache instead of re-reading every label (and no longer again at every split change), and the "no GPU" check, which has to load PyTorch, runs aside and shows its line when ready.

## [0.4.10] - 2026-10-15

### Added
- **The Train / Valid / Test split is also in the plain Train button** (YOLO-World and the other modes): its Training Configuration window has the same colour bar, a held-back test part scored at the end, and the "Mix the pictures before splitting" option. The default stays the old 70 / 30 with no test part.
- **Augment the validation (and test) pictures too.** The augmentation picker has an **Also augment** row with "the validation pictures" and "the test pictures" (off by default, because scores measured on real pictures are the honest ones). It works in the full YOLO training window, in the plain Train window and in Export Dataset; training is always augmented.

## [0.4.9] - 2026-10-15

### Added
- **Train / Valid / Test split in the full YOLO training window**, the same colour bar as Export Dataset (drag the two handles or type the percentages, a chip per split with its picture count). The test part is held back: training never sees it, and when the run ends the best model is scored on it once (mAP50 and mAP50-95 are shown in the window). The split is by position by default (first pictures train, then validation, then test, so near-duplicate video frames do not leak between them); **Mix the pictures before splitting** makes it random. The head-only training keeps its validation box.

### Changed
- **The training window now shows the data preparation.** Before the first epoch it says what it is doing (Step 1/3 reading the labels, 2/3 splitting, 3/3 copying the pictures and making the augmented copies), with a progress bar, the number of pictures done and the time left, then "Starting Ultralytics ..." while it checks the pictures. The bar switches to epochs when training starts.

## [0.4.8] - 2026-10-15

### Added
- **Export Dataset redesigned.** One scrolling window with numbered sections: format and destination, **Train / Valid / Test split** (draggable colour bar, a chip per split with its picture count and percentage), **Preprocessing** and **Augmentation**, and a footer that always says what will be written ("Export: Train 1530 · Valid 437 · Test 218", augmented copies included).
- **Preprocessing** in Export Dataset: **Auto-Orient** (on by default: the EXIF rotation tag is dropped and the picture saved upright) and **Resize** (**Stretch to** W x H, **Fit within** W x H keeping the shape, or **Fit with black edges** to an exact W x H). The boxes and polygons are recalculated with the same matrix, for YOLO, Pascal VOC and COCO, and for the augmented copies too. Your original pictures are never changed.
- **Visual augmentation picker.** Every augmentation is a tile with a preview made from a picture of your workspace; click one to open its settings window with the original next to examples that follow the slider: Flip, 90° Rotate, Rotation, Shear, Grayscale, Hue, Saturation, Brightness, Exposure, Contrast, Blur, Motion Blur, Noise, Cutout and JPEG compression. **Outputs per training image** and **Keep the original as one of them** say how many pictures each training image becomes (with the original dropped, only the augmented pictures are written). The preview picture is the first picture of the workspace until you choose another with **Choose preview picture...**; the choice is kept in the workspace config and falls back to the first picture when the file is gone.
- **Augmentation while training.** The full YOLO training window (Custom Model mode) and the Training Configuration window of the plain Train button have an **Augmentation** button with the same picker, so you can train on an augmented version of your data without exporting it first. The copies are made from the training pictures only (never the validation ones) in the run's scratch folder and removed afterwards. Flip, Rotation, Shear, Hue, Saturation and Brightness are greyed out there with a note that Ultralytics already does them while it trains.

### Changed
- Rotation and flips are separate settings, rotation takes any angle up to the limit you set, and the canvas grows instead of cropping.

## [0.4.7] - 2026-10-13

### Added
- **Video files in Add Workspace / Add Instance.** When the chosen folder contains videos (`.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`, `.m4v`, `.wmv`, `.flv`, `.mpg`, `.mpeg`), a new **Skip frames** box appears (10 by default) with a plain explanation and a live estimate ("about 30 picture(s) from 1 video(s)"). Jelibox saves one frame, skips the next N - 1, and so on: with 10 every 10th frame becomes a JPEG picture next to the images (`<video>_f<frame>.jpg`, or `<prefix><number>` when a prefix is set), so a 30 fps video gives 3 pictures per second; 1 keeps every frame. The pictures start without annotations, the videos themselves are not copied, and a video that cannot be read is reported while the rest still imports. A folder with only videos is accepted.

### Changed
- The Add Workspace window scrolls when it is taller than the screen (a 12-inch laptop); Cancel and Create stay visible.

## [0.4.6] - 2026-10-11

### Added
- **Finish early** in the training window (the head training, the full YOLO training and the plain Train button): it stops the training in the middle of the epoch in progress but ends it properly. The epoch is validated and saved, then Ultralytics runs its usual final evaluation of the best model, and the result is installed as after a normal run. The old Stop button is now **Stop now**: it ends the process at once (the best model so far is still kept).

### Fixed
- The training window cut off its buttons (Stop, Close) once the epoch table was added; the window now follows the size of its content, and the table and log are shorter on a small screen.

### Changed
- **The Train button of YOLO-World and the other modes now shows its progress in a window**, not in a terminal. It is the same window as the Custom Model's full training: a table with one row per epoch (box / cls / dfl loss, precision, recall, mAP50, mAP50-95, time), the best epoch, a progress bar, the time left and Stop. It does not lock the annotation window, so you can keep annotating while it trains. The best model is saved as `modelAssistant.pt` (the old one as `modelAssistant.prev.pt`) and selected as My trained model; the images are copied only into a scratch folder for the run and removed afterwards. The settings dialog before it is unchanged.

## [0.4.5] - 2026-10-11

### Added
- **Full YOLO training in the Custom Model mode.** The Train window now starts with a choice: **Head only** (as before) or **Full YOLO training**, which trains every layer with Ultralytics like the Train button of the other modes, with base model, square image size, epochs, batch, learning rate, freeze and an option to continue from the model trained here. The window shows a **table with one row per epoch** (box / cls / dfl loss, precision, recall, mAP50, mAP50-95, time) and the best epoch, a progress bar, the time left and Stop; the head run gets the same table. The best model is saved as `modelAssistant.pt` (the old one as `modelAssistant.prev.pt`) and selected as My trained model for YOLO-World mode.

### Fixed
- The plain **Train** button (YOLO-World and the other modes) no longer fails on a computer without an NVIDIA GPU: it trains on the CPU when PyTorch has no GPU.

## [0.4.4] - 2026-10-10

### Changed
- Screenshots in the README and on the website are now taken in a demo workspace (neutral workspace and image names).

## [0.4.3] - 2026-10-10

### Changed
- Change annotate.jpg

## [0.4.2] - 2026-10-10

### Changed
- **README and website rewritten** around what Jelibox does today: the four auto-label modes, importing any dataset, the Analyze dashboard, and augmented export to any drive, with new screenshots of each, a one-line install at the top of the README, a release badge, and a social preview image for shared links. The old annotation screenshot is replaced by a current one.

## [0.4.1] - 2026-10-10

### Added
- **Small object limit** in the Analyze Dataset window: a box next to the resize controls sets the size under which an object counts as small (32 px by default, 1-1024). The warning strip, the charts, the Resize table and **Remove these objects** all follow it, and the number is remembered for next time.

## [0.4.0] - 2026-10-10

### Changed
- **Import Dataset is now part of Add Workspace** (the Import Dataset button is gone, which frees room in the header). **Add Workspace** and the **+** on a workspace scan the chosen folder like Import Dataset did - every subfolder, images plus Pascal VOC, YOLO or COCO annotations (or images only) - and show what was found (images, format, classes) before anything is copied. There is a new optional **Filename prefix** field. A new workspace gets the classes found in the annotations as a starting point (still editable; the order you type comes first). An existing workspace keeps its classes: annotations of any other class are **skipped**, with an amber warning in the window before you press the button and a count afterwards. YOLO labels are now written again with the workspace's class numbers instead of being copied with the source dataset's numbers.
- **Export Dataset: choose where to save.** A new **Save to** row (Browse... / Default) lets you export to any folder, e.g. an HDD, so finished datasets are archived there instead of on the SSD. Into a chosen folder every export becomes a new `<workspace>-v<N>` folder (N = highest existing `<workspace>-v<number>` there + 1, starting at v1), never replacing or deleting anything; the choice is remembered, and if that folder is gone (an unplugged drive) the default folder is used instead, with a note in the window. Jelibox checks that the folder is writable and has room for the images before it starts. Without a chosen folder nothing changes: `export dataset/<workspace>/` as before.

### Added
- **Analyze Dataset**: click a workspace in the picker (its name is highlighted) and press **Analyze** for a dashboard window with charts: objects per class, images per dataset, image resolutions, objects per image, width / height histograms, a width x height scatter and the shortest side per class. Everything is read from the VOC XML files, no image is opened. A warning strip (not a dialog) appears when objects have a side under 32 px. **Simulate** calculates the same numbers for a model input size (640 or your own, letterbox or stretch) without resizing any image, and a table compares 320 to 1280. Charts use matplotlib (zoom, pan, save as PNG, tooltips on hover). A progress bar shows while the annotations are read. The warning strip has a **Remove these objects** button: after a warning (it changes and deletes objects in the current dataset - back it up first) it removes the flagged objects, measured at the original or the simulated size, from the VOC XML files and rewrites their YOLO labels; images are never touched.

## [0.3.0] - 2026-10-10

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

[Unreleased]: https://github.com/Jelibox/Jelibox-client/compare/v0.4.11...HEAD
[0.4.11]: https://github.com/Jelibox/Jelibox-client/compare/v0.4.10...v0.4.11
[0.4.10]: https://github.com/Jelibox/Jelibox-client/compare/v0.4.9...v0.4.10
[0.4.9]: https://github.com/Jelibox/Jelibox-client/compare/v0.4.8...v0.4.9
[0.4.8]: https://github.com/Jelibox/Jelibox-client/compare/v0.4.7...v0.4.8
[0.4.7]: https://github.com/Jelibox/Jelibox-client/compare/v0.4.6...v0.4.7
[0.4.6]: https://github.com/Jelibox/Jelibox-client/compare/v0.4.5...v0.4.6
[0.4.5]: https://github.com/Jelibox/Jelibox-client/compare/v0.4.4...v0.4.5
[0.4.4]: https://github.com/Jelibox/Jelibox-client/compare/v0.4.3...v0.4.4
[0.4.3]: https://github.com/Jelibox/Jelibox-client/compare/v0.4.2...v0.4.3
[0.4.2]: https://github.com/Jelibox/Jelibox-client/compare/v0.4.1...v0.4.2
[0.4.1]: https://github.com/Jelibox/Jelibox-client/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/Jelibox/Jelibox-client/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/Jelibox/Jelibox-client/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/Jelibox/Jelibox-client/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/Jelibox/Jelibox-client/releases/tag/v0.1.0
