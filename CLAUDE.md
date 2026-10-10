# Jelibox - notes for Claude

Local computer-vision annotation tool (Tkinter GUI, Ultralytics YOLO). Read `README.md` for the user-facing description.

## Working rules

- Do not commit, push, tag or release unless the user asks. Local edits only by default.
- Follow `RELEASING.md` when asked to release: GitHub Flow, `feature/*` branches, one `[Unreleased]` line in `CHANGELOG.md` per user-visible change.
- Keep docs truthful about permissions and system changes: README.md, index.html and the installer messages must match what the scripts really do (see "Installer principles").
- Match the existing code: short comments, no extra abstractions.

## Layout

- `utils/Annotator.py` - entry point. No args = workspace picker, folder arg = annotation window. `config.load_workspace()` must run before `AnnotationGUI` is imported.
- `utils/` - GUI (`AnnotationGUI.py`, `WorkspacePicker.py`), training (`training.py`, run as a subprocess), export, relocate ("Move Jelibox"), `collab/` (hidden behind `JELIBOX_COLLAB=1`).
- Data per workspace: `datasetsInput/<ws>-<n>/`, `vocdataset/<ws>/`, `YOLOdataset/<ws>/labels/`, `models/<ws>/`, `configs/<ws>.json`. Images are never copied.
- Installers: `install.ps1` / `install.sh` (one-liners that download a release and unpack it), `jelibox_windows_installation.bat` / `jelibox_linux_installation.bash` (set up the environment), `tools/setup_python.ps1` / `tools/setup_python.sh` (uv + private Python), `tools/release.py`.

## Installer principles

- Jelibox runs on its own Python 3.12 only: uv downloads it into `<install>/.python`, the virtual environment is `<install>/venv` (Windows) or `<install>/jelibox` (Linux). There is no system-Python fallback and no `JELIBOX_PYTHON=system`.
- uv is looked up (`JELIBOX_UV`, PATH, uv's default install folders) and installed with the official installer only when missing: Windows `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`, Linux/macOS `curl -LsSf https://astral.sh/uv/install.sh | sh`.
- Never put Python on `PATH`, never register it with Windows (`UV_PYTHON_INSTALL_REGISTRY=0`, `--no-bin`), never `activate` the environment: call `<venv>\Scripts\python.exe` / `<venv>/bin/python` by path.
- Elevation: nothing runs as administrator/sudo by default. Windows elevates only `VC_redist.x64.exe` (`:INSTALL_VCREDIST`), and only when the runtime is missing. Linux uses `sudo` only to install libGL/GLib when `import cv2` fails.
- GPU: the installers ask whether the machine has a discrete NVIDIA GPU (detection is only the default). The answer travels as `JELIBOX_GPU=nvidia|cpu` (`install.ps1`/`install.sh` ask first, the `.bat`/`.bash` ask only when it is unset). NVIDIA installs PyTorch from `whl/cu121`, CPU from `whl/cpu` (never PyPI's default Linux wheel, which is CUDA).
- `jelibox_windows_installation.bat` must keep CRLF line endings (tests check this); `*.sh` / `*.bash` must stay LF.

## Modes (front menu)

- Four modes, chosen on the picker's front menu and stored app-wide (`configs/_app.json` key `mode`, `utils/assistant_modes.py`): YOLO-World (original; provider `yolo_world` or `custom_model`), LocateAnything (`locate_anything`), Custom Model (called `custom_head` in the code and the config), SAM 2 Dynamic (`sam2_dynamic`). Per-workspace settings live in `configs/<ws>.json` under `label_assistant` (`utils/workspace_config.py`).
- `utils/inferenceObjectDetection.py` builds the predictor for the current mode; `G` / the Infer window (`AnnotationGUI.show_infer_dialog`, choice kept in `infer_scope`, default single image) and Auto-annotate all (`utils/auto_annotate.py`, GUI in `utils/AutoAnnotateDialog.py`) share it. Providers are config-free functions in `utils/assistant_providers.py`.
- Never load the 3B LocateAnything model in tests or scripts: tests fake the predictor; check imports with the processor/tokenizer only. Keep torch / transformers imports lazy (a test enforces it). Its remote code is pinned (`MODEL_REVISION`); change it only after reading the diff.
- Original mode can also use any Ultralytics model chosen with **Browse custom model** (`custom_model` block: path + class_map). A browsed model maps its class NAMES to workspace classes; the workspace's own `modelAssistant.pt` still maps by index (`inferenceObjectDetection.custom_model_file`, `build_predictor`).
- Workspace classes: use `inf.current_classes()` (live `class_manager`), never `config.CLASSLIST` in code that runs after the window opened. `CLASSLIST` is a snapshot from `load_workspace`; filtering predictions with it made Auto-annotate label nothing when a class was added later.
- Package limits live in `requirements.txt` only (`ultralytics>=8.4.68`, `transformers==4.57.6`, `huggingface_hub<1.0`); PyTorch is chosen by the installers. One environment serves all modes.
- Custom Model (code: custom head) = frozen detector (class ids are a setting) + trained head; sources it was ported from are in `TOBEADDED/` (kept as received, skipped by the hygiene test).
- Train in this mode (`AnnotationGUI._train_head`) trains the head, not a YOLO model: `utils/HeadTrainDialog.py` (setup + progress window) runs `python -m utils.head_train <job.json>` as a subprocess (`@@ {json}` event lines on stdout), `utils/head_data.py` is the torch-free data side (pairing, position-based split, taps per detector family). It installs `models/<ws>/head_best.pt` and points `custom_head` at it. Tests fake the detector; the real-detector run is done by hand (CPU, minutes per epoch).
- SAM 2 Dynamic (`utils/sam2_dynamic.py`): `ReferenceStore` keeps ONE record per image path (replaced whole on every `save_current`, so an annotation can never be a reference twice); `Sam2Engine.build` assembles the predictor's memory bank from the store (one entry per image, encodings cached per image + boxes + file stamp) and never appends blindly. One object slot per workspace class (`max_obj_num = len(classes)`, obj id = class position); a reference is given as ONE mask per class holding ALL its objects (`reference_masks`: polygons filled exactly, boxes first refined to silhouettes by `Sam2Engine._boxes_refiner` = SAM 2 in its normal one-image mode, rectangles only as a fallback; plain rectangles taught the background and merged/loosened results in a boots test) - SAM 2 remembers unmarked look-alikes as background, so a single box per class made it skip the second worker (measured). Polygon mode uses polygon items only, box mode uses boxes (+ polygon bounding boxes) - `ReferenceStore.select(..., polygon)`. The predictor must be created with `conf=0` (it drops objects scoring 0 and our `min_score` decides); output row i is slot `list(obj_idx_set)[boxes.cls[i]]`. Images whose annotations came from the assistant and were not edited (`AnnotationGUI._assist_unreviewed`) are not stored. Tests fake the predictor; the real run (`sam2.1_t.pt`, CPU, seconds) is done by hand.
- Add Workspace / the "+" on a workspace (`WorkspacePickerApp._open_add_workspace_dialog`) is the only way to bring a dataset in (there is no Import Dataset button): on Browse it runs `dataset_import.scan_dataset_folder` + `scan_classes` (the summary and the amber "SKIPPED" warning are shown before Create and follow the typed workspace name), then Create calls `dataset_import.import_dataset(folder, ws, prefix, progress_cb, classes)`. A workspace that already has classes is "locked": other classes are skipped and counted (`result["skipped"]`) and its config is never changed; a new one gets the typed classes first, then those found. YOLO labels are regenerated from the parsed objects with the workspace class list, never copied. `workspace_manager.create_workspace_instance` (images only, flat folder) is no longer used by the window.
- Export Dataset has a **Save to** row (`AnnotationGUI.show_export_dataset_dialog`; remembered in `configs/_app.json` key `export_dir`; a remembered or chosen folder that no longer exists, e.g. an unplugged drive, falls back to the default folder - at open (the setting is kept) and again at export time). `utils/export.py` holds the pure helpers: `plan_export_target` (no folder chosen = the old `export dataset/<ws>` folder, replaced each time; a chosen folder = a new `<ws>-v<N>` inside it, `next_version_name` counts on from the highest existing one, never deleting anything), `check_destination` (writable + free space for the images) and `files_size`. The export runs on a worker thread that reports with `root.after`, which only works while the Tk main loop runs: GUI tests start it from inside `root.mainloop()` (`_export_and_wait`), pumping with `update()` raises "main thread is not in main loop". The worker must not assign to a variable of the enclosing function (an earlier `target = ...` made it local and the export died with UnboundLocalError).
- Analyze Dataset (`utils/dataset_analysis.py` numbers, `utils/DatasetAnalysisDialog.py` matplotlib window): the picker keeps `selected` (workspace name without index, highlighted row) and enables the **Analyze** button from it. Data comes only from `vocdataset/<ws>/*.xml` (image size is in the file; polygons count through their bounding box; an XML without a matching image is ignored, one without a size is left out of the resize simulation). Resize is calculated (`view(data, size, mode)`: letterbox = size / longest side, stretch = per axis), never done on disk. Objects with a side under the limit (`SMALL_LIMIT` = 32 px by default; the "Small object limit" box in the window changes it, 1-1024, remembered in `configs/_app.json` key `small_limit`) raise a warning strip in the window; the charts, the table and Remove all use `self.limit`. matplotlib is imported inside the window only (`Figure` + `FigureCanvasTkAgg`, no pyplot); the GUI test skips without it. `da.remove_small` (the strip's **Remove these objects** button, after an `askyesno` warning that says to back up first; default answer No) rewrites the flagged XML files without those objects and regenerates their YOLO labels with the workspace class list through `file_handler._yolo_lines_from_annotations` (imported lazily: it pulls in `config`, so its test lives in the integration group); objects of an image whose size is unknown are only removed in the original-size view, and then no YOLO label is written. Reading the files reports progress (`analyze(..., progress_cb)`), shown by `utils/progress_popup.py` (also used by the picker's delete popups). Screenshots of it: the app is DPI-unaware, so do not call `SetProcessDPIAware` in the script (matplotlib then sizes the figure 1.25x too large); save the figures with `fig.savefig` instead.
- Dialogs taller than the screen use `utils/dialog_scroll.py` (`ScrollBody`): header and footer stay, the body scrolls.

## Commands

```bash
python tests/run.py                    # everything: repo, unit, integration, GUI
python tests/run.py --only repo,unit   # what CI runs (also runs on Linux and Windows) with only pillow, numpy and opencv-python-headless installed:
                                       # a test that needs torch / ultralytics must skip without them (the release workflow runs the same checks)
python tests/run.py --only repo -k python_setup
```

Real-environment check: build a scratch Python 3.12 venv with uv (torch from the CPU index, then `-r requirements.txt`) and run `tests/run.py` with its python. `JELIBOX_TEST_DOWNLOAD=1` also runs the custom-head test that downloads a 5 MB detector.

Run the app: `venv\Scripts\python utils\Annotator.py` (Windows) or `jelibox/bin/python -u utils/Annotator.py` (Linux).

Test an installer change locally without pushing: `JELIBOX_ARCHIVE` takes a local `.zip` (Windows) / `.tar.gz` (Linux) laid out like a GitHub archive (one top-level folder), so the one-liner can install the working tree. `JELIBOX_VERSION=main` installs GitHub's `main` instead of the latest release (the latest release can be older than `main`).

## Gotchas

- `jelibox-local.zip` (scratchpad) is a snapshot: rebuild it after every change you want the user to install. A stale archive once made a fixed bug look unfixed. The installed copy at `%USERPROFILE%\Jelibox` is also a snapshot, so check what it contains before diagnosing.
- Python 3.10 aborts (`Tcl_AsyncDelete`, "main thread is not in main loop") when a Tk variable is garbage-collected on a non-main thread. Avoid threads in dialog code (`subprocess.run(capture_output=True)` starts reader threads too; `HeadTrainDialog` polls a log file instead) and keep the `<Destroy>` -> `gc.collect` hook in `ModeDialogs._Dialog`.
- Original-mode Train (`utils/training.py`) hard-codes `device=0`, so it fails without an NVIDIA GPU. Not changed.
- The header shrinks in steps (`AnnotationGUI.HEADER_LEVELS`: tighter buttons, then no brand text, then icons with tooltips), measured with `_header_need`, so it fits a 12-inch screen (about 1280 px logical). Header buttons are made with `_hbtn`; change a label with `_set_hbtn_text`, never `config(text=...)`. Tests find them by their full label (`btn._jb['full']`).
- GUI checks by hand: copy the installed app to the scratchpad, drive the real window, screenshot with `SetProcessDPIAware()` (the user's display is 125%); `root.deiconify()` first or transient dialogs stay hidden. Never load the 3B model; a real head-training run on CPU takes minutes per epoch.
- The Bash tool mangles backslashes and quotes in heredocs: write edit scripts with the Write tool (UTF-8, CRLF-aware) instead.

- The one-line installers download from GitHub, not from the folder they sit in. Local changes are only installed through `JELIBOX_ARCHIVE`.
- `releases/latest` is what `install.ps1` installs by default; changes on `main` reach users only after a release is cut.
- `tests/repo/test_python_setup.py` checks installer wiring by reading the scripts as text; update it together with the scripts.
- `tools/setup_python.*` tests use a fake uv (`JELIBOX_UV`); the real uv branch was verified by hand, not in the suite.
