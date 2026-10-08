# Local changes compared to GitHub (`origin/main`, commit 7414a68)

Nothing here is committed or pushed. `git diff` shows the exact changes; this file explains them.

## Why

Installing through `install.ps1` replaced the user's Python 3.10 setup:

1. `install.ps1` installs the newest *release* (`v0.2.0`), which predates the private-Python commit `7414a68`, so it still ran the old `.bat` (system-wide Python 3.12 from python.org with `PrependPath=1`).
2. Even on `main`, the system-Python fallback could still install Python system-wide and change `PATH`, and the admin/sudo requirements were described wrongly.

## Changes

### Installers
- `install.ps1`, `install.sh`: `JELIBOX_VERSION=main` installs the latest `main` (was a 404 on `refs/tags/main`); a `main` install is never skipped as "up to date". `JELIBOX_PYTHON` removed from the header comments, wording about permissions corrected.
- `tools/setup_python.ps1`, `tools/setup_python.sh`:
  - Use uv from `JELIBOX_UV`, PATH or uv's default install folders; otherwise run the official installer (`irm https://astral.sh/uv/install.ps1 | iex` on Windows, `curl -LsSf https://astral.sh/uv/install.sh | sh` on Linux/macOS, `wget` as a fallback).
  - Removed the checksum-verified private copy of uv (`<install>/.uv`) and `JELIBOX_UV_VERSION`.
  - Python is installed only into `<install>/.python`; Windows additionally sets `UV_PYTHON_INSTALL_REGISTRY=0` so `py -3.12` does not see it.
- `jelibox_windows_installation.bat`:
  - Removed the system-wide Python install (python.org, `InstallAllUsers=1`, `PrependPath=1`), the `PATH` edit, `JELIBOX_PYTHON=system` and the `:REQUIRE_ADMIN` restart of the whole installer.
  - Administrator permission is requested only for `VC_redist.x64.exe`, only when the Visual C++ runtime is missing (`:INSTALL_VCREDIST`; accepts exit codes 0, 1638, 3010; reports a refused prompt).
  - No `activate.bat`: every `pip` call uses `"%VENV_PY%" -m pip`.
  - Setup failure now stops with a clear message instead of falling back.
- `jelibox_linux_installation.bash`:
  - Removed `use_system_python` (sudo up front, apt/dnf/pacman Python packages, deadsnakes PPA) and `JELIBOX_PYTHON=system`.
  - No `source activate`: pip and the desktop launcher use `$VENV_DIR/bin/python` by path.
  - New "system libraries" step: only when `import cv2` fails, installs libGL/GLib with the distribution's package manager (asks for `sudo`, or prints the exact command if sudo is unavailable).

### GPU question (discrete NVIDIA GPU or not)
- `install.ps1`, `install.sh`: before downloading anything, ask "discrete NVIDIA graphics card?" with the detected answer as the Enter default (Windows: `nvidia-smi`, then Win32_VideoController; Linux: `nvidia-smi`, then `lspci`). The answer is passed on as `JELIBOX_GPU=nvidia|cpu`. `JELIBOX_GPU` preset skips the question; any other value is refused before anything is unpacked. Not asked with `JELIBOX_NO_INSTALL=1` or on ARM.
- `jelibox_windows_installation.bat`, `jelibox_linux_installation.bash`: use `JELIBOX_GPU` when set, otherwise ask themselves (manual installs). The user's answer wins over detection; choosing NVIDIA without `nvidia-smi` prints a driver warning.
- PyTorch: NVIDIA -> `whl/cu121` as before; CPU -> `whl/cpu` on both platforms (Linux's default PyPI wheel is the CUDA build, which pulled in several GB of NVIDIA packages).

### Docs
- `README.md`, `index.html`: requirements, install steps and manual install now state the real permissions (Windows: admin only for the Visual C++ runtime; Linux: sudo only for OpenCV libraries), describe uv, drop "Run as administrator" and the system-Python option, run the app by the environment's python path.
- `CHANGELOG.md`: new "Changed" entries and the `JELIBOX_VERSION=main` entry under `[Unreleased]`.
- `CLAUDE.md` (new): project and installer rules for Claude. `LOCAL_CHANGES.md` (new): this file. `README.md` also gained a Modes section.

### Tests
- `tests/repo/test_quick_install.py`: the "only one question" test now expects the continue question plus the GPU question; new tests refuse an unknown `JELIBOX_GPU` in `install.sh` and `install.ps1`.
- `tests/repo/test_python_setup.py`: GPU wiring test (all four scripts ask, question before download, answer handed on, CPU/CUDA index URLs); wiring tests rewritten for the new design (no system Python, no PATH edit, no activate, single elevation point, uv install commands, sudo only in the system-library step, docs not promising wrong permissions); one test no longer assumes the host Python is 3.12.

## Three modes, LocateAnything, custom head, Auto-annotate all (second round)

- New modules: `utils/assistant_modes.py` (the 3 modes, app-wide choice), `utils/locate_parsing.py` + `utils/locate_anything.py` (LocateAnything-3B, ported from `TOBEADDED/locateanything/backend`; lazy torch/transformers imports, model pinned to commit `c32291ca...`, cache in `models/_huggingface`, free-memory guard, `check_requirements()`), `utils/custom_head.py` + `utils/detector_classes.py` (ported from `TOBEADDED/custom_dual_heads`; detector classes configurable instead of person-only, end-to-end detectors handled, `torch.load(weights_only=True)`), `utils/assistant_providers.py` (provider functions + `AssistantError`), `utils/auto_annotate.py` (GUI-independent batch runner + worker thread), `utils/AutoAnnotateDialog.py` (progress windows, download consent), `utils/ModeDialogs.py` (settings windows for the two new modes).
- Changed: `utils/workspace_config.py` (providers `locate_anything` / `custom_head`, blocks `locate_anything`, `custom_head`, `batch`, validated; removing a class prunes every mode), `utils/inferenceObjectDetection.py` (`build_predictor` / `predict_current` / `apply_predictions`; `inference_current` keeps its behaviour and signature), `utils/file_handler.py` (`save_annotations` writes VOC + YOLO without GUI state), `utils/WorkspacePicker.py` (front menu overlay, Mode button, `JELIBOX_SKIP_MODE_MENU` for self-restarts), `utils/AnnotationGUI.py` (Annotate all button, `Shift+G`, mode in the header, heavy providers run behind a progress window), `utils/LabelAssistantDialog.py` (opens the window of the current mode).
- Packages: new `requirements.txt` (`ultralytics>=8.4.68`, `transformers==4.57.6`, `huggingface_hub>=0.34.0,<1.0`, `peft`, `lmdb`, `decord` on Windows / x86-64 Linux, ...). Both installers now `pip install -r requirements.txt`. The LocateAnything pins that were NOT adopted: numpy 1.26.4, torch 2.9.0+cu130, opencv-python(-headless) 4.10, gradio / fastapi / uvicorn (web app only), the `nvidia-*` CUDA 13 wheels and the other transitive pins - pip resolves them freely and nothing conflicts, so no second environment (uv) is needed.
- Tests: new `tests/unit/test_locate_parsing.py`, `test_assistant_modes.py`, `test_auto_annotate.py`, `test_locate_requirements.py`, `test_custom_head.py`; `tests/integration/test_modes_flow.py`; `tests/gui/test_modes_gui.py`; `tests/repo/test_modes_wiring.py`. `tests/repo/test_repo_hygiene.py` skips `TOBEADDED/` (third-party source kept as received) and checks the new GUI modules for hard-coded colours. `TOBEADDED/` itself is untouched.

### Verification (second round)
- Resolved with uv for Windows and Linux (Python 3.12) and installed twice into scratch venvs: torch 2.5.1+cpu (what the CUDA 12.1 path installs) and torch 2.14.1+cpu. Both: transformers 4.57.6, huggingface_hub 0.36.2, tokenizers 0.22.2, ultralytics 8.4.174, `pip check` clean.
- LocateAnything remote code (downloaded as source files only, no weights): all 16 files compile; transformers' own import check finds nothing missing; the config, the model CLASS (not instantiated), tokenizer and processor load; the processor/tokenizer path of the engine ran on a real image. `model.generate` was NOT run - the 3B model was never loaded.
- Custom head: real `yolo11n` + a randomly initialised head on `bus.jpg`: 4 persons, 1 bus, 0 bottles, class filter, coordinates, `min_head_conf` and the frozen detector all verified.
- Whole suite inside the Python 3.12 scratch venv: 405 tests pass (repo, unit, integration, GUI). Latest-torch venv: 340 pass (GUI skipped there).

## Browse custom model (third round)

- Original mode, `utils/LabelAssistantDialog.py`: **Browse custom model ...** + **Use the workspace's own model** on the "My trained model" card; after browsing, the model's class names are read (`assistant_providers.model_class_names`) and shown as a model class -> workspace class table (same-name classes pre-mapped, `(skip)` for the rest). New config block `custom_model` {path, class_map} in `utils/workspace_config.py` (validated, pruned when a class is removed). `utils/inferenceObjectDetection.py`: `custom_model_file()` (browsed file, else `modelAssistant.pt`), mapping by NAME for a browsed model, unmapped classes skipped; the workspace's own model still maps by position. Works for `G` and Auto-annotate all.
- Custom head mode, `utils/ModeDialogs.py`: **Browse head ...** (was Browse) and **Browse custom model ...** next to Detector weights. A browsed detector's own class names replace COCO in "Detect these classes" (hint line lists them); typed paths are checked; bare stock names such as `yolov9c.pt` stay valid.
- Tests: `tests/unit/test_custom_model_config.py`, `tests/integration/test_browsed_model.py`, GUI classes `BrowseCustomModelTests` / `BrowseDetectorTests` in `tests/gui/test_modes_gui.py`. Real check: yolo11n as the browsed model on `bus.jpg`, person -> cat, bus -> dog: 4 cat + 1 dog.
- Shortcut question ("Jelibox shortcut still goes to the original mode"): not a code bug. The shortcut runs `venv\Scripts\python.exe utils\Annotator.py`, which opens the picker with the mode menu; verified by launching exactly that command (no arguments, no JELIBOX_ variables) and screenshotting the menu. The install in the user's profile folder (`%USERPROFILE%\Jelibox`) was made from an older local archive (no `assistant_modes.py`, no `requirements.txt`), so its shortcut cannot show a menu it does not contain. Fix = update that install from a current archive (see the README's JELIBOX_ARCHIVE note in CLAUDE.md).

- Auto-annotate bug: "nothing labelled" when a class was added after the window opened. `AutoAnnotateDialog` and `inferenceObjectDetection` used `config.CLASSLIST`, a snapshot taken at load, to filter predictions; they now read `inf.current_classes()` (live `class_manager`). Test: `AutoAnnotateWindowTests.test_a_class_added_after_the_window_opened_is_labelled`.

- Label Assistant dialogs scroll: new `utils/dialog_scroll.py` (`ScrollBody`). Header and Save / Cancel stay pinned, the body scrolls and the window is never taller than the screen (125% scale on 1080p leaves ~860 px; a 14-class head pushed Save off the screen). Used by `ModeDialogs._Dialog` (LocateAnything + Custom head) and `LabelAssistantDialog`. Tests: `DialogScrollTests`.
- Train in Custom head mode: `utils/head_data.py` (labels <-> images, position-based split, taps per detector family), `utils/head_train.py` (trainer subprocess: dataset/augmentation/jitter ported from `TOBEADDED/custom_dual_heads`, class weights ignore classes with no examples, stride check on the taps, detector fingerprint check, `@@ {json}` progress events), `utils/HeadTrainDialog.py` (setup + progress window, Stop keeps the best head so far, installs `models/<ws>/head_best.pt`, keeps the old one as `head_best.prev.pt`, selects it in Label Assistant), `AnnotationGUI._train_head`, `FrozenYOLO.strides_ok/fingerprint`. Defaults per the user: yolov8m.pt, 384x640, 50 epochs, batch 16, class weights on. Tests: `tests/unit/test_head_train.py`, `HeadTrainDialogTests` + two routing tests in `tests/gui/test_modes_gui.py`. Real run: 40 `testt` images, user's yolov8m.pt, CPU, 2 epochs through the window; the head then annotated `testt` (17 `standing` boxes on 10 images).

## Verification
- `python tests/run.py --only repo,unit`: all pass.
- `tools/setup_python.ps1` run for real on Windows twice, in a temp folder: with uv already installed, and with uv hidden so the official installer ran (uv into a temp dir). Both produced a working Python 3.12.10 with Tkinter inside the folder; the user PATH was unchanged and `py --list` still shows only Python 3.10.
- GPU prompt: ran `install.ps1` against a stand-in archive with answers n / y / Enter (results cpu / nvidia / detected=cpu), and the `.bat` GPU section in isolation for preset nvidia / cpu / invalid and typed y / n. The Linux prompt is only syntax-checked and covered by the invalid-value test.
- Not run: the full `.bat` (it downloads PyTorch) and the Linux scripts on a real Linux machine (syntax checked with `bash -n`, helper logic covered by the fake-uv tests).

## To ship
Cut a release (e.g. `v0.3.0`) after merging, otherwise `releases/latest` keeps serving v0.2.0 to people who run the one-liner.
