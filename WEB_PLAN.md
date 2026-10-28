# Jelibox 1.0 - the web version (plan and progress)

Goal: open Jelibox from a browser (`http://localhost:8420`, or `http://<server address>:8420` from a laptop on the same
network), sign in (`admin` / `admin` at first, changed on first sign in) and do everything the desktop window does,
except **Open Folder**. Runs on Windows and Linux; the installer starts the server at login / boot. The version
goes 0.4.15 -> **1.0.0** when all of it is done.

## Decisions (made with the maintainer)

| Topic | Decision |
|---|---|
| Desktop window | **Stays.** The web is an addition; both use the same files on disk and the same logic modules. |
| Dataset import | Add Workspace takes **a browser upload** (zip / many files, with progress) **or a folder that already exists on the server** (folder browser in the page). |
| Accounts | One `admin` at first (forced password change), the admin adds other users. Everyone sees every workspace; no per-workspace roles. |
| Autostart | **No administrator / sudo.** Windows: Startup (runs at login). Linux: `systemd --user` plus `loginctl enable-linger` (one `sudo`) so it survives a reboot without a login. |
| Server | Starlette + uvicorn (already in the environment through streamlit). No Node, nothing to build: the page is plain HTML / CSS / JavaScript in `utils/web/static/`. |
| Port | 8420 (`web_port` in `configs/_app.json`, `--port`). All interfaces by default (`web_host`, `--host 127.0.0.1` for this computer only). |

## The one hard problem: global state

The desktop code keeps the open workspace in module globals (`config.load_workspace` fills `input_folder`,
`class_manager`, `state` ...; modules do `from .config import x` at import time). A server serves several workspaces
and several people at once, so **web code must not use those globals**. Rules:

- New code takes paths and classes as arguments (a small workspace context object), and calls the pure helpers that
  already exist (`file_handler.build_voc_xml`, `_yolo_lines_from_annotations`, `_parse_voc_objects`,
  `workspace_config`, `workspace_manager`, `workspace_counts`, `head_data`, `export_preprocess`, `augment_*`).
- Where a desktop module reads the globals (`inferenceObjectDetection`, `auto_annotate`, `export`, the `_export_*`
  methods of `AnnotationGUI`), the logic is first moved into a context-taking function that both the desktop and the
  web call. The desktop behaviour must not change (its 264 GUI tests are the guard).
- Heavy work (training, auto-annotate, export) runs as a job with progress events, never inside a request.
- Nothing in `utils/web` imports tkinter, torch or ultralytics at import time (tests run in CI without them).

## Security

PBKDF2 password hashes in `configs/_users.json` (never committed: `configs/*` is ignored), signed `HttpOnly`
`SameSite=Strict` session cookie, a session ends when the password changes, login throttle, `X-Jelibox` header + Origin
check on every change, no path from the URL reaches the disk without going through `dataset root` validation, security
headers on every answer. Plain HTTP on a LAN (HTTPS is out of scope for 1.0; say so in the docs).

## Steps

1. [x] **Foundation**: server, accounts, login + forced password change, user management, workspace list, mode menu.
   (`utils/server.py`, `utils/web/`, tests `tests/unit/test_web_*.py`)
2. [x] **Workspaces**: done: Add Workspace (browser upload of a folder / files / .zip, or a folder on the server with a
   folder browser; Pascal VOC / YOLO / COCO detected, classes, prefix, videos with Skip frames), add a dataset to an
   existing workspace, delete workspace / dataset (jobs with progress; refused while that workspace trains).
   **Analyze** (charts, simulated resize, small-object limit, Remove these objects: `utils/web/analysis.py`, `analysis_api.py`, `static/analyze.js`, `tests/unit/test_web_analysis.py`). **Move Jelibox is not in the web** (decided with the maintainer).
   (`utils/web/workspaces_api.py`, `importing.py`, `jobs.py`; `utils/annotation_io.py` is the tk-free half of
   `file_handler`; tests `tests/unit/test_web_workspaces.py`; real-browser run with Playwright + Edge, see CLAUDE.md)
3. [x] **Annotation editor**: image list + search, canvas with zoom / pan (wheel, middle button or Space + drag), boxes (draw,
   move, 8 resize handles, Shift + drag multi-select, B forces a new box), polygons (points, move / delete a point,
   close on the first point, Enter / double-click), ZeroFill (written into the picture file on save), classes (add,
   delete with every label file rewritten, visibility, counts, change class by S / double-click), undo / redo, Repeat,
   delete picture, info panel, autosave (2.5 s) plus a save on every move, a keepalive save when the tab closes, and a
   version check so two people cannot silently overwrite each other (409 -> keep mine / take theirs). Shortcuts as on
   the desktop (A D arrows, R, S, E, B, M, 1-9, Del, Esc, Enter, Ctrl+Z / Y). Infer / Label Assistant / Train / Export
   buttons are there but disabled until steps 4 and 5. The desktop's **Stream** button is not in the web (removed on purpose: nothing uses it).
   (`utils/web/editing.py` = `Workspace`, `editor_api.py`, `static/editor.js`; tests `tests/unit/test_web_editor.py`;
   real-browser run in the scratchpad, see CLAUDE.md)
4. [x] **Label Assistant**: the four modes through one context-taking core (`utils/web/assistant.py`: `Ctx`, `build_predictor`,
   `infer_one`, `autoannotate`; routes in `assistant_api.py`; the desktop `inferenceObjectDetection` is untouched). Infer
   (G / Infer window: one picture or the whole dataset) sends the annotations on screen and gets back only the NEW ones
   (same merge as the desktop; undoable; marked "unreviewed" until edited, so SAM 2 does not learn from them). Auto-annotate
   all is a job with Stop and a summary. A model that has to be downloaded asks first (409 `needs_confirm`). One model job
   at a time, server-wide (409 `busy` names who). Settings windows for all four modes (targets, own model + class mapping,
   LocateAnything, head + detector, SAM 2) with a server-side file browser for `.pt` files. SAM 2 references are one
   `ReferenceStore` per workspace (`assistant.sam_store`), fed by reviewed saves, emptied by delete image / Forget all.
   Not in the web: uploading a `.pt` from the laptop (pick a file already on the server). The Stream page is not in the web either.
5. [x] **Training and export**: Export Dataset (YOLO / Pascal VOC / COCO: split bar, Auto-Orient / Resize, the whole augmentation picker
   with tile previews, results are only downloaded to the browser (no "Save to", nothing kept on the server), a folder as a .zip), Export Model (all Ultralytics formats and options, result downloadable), the Train window (full YOLO in every mode,
   head choice in Custom Model mode, counts only on Refresh, base-model download and "replace the trained model" asked first,
   epoch table, Finish early, Stop now, installs the best model and selects it). The exporters and the training helpers were
   moved out of the Tk code into `utils/dataset_export.py` and `utils/train_support.py` (the desktop calls the same functions);
   web side: `utils/web/exporting.py`, `training.py` (`TrainingManager`: one training at a time server-wide, it belongs to the
   server and goes on when the tab is closed), `export_api.py`; page: `static/export.js`, `static/train.js`. Tests:
   `tests/unit/test_web_export.py`, `test_web_training.py` (a fake process that speaks the real `@@ {json}` protocol); a REAL
   CPU training with a local .pt as base model ran through the browser. Not in the web: upload a .pt from the laptop.
6. [x] **Installers and service**: `install.ps1` / `install.sh` (the same one-liners) ask on every install and update "App + web" or "App only" (`JELIBOX_WEB`, default = last answer); yes installs `requirements-web.txt`, registers and starts the server (Windows: Startup shortcut running hidden `pythonw`, at sign in; Linux: systemd user unit + linger), no removes it; updates stop the old server first; the firewall is only explained, never changed. `tools/web_service.ps1` / `.sh` (install, remove, start, stop, status) can be run by hand. Tests: `tests/repo/test_web_install.py`; a real Windows install / restart / stop / remove cycle was run with a hidden pythonw (scratch folder); Linux is only tested with a fake systemctl.
7. [x] **Docs and release**: README ("Web version", installer question, service, firewall, security), `index.html` (`#web` section, install notes), CHANGELOG, CLAUDE.md. Not done on purpose: `utils.__version__ = 1.0.0` and the release itself (`tools/release.py` does the bump when the maintainer asks).

Every step ends with `python tests/run.py` green and a changelog line; nothing is committed or released until asked.
