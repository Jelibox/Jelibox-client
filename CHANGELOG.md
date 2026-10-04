# Changelog

All notable changes to Jelibox are written here, newest first.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/): `MAJOR.MINOR.PATCH`.

Add a line under **[Unreleased]** in every pull request that a user would notice.
`python tools/release.py` turns that section into a version when you cut a release (see [RELEASING.md](RELEASING.md)).

## [Unreleased]

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

[Unreleased]: https://github.com/Jelibox/Jelibox-client/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Jelibox/Jelibox-client/releases/tag/v0.1.0
