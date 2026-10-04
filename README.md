# Jelibox

**Jelibox is a local computer vision annotation tool for creating object detection and image segmentation datasets.** It runs on your own computer, supports bounding boxes and polygons, and can use Ultralytics YOLO models for inference and training.

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Platform: Linux](https://img.shields.io/badge/platform-Linux-lightgrey)](jelibox_linux_installation.bash)
[![Platform: Windows](https://img.shields.io/badge/platform-Windows-lightgrey)](jelibox_windows_installation.bat)

Jelibox is designed for individuals and teams that need a private, offline-first workflow for labeling image datasets without uploading images to a third-party service.

![Jelibox annotation interface](assets/visualize.png)

## Features

- Local annotation workflow with no required cloud account
- A VS Code-style workspace picker for switching between datasets
- Bounding box annotation for object detection
- Polygon annotation for image segmentation
- AI-assisted annotation with an Ultralytics YOLO model
- Model training from the annotation workspace, reading images straight from `datasetsInput/` (no duplicate copies)
- NVIDIA CUDA and CPU workflows, depending on the installed PyTorch build
- Class management, visibility toggles, image search, zoom, and multi-selection
- Repeat annotations from the previous image
- ZeroFill masking for removing sensitive image regions locally
- Dataset export for YOLO and Pascal VOC XML
- Live camera or video inference through Streamlit

## Requirements

### Linux

- Debian/Ubuntu/Mint, Fedora/RHEL/CentOS, or Arch/Manjaro
- Python **3.11.9** for the pinned installer
- Tkinter and Python virtual-environment support
- Internet access during installation
- An NVIDIA driver and `nvidia-smi` for the CUDA PyTorch path

### Windows

- 64-bit Windows is recommended
- Python 3.12 is installed by the Windows installer when needed
- Microsoft Visual C++ Redistributable from [`VC_redist/`](VC_redist/)
- Internet access during installation

The application can run on CPU. NVIDIA GPU support requires a compatible NVIDIA driver and a PyTorch build with CUDA support.

## Installation

### Linux

1. Install git (if you haven't install git)
```bash
sudo apt install git
```
2. Clone this repository
```bash
git clone https://github.com/Jelibox/Jelibox-client.git 
```
3. Add access to linux installation script
```bash
chmod +x jelibox_linux_installation.bash
```
4. Run Jelibox installation script
```bash
./jelibox_linux_installation.bash
```

The installer asks for your `sudo` password at the beginning, installs Python with Tkinter and venv support, creates the `jelibox/` virtual environment, installs dependencies, and creates a desktop launcher.

### Windows

1. Open [`VC_redist/`](VC_redist/) and install the package matching your system architecture.
2. Right-click [`jelibox_windows_installation.bat`](jelibox_windows_installation.bat) and choose **Run as administrator**.
3. Follow the installer prompts.
4. Open the generated `Jelibox.lnk` shortcut.

The Windows installer checks for an NVIDIA GPU through `nvidia-smi` and uses Windows device information as a fallback. GPU detection does not guarantee that the installed PyTorch package has CUDA enabled.

## Running Jelibox Manually

### Linux

```bash
source jelibox/bin/activate
python -u utils/Annotator.py
```

### Windows

```bat
venv\Scripts\activate
python utils\Annotator.py
```

Running `Annotator.py` with no arguments opens the **workspace picker** - a VS
Code-style "no folder opened" screen that lists every workspace found in
`datasetsInput/`. Folders are grouped by name, so `weapon-1` and `weapon-2`
both appear under one `weapon` workspace; picking an instance launches the
annotation window for it. Inside the annotation window, the **◀** button next
to the Jelibox logo closes that workspace and returns to the picker so you can
switch datasets without restarting the app.

## Keyboard Shortcuts

| Key | Action |
| --- | --- |
| `A` / `Left Arrow` | Previous image |
| `D` / `Right Arrow` | Next image |
| `Delete` | Delete the current image |
| `M` | Toggle bounding box and polygon mode |
| `B` | Force a new bounding box |
| `G` | Run inference on the current image |
| `T` | Open the training workflow |
| `S` | Change the selected annotation class |
| `R` | Delete the selected annotation |
| `E` | Repeat annotations from the previous image |
| `Esc` | Cancel the current operation or exit |

In polygon mode, click to add points, double-click or press `Enter` to finish, and right-click a point to delete it.

## Workspace Structure

Jelibox keeps data, annotations, and models separated by workspace. Images are
never copied - every format reads them straight from `datasetsInput/`:

```text
datasetsInput/<workspace>-<index>/   Input images for annotation (read-only source)
vocdataset/<workspace>/              Pascal VOC XML annotations
YOLOdataset/<workspace>/labels/      YOLO .txt labels (keyed by image filename)
models/<workspace>/                  Trained YOLO models
configs/<workspace>.txt              Workspace class configuration
export dataset/<workspace>/          Exported datasets
export model/<workspace>/            Exported model files
```

`YOLOdataset/<workspace>/` also doubles as scratch space during training: a
`train/`, `val/`, and `data.yaml` are generated there temporarily and removed
once training finishes.

Every time a workspace is opened, Jelibox checks each Pascal VOC XML file in
`vocdataset/<workspace>/` for a matching YOLO label and generates any that are
missing. This keeps `YOLOdataset/` in sync automatically - including a
one-time backfill for datasets annotated before this folder existed - without
requiring you to re-save every image. Already-synced workspaces open
instantly; only the first sync of a large, previously-unsynced dataset takes
noticeably longer, and a progress screen is shown while it runs.

Example workspace:

```text
datasetsInput/cat-2/
vocdataset/cat/
YOLOdataset/cat/
models/cat/
configs/cat.txt
```

## Annotation Formats

Jelibox supports:

- Bounding boxes for YOLO object detection datasets
- Polygons for segmentation workflows
- Pascal VOC XML annotations in the `vocdataset/` workspace folder
- YOLO labels in the `YOLOdataset/<workspace>/labels/` folder, paired with images from `datasetsInput/` at training/export time
- COCO JSON YOLO exports with bounding boxes and polygon segmentations

## Exporting a Dataset

Use the **Export Dataset** action in the application to export YOLO, Pascal VOC, or COCO data with train, validation, and test splits.

COCO exports use this structure:

```text
<workspace>/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── annotations/
	├── instances_train.json
	├── instances_val.json
	└── instances_test.json
```

The COCO exporter reads images from all indexed folders matching `datasetsInput/<workspace>-<index>/` and annotations from `vocdataset/<workspace>/`. The YOLO exporter does the same for images, pairing each one with its label in `YOLOdataset/<workspace>/labels/`.

The standalone YOLOX conversion utility can be run with:

```bash
python exportTools/export2YOLOX.py
```

## GPU and PyTorch Troubleshooting

Check the NVIDIA driver first:

```bash
nvidia-smi
```

Then check whether the active Python environment can use CUDA:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda)"
```

`nvidia-smi` detecting a GPU does not mean that PyTorch has CUDA enabled. The driver, PyTorch build, CUDA runtime, and GPU architecture must be compatible.

For training failures caused by limited memory, try a smaller image size, a smaller batch size, a smaller model, or a smaller dataset split.

## Screenshots

![Jelibox annotation interface](assets/visualize.png)

## Contributing

Issues, documentation improvements, bug fixes, and feature contributions are welcome.

```bash
git clone https://github.com/Jelibox/Jelibox-client.git
cd Jelibox-client
git checkout -b feature/your-feature-name
```

Before opening a pull request, test the installer or application flow affected by your change and describe the platform used.

## Testing

One command checks everything (no data of yours is touched - tests run in a temp workspace):

```bash
python tests/run.py            # repo hygiene + unit + integration + GUI
python tests/run.py --fast     # skip the GUI group (works without a display)
python tests/run.py --only unit,repo
python tests/run.py -k assistant -v
```

Run it with the project's virtual environment (`venv\Scripts\python` on Windows, `jelibox/bin/python` on Linux). It uses only the standard `unittest` module, so there is nothing extra to install.

## License

Jelibox is released under the [Apache License 2.0](LICENSE).
