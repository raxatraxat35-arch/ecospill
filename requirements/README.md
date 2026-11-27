# Dependency environments

| File | Purpose | Install command from the repository root |
| --- | --- | --- |
| `../requirements.txt` | Main training, evaluation, inference, and benchmark environment | `python -m pip install -r requirements.txt` |
| `base.txt` | Runtime dependencies included by the root file | `python -m pip install -r requirements/base.txt` |
| `data.txt` | Roboflow downloader in its own environment | `.venv-data/Scripts/python.exe -m pip install -r requirements/data.txt` |

Use Python 3.11 for a new desktop environment. Install matching `torch` and `torchvision` builds for your hardware using the [official PyTorch installer](https://pytorch.org/get-started/locally/) before the main requirements. Lower bounds reflect Ultralytics metadata, not a guarantee that every older build works on every device. The Windows exclusion for Torch 2.4.0 is applied across platforms here to keep one common requirements file.

Ultralytics stays at the repository's existing `8.4.212` version. Roboflow is pinned independently for the download environment. Other support packages are resolved by pip; these files are dependency specifications, not a complete environment lock. Run `python -m pip check` after installation and record `python -m pip freeze` with experiment results when exact reproduction is needed.

Keep Roboflow separate because its `opencv-python-headless` dependency and the runtime's `opencv-python` both install `cv2`. Do not install both variants in the main environment. On Linux/macOS the downloader interpreter is `.venv-data/bin/python`.

TensorRT, CUDA platform packages, and Jetson-specific PyTorch wheels depend on hardware and JetPack and are not installed by these files. See the root [README](../README.md) for deployment notes.
