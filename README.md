# EcoSpill Vision

Oil spill detection in images and video using Ultralytics YOLOv8s and YOLO26s. This research project includes training, held-out evaluation, clean-water background extraction, false-alarm analysis, and an FPS benchmark for deployment experiments.

## Included in this repository

- YOLOv8s baseline, fine-tuned YOLOv8s, and YOLO26s checkpoints under [`runs/`](runs/).
- Training histories, evaluation plots, and example false-alarm artifacts.
- Scripts for training, inference, evaluation, and performance measurement.

Datasets, input videos, and TensorRT engines are not included. Run the commands below from the repository root.

## Installation

Use Python 3.11 for a new desktop environment. Training, evaluation, and video scripts select NVIDIA GPU `device=0`, so they require an NVIDIA driver and CUDA-enabled PyTorch. For CPU use, change these scripts to `device="cpu"`; processing will be slower.

```powershell
git clone https://github.com/raxatraxat35-arch/ecospill.git
cd ecospill
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

On Linux/macOS, activate with `source .venv/bin/activate`. Install matching `torch` and `torchvision` builds using the command from the [official PyTorch installer](https://pytorch.org/get-started/locally/) for your hardware, then run:

```powershell
python -m pip install -r requirements.txt
python -m pip check
python check_gpu.py
```

The GPU check should report CUDA availability and your GPU name. If CUDA is unavailable, check the driver and PyTorch build before running GPU scripts. Dependency groups are documented in [`requirements/`](requirements/).

## Quick inference

Place an input video at `test_video.mp4`, then run the supplied YOLO26s checkpoint:

```powershell
python "test_video 26.py"
```

This uses confidence `0.45`, GPU 0, and a live preview. Annotated output is saved beneath `runs/detect/`; Ultralytics prints the exact directory. To predict on an image or a custom source:

```powershell
yolo predict model="runs/detect/runs/detect/baseline_yolo26s/weights/best.pt" source="path/to/image.jpg" conf=0.45 device=0 save=True
```

For a machine without a display, set `show=False` in the video script. The CLI example does not enable a preview.

## Download the dataset

[`downlaod_data.py`](downlaod_data.py) downloads Roboflow workspace `prueba-bl8vt`, project `oil-spill-sfzvs`, version **6**, in YOLOv8 format. The filename is spelled `downlaod_data.py` in this repository.

Use a separate download environment. Roboflow installs `opencv-python-headless`; the main environment needs GUI-enabled `opencv-python` for video previews. Both provide `cv2` and should not be installed together.

```powershell
python -m venv .venv-data
.\.venv-data\Scripts\python.exe -m pip install -r requirements/data.txt
```

Replace the placeholder `Key` in `downlaod_data.py` with your Roboflow API key locally, then run:

```powershell
.\.venv-data\Scripts\python.exe downlaod_data.py
```

On Linux/macOS, use `.venv-data/bin/python` instead. Restore the placeholder before committing; never publish your API key. Dataset access may require Roboflow permissions. Expected layout:

```text
data/oil_spill/
  data.yaml
  train/
    images/
    labels/
  valid/
    images/
    labels/
  test/
    images/
    labels/
```

Check that `data.yaml` paths and class names match the local export. Evaluation requires a `test` split kept separate from training data.

## Train and evaluate

Activate the main `.venv` before running these commands.

| Experiment | Train | Evaluate |
| --- | --- | --- |
| YOLOv8s baseline | `python train_baseline.py` | `python eval_test.py` |
| Fine-tuned YOLOv8s | `python finetune.py` | `python eval_test1.py` |
| YOLO26s baseline | `python train_yolo26s.py` | `python eval_yolo26s.py` |

Both baseline scripts request 100 epochs, image size 640, and batch size 8. YOLO26s also uses early stopping with patience 20. Fine-tuning requests 40 epochs and learning rate `0.001`. Reduce the batch size in the relevant script if GPU memory is insufficient. Ultralytics downloads initial pretrained weights when needed.

For the second stage, the original experiment added 300 labeled images from Roboflow project `oil-spill-ujfym` across training and validation. Its workspace, version, and exact split are not recorded here, so this stage cannot be reproduced exactly from the repository alone. Keep added training images separate from the held-out test set.

To add clean-water backgrounds, place `clean_water.<extension>` and `clean_water1.<extension>` in the root, then run:

```powershell
python extract_backgrounds.py
python finetune.py
```

The extractor saves up to 30 resized frames per video into the training split with empty label files. Running it again uses the same output filenames.

### Checkpoint paths

The scripts currently reference these supplied checkpoints:

| Model | Checkpoint |
| --- | --- |
| YOLOv8s baseline | `runs/detect/runs/baseline_yolov8s/weights/best.pt` |
| Fine-tuned YOLOv8s | `runs/detect/runs/detect/finetuned_yolov8s-2/weights/best.pt` |
| YOLO26s baseline | `runs/detect/runs/detect/baseline_yolo26s/weights/best.pt` |

After retraining, use the save directory printed by Ultralytics and update checkpoint paths in evaluation, inference, and fine-tuning scripts. New paths can differ from these historical paths, and repeated runs can receive numeric suffixes.

## Video scripts

| Script | Model | Expected input |
| --- | --- | --- |
| `test_video.py` | Fine-tuned YOLOv8s | `test_video.mp4` |
| `test_video1.py` | Fine-tuned YOLOv8s | `test_video1.mp4` |
| `test_video2.py` | Baseline YOLOv8s | `test_video2.mp4` |
| `test_video 26.py` | YOLO26s | `test_video.mp4` |
| `test_video2 26.py` | Fine-tuned YOLOv8s, despite its filename | `test_video2.mp4` |
| `test_clean_video.py` | Fine-tuned YOLOv8s | `clean_video1.mp4` |
| `test_clean_video1.py` | YOLO26s | `clean_video.mp4` |

Quote filenames containing spaces. Edit `source` and the checkpoint path to test other files. `python test_model.py` prints baseline YOLOv8s model information.

## Measure false alarms

Use a video or image directory known to contain **no oil spills**:

```powershell
python false_positive.py --weights runs/detect/runs/detect/baseline_yolo26s/weights/best.pt --source clean_video1.mp4 --conf 0.45 --stride 5
```

This measures the fraction of sampled frames containing any detection, not per-box precision. `--stride 5` samples every fifth video frame. `--limit` defaults to `0.20`. The CSV and annotated flagged frames go to `fp_test_results/`; use `--out` to preserve the committed example results. `PASS`/`FAIL` is printed but does not set the process exit status. The CSV `frame` column counts sampled results, not original video frame numbers.

The committed log contains **5 flagged frames out of 131 sampled frames (3.8%)**. It does not record the model, stride, or full configuration, so treat it as an example artifact rather than a reproducible model comparison.

## FPS benchmark and Jetson deployment

The benchmark file is named [`fps test`](fps%20test), with a space and no `.py` extension:

```powershell
python "fps test" --weights runs/detect/runs/detect/baseline_yolo26s/weights/best.pt --source test_video.mp4 --seconds 60
```

After 20 warm-up frames, it measures video reading, inference, and post-processing. It reports mean FPS, the lowest completed one-second interval, median frame time, and a CSV (`fps_log.csv` by default). The default target is 20 FPS. File sources loop; use `--source 0` for a camera. Preview rendering and video encoding are not measured.

For TensorRT experiments, install the PyTorch/TensorRT stack appropriate to your JetPack release on the target Jetson, then export on that device:

```bash
yolo export model="runs/detect/runs/detect/baseline_yolo26s/weights/best.pt" format=engine half=True device=0
python "fps test" --weights runs/detect/runs/detect/baseline_yolo26s/weights/best.engine --source test_video.mp4 --seconds 60
```

TensorRT and Jetson-specific packages are outside the desktop requirements. Follow the [Ultralytics Jetson guide](https://docs.ultralytics.com/guides/nvidia-jetson/) for the target environment. Measure performance on the actual deployment hardware before making real-time claims.

## Results and reproducibility

Earlier project notes reported approximately 78.5% test mAP@50 for baseline YOLOv8s, 44.9% for fine-tuned YOLOv8s, and 79.8% for YOLO26s. These figures were not independently rerun during documentation preparation. The committed training `results.csv` files contain training-time validation metrics, not fresh held-out test measurements.

Compare models using the same held-out dataset and record checkpoints, settings, and evaluation output. Dataset access, missing second-stage dataset details, hard-coded paths, and hardware differences limit exact reproduction.

## Project layout

```text
requirements.txt       Main environment entry point
requirements/          Runtime and dataset-download dependencies
downlaod_data.py        Roboflow dataset download
train_*.py             Baseline training
finetune.py            Second-stage YOLOv8s training
eval_*.py              Held-out test evaluation
test*.py               Model inspection and video demonstrations
extract_backgrounds.py Clean-water training frame extraction
false_positive.py      Frame-level false-alarm analysis
fps test               Sustained FPS benchmark
runs/                  Checkpoints, training histories, and plots
fp_test_results/       Example false-alarm artifacts
```

## References

- [Ultralytics documentation](https://docs.ultralytics.com/) and [installation guide](https://docs.ultralytics.com/quickstart/)
- [PyTorch installation](https://pytorch.org/get-started/locally/)
- [Roboflow Python SDK](https://github.com/roboflow/roboflow-python)

No project license file is included. Check the applicable model, library, and dataset terms before redistribution or deployment.
