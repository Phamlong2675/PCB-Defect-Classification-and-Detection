# PCB Defect Detection

PCB defect detection and classification for six defect types: open, short, mousebite, spur, copper, and pin-hole.

The repository contains an EDA notebook, the team's Simple CNN, Complex CNN, and ResNet18 classification scripts, and three YOLOv8 detection experiments: YOLOv8n, YOLOv8s, and YOLOv8s-P2. The YOLO scripts use a frozen-backbone transfer stage followed by fine-tuning, with three recorded random seeds.

## Repository layout

```text
Deep-Learning-PCB-Defect-Detection/
  Data/
    processed/
      classification/              Current classification dataset
      detection/                   YOLO dataset: 1,199 train / 150 valid / 150 test
  PCB_learning/
    README.md                      Detailed training and testing instructions
    yolo_pcb_aug.py                 YOLOv8n
    Yolo8s_pcb_aug.py               YOLOv8s
    Yolo8p2_pcb_aug.py              YOLOv8s-P2
    pcb_augmentation.py            Right-angle rotation and saved seeds
    test_pcb.py                     Test one model / phase / seed
    pcb_aug_seed_run1.txt
    pcb_aug_seed_run2.txt
    pcb_aug_seed_run3.txt
  EDA.ipynb
  simple_cnn_model.py
  complex_cnn.py
  resnet18.py
  predict_complex_cnn.py
  predict_resnet18.py
  requirements.txt
```

New YOLO checkpoints, virtual environments, machine-specific configuration, and generated training/testing outputs are local files, excluded from Git. Existing files committed by other contributors are retained.

## Installation

Use Python 3.12, the Python minor version used for the recorded Windows runs. Run the following commands from the repository root in VS Code's PowerShell terminal:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install a matching **torch + torchvision** pair for your operating system and GPU using the [official PyTorch installation selector](https://pytorch.org/get-started/locally/). Run its installation command inside the activated environment before installing the remaining requirements. Choose a CUDA build compatible with your NVIDIA driver; the original machine's CUDA build is not a universal requirement.

Then install the project dependencies:

```powershell
python -m pip install -r requirements.txt
python -c "import torch, torchvision, ultralytics, albumentations, yaml; print('torch:', torch.__version__); print('torchvision:', torchvision.__version__); print('ultralytics:', ultralytics.__version__); print('CUDA:', torch.cuda.is_available())"
```

The YOLO-specific dependencies are `ultralytics==8.4.174`, `albumentations==2.0.8`, and `PyYAML==6.0.3`, alongside PyTorch/torchvision. The other requirements provide the shared data-analysis libraries. `complex_cnn.py` and `simple_cnn_model.py` additionally import TensorFlow; the Simple CNN also imports seaborn. Install those dependencies in an appropriate environment for these separate classification models.

The recorded local environment was Python 3.12.10, torch 2.14.0+cu130, torchvision 0.29.1+cu130, and an NVIDIA RTX 3060 Laptop GPU. These are environment records, not a requirement to install the same CUDA wheel on every computer.

The scripts automatically use the first CUDA GPU when available, otherwise the CPU. Training on CPU is considerably slower. The recorded settings retain `workers=0` and `amp=False`.

## Dataset and preprocessing

The current repository tracks `Data/processed/classification/` and `Data/processed/detection/`. The classification export contains defect crops; the detection export contains PCB images and normalized YOLO bounding boxes. Keep existing splits and class IDs when comparing models.

The detection dataset contains **1,199 training images, 150 validation images, and 150 test images** at 640 x 640. One faulty training image was removed from the earlier 1,200-image export. Use the corrected detection dataset for new runs; no historical dataset restoration is needed.

Processed YOLO labels contain normalized `class_id x_center y_center width height` values. The scripts read `Data/processed/detection/data.yaml` and automatically replace its machine-specific dataset path with the path of the current clone. They write a local, Git-ignored `PCB_learning/pcb_640_local.yaml`; no manual path edits are required.

Previously recorded YOLO metrics came from the earlier export. Removing one image may have a small effect, but its impact has not been measured; new runs on the corrected dataset may yield different metrics. The current analysis notebook is `EDA.ipynb`.

## Run the YOLO experiments

Follow [PCB_learning/README.md](PCB_learning/README.md) to:

1. Install dependencies and keep the included `Data/processed/detection/` dataset. Pretrained YOLOv8 weights download automatically on first transfer training (Internet access required).
2. Select a model script, set `MODE="transfer"` and `RUN_ID`, and train the frozen-backbone stage.
3. Set `MODE="finetune"` with the same `RUN_ID` to continue from its transfer `best.pt`.
4. Set `MODEL`, `PHASE`, and `SEED` in `test_pcb.py` to evaluate one checkpoint on the test split.

The repository includes the **two best YOLOv8s checkpoints**, selected independently using validation mAP@0.5:0.95: transfer Run 3 and fine-tuning Run 1. Their original training results are included in the same run folders. See [the selected results](reports/yolo/README.md) for checkpoint details and freshly evaluated test results.

After installing dependencies, test the included fine-tuned checkpoint directly from the repository root:

```powershell
.\.venv\Scripts\python.exe .\PCB_learning\test_pcb.py
```

No training or pretrained-weight download is needed for this test. The script defaults to YOLOv8s fine-tuning Run 1 and automatically chooses CUDA or CPU. To test the included transfer checkpoint, set `PHASE="transfer"` and `SEED=2052798190` in that script; keep `MODEL="yolov8s"`.

Training scripts default to `MODE="transfer"`, `RUN_ID=1` for a fresh clone. Change these two values to choose another stage or run.

## What belongs in Git

Commit source code, both READMEs, `requirements.txt`, `.gitignore`, and the three saved seed files. Keep the existing tracked datasets.

The only published YOLO checkpoints are `best.pt` in `PCB_learning/runs/yolov8s_640_aug_frozen_run3/` and `PCB_learning/runs/yolov8s_640_aug_finetune_run1/`, together with their training results. `.gitignore` makes explicit exceptions for these two runs.

Keep other checkpoints, `last.pt`, `.venv/`, new `runs/` and `testing/` outputs, label caches, and generated checkpoint-pointer text files local. `pcb_640_local.yaml` is generated automatically for each clone and contains a machine-specific absolute path. The selected test CSV tables, curves, confusion matrices, and sample predictions are published under `reports/yolo/test/`.
