# Multimodal Deception Detection Pipeline

Strict OpenFace remediation for the capstone pipeline at `E:\deception_pipeline`.

This repo now treats any feature cache without strict OpenFace metadata as provisional. Existing cached features, checkpoints, XAI figures, and `working/models/cross_dataset_results.json` should be overwritten before you use results in the paper or dashboard.

## Official runtime

Use this repo in two ways:

1. Keep the workspace on Windows at `E:\deception_pipeline`.
2. Run the official extraction and recommended full notebook workflow from WSL2 Ubuntu at `/mnt/e/deception_pipeline`.

Native Windows is still fine for code inspection or for training/evaluation on already regenerated official caches, but it is not the supported path for strict OpenFace extraction.

## Prerequisites

### 1. Install WSL2 Ubuntu

Run this from an elevated PowerShell session:

```powershell
wsl.exe --install -d Ubuntu
```

Reboot if Windows asks for it, then open Ubuntu and finish the first-run setup.

### 2. Install Conda or Miniconda with Python 3.10 support

The pipeline environment expects `conda`. If `conda --version` fails, install Miniconda or Anaconda first.

### 3. Create the Python environment

From either Windows PowerShell or WSL, in the repo root:

```bash
conda env create -f environment.yml
conda activate deception
python -m ipykernel install --user --name deception --display-name "Deception (Python 3.10)"
```

The pipeline defaults to CPU. If you later add a supported accelerator, override with `DECEPTION_DEVICE=cuda`.

## OpenFace build in WSL2 Ubuntu

Open Ubuntu, then build OpenFace against the shared repo path:

```bash
cd /mnt/e/deception_pipeline

sudo apt-get update
sudo apt-get install -y \
    build-essential cmake git wget \
    libopencv-dev \
    libopenblas-dev liblapack-dev libatlas-base-dev \
    libboost-all-dev \
    libgtk2.0-dev pkg-config \
    libavcodec-dev libavformat-dev libswscale-dev \
    ffmpeg

git clone --depth 1 https://github.com/TadasBaltrusaitis/OpenFace.git OpenFace

dpkg -L libopencv-dev | grep OpenCVConfig.cmake

mkdir -p OpenFace/build
cd OpenFace/build
cmake -D CMAKE_BUILD_TYPE=RELEASE \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D OpenCV_DIR=/usr/lib/x86_64-linux-gnu/cmake/opencv4 \
      ..
make -j"$(nproc)"

cd ..
bash download_models.sh

./build/bin/FeatureExtraction -help
```

The most common build failure is missing C++ OpenCV headers. `libopencv-dev` must be installed before running `cmake`.

## Dataset layout

The repo is expected to look like this:

```text
E:\deception_pipeline
|-- pipeline/
|-- notebooks/
|-- working/
|-- dolos-dataset/
|   |-- videos/
|   `-- labels.csv
`-- real-life-trial/
    |-- videos/
    `-- labels.csv
```

`labels.csv` format:

```csv
video_id,label
BRI_WILTY_EP60_lie_19,1
SB_WILTY_EP42_truth13,0
trial_lie_028,1
trial_truth_056,0
```

## Official run order

After OpenFace is built in WSL and the Python environment exists:

```bash
cd /mnt/e/deception_pipeline
conda activate deception
jupyter notebook notebooks/deception_pipeline.ipynb
```

Run cells 1 through 16 in order.

What happens on the first strict run:

- The notebook checks for a native OpenFace binary.
- Any legacy cache without `features_meta.json` is treated as provisional and rebuilt.
- Strict OpenFace features are extracted and saved with metadata.
- DOLOS and Trial models are retrained.
- Cross-dataset evaluation, SHAP, saliency, ablation, and artifact export are rerun.

What happens on later runs:

- Official caches with OpenFace metadata are reused.
- Feature extraction is skipped unless you force a rebuild.

## Strict cache behavior

Each official dataset cache now includes:

- `working/features/dolos/features_meta.json`
- `working/features/trial/features_meta.json`

The pipeline only trusts a cache when that metadata declares:

- `video_backend = "openface"`
- `feature_count = 92`
- `cache_schema_version = 2`

Any old cache without metadata is refused on purpose.

## Key outputs

After a complete strict run, these artifacts are the ones to keep:

- `working/models/deception_model.pth`
- `working/models/deception_model_dolos.pth`
- `working/models/deception_model_trial.pth`
- `working/models/feature_scaler.pkl`
- `working/models/feature_scaler_dolos.pkl`
- `working/models/feature_scaler_trial.pkl`
- `working/models/feature_names.json`
- `working/models/cross_dataset_results.json`
- `working/xai/shap_summary.png`
- `working/xai/cross_dataset_summary.png`
- `working/xai/single_sample_xai.png`

## Notes

- The repo now supports both strict OpenFace mode (via WSL) and OpenCV proxy fallback.

```powershell
python verify_imports.py
```
