"""Central configuration for the deception detection pipeline."""

from __future__ import annotations

import os
from pathlib import Path

import torch


def windows_to_wsl_path(path: str) -> str:
    """Convert ``E:\\repo`` style paths to ``/mnt/e/repo`` for WSL use."""
    normalized = os.path.abspath(path)
    drive, tail = os.path.splitdrive(normalized)
    if not drive:
        return normalized.replace("\\", "/")
    return f"/mnt/{drive[0].lower()}{tail.replace('\\', '/')}"


PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
WSL_PROJECT_ROOT = windows_to_wsl_path(PROJECT_ROOT)

BASE_DIR = os.path.join(PROJECT_ROOT, "working")
DATA_DIR = PROJECT_ROOT

DOLOS_VIDEO_DIR = os.path.join(DATA_DIR, "dolos-dataset", "videos")
DOLOS_LABEL_CSV = os.path.join(DATA_DIR, "dolos-dataset", "labels.csv")
TRIAL_VIDEO_DIR = os.path.join(DATA_DIR, "real-life-trial", "videos")
TRIAL_LABEL_CSV = os.path.join(DATA_DIR, "real-life-trial", "labels.csv")

FEATURES_DIR = os.path.join(BASE_DIR, "features")
DOLOS_FEAT_DIR = os.path.join(FEATURES_DIR, "dolos")
TRIAL_FEAT_DIR = os.path.join(FEATURES_DIR, "trial")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "deception_model.pth")
SCALER_PATH = os.path.join(MODEL_DIR, "feature_scaler.pkl")
FEAT_NAMES_PATH = os.path.join(MODEL_DIR, "feature_names.json")

OPENFACE_BIN_DIR = os.path.join(PROJECT_ROOT, "OpenFace", "build", "bin")
OPENFACE_BIN = os.path.join(
    OPENFACE_BIN_DIR,
    "FeatureExtraction.exe" if os.name == "nt" else "FeatureExtraction",
)
OPENFACE_WSL_BIN = f"{WSL_PROJECT_ROOT}/OpenFace/build/bin/FeatureExtraction"
OPENFACE_OUTDIR = os.path.join(BASE_DIR, "openface_raw")
XAI_DIR = os.path.join(BASE_DIR, "xai")

FEATURE_BACKEND = "mediapipe_blendshapes"
REQUIRE_OPENFACE = False
FEATURE_CACHE_SCHEMA_VERSION = 2

OPENFACE_AU_COLS = [
    "AU01_r",
    "AU02_r",
    "AU04_r",
    "AU05_r",
    "AU06_r",
    "AU07_r",
    "AU09_r",
    "AU10_r",
    "AU12_r",
    "AU14_r",
    "AU15_r",
    "AU17_r",
    "AU20_r",
    "AU23_r",
    "AU25_r",
    "AU26_r",
    "AU45_r",
    "pose_Tx",
    "pose_Ty",
    "pose_Tz",
    "pose_Rx",
    "pose_Ry",
    "pose_Rz",
    "gaze_0_x",
    "gaze_0_y",
    "gaze_1_x",
    "gaze_1_y",
    "gaze_angle_x",
    "gaze_angle_y",
]

SAMPLE_RATE = 16000
N_MFCC = 13
HOP_LENGTH = 512
N_FFT = 2048

VIDEO_INPUT_DIM = 104
AUDIO_INPUT_DIM = 34
FUSION_DIM = 128
HIDDEN_DIM = 64
NUM_CLASSES = 2
DROPOUT = 0.4

BATCH_SIZE = 32
EPOCHS = 60
LR = 3e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 15
RANDOM_SEED = 42
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
FEW_SHOT_FRAC = 0.10
LABEL_SMOOTH = 0.1

DEVICE = os.environ.get("DECEPTION_DEVICE", "cpu").strip().lower() or "cpu"
if DEVICE == "cuda" and not torch.cuda.is_available():
    DEVICE = "cpu"

LABEL_MAP = {0: "Truthful", 1: "Deceptive"}

for directory in [
    FEATURES_DIR,
    DOLOS_FEAT_DIR,
    TRIAL_FEAT_DIR,
    MODEL_DIR,
    OPENFACE_OUTDIR,
    XAI_DIR,
]:
    os.makedirs(directory, exist_ok=True)
