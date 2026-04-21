"""Quick import and environment readiness check for the pipeline."""

import os
import sys

os.environ["PYTHONIOENCODING"] = "utf-8"

print(f"Python: {sys.version}")

errors = []
for module_name in [
    "torch",
    "librosa",
    "shap",
    "sklearn",
    "pandas",
    "numpy",
    "seaborn",
    "matplotlib",
    "joblib",
    "tqdm",
    "soundfile",
]:
    try:
        __import__(module_name)
        print(f"  OK: {module_name}")
    except ImportError as exc:
        print(f"  FAIL: {module_name}: {exc}")
        errors.append(module_name)

import torch

print(f"\nPyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Configured device default: {os.environ.get('DECEPTION_DEVICE', 'cpu')}")

sys.path.insert(0, r"e:\deception_pipeline")

try:
    from pipeline.config import DEVICE, DOLOS_VIDEO_DIR, PROJECT_ROOT, TRIAL_VIDEO_DIR
    from pipeline.feature_extraction import OPENFACE_AVAILABLE, get_openface_binary_path

    print(f"\nProject root: {PROJECT_ROOT}")
    print(f"Resolved device: {DEVICE}")
    print(f"OpenFace binary: {get_openface_binary_path()}")
    print(f"OpenFace available: {OPENFACE_AVAILABLE}")
    print(
        f"DOLOS videos: {len([f for f in os.listdir(DOLOS_VIDEO_DIR) if f.endswith('.mp4')])}"
    )
    print(
        f"Trial videos: {len([f for f in os.listdir(TRIAL_VIDEO_DIR) if f.endswith('.mp4')])}"
    )
    print("  OK: pipeline.config")
except Exception as exc:
    print(f"  FAIL: pipeline.config: {exc}")
    errors.append("pipeline.config")

for module_name, import_stmt in [
    ("pipeline.model", "from pipeline.model import DeceptionDetector, count_parameters"),
    ("pipeline.feature_extraction", "from pipeline.feature_extraction import build_feature_names"),
    ("pipeline.dataset", "from pipeline.dataset import DeceptionDataset"),
    ("pipeline.train", "from pipeline.train import train"),
    ("pipeline.evaluate", "from pipeline.evaluate import protocol_A, protocol_B"),
    ("pipeline.xai", "from pipeline.xai import plot_shap_summary, explain_single_sample"),
]:
    try:
        namespace = {}
        exec(import_stmt, namespace)
        if module_name == "pipeline.model":
            model = namespace["DeceptionDetector"]()
            count_parameters = namespace["count_parameters"]
            print(f"  OK: {module_name} ({count_parameters(model):,} params)")
        elif module_name == "pipeline.feature_extraction":
            names = namespace["build_feature_names"]()
            print(f"  OK: {module_name} ({len(names)} features)")
        else:
            print(f"  OK: {module_name}")
    except Exception as exc:
        print(f"  FAIL: {module_name}: {exc}")
        errors.append(module_name)

if errors:
    print(f"\nFAILED: {len(errors)} import checks failed: {errors}")
else:
    print("\nSUCCESS: All imports OK")
    if "OPENFACE_AVAILABLE" in globals() and OPENFACE_AVAILABLE:
        print("Strict OpenFace extraction is ready.")
    else:
        print("Strict OpenFace extraction is NOT ready until the binary is built.")
