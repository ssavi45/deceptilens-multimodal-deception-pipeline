"""Run the full deception detection pipeline outside Jupyter."""

from __future__ import annotations

import json
import os
import shutil
import sys
import warnings

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if os.name == "nt":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import torch

from pipeline.config import (
    DEVICE,
    DOLOS_FEAT_DIR,
    DOLOS_LABEL_CSV,
    DOLOS_VIDEO_DIR,
    FEAT_NAMES_PATH,
    FEW_SHOT_FRAC,
    MODEL_DIR,
    MODEL_PATH,
    SCALER_PATH,
    TRIAL_FEAT_DIR,
    TRIAL_LABEL_CSV,
    TRIAL_VIDEO_DIR,
    XAI_DIR,
)
from pipeline.dataset import load_features
from pipeline.evaluate import (
    plot_comparison_chart,
    print_summary_table,
    protocol_A,
    protocol_B,
    save_all_results,
)
from pipeline.feature_extraction import (
    OPENFACE_AVAILABLE,
    build_feature_names,
    extract_dataset_features,
    get_feature_cache_metadata_path,
    get_openface_binary_path,
    is_official_feature_cache,
)
from pipeline.model import DeceptionDetector, count_parameters
from pipeline.train import train
from pipeline.xai import (
    explain_single_sample,
    load_model_and_scaler,
    modality_ablation,
    plot_shap_summary,
    plot_single_sample_explanation,
)


print("=" * 70)
print("MULTIMODAL DECEPTION DETECTION PIPELINE")
print("Strict OpenFace remediation mode")
print("=" * 70)

print("\n[1/16] Verifying paths...")
print(f"Device: {DEVICE}")
print(f"OpenFace binary: {get_openface_binary_path()}")
print(f"OpenFace available: {OPENFACE_AVAILABLE}")
print(f"DOLOS videos: {len([f for f in os.listdir(DOLOS_VIDEO_DIR) if f.endswith('.mp4')])}")
print(f"Trial videos: {len([f for f in os.listdir(TRIAL_VIDEO_DIR) if f.endswith('.mp4')])}")

print("\n[2/16] Importing modules...")
print(f"Model params: {count_parameters(DeceptionDetector()):,}")
print(f"Feature vector size: {len(build_feature_names())}")

if not OPENFACE_AVAILABLE:
    print("\n[3/16] Running with OpenCV fallback (Strict OpenFace mode bypassed).")
else:
    print("\n[3/16] OpenFace smoke test is performed during dataset extraction.")

print("\n[4/16] Extracting DOLOS features...")
dolos_force_rebuild = not is_official_feature_cache(DOLOS_FEAT_DIR)
print(f"Official DOLOS cache present: {not dolos_force_rebuild}")
dolos_df = extract_dataset_features(
    video_dir=DOLOS_VIDEO_DIR,
    label_csv=DOLOS_LABEL_CSV,
    out_feat_dir=DOLOS_FEAT_DIR,
    dataset_name="DOLOS",
    force_rebuild=dolos_force_rebuild,
)
print(f"DOLOS features shape: {dolos_df.shape}")
print(f"DOLOS metadata: {get_feature_cache_metadata_path(DOLOS_FEAT_DIR)}")

print("\n[5/16] Extracting Trial features...")
trial_force_rebuild = not is_official_feature_cache(TRIAL_FEAT_DIR)
print(f"Official Trial cache present: {not trial_force_rebuild}")
trial_df = extract_dataset_features(
    video_dir=TRIAL_VIDEO_DIR,
    label_csv=TRIAL_LABEL_CSV,
    out_feat_dir=TRIAL_FEAT_DIR,
    dataset_name="RealLifeTrial",
    force_rebuild=trial_force_rebuild,
)
print(f"Trial features shape: {trial_df.shape}")
print(f"Trial metadata: {get_feature_cache_metadata_path(TRIAL_FEAT_DIR)}")

print("\n[6/16] EDA...")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for axis, (name, df) in zip(axes, [("DOLOS", dolos_df), ("Real-Life Trial", trial_df)]):
    counts = df["label"].value_counts().sort_index()
    axis.bar(["Truthful", "Deceptive"], counts.values, color=["#34A853", "#EA4335"])
    axis.set_title(name)
    axis.set_ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(XAI_DIR, "class_distribution.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

dolos_feat_csv = os.path.join(DOLOS_FEAT_DIR, "features.csv")
trial_feat_csv = os.path.join(TRIAL_FEAT_DIR, "features.csv")
dolos_model_path = os.path.join(MODEL_DIR, "deception_model_dolos.pth")
dolos_scaler_path = os.path.join(MODEL_DIR, "feature_scaler_dolos.pkl")
trial_model_path = os.path.join(MODEL_DIR, "deception_model_trial.pth")
trial_scaler_path = os.path.join(MODEL_DIR, "feature_scaler_trial.pkl")

print("\n[7/16] Training on DOLOS...")
dolos_results = train(
    feat_csv=dolos_feat_csv,
    run_name="dolos_baseline",
    device_str=DEVICE,
    model_save_path=dolos_model_path,
    scaler_save_path=dolos_scaler_path,
)
shutil.copy2(dolos_model_path, MODEL_PATH)
shutil.copy2(dolos_scaler_path, SCALER_PATH)

print("\n[8/16] Training on Trial...")
trial_results = train(
    feat_csv=trial_feat_csv,
    run_name="trial_baseline",
    device_str=DEVICE,
    model_save_path=trial_model_path,
    scaler_save_path=trial_scaler_path,
)

print("\n[9/16] Protocol A...")
proto_a_dolos_trial = protocol_A(
    source_csv=dolos_feat_csv,
    target_csv=trial_feat_csv,
    source_name="DOLOS",
    target_name="Trial",
    model_path=dolos_model_path,
    scaler_path=dolos_scaler_path,
)
proto_a_trial_dolos = protocol_A(
    source_csv=trial_feat_csv,
    target_csv=dolos_feat_csv,
    source_name="Trial",
    target_name="DOLOS",
    model_path=trial_model_path,
    scaler_path=trial_scaler_path,
)

print("\n[10/16] Protocol B...")
proto_b = protocol_B(
    source_csv=dolos_feat_csv,
    target_csv=trial_feat_csv,
    source_name="DOLOS",
    target_name="Trial",
    few_shot_frac=FEW_SHOT_FRAC,
    model_path=dolos_model_path,
    scaler_path=dolos_scaler_path,
)

print("\n[11/16] Summary...")
all_results = [
    {
        "title": "Protocol C: DOLOS Baseline",
        "acc": dolos_results["test_acc"],
        "f1": dolos_results["test_f1"],
        "auc": dolos_results["test_auc"],
    },
    {
        "title": "Protocol C: Trial Baseline",
        "acc": trial_results["test_acc"],
        "f1": trial_results["test_f1"],
        "auc": trial_results["test_auc"],
    },
    proto_a_dolos_trial,
    proto_a_trial_dolos,
    proto_b,
]
print_summary_table(all_results)
save_all_results(all_results)
plot_comparison_chart(all_results)

print("\n[12/16] SHAP...")
plot_shap_summary(
    feat_csv=dolos_feat_csv,
    n_background=50,
    n_explain=100,
    model_path=dolos_model_path,
    device_str=DEVICE,
    top_k=20,
)

print("\n[13/16] Modality ablation...")
modality_ablation(feat_csv=dolos_feat_csv, ablation_name="DOLOS", model_path=dolos_model_path)
modality_ablation(
    feat_csv=trial_feat_csv,
    ablation_name="RealLifeTrial",
    model_path=dolos_model_path,
)

print("\n[14/16] Single sample XAI...")
model, scaler, feat_names, device = load_model_and_scaler(dolos_model_path, DEVICE)
X_dolos, y_dolos = load_features(dolos_feat_csv)
X_dolos_scaled = scaler.transform(X_dolos)
np.nan_to_num(X_dolos_scaled, copy=False, nan=0.0)
single_explanation = explain_single_sample(
    feature_vector=X_dolos_scaled[0],
    model=model,
    feat_names=feat_names,
    device=device,
    top_k=10,
)
plot_single_sample_explanation(single_explanation)

print("\n[15/16] Batch demo...")
X_trial, y_trial = load_features(trial_feat_csv)
X_trial_scaled = scaler.transform(X_trial)
np.nan_to_num(X_trial_scaled, copy=False, nan=0.0)
demo_count = min(10, len(X_trial_scaled))
correct = 0
for idx in range(demo_count):
    explanation = explain_single_sample(
        feature_vector=X_trial_scaled[idx],
        model=model,
        feat_names=feat_names,
        device=device,
        top_k=3,
    )
    predicted_ok = explanation["predicted_class"] == int(y_trial[idx])
    correct += int(predicted_ok)
    print(
        f"[{idx}] true={int(y_trial[idx])} pred={explanation['predicted_class']} "
        f"conf={explanation['confidence']:.1%}"
    )
print(f"Batch accuracy on {demo_count} samples: {correct / demo_count:.1%}")

print("\n[16/16] Verifying artifacts...")
with open(FEAT_NAMES_PATH, "w", encoding="utf-8") as handle:
    json.dump(build_feature_names(), handle, indent=2)

artifacts = {
    "Primary model": MODEL_PATH,
    "DOLOS model": dolos_model_path,
    "Trial model": trial_model_path,
    "Feature scaler": SCALER_PATH,
    "Feature names": FEAT_NAMES_PATH,
    "Cross-dataset JSON": os.path.join(MODEL_DIR, "cross_dataset_results.json"),
    "DOLOS features": dolos_feat_csv,
    "Trial features": trial_feat_csv,
}
for name, path in artifacts.items():
    exists = os.path.isfile(path)
    size = os.path.getsize(path) if exists else 0
    print(f"[{'OK' if exists else 'MISS':>4}] {name:<20} {size:>10} {path}")

print("\nPipeline run complete.")
