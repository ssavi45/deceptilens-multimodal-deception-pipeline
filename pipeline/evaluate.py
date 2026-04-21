"""
evaluate.py — Evaluation protocols for cross-dataset generalization.

Protocols:
  A: Zero-shot transfer (train on source, test on target)
  B: Few-shot adaptation (fine-tune classifier head with 10% target)
  C: Within-dataset baseline (70/15/15 split) — handled by train.py

Also provides:
  - Confusion matrix + ROC curve visualization
  - Classification report
  - Summary table printer
"""

import os
import json
import logging

import numpy as np
import torch
import torch.nn as nn
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    f1_score, roc_auc_score, accuracy_score,
)
from tqdm import tqdm

from pipeline.config import (
    DEVICE, MODEL_DIR, XAI_DIR, SCALER_PATH,
    VIDEO_INPUT_DIM, AUDIO_INPUT_DIM, BATCH_SIZE, LABEL_MAP,
    FEW_SHOT_FRAC, LABEL_SMOOTH, WEIGHT_DECAY,
)
from pipeline.model import DeceptionDetector
from pipeline.dataset import (
    load_features, make_loader, prepare_cross_dataset,
)
from pipeline.train import finetune_few_shot, evaluate as eval_loader

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────────
#  Model loading
# ────────────────────────────────────────────────────────────────────────

def load_model(path: str, device_str: str = DEVICE) -> DeceptionDetector:
    """Load a trained DeceptionDetector from a checkpoint file."""
    device = torch.device(device_str)
    model = DeceptionDetector().to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.eval()
    print(f"✓ Model loaded from {path}")
    return model


# ────────────────────────────────────────────────────────────────────────
#  Full evaluation (metrics + plots)
# ────────────────────────────────────────────────────────────────────────

def full_evaluation(
    model: DeceptionDetector,
    X: np.ndarray,
    y: np.ndarray,
    device: str = DEVICE,
    title: str = "Evaluation",
) -> dict:
    """
    Comprehensive evaluation: classification report, confusion matrix, ROC curve.
    Saves a side-by-side PNG (confusion matrix + ROC).
    Returns dict with {acc, f1, auc, preds, labels, probs}.
    """
    # Run inference
    loader = make_loader(X, y, batch_size=BATCH_SIZE, shuffle=False, balance=False)
    all_preds, all_labels, all_probs = [], [], []

    model.eval()
    with torch.no_grad():
        for video, audio, labels in tqdm(loader, desc=f"Eval: {title}", leave=False):
            video = video.to(device)
            audio = audio.to(device)
            logits, _, _ = model(video, audio)
            probs = torch.softmax(logits, dim=-1)
            all_preds.extend(logits.argmax(dim=-1).cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs  = np.array(all_probs)

    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    try:
        auc_val = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc_val = 0.0

    # Print classification report
    print(f"\n{'─'*50}")
    print(f"  {title}")
    print(f"{'─'*50}")
    target_names = [LABEL_MAP[0], LABEL_MAP[1]]
    print(classification_report(all_labels, all_preds,
                                target_names=target_names, zero_division=0))
    print(f"  AUC-ROC: {auc_val:.4f}")

    # ── Plot confusion matrix + ROC curve ───────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    im = ax1.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax1.set_title("Confusion Matrix")
    ax1.set_ylabel("True Label")
    ax1.set_xlabel("Predicted Label")
    ax1.set_xticks([0, 1])
    ax1.set_yticks([0, 1])
    ax1.set_xticklabels(target_names)
    ax1.set_yticklabels(target_names)
    fig.colorbar(im, ax=ax1, shrink=0.8)

    # Annotate cells
    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax1.text(j, i, str(cm[i, j]),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black",
                     fontsize=16, fontweight="bold")

    # ROC curve
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    ax2.plot(fpr, tpr, color="darkorange", lw=2,
             label=f"AUC = {roc_auc:.3f}")
    ax2.plot([0, 1], [0, 1], "k--", lw=1)
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1.05])
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.set_title("ROC Curve")
    ax2.legend(loc="lower right")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    safe_title = title.replace(" ", "_").replace("→", "to").replace("/", "_")
    save_path = os.path.join(XAI_DIR, f"{safe_title}.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Evaluation plot saved to {save_path}")

    return {
        "title": title,
        "acc": float(acc),
        "f1": float(f1),
        "auc": float(auc_val),
        "preds": all_preds.tolist(),
        "labels": all_labels.tolist(),
        "probs": all_probs.tolist(),
    }


# ────────────────────────────────────────────────────────────────────────
#  Protocol A — Zero-shot cross-dataset transfer
# ────────────────────────────────────────────────────────────────────────

def protocol_A(
    source_csv: str,
    target_csv: str,
    source_name: str,
    target_name: str,
    model_path: str | None = None,
    scaler_path: str | None = None,
    device_str: str = DEVICE,
) -> dict:
    """
    Zero-shot transfer: model trained on source, evaluated on full target.
    Uses scaler from source training.
    """
    print(f"\n{'='*60}")
    print(f"  Protocol A: {source_name} → {target_name} (Zero-Shot)")
    print(f"{'='*60}")

    # Load model
    mpath = model_path or os.path.join(MODEL_DIR, f"deception_model_{source_name.lower()}.pth")
    spath = scaler_path or os.path.join(MODEL_DIR, f"feature_scaler_{source_name.lower()}.pkl")
    model = load_model(mpath, device_str)

    # Load & normalize target with source scaler
    data = prepare_cross_dataset(source_csv, target_csv,
                                 few_shot_frac=0.0,
                                 source_scaler_path=spath)

    title = f"Protocol A: {source_name} → {target_name}"
    result = full_evaluation(model, data["X_target"], data["y_target"],
                             device_str, title)
    result["protocol"] = "A"
    result["source"] = source_name
    result["target"] = target_name
    return result


# ────────────────────────────────────────────────────────────────────────
#  Protocol B — Few-shot adaptation
# ────────────────────────────────────────────────────────────────────────

def protocol_B(
    source_csv: str,
    target_csv: str,
    source_name: str,
    target_name: str,
    few_shot_frac: float = FEW_SHOT_FRAC,
    model_path: str | None = None,
    scaler_path: str | None = None,
    device_str: str = DEVICE,
) -> dict:
    """
    Few-shot adaptation: Train on source, fine-tune classifier head
    on few_shot_frac of target, evaluate on remaining target.
    """
    print(f"\n{'='*60}")
    print(f"  Protocol B: {source_name} → {target_name} "
          f"(Few-Shot {few_shot_frac:.0%})")
    print(f"{'='*60}")

    mpath = model_path or os.path.join(MODEL_DIR, f"deception_model_{source_name.lower()}.pth")
    spath = scaler_path or os.path.join(MODEL_DIR, f"feature_scaler_{source_name.lower()}.pkl")

    data = prepare_cross_dataset(source_csv, target_csv,
                                 few_shot_frac=few_shot_frac,
                                 source_scaler_path=spath)

    print(f"  Few-shot samples: {len(data['y_few'])}  "
          f"Test samples: {len(data['y_target'])}")

    # Fine-tune
    ft_metrics = finetune_few_shot(
        model_path=mpath,
        X_few=data["X_few"],
        y_few=data["y_few"],
        X_test=data["X_target"],
        y_test=data["y_target"],
        device_str=device_str,
    )

    # Full evaluation with the fine-tuned model
    ft_model_path = mpath.replace(".pth", "_finetuned.pth")
    model = load_model(ft_model_path, device_str)

    title = f"Protocol B: {source_name} → {target_name} (Few-Shot)"
    result = full_evaluation(model, data["X_target"], data["y_target"],
                             device_str, title)
    result["protocol"] = "B"
    result["source"] = source_name
    result["target"] = target_name
    result["few_shot_frac"] = few_shot_frac
    result["few_shot_n"] = len(data["y_few"])
    return result


# ────────────────────────────────────────────────────────────────────────
#  Summary table
# ────────────────────────────────────────────────────────────────────────

def print_summary_table(results: list[dict]) -> None:
    """Print a formatted ASCII summary table of all experiment results."""
    header = f"{'Experiment':<45} {'Acc':>7} {'F1':>7} {'AUC':>7}"
    sep = "─" * len(header)

    print(f"\n{sep}")
    print(f"  CROSS-DATASET GENERALIZATION RESULTS")
    print(f"{sep}")
    print(header)
    print(sep)

    for r in results:
        name = r.get("title", r.get("run_name", "?"))
        acc = r.get("acc", r.get("test_acc", 0))
        f1  = r.get("f1", r.get("test_f1", 0))
        auc_val = r.get("auc", r.get("test_auc", 0))
        print(f"{name:<45} {acc:>7.4f} {f1:>7.4f} {auc_val:>7.4f}")

    print(sep)


def save_all_results(results: list[dict], path: str | None = None) -> str:
    """Save all results to a JSON file."""
    save_path = path or os.path.join(MODEL_DIR, "cross_dataset_results.json")
    # Make results JSON-serializable (strip numpy arrays)
    clean = []
    for r in results:
        entry = {}
        for k, v in r.items():
            if k in ("preds", "labels", "probs", "history", "data"):
                continue
            entry[k] = v
        clean.append(entry)
    with open(save_path, "w") as f:
        json.dump(clean, f, indent=2)
    print(f"✓ All results saved to {save_path}")
    return save_path


def plot_comparison_chart(results: list[dict]) -> str:
    """Create a grouped bar chart comparing all experiments."""
    names = [r.get("title", r.get("run_name", "?"))[:30] for r in results]
    accs  = [r.get("acc", r.get("test_acc", 0)) for r in results]
    f1s   = [r.get("f1", r.get("test_f1", 0)) for r in results]
    aucs  = [r.get("auc", r.get("test_auc", 0)) for r in results]

    x = np.arange(len(names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - width, accs, width, label="Accuracy", color="#4285F4")
    ax.bar(x,         f1s,  width, label="F1 (macro)", color="#EA4335")
    ax.bar(x + width, aucs, width, label="AUC-ROC", color="#34A853")

    ax.set_ylabel("Score")
    ax.set_title("Cross-Dataset Generalization Comparison", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right", fontsize=9)
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    save_path = os.path.join(XAI_DIR, "cross_dataset_summary.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Comparison chart saved to {save_path}")
    return save_path
