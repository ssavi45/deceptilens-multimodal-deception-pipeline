"""
xai.py — Explainability: SHAP, Gradient×Input Saliency, Modality Ablation.

Produces:
  - SHAP top-20 feature importance bar chart (color-coded by modality)
  - Per-sample gradient×input saliency + attention weights (for dashboard)
  - Modality ablation analysis (full vs. video-only vs. audio-only)
"""

import os
import json
import logging
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from pipeline.config import (
    DEVICE, MODEL_PATH, SCALER_PATH, FEAT_NAMES_PATH,
    XAI_DIR, MODEL_DIR,
    VIDEO_INPUT_DIM, AUDIO_INPUT_DIM, BATCH_SIZE, LABEL_MAP,
)
from pipeline.model import DeceptionDetector
from pipeline.dataset import load_features, load_named_features, make_loader

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────────
#  Helpers
# ────────────────────────────────────────────────────────────────────────

def load_model_and_scaler(
    model_path: str = MODEL_PATH,
    device_str: str = DEVICE,
) -> tuple:
    """Load model, scaler, feature names, and device."""
    device = torch.device(device_str)
    model = DeceptionDetector().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    scaler = joblib.load(SCALER_PATH)

    with open(FEAT_NAMES_PATH) as f:
        feat_names = json.load(f)

    return model, scaler, feat_names, device


# ────────────────────────────────────────────────────────────────────────
#  SHAP wrapper — DeepExplainer needs a single-tensor input
# ────────────────────────────────────────────────────────────────────────

class ModelWrapperForSHAP(nn.Module):
    """
    Wraps DeceptionDetector to accept a single concatenated (B, 92) tensor.
    Needed because SHAP's DeepExplainer cannot handle multi-input models.
    """

    def __init__(self, model: DeceptionDetector):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        video = x[:, :VIDEO_INPUT_DIM]
        audio = x[:, VIDEO_INPUT_DIM:]
        logits, _, _ = self.model(video, audio)
        return logits


# ────────────────────────────────────────────────────────────────────────
#  SHAP feature importance
# ────────────────────────────────────────────────────────────────────────

def compute_shap_values(
    feat_csv: str,
    n_background: int = 50,
    n_explain: int = 100,
    model_path: str = MODEL_PATH,
    device_str: str = DEVICE,
) -> tuple:
    """
    Compute SHAP values using DeepExplainer.

    Args:
        feat_csv: path to features.csv
        n_background: number of background samples
        n_explain: number of samples to explain

    Returns:
        (shap_values, X_sample, feat_names)
    """
    import shap

    model, scaler, feat_names, device = load_model_and_scaler(model_path, device_str)
    wrapper = ModelWrapperForSHAP(model).to(device)
    wrapper.eval()

    X, y = load_features(feat_csv)
    X = scaler.transform(X)
    np.nan_to_num(X, copy=False, nan=0.0)

    # Random subset
    rng = np.random.RandomState(42)
    bg_idx = rng.choice(len(X), size=min(n_background, len(X)), replace=False)
    ex_idx = rng.choice(len(X), size=min(n_explain, len(X)), replace=False)

    bg_tensor = torch.tensor(X[bg_idx], dtype=torch.float32).to(device)
    ex_tensor = torch.tensor(X[ex_idx], dtype=torch.float32).to(device)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        explainer = shap.DeepExplainer(wrapper, bg_tensor)
        shap_values = explainer.shap_values(ex_tensor, check_additivity=False)

    # shap_values is a list of 2 arrays (one per class), each (n_explain, 92)
    # We use the absolute mean across class 1 (Deceptive)
    if isinstance(shap_values, list):
        sv = np.array(shap_values[1])  # class 1 = Deceptive
    else:
        sv = np.array(shap_values)
        # DeepExplainer on PyTorch sometimes returns (batch, feats, classes)
        if sv.ndim == 3:
            sv = sv[:, :, 1]
    
    return sv, X[ex_idx], feat_names


def plot_shap_summary(
    feat_csv: str,
    n_background: int = 50,
    n_explain: int = 100,
    model_path: str = MODEL_PATH,
    device_str: str = DEVICE,
    top_k: int = 20,
) -> dict:
    """
    Create a top-k SHAP importance bar chart, color-coded by modality.
    Video features = coral/red, Audio features = steelblue.

    Returns dict of {feature_name: mean_abs_shap_value}.
    """
    sv, X_sample, feat_names = compute_shap_values(
        feat_csv, n_background, n_explain, model_path, device_str
    )

    mean_abs = np.abs(sv).mean(axis=0)
    importance = dict(zip(feat_names, mean_abs.tolist()))

    # Sort and take top_k
    sorted_feats = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    top_feats = sorted_feats[:top_k]

    names  = [f[0] for f in top_feats][::-1]  # reverse for horizontal bar
    values = [f[1] for f in top_feats][::-1]

    # Color by modality
    colors = []
    for name in names:
        if any(name.startswith(prefix) for prefix in
               ["AU", "pose_", "gaze_", "gaze_angle_"]):
            colors.append("#E8453C")  # video = red
        elif name.endswith("_mean") and any(
            name.startswith(p) for p in ["AU", "pose_", "gaze_"]
        ):
            colors.append("#E8453C")
        elif name.endswith("_std") and any(
            name.startswith(p) for p in ["AU", "pose_", "gaze_"]
        ):
            colors.append("#E8453C")
        else:
            colors.append("#4285F4")  # audio = blue

    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(range(len(names)), values, color=colors)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Mean |SHAP value|", fontsize=12)
    ax.set_title(f"Top-{top_k} Feature Importance (SHAP)", fontsize=14,
                 fontweight="bold")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#E8453C", label="Video (AU/Pose/Gaze)"),
        Patch(facecolor="#4285F4", label="Audio (MFCC/Pitch/RMS/ZCR/SC)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    save_path = os.path.join(XAI_DIR, "shap_summary.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ SHAP summary saved to {save_path}")

    return importance


# ────────────────────────────────────────────────────────────────────────
#  Gradient × Input saliency (per sample)
# ────────────────────────────────────────────────────────────────────────

def explain_single_sample(
    feature_vector: np.ndarray,
    model: DeceptionDetector | None = None,
    feat_names: list[str] | None = None,
    device: str | torch.device = DEVICE,
    top_k: int = 10,
    model_path: str = MODEL_PATH,
) -> dict:
    """
    Explain a single 92-dim sample using gradient×input saliency + attention.

    Args:
        feature_vector: (92,) numpy array (already scaled)
        model: loaded model (if None, loads from model_path)
        feat_names: feature names (if None, loads from FEAT_NAMES_PATH)
        device: torch device
        top_k: number of top features to return

    Returns:
        JSON-serializable dict for the FastAPI dashboard:
        {
            prediction, confidence, prob_truthful, prob_deceptive,
            top_features: [{name, value, attribution, modality}],
            attention_v2a, attention_a2v, dominant_modality,
            explanation_text
        }
    """
    if model is None:
        model, _, feat_names, device = load_model_and_scaler(model_path, str(device))
    elif feat_names is None:
        with open(FEAT_NAMES_PATH) as f:
            feat_names = json.load(f)

    if isinstance(device, str):
        device = torch.device(device)

    model.eval()

    # Prepare input
    x = torch.tensor(feature_vector, dtype=torch.float32).unsqueeze(0).to(device)
    video_input = x[:, :VIDEO_INPUT_DIM].clone().requires_grad_(True)
    audio_input = x[:, VIDEO_INPUT_DIM:].clone().requires_grad_(True)

    # Forward pass (with gradients)
    logits, attn_v2a, attn_a2v = model(video_input, audio_input)
    probs = F.softmax(logits, dim=-1)

    predicted_class = probs.argmax(dim=-1).item()
    confidence = probs[0, predicted_class].item()

    # Backward pass on predicted class prob
    model.zero_grad()
    probs[0, predicted_class].backward()

    # Gradient × Input
    video_grad = video_input.grad.detach().cpu().numpy().flatten()
    audio_grad = audio_input.grad.detach().cpu().numpy().flatten()
    video_val  = video_input.detach().cpu().numpy().flatten()
    audio_val  = audio_input.detach().cpu().numpy().flatten()

    video_saliency = video_grad * video_val  # (58,)
    audio_saliency = audio_grad * audio_val  # (34,)
    all_saliency   = np.concatenate([video_saliency, audio_saliency])  # (92,)

    # Sort by absolute attribution
    abs_sal = np.abs(all_saliency)
    top_indices = abs_sal.argsort()[::-1][:top_k]

    top_features = []
    for idx in top_indices:
        name = feat_names[idx]
        modality = "video" if idx < VIDEO_INPUT_DIM else "audio"
        top_features.append({
            "name": name,
            "value": float(feature_vector[idx]),
            "attribution": float(all_saliency[idx]),
            "modality": modality,
        })

    # Determine dominant modality
    video_total = float(np.abs(video_saliency).sum())
    audio_total = float(np.abs(audio_saliency).sum())
    total = video_total + audio_total + 1e-8
    dominant = "video" if video_total > audio_total else "audio"

    explanation_text = (
        f"Model predicts {LABEL_MAP[predicted_class]} with {confidence:.1%} confidence. "
        f"Video features contribute {video_total/total:.1%} of the attribution, "
        f"audio features contribute {audio_total/total:.1%}. "
        f"Top feature: {top_features[0]['name']} "
        f"(attribution={top_features[0]['attribution']:.4f})."
    )

    return {
        "prediction": LABEL_MAP[predicted_class],
        "predicted_class": predicted_class,
        "confidence": float(confidence),
        "prob_truthful": float(probs[0, 0].item()),
        "prob_deceptive": float(probs[0, 1].item()),
        "top_features": top_features,
        "attention_v2a": float(attn_v2a[0, 0].item()),
        "attention_a2v": float(attn_a2v[0, 0].item()),
        "dominant_modality": dominant,
        "video_contribution": float(video_total / total),
        "audio_contribution": float(audio_total / total),
        "explanation_text": explanation_text,
    }


def plot_single_sample_explanation(explanation: dict, save_path: str | None = None) -> str:
    """Visualize a single sample explanation as a multi-panel figure."""
    top_features = explanation["top_features"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        f"Prediction: {explanation['prediction']} "
        f"({explanation['confidence']:.1%} confidence)",
        fontsize=14, fontweight="bold"
    )

    # Panel 1: Top feature attributions
    names  = [f["name"] for f in top_features][::-1]
    attrs  = [f["attribution"] for f in top_features][::-1]
    colors = ["#E8453C" if f["modality"] == "video" else "#4285F4"
              for f in top_features][::-1]

    axes[0].barh(range(len(names)), attrs, color=colors)
    axes[0].set_yticks(range(len(names)))
    axes[0].set_yticklabels(names, fontsize=8)
    axes[0].set_xlabel("Attribution (grad × input)")
    axes[0].set_title("Top Feature Attributions")
    axes[0].grid(True, alpha=0.3, axis="x")

    # Panel 2: Modality contribution pie
    video_pct = explanation["video_contribution"]
    audio_pct = explanation["audio_contribution"]
    axes[1].pie(
        [video_pct, audio_pct],
        labels=["Video", "Audio"],
        colors=["#E8453C", "#4285F4"],
        autopct="%1.1f%%",
        startangle=90,
        textprops={"fontsize": 12},
    )
    axes[1].set_title("Modality Contribution")

    # Panel 3: Attention weights
    attn_labels = ["Video→Audio\n(v2a)", "Audio→Video\n(a2v)"]
    attn_vals = [explanation["attention_v2a"], explanation["attention_a2v"]]
    bars = axes[2].bar(attn_labels, attn_vals, color=["#E8453C", "#4285F4"],
                       width=0.5)
    axes[2].set_ylim(0, 1)
    axes[2].set_ylabel("Attention Weight")
    axes[2].set_title("Cross-Attention Weights")
    for bar, val in zip(bars, attn_vals):
        axes[2].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                     f"{val:.3f}", ha="center", fontsize=12, fontweight="bold")
    axes[2].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    save_path = save_path or os.path.join(XAI_DIR, "single_sample_xai.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Single sample XAI saved to {save_path}")
    return save_path


# ────────────────────────────────────────────────────────────────────────
#  Modality ablation
# ────────────────────────────────────────────────────────────────────────

def modality_ablation(
    feat_csv: str,
    ablation_name: str = "ablation",
    model_path: str = MODEL_PATH,
    device_str: str = DEVICE,
) -> dict:
    """
    Compare full model vs. video-only (zero audio) vs. audio-only (zero video).

    Returns dict with accuracy/f1/auc for each condition.
    """
    from sklearn.metrics import f1_score as sk_f1, roc_auc_score as sk_auc

    model, scaler, feat_names, device = load_model_and_scaler(model_path, device_str)
    X, y = load_features(feat_csv)
    X = scaler.transform(X)
    np.nan_to_num(X, copy=False, nan=0.0)

    conditions = {
        "Full Model": X.copy(),
        "Video Only": np.hstack([X[:, :VIDEO_INPUT_DIM],
                                  np.zeros((len(X), AUDIO_INPUT_DIM), dtype=np.float32)]),
        "Audio Only": np.hstack([np.zeros((len(X), VIDEO_INPUT_DIM), dtype=np.float32),
                                  X[:, VIDEO_INPUT_DIM:]]),
    }

    results = {}
    for cond_name, X_cond in conditions.items():
        loader = make_loader(X_cond, y, batch_size=BATCH_SIZE,
                             shuffle=False, balance=False)
        all_preds, all_labels, all_probs = [], [], []

        with torch.no_grad():
            for video, audio, labels in loader:
                video = video.to(device)
                audio = audio.to(device)
                logits, _, _ = model(video, audio)
                probs = torch.softmax(logits, dim=-1)
                all_preds.extend(logits.argmax(dim=-1).cpu().numpy())
                all_labels.extend(labels.numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())

        acc = float(np.mean(np.array(all_preds) == np.array(all_labels)))
        f1  = float(sk_f1(all_labels, all_preds, average="macro", zero_division=0))
        try:
            auc_val = float(sk_auc(all_labels, all_probs))
        except ValueError:
            auc_val = 0.0

        results[cond_name] = {"acc": acc, "f1": f1, "auc": auc_val}
        print(f"  {cond_name}: Acc={acc:.4f}  F1={f1:.4f}  AUC={auc_val:.4f}")

    # ── Bar chart ───────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    cond_names = list(results.keys())
    x = np.arange(len(cond_names))
    width = 0.25

    accs = [results[c]["acc"] for c in cond_names]
    f1s  = [results[c]["f1"]  for c in cond_names]
    aucs = [results[c]["auc"] for c in cond_names]

    ax.bar(x - width, accs, width, label="Accuracy", color="#4285F4")
    ax.bar(x,         f1s,  width, label="F1 (macro)", color="#EA4335")
    ax.bar(x + width, aucs, width, label="AUC-ROC", color="#34A853")

    ax.set_ylabel("Score")
    ax.set_title(f"Modality Ablation — {ablation_name}", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(cond_names, fontsize=11)
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis="y")

    # Value labels on bars
    for bars_group in [
        ax.containers[0], ax.containers[1], ax.containers[2]
    ]:
        ax.bar_label(bars_group, fmt="%.3f", fontsize=8, padding=2)

    plt.tight_layout()
    save_path = os.path.join(XAI_DIR, f"ablation_{ablation_name}.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Ablation chart saved to {save_path}")

    return results
