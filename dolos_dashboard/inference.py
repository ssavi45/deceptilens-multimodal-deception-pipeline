"""
inference.py
------------
Full feature extraction + ensemble prediction pipeline.
Mirrors the notebook's extract_single_video() and ensemble logic exactly.
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
import librosa
import joblib
import json
import warnings
warnings.filterwarnings("ignore")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MEDIAPIPE_BLENDSHAPE_NAMES = [
    "browDownLeft",
    "browDownRight",
    "browInnerUp",
    "browOuterUpLeft",
    "browOuterUpRight",
    "cheekPuff",
    "cheekSquintLeft",
    "cheekSquintRight",
    "eyeBlinkLeft",
    "eyeBlinkRight",
    "eyeLookDownLeft",
    "eyeLookDownRight",
    "eyeLookInLeft",
    "eyeLookInRight",
    "eyeLookOutLeft",
    "eyeLookOutRight",
    "eyeLookUpLeft",
    "eyeLookUpRight",
    "eyeSquintLeft",
    "eyeSquintRight",
    "eyeWideLeft",
    "eyeWideRight",
    "jawForward",
    "jawLeft",
    "jawOpen",
    "jawRight",
    "mouthClose",
    "mouthDimpleLeft",
    "mouthDimpleRight",
    "mouthFrownLeft",
    "mouthFrownRight",
    "mouthFunnel",
    "mouthLeft",
    "mouthLowerDownLeft",
    "mouthLowerDownRight",
    "mouthPressLeft",
    "mouthPressRight",
    "mouthPucker",
    "mouthRight",
    "mouthRollLower",
    "mouthRollUpper",
    "mouthShrugLower",
    "mouthShrugUpper",
    "mouthSmileLeft",
    "mouthSmileRight",
    "mouthStretchLeft",
    "mouthStretchRight",
    "mouthUpperUpLeft",
    "mouthUpperUpRight",
    "noseSneerLeft",
    "noseSneerRight",
    "tongueOut",
]

# ─────────────────────────────────────────────
# 1.  PyTorch Model Architecture (must match notebook exactly)
# ─────────────────────────────────────────────

class ResidualBlock1D(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.GELU(), nn.Dropout(0.4),
            nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.Dropout(0.4)
        )
        self.relu = nn.GELU()

    def forward(self, x):
        return self.relu(self.net(x) + x)


class ResNetDeceptionDetector(nn.Module):
    def __init__(self):
        super().__init__()
        dim = 512
        # video = mediapipe (104) + resnet (512)
        self.video_enc = nn.Sequential(
            nn.Linear(104 + 512, dim), nn.BatchNorm1d(dim), nn.GELU(),
            ResidualBlock1D(dim)
        )
        self.audio_enc = nn.Sequential(
            nn.Linear(34, dim), nn.BatchNorm1d(dim), nn.GELU(),
            ResidualBlock1D(dim)
        )
        self.cross_v2a = nn.MultiheadAttention(dim, num_heads=4, batch_first=True)
        self.cross_a2v = nn.MultiheadAttention(dim, num_heads=4, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(dim * 2, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.4),
            nn.Linear(256, 2)
        )

    def forward(self, v, a):
        v = self.video_enc(v).unsqueeze(1)
        a = self.audio_enc(a).unsqueeze(1)
        v_attn, _ = self.cross_v2a(v, a, a)
        a_attn, _ = self.cross_a2v(a, v, v)
        out = torch.cat(
            [v.squeeze(1) + v_attn.squeeze(1),
             a.squeeze(1) + a_attn.squeeze(1)], dim=-1
        )
        return self.classifier(out)


# ─────────────────────────────────────────────
# 2.  Model Loader
# ─────────────────────────────────────────────

class DeceptionEnsemble:
    """Loads all 4 models + scaler from saved_models/ directory."""

    def __init__(self, model_dir: str = "saved_models"):
        self.model_dir = model_dir
        self._load_all()

    def _load_all(self):
        d = self.model_dir

        # Sklearn models
        self.hgb    = joblib.load(os.path.join(d, "hgb_model.joblib"))
        self.svm    = joblib.load(os.path.join(d, "svm_model.joblib"))
        self.rf     = joblib.load(os.path.join(d, "rf_model.joblib"))
        self.scaler = joblib.load(os.path.join(d, "scaler.joblib"))

        # PyTorch model (best performer: 82.11%)
        self.pt_model = ResNetDeceptionDetector().to(DEVICE)
        weights_path  = os.path.join(d, "pytorch_model.pth")
        self.pt_model.load_state_dict(
            torch.load(weights_path, map_location=DEVICE)
        )
        self.pt_model.eval()

        # Feature names
        with open(os.path.join(d, "feature_names.json")) as f:
            self.feature_names = json.load(f)

        print(f"✅ All models loaded from '{self.model_dir}' | Device: {DEVICE}")


# ─────────────────────────────────────────────
# 3.  Feature Extraction  (mirrors notebook exactly)
# ─────────────────────────────────────────────

def _extract_audio_features(video_path: str) -> list:
    """34-dimensional Librosa audio features."""
    try:
        y, sr = librosa.load(video_path, sr=16000)
        mfcc       = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        rms        = librosa.feature.rms(y=y)
        zcr        = librosa.feature.zero_crossing_rate(y=y)
        spec_cent  = librosa.feature.spectral_centroid(y=y, sr=sr)

        return (
            [float(np.mean(mfcc)), float(np.std(mfcc)),
             float(np.mean(rms)),  float(np.std(rms)),
             float(np.mean(zcr)),  float(np.std(zcr)),
             float(np.mean(spec_cent)), float(np.std(spec_cent))]
            + [float(x) for x in np.mean(mfcc, axis=1)]
            + [float(x) for x in np.std(mfcc, axis=1)]
        )
    except Exception:
        return [0.0] * 34


def _extract_mediapipe_features(video_path: str) -> list:
    """104-dimensional MediaPipe face blendshape features."""
    try:
        import mediapipe as mp
        BaseOptions       = mp.tasks.BaseOptions
        FaceLandmarker    = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions

        TASK_PATH = "face_landmarker.task"
        if not os.path.exists(TASK_PATH):
            import urllib.request
            url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
            urllib.request.urlretrieve(url, TASK_PATH)

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=TASK_PATH),
            output_face_blendshapes=True
        )

        cap = cv2.VideoCapture(video_path)
        blendshape_frames = []

        with FaceLandmarker.create_from_options(options) as landmarker:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                res   = landmarker.detect(mp_img)
                if res.face_blendshapes:
                    scores = [c.score for c in res.face_blendshapes[0]]
                    blendshape_frames.append(scores)
        cap.release()

        if blendshape_frames:
            arr = np.array(blendshape_frames)
            return list(np.mean(arr, axis=0)) + list(np.std(arr, axis=0))
    except Exception:
        pass

    return [0.0] * 104


def _extract_resnet_features(video_path: str) -> list:
    """512-dimensional ResNet-18 spatial embeddings (sampled 20 frames)."""
    try:
        weights    = ResNet18_Weights.IMAGENET1K_V1
        model      = resnet18(weights=weights)
        model.fc   = nn.Identity()
        model      = model.to(DEVICE)
        model.eval()
        preprocess = weights.transforms()

        cap         = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sample_idx  = set(np.linspace(0, max(0, frame_count - 1), 20, dtype=int))
        frames      = []

        for i in range(frame_count):
            ret, frame = cap.read()
            if not ret:
                break
            if i in sample_idx:
                t = (torch.from_numpy(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                       .permute(2, 0, 1).float() / 255.0)
                frames.append(preprocess(t))
        cap.release()

        if frames:
            batch = torch.stack(frames).to(DEVICE)
            with torch.no_grad():
                return model(batch).mean(dim=0).cpu().numpy().tolist()
    except Exception:
        pass

    return [0.0] * 512


def extract_features(video_path: str) -> np.ndarray:
    """
    Extract 650-dimensional feature vector from a video file.
    Order: MediaPipe (104) + Audio (34) + ResNet-18 (512) = 650
    """
    mp_feats     = _extract_mediapipe_features(video_path)   # 0:104
    audio_feats  = _extract_audio_features(video_path)       # 104:138
    resnet_feats = _extract_resnet_features(video_path)      # 138:650

    return np.array(mp_feats + audio_feats + resnet_feats, dtype=np.float32)


def _diagnose_modality(features: np.ndarray, name: str) -> dict:
    """Summarize extraction quality for a feature slice."""
    total = int(features.size)
    nonzero = int(np.count_nonzero(np.abs(features) > 1e-8))
    nonzero_pct = float((nonzero / total) * 100.0) if total else 0.0
    all_zero_fallback = nonzero == 0
    status = "fallback" if all_zero_fallback else "ok"
    return {
        "name": name,
        "total_dims": total,
        "nonzero_dims": nonzero,
        "nonzero_pct": nonzero_pct,
        "all_zero_fallback": all_zero_fallback,
        "status": status,
    }


def _predict_scores_from_scaled(ensemble: DeceptionEnsemble, scaled: np.ndarray) -> dict:
    """Run all 4 models on already-scaled features and return deception probabilities."""
    hgb_p = float(ensemble.hgb.predict_proba(scaled)[0, 1])
    svm_p = float(ensemble.svm.predict_proba(scaled)[0, 1])
    rf_p = float(ensemble.rf.predict_proba(scaled)[0, 1])

    v_in = np.concatenate([scaled[:, :104], scaled[:, 138:]], axis=1)
    a_in = scaled[:, 104:138]
    v_t = torch.FloatTensor(v_in).to(DEVICE)
    a_t = torch.FloatTensor(a_in).to(DEVICE)

    ensemble.pt_model.eval()
    with torch.no_grad():
        logits = ensemble.pt_model(v_t, a_t)
        pt_p = float(torch.softmax(logits, dim=1)[0, 1].item())

    return {
        "PyTorch Dual-Stream (Best)": pt_p,
        "HistGradientBoosting": hgb_p,
        "Support Vector Machine": svm_p,
        "Random Forest": rf_p,
    }


def _ensemble_deception_probability(model_scores: dict) -> float:
    return float(np.mean(list(model_scores.values())))


def _humanize_feature_name(raw_name: str) -> str:
    """Convert technical feature names to user-facing labels."""
    if raw_name.startswith("audio_"):
        idx = int(raw_name.split("_")[1])
        audio_map = {
            0: "MFCC mean",
            1: "MFCC std",
            2: "RMS energy mean",
            3: "RMS energy std",
            4: "Zero-crossing rate mean",
            5: "Zero-crossing rate std",
            6: "Spectral centroid mean",
            7: "Spectral centroid std",
        }
        if idx in audio_map:
            return audio_map[idx]
        if 8 <= idx <= 20:
            return f"MFCC coefficient mean #{idx - 7}"
        if 21 <= idx <= 33:
            return f"MFCC coefficient std #{idx - 20}"
        return f"Audio feature #{idx + 1}"

    if raw_name.startswith("mp_mean_"):
        idx = int(raw_name.split("_")[-1])
        if 0 <= idx < len(MEDIAPIPE_BLENDSHAPE_NAMES):
            return f"{MEDIAPIPE_BLENDSHAPE_NAMES[idx]} average intensity"
        return f"Face blendshape mean #{idx + 1}"

    if raw_name.startswith("mp_std_"):
        idx = int(raw_name.split("_")[-1])
        if 0 <= idx < len(MEDIAPIPE_BLENDSHAPE_NAMES):
            return f"{MEDIAPIPE_BLENDSHAPE_NAMES[idx]} variability"
        return f"Face blendshape variability #{idx + 1}"

    if raw_name.startswith("resnet_"):
        idx = int(raw_name.split("_")[1])
        return f"Visual latent pattern unit #{idx + 1}"

    return raw_name.replace("_", " ").strip().title()


def _feature_category(raw_name: str) -> str:
    if raw_name.startswith("audio_"):
        return "audio"
    if raw_name.startswith("mp_mean_") or raw_name.startswith("mp_std_"):
        return "face"
    if raw_name.startswith("resnet_"):
        return "visual_latent"
    return "other"


def _feature_description(raw_name: str) -> str:
    if raw_name.startswith("audio_"):
        idx = int(raw_name.split("_")[1])
        if idx in [0, 1]:
            return "Overall speech timbre/shape captured by MFCC summary."
        if idx in [2, 3]:
            return "Loudness/energy behavior of the clip audio."
        if idx in [4, 5]:
            return "How noisy vs tonal the speech signal is over time."
        if idx in [6, 7]:
            return "Brightness of the speech spectrum over time."
        if 8 <= idx <= 33:
            return "Fine-grained voice spectral pattern component."
        return "Audio-derived cue from the clip."

    if raw_name.startswith("mp_mean_"):
        return "Average strength of this facial action across frames."

    if raw_name.startswith("mp_std_"):
        return "How much this facial action varies over time."

    if raw_name.startswith("resnet_"):
        return (
            "Latent CNN visual unit (not directly human-named); represents a compressed visual pattern "
            "from sampled video frames."
        )

    return "Derived model feature."


def _compute_local_xai(
    ensemble: DeceptionEnsemble,
    scaled: np.ndarray,
    feature_names: list,
    base_ensemble_prob: float,
    candidate_top_n: int = 40,
    return_top_k: int = 10,
) -> dict:
    """
    Local perturbation XAI:
    mask features to baseline (scaled=0) and measure probability shift.
    """
    group_slices = [
        ("Face expression means", slice(0, 52)),
        ("Face expression variability", slice(52, 104)),
        ("Audio summary cues", slice(104, 112)),
        ("Audio MFCC mean profile", slice(112, 125)),
        ("Audio MFCC variability", slice(125, 138)),
        ("Visual frame embeddings", slice(138, 650)),
    ]

    group_contributions = []
    for label, s in group_slices:
        masked = scaled.copy()
        masked[:, s] = 0.0
        masked_scores = _predict_scores_from_scaled(ensemble, masked)
        masked_prob = _ensemble_deception_probability(masked_scores)
        delta = base_ensemble_prob - masked_prob
        group_contributions.append(
            {
                "group": label,
                "contribution_pp": float(delta * 100.0),
                "direction": "deception" if delta > 0 else "truth",
                "abs_contribution_pp": float(abs(delta) * 100.0),
            }
        )

    ranked_indices = np.argsort(np.abs(scaled[0]))[::-1][:candidate_top_n]
    feature_contributions = []
    for idx in ranked_indices:
        masked = scaled.copy()
        masked[0, idx] = 0.0
        masked_scores = _predict_scores_from_scaled(ensemble, masked)
        masked_prob = _ensemble_deception_probability(masked_scores)
        delta = base_ensemble_prob - masked_prob

        raw_name = (
            feature_names[idx]
            if isinstance(feature_names, list) and idx < len(feature_names)
            else f"feature_{idx}"
        )
        feature_contributions.append(
            {
                "index": int(idx),
                "raw_name": raw_name,
                "feature": _humanize_feature_name(raw_name),
                "category": _feature_category(raw_name),
                "description": _feature_description(raw_name),
                "is_human_interpretable": _feature_category(raw_name) in {"audio", "face"},
                "z_value": float(scaled[0, idx]),
                "contribution_pp": float(delta * 100.0),
                "direction": "deception" if delta > 0 else "truth",
                "abs_contribution_pp": float(abs(delta) * 100.0),
            }
        )

    group_contributions.sort(key=lambda x: x["abs_contribution_pp"], reverse=True)
    feature_contributions.sort(key=lambda x: x["abs_contribution_pp"], reverse=True)
    top_human_features = [
        row for row in feature_contributions if row.get("is_human_interpretable", False)
    ][:return_top_k]
    top_latent_features = [
        row for row in feature_contributions if row.get("category") == "visual_latent"
    ][:return_top_k]

    return {
        "method": "Local masking to baseline (scaled=0)",
        "group_contributions": group_contributions,
        "top_features": feature_contributions[:return_top_k],
        "top_human_features": top_human_features,
        "top_latent_features": top_latent_features,
    }


# ─────────────────────────────────────────────
# 4.  Prediction
# ─────────────────────────────────────────────

def predict(ensemble: DeceptionEnsemble, video_path: str) -> dict:
    """
    Run full ensemble prediction on a video file.

    Returns:
        {
          "label":        "Truth" | "Deception",
          "confidence":   0.0–1.0  (deception probability),
          "model_scores": {model_name: prob_deception},
          "hard_vote":    [0 or 1 per model]
        }
    """
    # Feature extraction
    raw   = extract_features(video_path).reshape(1, -1)
    mp_raw = raw[:, :104]
    audio_raw = raw[:, 104:138]
    resnet_raw = raw[:, 138:]

    diagnostics = {
        "mediapipe": _diagnose_modality(mp_raw, "MediaPipe Blendshapes"),
        "audio": _diagnose_modality(audio_raw, "Librosa Audio"),
        "resnet": _diagnose_modality(resnet_raw, "ResNet-18 Spatial"),
    }
    fallback_modalities = [
        details["name"]
        for details in diagnostics.values()
        if details["all_zero_fallback"]
    ]

    scaled = ensemble.scaler.transform(raw)

    # Model predictions
    votes = _predict_scores_from_scaled(ensemble, scaled)

    hard_votes   = {k: int(v >= 0.5) for k, v in votes.items()}
    deception_votes = sum(hard_votes.values())
    final_label  = "Deception" if deception_votes >= 2 else "Truth"

    # Ensemble confidence = average probability
    ensemble_confidence = _ensemble_deception_probability(votes)
    pt_p = float(votes.get("PyTorch Dual-Stream (Best)", 0.0))

    xai = _compute_local_xai(
        ensemble=ensemble,
        scaled=scaled,
        feature_names=getattr(ensemble, "feature_names", []),
        base_ensemble_prob=ensemble_confidence,
    )

    return {
        "label":          final_label,
        "confidence":     ensemble_confidence,
        "pt_confidence":  pt_p,           # Best model standalone confidence
        "model_scores":   votes,
        "hard_votes":     hard_votes,
        "deception_votes": deception_votes,
        "feature_diagnostics": diagnostics,
        "fallback_modalities": fallback_modalities,
        "xai": xai,
    }
