"""
model.py — Dual-Stream Cross-Attention Fusion Network for deception detection.

Architecture:
  VideoStream (58d) → StreamEncoder → 128d
  AudioStream (34d) → StreamEncoder → 128d
        ↓                         ↓
  CrossAttention(video→audio)  CrossAttention(audio→video)
        ↓                         ↓
  v_out = v_enc + v_attended   a_out = a_enc + a_attended
                  ↓
        Concat [v_out, a_out] → 256d
                  ↓
        Classifier → Linear(64) → Linear(2) → Softmax
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from pipeline.config import (
    VIDEO_INPUT_DIM, AUDIO_INPUT_DIM, FUSION_DIM, HIDDEN_DIM,
    NUM_CLASSES, DROPOUT,
)


class CrossAttention(nn.Module):
    """
    Single-head cross-attention.
    Query comes from stream A, Key/Value from stream B.
    Returns (attended_output, attention_weights).
    """

    def __init__(self, dim: int):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key   = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.scale = dim ** 0.5

    def forward(
        self,
        query_input: torch.Tensor,   # (B, dim) — stream A
        kv_input: torch.Tensor,      # (B, dim) — stream B
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            attended: (B, dim) — attended representation
            attn_weights: (B, 1) — attention scalar
        """
        Q = self.query(query_input)   # (B, dim)
        K = self.key(kv_input)        # (B, dim)
        V = self.value(kv_input)      # (B, dim)

        # Dot-product attention: score = Q·Kᵀ / √d
        # For single-vector inputs (no sequence dim), this is a scalar per sample
        attn_scores = (Q * K).sum(dim=-1, keepdim=True) / self.scale  # (B, 1)
        attn_weights = torch.sigmoid(attn_scores)  # (B, 1)

        attended = attn_weights * V   # (B, dim)
        return attended, attn_weights


class StreamEncoder(nn.Module):
    """
    Two-layer encoder for a single modality stream.
    input_dim → FUSION_DIM with LayerNorm, ReLU, Dropout.
    """

    def __init__(self, input_dim: int, hidden_dim: int = FUSION_DIM,
                 dropout: float = DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DeceptionDetector(nn.Module):
    """
    Full dual-stream cross-attention fusion model.

    forward() returns:
        logits     (B, NUM_CLASSES)
        attn_v2a   (B, 1)  — how much video attends to audio
        attn_a2v   (B, 1)  — how much audio attends to video
    """

    def __init__(
        self,
        video_dim: int = VIDEO_INPUT_DIM,
        audio_dim: int = AUDIO_INPUT_DIM,
        fusion_dim: int = FUSION_DIM,
        hidden_dim: int = HIDDEN_DIM,
        num_classes: int = NUM_CLASSES,
        dropout: float = DROPOUT,
    ):
        super().__init__()

        # Stream encoders
        self.video_encoder = StreamEncoder(video_dim, fusion_dim, dropout)
        self.audio_encoder = StreamEncoder(audio_dim, fusion_dim, dropout)

        # Cross-attention
        self.cross_attn_v2a = CrossAttention(fusion_dim)  # video queries, audio provides K/V
        self.cross_attn_a2v = CrossAttention(fusion_dim)  # audio queries, video provides K/V

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self,
        video: torch.Tensor,   # (B, 58)
        audio: torch.Tensor,   # (B, 34)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            logits:   (B, NUM_CLASSES)
            attn_v2a: (B, 1) — video-to-audio attention weight
            attn_a2v: (B, 1) — audio-to-video attention weight
        """
        # Encode
        v_enc = self.video_encoder(video)   # (B, fusion_dim)
        a_enc = self.audio_encoder(audio)   # (B, fusion_dim)

        # Cross-attend
        v_attended, attn_v2a = self.cross_attn_v2a(v_enc, a_enc)  # video attends to audio
        a_attended, attn_a2v = self.cross_attn_a2v(a_enc, v_enc)  # audio attends to video

        # Residual fusion
        v_out = v_enc + v_attended
        a_out = a_enc + a_attended

        # Concatenate and classify
        fused = torch.cat([v_out, a_out], dim=-1)  # (B, fusion_dim * 2)
        logits = self.classifier(fused)

        return logits, attn_v2a, attn_a2v

    @torch.no_grad()
    def predict_proba(
        self,
        video: torch.Tensor,
        audio: torch.Tensor,
    ) -> dict:
        """
        Convenience method for single-sample or batch inference.

        Returns dict:
            probs:     (B, 2) — softmax probabilities
            predicted: (B,)   — argmax class indices
            attn_v2a:  (B, 1)
            attn_a2v:  (B, 1)
        """
        self.eval()
        logits, attn_v2a, attn_a2v = self.forward(video, audio)
        probs = F.softmax(logits, dim=-1)
        predicted = probs.argmax(dim=-1)
        return {
            "probs": probs,
            "predicted": predicted,
            "attn_v2a": attn_v2a,
            "attn_a2v": attn_a2v,
        }


def count_parameters(model: nn.Module) -> int:
    """Return total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
