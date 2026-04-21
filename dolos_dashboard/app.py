"""
app.py — Deception Detection Dashboard
Run with: streamlit run app.py
"""

import streamlit as st
import tempfile
import os
import time
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from inference import DeceptionEnsemble, predict, extract_features

# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="DeceptiLens — Truth Analysis",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');

  html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background-color: #0a0e1a;
    color: #e2e8f0;
  }

  .main { background-color: #0a0e1a; }

  h1, h2, h3 { font-family: 'Syne', sans-serif; font-weight: 800; }

  .hero-title {
    font-size: 3.2rem;
    font-weight: 800;
    letter-spacing: -1px;
    background: linear-gradient(135deg, #60a5fa, #a78bfa, #f472b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.1;
    margin-bottom: 0.5rem;
  }

  .hero-sub {
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
    color: #64748b;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 2rem;
  }

  .verdict-truth {
    background: linear-gradient(135deg, #064e3b, #065f46);
    border: 1px solid #10b981;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
  }

  .verdict-deception {
    background: linear-gradient(135deg, #450a0a, #7f1d1d);
    border: 1px solid #ef4444;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
  }

  .verdict-label {
    font-size: 3rem;
    font-weight: 800;
    letter-spacing: -1px;
  }

  .verdict-confidence {
    font-family: 'Space Mono', monospace;
    font-size: 0.9rem;
    opacity: 0.7;
    margin-top: 0.5rem;
  }

  .model-card {
    background: #111827;
    border: 1px solid #1f2937;
    border-radius: 12px;
    padding: 1.2rem;
    margin-bottom: 0.8rem;
    transition: border-color 0.2s;
  }

  .model-card:hover { border-color: #3b82f6; }

  .model-name {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 0.4rem;
  }

  .upload-zone {
    background: #111827;
    border: 2px dashed #1f2937;
    border-radius: 16px;
    padding: 3rem;
    text-align: center;
    transition: border-color 0.3s;
  }

  .stProgress > div > div > div { background: linear-gradient(90deg, #3b82f6, #8b5cf6); }

  .metric-box {
    background: #111827;
    border: 1px solid #1f2937;
    border-radius: 12px;
    padding: 1rem 1.5rem;
    text-align: center;
  }

  .metric-value {
    font-size: 2rem;
    font-weight: 800;
    font-family: 'Space Mono', monospace;
  }

  .metric-label {
    font-size: 0.75rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 1px;
  }

  .tag-truth { color: #10b981; }
  .tag-deception { color: #ef4444; }

  div[data-testid="stFileUploader"] {
    background: #111827;
    border: 2px dashed #374151;
    border-radius: 16px;
    padding: 1rem;
  }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Load models (cached)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_models():
    return DeceptionEnsemble(model_dir="saved_models")


# ─────────────────────────────────────────────
# Gauge chart helper
# ─────────────────────────────────────────────
def make_gauge(confidence: float, label: str) -> go.Figure:
    color = "#ef4444" if label == "Deception" else "#10b981"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(confidence * 100, 1),
        number={"suffix": "%", "font": {"size": 36, "family": "Space Mono", "color": "#e2e8f0"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#374151", "tickfont": {"color": "#64748b"}},
            "bar":  {"color": color, "thickness": 0.25},
            "bgcolor": "#1f2937",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 40],   "color": "#064e3b"},
                {"range": [40, 60],  "color": "#1e3a5f"},
                {"range": [60, 100], "color": "#450a0a"},
            ],
            "threshold": {
                "line":  {"color": color, "width": 3},
                "thickness": 0.85,
                "value": confidence * 100
            },
        },
        title={"text": "Deception Probability", "font": {"size": 13, "color": "#94a3b8"}},
    ))
    fig.update_layout(
        paper_bgcolor="#111827",
        plot_bgcolor="#111827",
        margin=dict(t=40, b=10, l=30, r=30),
        height=220,
    )
    return fig


def make_bar_chart(model_scores: dict) -> go.Figure:
    models = list(model_scores.keys())
    scores = [v * 100 for v in model_scores.values()]
    colors = ["#ef4444" if s >= 50 else "#10b981" for s in scores]

    fig = go.Figure(go.Bar(
        x=scores,
        y=models,
        orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{s:.1f}%" for s in scores],
        textposition="outside",
        textfont=dict(color="#94a3b8", size=12, family="Space Mono"),
    ))
    fig.add_vline(x=50, line_dash="dash", line_color="#374151", line_width=1.5)
    fig.update_layout(
        paper_bgcolor="#111827",
        plot_bgcolor="#111827",
        xaxis=dict(range=[0, 110], showgrid=False, tickfont=dict(color="#64748b"), zeroline=False),
        yaxis=dict(showgrid=False, tickfont=dict(color="#e2e8f0", size=11)),
        margin=dict(t=10, b=10, l=10, r=60),
        height=220,
        showlegend=False,
    )
    return fig


# ─────────────────────────────────────────────
# Main UI
# ─────────────────────────────────────────────
st.markdown('<div class="hero-title">DeceptiLens</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Multimodal Deception Detection · 82.1% Accuracy</div>', unsafe_allow_html=True)

# Model status
with st.spinner("Loading models..."):
    try:
        ensemble = load_models()
        st.success("✅ All 4 models loaded | PyTorch Dual-Stream · SVM · HGB · Random Forest")
    except Exception as e:
        st.error(f"❌ Could not load models: {e}")
        st.info("Make sure `saved_models/` folder is in the same directory as `app.py`")
        st.stop()

st.markdown("---")

# Upload section
col_upload, col_info = st.columns([3, 2], gap="large")

with col_upload:
    st.markdown("### 📤 Upload Video Clip")
    uploaded = st.file_uploader(
        "Drop an .mp4 or .avi clip here",
        type=["mp4", "avi", "mov", "mkv"],
        help="For best results: clear face view, 5–30 seconds, good lighting"
    )

with col_info:
    st.markdown("### 🧠 Model Architecture")
    st.markdown("""
    <div class="model-card">
      <div class="model-name">Feature Extraction (650 dims)</div>
      ResNet-18 · MediaPipe Blendshapes · Librosa Audio
    </div>
    <div class="model-card">
      <div class="model-name">PyTorch Dual-Stream ⭐ Best</div>
      CrossAttention · Video + Audio Streams · 82.1% Acc
    </div>
    <div class="model-card">
      <div class="model-name">Ensemble Hard Voting</div>
      4 Models · Majority Decision
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Prediction
# ─────────────────────────────────────────────
if uploaded:
    st.markdown("---")

    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    # Preview + progress
    c1, c2 = st.columns([1, 2])
    with c1:
        st.video(tmp_path)

    with c2:
        st.markdown("### ⚙️ Processing Pipeline")
        bar    = st.progress(0)
        status = st.empty()

        stages = [
            ("🎵 Extracting Librosa audio features...", 20),
            ("👁️  Running MediaPipe blendshape analysis...", 45),
            ("🖼️  Sampling frames through ResNet-18...", 70),
            ("🤖 Running ensemble inference...", 90),
            ("✅ Done!", 100),
        ]

        for msg, pct in stages:
            status.markdown(f"`{msg}`")
            bar.progress(pct)
            if pct < 100:
                time.sleep(0.3)   # visual pacing; actual work is below

    # Run actual inference
    with st.spinner("Running full inference (may take 30–60s)..."):
        try:
            result = predict(ensemble, tmp_path)
        except Exception as e:
            st.error(f"Inference failed: {e}")
            os.unlink(tmp_path)
            st.stop()

    os.unlink(tmp_path)

    st.markdown("---")
    st.markdown("## 🔍 Analysis Results")

    # ── Verdict banner ──
    label      = result["label"]
    confidence = result["pt_confidence"]   # Best model confidence
    css_class  = "verdict-deception" if label == "Deception" else "verdict-truth"
    emoji      = "🔴" if label == "Deception" else "🟢"
    tag_class  = "tag-deception" if label == "Deception" else "tag-truth"

    st.markdown(f"""
    <div class="{css_class}">
      <div class="verdict-label">{emoji} {label.upper()}</div>
      <div class="verdict-confidence">
        PyTorch Dual-Stream Confidence: {confidence * 100:.1f}% &nbsp;|&nbsp;
        Ensemble Vote: {result['deception_votes']}/4 models flagged deception
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Charts ──
    col_gauge, col_bar = st.columns(2, gap="large")

    with col_gauge:
        st.markdown("#### Deception Probability Gauge")
        st.plotly_chart(
            make_gauge(result["confidence"], label),
            use_container_width=True, config={"displayModeBar": False}
        )

    with col_bar:
        st.markdown("#### Per-Model Deception Probability")
        st.plotly_chart(
            make_bar_chart(result["model_scores"]),
            use_container_width=True, config={"displayModeBar": False}
        )

    # ── Per-model breakdown ──
    st.markdown("#### Model Breakdown")
    cols = st.columns(4)
    model_order = [
        ("PyTorch Dual-Stream (Best)", "⭐"),
        ("HistGradientBoosting",       "📊"),
        ("Support Vector Machine",     "🔵"),
        ("Random Forest",              "🌲"),
    ]

    for i, (name, icon) in enumerate(model_order):
        prob  = result["model_scores"].get(name, 0.0)
        vote  = result["hard_votes"].get(name, 0)
        v_str = "🔴 LIE" if vote else "🟢 TRUTH"
        with cols[i]:
            st.markdown(f"""
            <div class="metric-box">
              <div class="metric-label">{icon} {name.split("(")[0].strip()}</div>
              <div class="metric-value {'tag-deception' if vote else 'tag-truth'}">{prob*100:.1f}%</div>
              <div class="metric-label" style="margin-top:4px">{v_str}</div>
            </div>
            """, unsafe_allow_html=True)

    # ── Feature insight ──
    st.markdown("---")
    st.markdown("#### 📋 Clip Analysis Notes")
    note_cols = st.columns(3)
    with note_cols[0]:
        st.info("**Audio Stream**\nMFCC, ZCR, RMS & spectral centroid extracted via Librosa (34 dims)")
    with note_cols[1]:
        st.info("**Facial Micro-expressions**\n52 MediaPipe blendshapes (mean + std = 104 dims) tracked frame-by-frame")
    with note_cols[2]:
        st.info("**Spatial Embeddings**\n20 sampled frames → ResNet-18 latent space (512 dims)")

else:
    st.markdown("""
    <div style="text-align:center; padding: 4rem; color: #374151;">
      <div style="font-size: 4rem;">🎬</div>
      <div style="font-family: 'Space Mono', monospace; font-size:0.85rem; letter-spacing:2px; text-transform:uppercase; margin-top:1rem;">
        Upload a video clip above to begin analysis
      </div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="font-family: 'Space Mono', monospace; font-size: 0.7rem; color: #374151; text-align: center;">
  DeceptiLens · Ensemble: PyTorch Dual-Stream + HGB + SVM + RF · Features: ResNet-18 + MediaPipe + Librosa · 82.1% Accuracy
</div>
""", unsafe_allow_html=True)
