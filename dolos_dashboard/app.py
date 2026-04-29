"""
app.py - Deception Detection Dashboard
Run with: streamlit run app.py
"""

import base64
from pathlib import Path
import os
import tempfile
import time

import plotly.graph_objects as go
import streamlit as st

from inference import DeceptionEnsemble, predict, validate_human_face, validate_speech_activity


APP_DIR = Path(__file__).resolve().parent


def resolve_logo_path() -> Path | None:
    candidates = [
        APP_DIR / "deceptiLens.png",
        APP_DIR / "DeceptiLens.png",
        APP_DIR.parent / "deceptiLens.png",
        APP_DIR.parent / "DeceptiLens.png",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


LOGO_PATH = resolve_logo_path()


st.set_page_config(
    page_title="DeceptiLens | Truth Analysis",
    page_icon=str(LOGO_PATH) if LOGO_PATH else ":material/monitoring:",
    layout="wide",
    initial_sidebar_state="collapsed",
)


THEMES = {
    "dark": {
        "bg": "#0f141c",
        "surface": "#171d27",
        "surface_alt": "#1c2430",
        "text": "#ecf1f7",
        "muted": "#92a0b3",
        "border": "#2a3442",
        "accent": "#4f8cff",
        "accent_soft": "#dbe7ff",
        "success": "#2f8f6d",
        "success_soft": "#d9f3e8",
        "danger": "#c55d5d",
        "danger_soft": "#f6dddd",
        "warning": "#d0a44e",
        "shadow": "0 18px 40px rgba(0, 0, 0, 0.18)",
    },
    "light": {
        "bg": "#f3f6fb",
        "surface": "#ffffff",
        "surface_alt": "#eef3f9",
        "text": "#18202b",
        "muted": "#5f6f82",
        "border": "#d5dee9",
        "accent": "#2f5fd2",
        "accent_soft": "#dce7ff",
        "success": "#28785d",
        "success_soft": "#dff1e9",
        "danger": "#b64c4c",
        "danger_soft": "#f7e1e1",
        "warning": "#b58427",
        "shadow": "0 16px 34px rgba(24, 32, 43, 0.08)",
    },
}


MODEL_ORDER = [
    "PyTorch Dual-Stream (Best)",
    "HistGradientBoosting",
    "Support Vector Machine",
    "Random Forest",
]


def svg_icon(name: str) -> str:
    icons = {
        "brand": """
        <svg viewBox="0 0 24 24" aria-hidden="true">
          <rect x="3" y="4" width="18" height="16" rx="4" fill="none" stroke="currentColor" stroke-width="1.7"/>
          <path d="M7 12c1.5-2.4 3.4-3.6 5-3.6s3.5 1.2 5 3.6c-1.5 2.4-3.4 3.6-5 3.6S8.5 14.4 7 12Z" fill="none" stroke="currentColor" stroke-width="1.7"/>
          <circle cx="12" cy="12" r="1.8" fill="currentColor"/>
        </svg>
        """,
        "theme": """
        <svg viewBox="0 0 24 24" aria-hidden="true">
          <path d="M12 3.5v2.2M12 18.3v2.2M5.6 5.6l1.6 1.6M16.8 16.8l1.6 1.6M3.5 12h2.2M18.3 12h2.2M5.6 18.4l1.6-1.6M16.8 7.2l1.6-1.6" fill="none" stroke="currentColor" stroke-linecap="round" stroke-width="1.7"/>
          <circle cx="12" cy="12" r="4" fill="none" stroke="currentColor" stroke-width="1.7"/>
        </svg>
        """,
        "upload": """
        <svg viewBox="0 0 24 24" aria-hidden="true">
          <path d="M12 16V7.2M8.8 10.4 12 7.2l3.2 3.2" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" stroke-width="1.8"/>
          <path d="M6 18.5h12" fill="none" stroke="currentColor" stroke-linecap="round" stroke-width="1.8"/>
          <rect x="4" y="4" width="16" height="16" rx="4" fill="none" stroke="currentColor" stroke-width="1.6"/>
        </svg>
        """,
        "stack": """
        <svg viewBox="0 0 24 24" aria-hidden="true">
          <path d="M12 4 4.5 8 12 12 19.5 8 12 4Z" fill="none" stroke="currentColor" stroke-linejoin="round" stroke-width="1.7"/>
          <path d="M4.5 12 12 16l7.5-4M4.5 16 12 20l7.5-4" fill="none" stroke="currentColor" stroke-linejoin="round" stroke-width="1.7"/>
        </svg>
        """,
        "result": """
        <svg viewBox="0 0 24 24" aria-hidden="true">
          <path d="M5 19h14M8 15l3-3 2 2 4-5" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" stroke-width="1.8"/>
          <path d="M6 19V7.5a1.5 1.5 0 0 1 1.5-1.5h9A1.5 1.5 0 0 1 18 7.5V19" fill="none" stroke="currentColor" stroke-width="1.6"/>
        </svg>
        """,
        "insight": """
        <svg viewBox="0 0 24 24" aria-hidden="true">
          <path d="M9.5 21h5M10 18h4M8.5 14.2a5.5 5.5 0 1 1 7 0c-.9.8-1.5 1.9-1.7 3H10.2c-.2-1.1-.8-2.2-1.7-3Z" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" stroke-width="1.7"/>
        </svg>
        """,
        "diagnostics": """
        <svg viewBox="0 0 24 24" aria-hidden="true">
          <path d="M5 18.5h14M7.5 15.5V12M12 15.5V8.5M16.5 15.5v-5" fill="none" stroke="currentColor" stroke-linecap="round" stroke-width="1.8"/>
          <rect x="4" y="4" width="16" height="16" rx="4" fill="none" stroke="currentColor" stroke-width="1.6"/>
        </svg>
        """,
        "ready": """
        <svg viewBox="0 0 24 24" aria-hidden="true">
          <path d="M7 12.5 10.2 15.7 17 9" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" stroke-width="1.9"/>
          <circle cx="12" cy="12" r="9" fill="none" stroke="currentColor" stroke-width="1.6"/>
        </svg>
        """,
        "error": """
        <svg viewBox="0 0 24 24" aria-hidden="true">
          <path d="M12 8v5M12 16.5h.01" fill="none" stroke="currentColor" stroke-linecap="round" stroke-width="1.9"/>
          <path d="M10.4 4.8 3.8 16.2A1.4 1.4 0 0 0 5 18.3h14a1.4 1.4 0 0 0 1.2-2.1L13.6 4.8a1.8 1.8 0 0 0-3.2 0Z" fill="none" stroke="currentColor" stroke-linejoin="round" stroke-width="1.6"/>
        </svg>
        """,
        "truth": """
        <svg viewBox="0 0 24 24" aria-hidden="true">
          <path d="M6.8 12.5 10.1 15.8 17.2 8.7" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" stroke-width="1.9"/>
          <circle cx="12" cy="12" r="9" fill="none" stroke="currentColor" stroke-width="1.6"/>
        </svg>
        """,
        "deception": """
        <svg viewBox="0 0 24 24" aria-hidden="true">
          <path d="m8.4 8.4 7.2 7.2M15.6 8.4l-7.2 7.2" fill="none" stroke="currentColor" stroke-linecap="round" stroke-width="1.9"/>
          <circle cx="12" cy="12" r="9" fill="none" stroke="currentColor" stroke-width="1.6"/>
        </svg>
        """,
        "empty": """
        <svg viewBox="0 0 24 24" aria-hidden="true">
          <path d="M12 16V7.2M8.8 10.4 12 7.2l3.2 3.2" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" stroke-width="1.8"/>
          <path d="M5.5 18h13" fill="none" stroke="currentColor" stroke-linecap="round" stroke-width="1.8"/>
          <path d="M6.5 5.5h11a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2h-11a2 2 0 0 1-2-2v-9a2 2 0 0 1 2-2Z" fill="none" stroke="currentColor" stroke-width="1.6"/>
        </svg>
        """,
    }
    return "".join(line.strip() for line in icons[name].splitlines())


def brand_logo_markup() -> str:
    if LOGO_PATH:
        mime = "image/png" if LOGO_PATH.suffix.lower() == ".png" else "image/jpeg"
        encoded = base64.b64encode(LOGO_PATH.read_bytes()).decode("ascii")
        return (
            f'<img src="data:{mime};base64,{encoded}" '
            'alt="DeceptiLens logo" class="hero-logo-image"/>'
        )
    return svg_icon("brand")


def theme_button_config(theme_name: str) -> tuple[str, str]:
    if theme_name == "light":
        return ":material/dark_mode: Dark Mode", "dark"
    return ":material/light_mode: Light Mode", "light"


def render_css(theme_name: str) -> None:
    palette = THEMES[theme_name]
    st.markdown(
        f"""
        <style>
          :root {{
            color-scheme: {theme_name};
          }}

          html, body, [class*="css"] {{
            font-family: Inter, "Segoe UI", sans-serif;
          }}

          .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {{
            background: {palette["bg"]};
            color: {palette["text"]};
          }}

          * {{
            transition:
              background-color 320ms ease,
              color 320ms ease,
              border-color 320ms ease,
              box-shadow 320ms ease,
              fill 320ms ease,
              stroke 320ms ease;
          }}

          [data-testid="stHeader"] {{
            background: rgba(0, 0, 0, 0);
          }}

          [data-testid="stToolbar"] {{
            right: 1rem;
          }}

          .block-container {{
            padding-top: 1.25rem;
            padding-bottom: 1.5rem;
            max-width: 1380px;
          }}

          h1, h2, h3, h4, p, li, label, div {{
            color: {palette["text"]};
          }}

          .hero-shell {{
            padding: 0.1rem 0 0.7rem 0;
          }}

          .hero-grid {{
            display: flex;
            align-items: center;
            gap: 1rem;
          }}

          .hero-mark {{
            width: 72px;
            height: 72px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            border-radius: 8px;
            color: {palette["accent"]};
            background: {palette["surface"]};
            border: 1px solid {palette["border"]};
            overflow: hidden;
            flex: 0 0 auto;
          }}

          .hero-mark svg,
          .section-icon svg,
          .status-icon svg,
          .verdict-icon svg,
          .empty-icon svg {{
            width: 24px;
            height: 24px;
          }}

          .hero-logo-image {{
            width: 100%;
            height: 100%;
            object-fit: contain;
            display: block;
          }}

          .hero-title {{
            font-size: 2.75rem;
            font-weight: 700;
            letter-spacing: 0;
            line-height: 1.05;
            margin-bottom: 0.35rem;
          }}

          .hero-subtitle {{
            color: {palette["muted"]};
            font-size: 1rem;
            max-width: 760px;
            line-height: 1.55;
          }}

          .stat-card,
          .info-card,
          .summary-card,
          .metric-card,
          .placeholder-panel,
          .verdict-panel,
          .status-strip {{
            background: {palette["surface"]};
            border: 1px solid {palette["border"]};
            border-radius: 8px;
            box-shadow: {palette["shadow"]};
          }}

          .stat-card,
          .info-card,
          .summary-card,
          .metric-card {{
            padding: 0.9rem 1rem;
            min-height: 96px;
          }}

          .stat-label,
          .info-label,
          .metric-label,
          .eyebrow {{
            color: {palette["muted"]};
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 0.6rem;
          }}

          .stat-value,
          .metric-value {{
            font-size: 1.75rem;
            font-weight: 700;
            line-height: 1.1;
          }}

          .stat-copy,
          .info-copy,
          .summary-copy,
          .metric-copy {{
            color: {palette["muted"]};
            font-size: 0.92rem;
            line-height: 1.5;
          }}

          .summary-card {{
            height: 188px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            margin-bottom: 0.95rem;
          }}

          .summary-title {{
            font-size: 1.15rem;
            font-weight: 700;
            line-height: 1.25;
            margin-bottom: 0.45rem;
          }}

          .summary-copy {{
            min-height: 4.6rem;
          }}

          .summary-card.featured {{
            background: {palette["surface_alt"]};
          }}

          .pipeline-card {{
            background: {palette["surface"]};
            border: 1px solid {palette["border"]};
            border-radius: 8px;
            box-shadow: {palette["shadow"]};
            padding: 0.95rem 1rem 1rem 1rem;
            margin-top: 1.1rem;
          }}

          .pipeline-head {{
            display: flex;
            align-items: flex-start;
            gap: 0.8rem;
            margin-bottom: 0.8rem;
          }}

          .pipeline-icon {{
            width: 36px;
            height: 36px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            border-radius: 8px;
            color: {palette["accent"]};
            background: {palette["accent_soft"]};
            border: 1px solid {palette["border"]};
            flex: 0 0 auto;
          }}

          .pipeline-icon svg {{
            width: 20px;
            height: 20px;
          }}

          .pipeline-title {{
            font-size: 1.02rem;
            font-weight: 700;
            line-height: 1.2;
            margin-bottom: 0.2rem;
          }}

          .pipeline-copy,
          .pipeline-stage {{
            color: {palette["muted"]};
            font-size: 0.92rem;
            line-height: 1.45;
          }}

          .pipeline-track {{
            width: 100%;
            height: 9px;
            border-radius: 999px;
            background: {palette["surface_alt"]};
            overflow: hidden;
            margin: 0.2rem 0 0.7rem 0;
          }}

          .pipeline-fill {{
            height: 100%;
            border-radius: 999px;
            background: linear-gradient(90deg, {palette["accent"]}, {palette["success"]});
          }}

          .pipeline-meta {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 0.8rem;
          }}

          .pipeline-percent {{
            font-size: 0.8rem;
            font-weight: 600;
            color: {palette["accent"]};
            white-space: nowrap;
          }}

          .chart-row-spacer {{
            height: 1.15rem;
          }}

          .section-heading {{
            display: flex;
            align-items: center;
            gap: 0.8rem;
            margin: 0.15rem 0 0.7rem 0;
          }}

          .section-icon {{
            width: 38px;
            height: 38px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            border-radius: 8px;
            color: {palette["accent"]};
            background: {palette["accent_soft"]};
            border: 1px solid {palette["border"]};
            flex: 0 0 auto;
          }}

          .section-title {{
            font-size: 1.12rem;
            font-weight: 600;
            margin: 0;
          }}

          .section-copy {{
            color: {palette["muted"]};
            font-size: 0.92rem;
            margin-top: 0.15rem;
          }}

          .status-strip {{
            display: flex;
            align-items: center;
            gap: 0.85rem;
            padding: 0.95rem 1rem;
            margin: 0.5rem 0 0.9rem 0;
          }}

          .status-ready .status-icon {{
            color: {palette["success"]};
            background: {palette["success_soft"]};
          }}

          .status-error .status-icon {{
            color: {palette["danger"]};
            background: {palette["danger_soft"]};
          }}

          .status-icon {{
            width: 34px;
            height: 34px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            border-radius: 8px;
            flex: 0 0 auto;
          }}

          .status-title {{
            font-weight: 600;
            margin-bottom: 0.15rem;
          }}

          .status-copy {{
            color: {palette["muted"]};
            font-size: 0.92rem;
          }}

          .verdict-panel {{
            padding: 1.15rem 1.25rem;
            margin-bottom: 0.8rem;
          }}

          .verdict-grid {{
            display: flex;
            align-items: center;
            gap: 1rem;
          }}

          .verdict-truth {{
            border-left: 4px solid {palette["success"]};
          }}

          .verdict-deception {{
            border-left: 4px solid {palette["danger"]};
          }}

          .verdict-icon {{
            width: 42px;
            height: 42px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            border-radius: 8px;
            flex: 0 0 auto;
          }}

          .verdict-truth .verdict-icon {{
            color: {palette["success"]};
            background: {palette["success_soft"]};
          }}

          .verdict-deception .verdict-icon {{
            color: {palette["danger"]};
            background: {palette["danger_soft"]};
          }}

          .verdict-label {{
            font-size: 1.5rem;
            font-weight: 700;
            line-height: 1.1;
            margin-bottom: 0.2rem;
          }}

          .verdict-copy {{
            color: {palette["muted"]};
            font-size: 0.96rem;
            line-height: 1.5;
          }}

          .placeholder-panel {{
            padding: 2rem 1.4rem;
            text-align: center;
            margin-top: 0.5rem;
          }}

          .empty-icon {{
            width: 48px;
            height: 48px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            color: {palette["accent"]};
            background: {palette["accent_soft"]};
            border: 1px solid {palette["border"]};
            border-radius: 8px;
            margin-bottom: 1rem;
          }}

          .placeholder-title {{
            font-size: 1.05rem;
            font-weight: 600;
            margin-bottom: 0.25rem;
          }}

          .placeholder-copy {{
            color: {palette["muted"]};
            max-width: 540px;
            margin: 0 auto;
            line-height: 1.55;
          }}

          [data-testid="stFileUploaderDropzone"] {{
            background: {palette["surface"]};
            border: 1.5px dashed {palette["border"]};
            border-radius: 8px;
            padding: 0.9rem 1rem;
          }}

          [data-testid="stFileUploaderDropzone"]:hover {{
            border-color: {palette["accent"]};
          }}

          [data-testid="stFileUploaderDropzoneInstructions"] > div,
          [data-testid="stFileUploaderDropzoneInstructions"] span {{
            color: {palette["muted"]};
          }}

          .stButton > button {{
            border-radius: 8px;
            border: 1px solid {palette["accent"]};
            background: {palette["accent"]};
            color: #ffffff;
            min-height: 42px;
            font-weight: 600;
            box-shadow: none;
          }}

          .stButton > button:hover {{
            border-color: {palette["accent"]};
            background: {palette["text"]};
            color: {palette["bg"]};
          }}

          .stButton > button:focus,
          .stButton > button:focus-visible {{
            outline: none;
            border-color: {palette["accent"]};
            box-shadow: 0 0 0 0.2rem rgba(79, 140, 255, 0.18);
          }}

          [data-testid="stVideo"] {{
            border-radius: 8px;
            overflow: hidden;
            border: 1px solid {palette["border"]};
          }}

          .stProgress > div > div > div > div {{
            background: {palette["accent"]};
          }}

          [data-testid="stFileUploaderDropzone"] button {{
            border-radius: 8px;
            border: 1px solid {palette["accent"]};
            background: {palette["accent"]};
            color: #ffffff;
            font-weight: 600;
          }}

          [data-testid="stFileUploaderDropzone"] button:hover {{
            background: {palette["text"]};
            color: {palette["bg"]};
            border-color: {palette["text"]};
          }}

          [data-testid="stMarkdownContainer"] a {{
            color: {palette["accent"]};
          }}

          hr {{
            border-color: {palette["border"]};
          }}

          .footer-copy {{
            color: {palette["muted"]};
            text-align: center;
            font-size: 0.84rem;
            line-height: 1.6;
            padding: 0.15rem 0 0 0;
          }}
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def load_models():
    return DeceptionEnsemble(model_dir=str(APP_DIR / "saved_models"))


def make_gauge(confidence: float, label: str, theme_name: str) -> go.Figure:
    palette = THEMES[theme_name]
    primary = palette["danger"] if label == "Deception" else palette["success"]

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=round(confidence * 100, 2),
            number={
                "suffix": "%",
                "font": {"size": 34, "color": palette["text"]},
            },
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickcolor": palette["muted"],
                    "tickfont": {"color": palette["muted"]},
                },
                "bar": {"color": primary, "thickness": 0.28},
                "bgcolor": palette["surface_alt"],
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 40], "color": palette["success_soft"]},
                    {"range": [40, 60], "color": palette["accent_soft"]},
                    {"range": [60, 100], "color": palette["danger_soft"]},
                ],
                "threshold": {
                    "line": {"color": primary, "width": 3},
                    "thickness": 0.84,
                    "value": confidence * 100,
                },
            },
            title={
                "text": "Ensemble deception probability",
                "font": {"size": 13, "color": palette["muted"]},
            },
        )
    )
    fig.update_layout(
        paper_bgcolor=palette["surface"],
        plot_bgcolor=palette["surface"],
        margin=dict(t=40, b=10, l=24, r=24),
        height=240,
    )
    return fig


def make_bar_chart(model_scores: dict, theme_name: str) -> go.Figure:
    palette = THEMES[theme_name]
    labels = list(model_scores.keys())
    values = [score * 100 for score in model_scores.values()]
    colors = [palette["danger"] if score >= 50 else palette["success"] for score in values]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker=dict(color=colors, line=dict(width=0)),
            text=[f"{score:.1f}%" for score in values],
            textposition="outside",
            textfont=dict(color=palette["muted"], size=12),
        )
    )
    fig.add_vline(x=50, line_dash="dash", line_color=palette["border"], line_width=1.5)
    fig.update_layout(
        paper_bgcolor=palette["surface"],
        plot_bgcolor=palette["surface"],
        margin=dict(t=10, b=10, l=10, r=60),
        height=240,
        showlegend=False,
        xaxis=dict(
            range=[0, 110],
            showgrid=False,
            zeroline=False,
            tickfont=dict(color=palette["muted"]),
        ),
        yaxis=dict(showgrid=False, tickfont=dict(color=palette["text"], size=11)),
    )
    return fig


def make_xai_group_chart(group_rows: list, theme_name: str) -> go.Figure:
    palette = THEMES[theme_name]
    labels = [row.get("group", "") for row in group_rows]
    values = [float(row.get("contribution_pp", 0.0)) for row in group_rows]
    colors = [palette["danger"] if value > 0 else palette["success"] for value in values]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker=dict(color=colors),
            text=[f"{value:+.2f} pp" for value in values],
            textposition="outside",
            textfont=dict(color=palette["muted"], size=11),
        )
    )
    fig.add_vline(x=0, line_dash="dash", line_color=palette["border"], line_width=1.4)
    fig.update_layout(
        paper_bgcolor=palette["surface"],
        plot_bgcolor=palette["surface"],
        margin=dict(t=10, b=10, l=10, r=90),
        height=270,
        showlegend=False,
        xaxis=dict(
            title="Contribution to deception probability (pp)",
            showgrid=False,
            tickfont=dict(color=palette["muted"]),
            title_font=dict(color=palette["muted"], size=12),
        ),
        yaxis=dict(showgrid=False, tickfont=dict(color=palette["text"], size=11)),
    )
    return fig


def build_xai_summary(result: dict) -> list[str]:
    xai = result.get("xai", {})
    top_groups = xai.get("group_contributions", [])[:3]
    top_features = xai.get("top_human_features", [])[:3]
    lines = []

    for row in top_groups:
        group = row.get("group", "Unknown group")
        contribution = float(row.get("contribution_pp", 0.0))
        direction = "toward deception" if contribution > 0 else "toward truth"
        lines.append(
            f"{group} shifted the score {direction} by {abs(contribution):.2f} percentage points."
        )

    for row in top_features:
        name = row.get("feature", row.get("raw_name", "Unknown feature"))
        contribution = float(row.get("contribution_pp", 0.0))
        z_value = float(row.get("z_value", 0.0))
        direction = "toward deception" if contribution > 0 else "toward truth"
        lines.append(
            f"{name} (standardized value {z_value:+.2f}) moved the score "
            f"{direction} by {abs(contribution):.2f} percentage points."
        )

    if not top_features:
        lines.append(
            "Most dominant feature drivers were latent visual units; see the technical section for those details."
        )

    fallback_modalities = result.get("fallback_modalities", [])
    if fallback_modalities:
        lines.append(
            "Interpretability warning: fallback all-zero features were used for "
            + ", ".join(fallback_modalities)
            + ", so attributions may be less reliable."
        )

    return lines


def section_heading(title: str, copy: str, icon: str) -> None:
    st.markdown(
        f"""
        <div class="section-heading">
          <div class="section-icon">{svg_icon(icon)}</div>
          <div>
            <div class="section-title">{title}</div>
            <div class="section-copy">{copy}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def info_card(label: str, title: str, copy: str) -> None:
    st.markdown(
        f"""
        <div class="info-card">
          <div class="info-label">{label}</div>
          <div class="stat-value">{title}</div>
          <div class="info-copy">{copy}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def summary_card(label: str, title: str, copy: str, featured: bool = False) -> None:
    featured_class = " featured" if featured else ""
    st.markdown(
        f"""
        <div class="summary-card{featured_class}">
          <div>
            <div class="info-label">{label}</div>
            <div class="summary-title">{title}</div>
          </div>
          <div class="summary-copy">{copy}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def stat_card(label: str, value: str, copy: str) -> None:
    st.markdown(
        f"""
        <div class="stat-card">
          <div class="stat-label">{label}</div>
          <div class="stat-value">{value}</div>
          <div class="stat-copy">{copy}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def verdict_panel(result: dict) -> None:
    label = result["label"]
    confidence = result["pt_confidence"]
    vote_count = result["deception_votes"]
    verdict_class = "verdict-deception" if label == "Deception" else "verdict-truth"
    icon = "deception" if label == "Deception" else "truth"

    st.markdown(
        f"""
        <div class="verdict-panel {verdict_class}">
          <div class="verdict-grid">
            <div class="verdict-icon">{svg_icon(icon)}</div>
            <div>
              <div class="eyebrow">Primary verdict</div>
              <div class="verdict-label">{label}</div>
              <div class="verdict-copy">
                PyTorch Dual-Stream confidence: {confidence * 100:.2f}% |
                Ensemble vote: {vote_count}/4 models flagged deception
              </div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def model_metric_card(name: str, probability: float, vote: int) -> None:
    decision = "Deception" if vote else "Truth"
    st.markdown(
        f"""
        <div class="metric-card">
          <div class="metric-label">{name}</div>
          <div class="metric-value">{probability * 100:.2f}%</div>
          <div class="metric-copy">Hard vote: {decision}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def pipeline_card(message: str, percent: int) -> str:
    safe_percent = max(0, min(percent, 100))
    return f"""
    <div class="pipeline-card">
      <div class="pipeline-head">
        <div class="pipeline-icon">{svg_icon("result")}</div>
        <div>
          <div class="pipeline-title">Processing Pipeline</div>
          <div class="pipeline-copy">Feature extraction and ensemble scoring are staged here while the full inference runs.</div>
        </div>
      </div>
      <div class="pipeline-track">
        <div class="pipeline-fill" style="width: {safe_percent}%;"></div>
      </div>
      <div class="pipeline-meta">
        <div class="pipeline-stage">{message}</div>
        <div class="pipeline-percent">{safe_percent}%</div>
      </div>
    </div>
    """


def status_strip(kind: str, title: str, copy: str) -> None:
    icon = "ready" if kind == "ready" else "error"
    css = "status-ready" if kind == "ready" else "status-error"
    st.markdown(
        f"""
        <div class="status-strip {css}">
          <div class="status-icon">{svg_icon(icon)}</div>
          <div>
            <div class="status-title">{title}</div>
            <div class="status-copy">{copy}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


if "theme_name" not in st.session_state:
    st.session_state.theme_name = "dark"

theme_name = st.session_state.theme_name
render_css(theme_name)
theme_label, next_theme = theme_button_config(theme_name)


toolbar_left, toolbar_right = st.columns([6.5, 1.5], vertical_alignment="center")

with toolbar_left:
    st.markdown(
        f"""
        <div class="hero-shell">
          <div class="hero-grid">
            <div class="hero-mark">{brand_logo_markup()}</div>
            <div>
              <div class="hero-title">DeceptiLens</div>
              <div class="hero-subtitle">
                Analyze interview and trial videos with ensemble-based deception scoring and clip-level explanations.
              </div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with toolbar_right:
    if st.button(
        theme_label,
        key="theme_switch",
        use_container_width=True,
    ):
        st.session_state.theme_name = next_theme
        st.rerun()


hero_stats = st.columns(4, gap="medium")
with hero_stats[0]:
    stat_card("Reported Accuracy", "81.27%", "Mean accuracy across 10-fold cross-validation.")
with hero_stats[1]:
    stat_card("Decision Stack", "4 Models", "PyTorch, HGB, SVM, and Random Forest voting ensemble.")
with hero_stats[2]:
    stat_card("Feature Space", "650 Dims", "ResNet-18, MediaPipe blendshapes, and Librosa audio features.")
with hero_stats[3]:
    stat_card("Primary Use", "Clip Review", "Single-video analysis with per-model probabilities and local XAI.")


with st.spinner("Loading saved models..."):
    try:
        ensemble = load_models()
    except Exception as exc:
        status_strip(
            "error",
            "Model initialization failed",
            f"Could not load the saved ensemble artifacts: {exc}",
        )
        st.info("Place the `saved_models/` directory beside `app.py` before running the dashboard.")
        st.stop()

status_strip(
    "ready",
    "System ready",
    "All four models loaded. The dashboard is ready for inference.",
)

st.divider()

workflow_left, workflow_right = st.columns([1.18, 0.9], gap="large")

with workflow_left:
    section_heading(
        "Clip Input",
        "Upload a video clip for inference. Supported formats: mp4, avi, mov, mkv.",
        "upload",
    )
    uploaded = st.file_uploader(
        "Select a video clip",
        type=["mp4", "avi", "mov", "mkv"],
        help="For best results use a clear face view, stable framing, and 5 to 30 seconds of speech.",
        label_visibility="collapsed",
    )
    st.caption("Recommended: frontal face visibility, audible speech, and moderate clip length.")

tmp_path = None
result = None
if uploaded:
    suffix = Path(uploaded.name).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

with workflow_left:
    if uploaded and tmp_path:
        section_heading(
            "Clip Preview",
            "Review the uploaded video before running the final inference step.",
            "upload",
        )
        st.video(tmp_path)

with workflow_right:
    section_heading(
        "System Summary",
        "The dashboard combines feature extraction, ensemble inference, and explanation outputs.",
        "stack",
    )
    summary_top = st.columns(2, gap="medium")
    with summary_top[0]:
        summary_card(
            "Feature Extraction",
            "ResNet-18 + MediaPipe + Librosa",
            "Visual embeddings, face blendshape statistics, and acoustic descriptors are fused into one feature vector.",
        )
    with summary_top[1]:
        summary_card(
            "Decision Layer",
            "Voting Ensemble",
            "The PyTorch dual-stream model leads the score while the full ensemble determines the final verdict.",
        )
    summary_card(
        "Validation",
        "10-Fold Cross-Validation",
        "Reported mean accuracy: 81.27% across the evaluation folds.",
        featured=True,
    )

    if uploaded and tmp_path:
        pipeline_slot = st.empty()
        pipeline_slot.markdown(
            pipeline_card("Validating human subject", 5), unsafe_allow_html=True
        )

        with st.spinner("Checking for a human face in the video…"):
            face_check = validate_human_face(tmp_path)

        if not face_check["face_detected"]:
            pipeline_slot.empty()
            os.unlink(tmp_path)
            status_strip(
                "error",
                "No human face detected",
                f"Scanned {face_check['frames_checked']} frames — "
                f"found a face in {face_check['frames_with_face']} "
                f"({face_check['detection_ratio'] * 100:.0f}%). "
                "DeceptiLens requires a clearly visible human subject.",
            )
            st.info(
                "Upload a video that contains a frontal view of a human face "
                "with audible speech for accurate deception analysis."
            )
            st.stop()

        pipeline_slot.markdown(
            pipeline_card("Checking for speech activity", 10), unsafe_allow_html=True
        )

        with st.spinner("Analyzing audio for speech activity…"):
            speech_check = validate_speech_activity(tmp_path)

        if not speech_check["speech_detected"]:
            pipeline_slot.empty()
            os.unlink(tmp_path)
            status_strip(
                "error",
                "No speech detected",
                f"Audio duration: {speech_check['duration_seconds']:.1f}s | "
                f"Mean energy: {speech_check['mean_rms']:.4f} | "
                f"Voice activity: {speech_check['voice_activity_ratio'] * 100:.0f}%. "
                "DeceptiLens requires audible speech to analyze deception cues.",
            )
            st.info(
                "Upload a video where a person is clearly speaking — "
                "interview clips, trial recordings, or statement videos work best."
            )
            st.stop()

        stages = [
            ("Extracting acoustic features", 20),
            ("Running face blendshape analysis", 45),
            ("Sampling visual embeddings", 70),
            ("Running ensemble inference", 90),
            ("Packaging explanation outputs", 100),
        ]
        for message, percent in stages:
            pipeline_slot.markdown(pipeline_card(message, percent), unsafe_allow_html=True)
            if percent < 100:
                time.sleep(0.25)

        with st.spinner("Running inference. This can take 30 to 60 seconds on CPU."):
            try:
                result = predict(ensemble, tmp_path)
            except Exception as exc:
                os.unlink(tmp_path)
                st.error(f"Inference failed: {exc}")
                st.stop()

        os.unlink(tmp_path)

if result:
    st.divider()
    section_heading(
        "Analysis Result",
        "Primary verdict, confidence distribution, and explainability outputs for the uploaded clip.",
        "result",
    )
    verdict_panel(result)

    metric_cols = st.columns(3, gap="medium")
    with metric_cols[0]:
        stat_card("Ensemble Verdict", result["label"], "Final decision after majority voting.")
    with metric_cols[1]:
        stat_card("Lead Model Confidence", f"{result['pt_confidence'] * 100:.2f}%", "PyTorch Dual-Stream output.")
    with metric_cols[2]:
        stat_card("Deception Votes", f"{result['deception_votes']} / 4", "Number of models voting for deception.")

    st.markdown('<div class="chart-row-spacer"></div>', unsafe_allow_html=True)

    chart_left, chart_right = st.columns(2, gap="large")
    with chart_left:
        section_heading(
            "Probability Gauge",
            "Aggregate ensemble probability calibrated on the current clip.",
            "result",
        )
        st.plotly_chart(
            make_gauge(result["confidence"], result["label"], theme_name),
            use_container_width=True,
            config={"displayModeBar": False},
        )

    with chart_right:
        section_heading(
            "Per-Model Probabilities",
            "Each model contributes a deception probability to the final vote.",
            "stack",
        )
        st.plotly_chart(
            make_bar_chart(result["model_scores"], theme_name),
            use_container_width=True,
            config={"displayModeBar": False},
        )

    st.divider()
    section_heading(
        "Explainability",
        "Feature groups and human-readable cues that moved the final probability for this clip.",
        "insight",
    )
    xai = result.get("xai", {})
    st.caption(
        "Method: "
        + xai.get("method", "Local masking")
        + ". Positive values push the decision toward deception; negative values push toward truth."
    )

    group_rows = xai.get("group_contributions", [])
    if group_rows:
        st.plotly_chart(
            make_xai_group_chart(group_rows[:6], theme_name),
            use_container_width=True,
            config={"displayModeBar": False},
        )

    st.markdown("#### Human-interpretable cues")
    human_features = xai.get("top_human_features", [])
    if human_features:
        for row in human_features[:8]:
            direction = "deception" if float(row.get("contribution_pp", 0.0)) > 0 else "truth"
            st.markdown(
                f"- **{row.get('feature', row.get('raw_name', 'feature'))}**: "
                f"{float(row.get('contribution_pp', 0.0)):+.2f} pp toward {direction} "
                f"(z={float(row.get('z_value', 0.0)):+.2f})"
            )
            description = row.get("description", "")
            if description:
                st.caption(description)
    else:
        st.info("No high-impact human-readable cues ranked in the top masked features for this clip.")

    with st.expander("Technical view: latent visual units", expanded=False):
        st.caption(
            "These are internal CNN latent dimensions. They influence the prediction but do not map directly to named facial actions."
        )
        latent_features = xai.get("top_latent_features", [])
        if latent_features:
            for row in latent_features[:6]:
                direction = "deception" if float(row.get("contribution_pp", 0.0)) > 0 else "truth"
                st.markdown(
                    f"- **{row.get('feature', row.get('raw_name', 'feature'))}**: "
                    f"{float(row.get('contribution_pp', 0.0)):+.2f} pp toward {direction} "
                    f"(z={float(row.get('z_value', 0.0)):+.2f})"
                )
                description = row.get("description", "")
                if description:
                    st.caption(description)
        else:
            st.write("No dominant latent visual units were highlighted for this clip.")

    for summary_line in build_xai_summary(result):
        st.markdown(f"- {summary_line}")

    st.divider()
    section_heading(
        "Model Breakdown",
        "Probability and hard-vote output for each model in the ensemble.",
        "stack",
    )
    breakdown_cols = st.columns(4, gap="medium")
    for index, model_name in enumerate(MODEL_ORDER):
        with breakdown_cols[index]:
            model_metric_card(
                model_name.replace(" (Best)", ""),
                result["model_scores"].get(model_name, 0.0),
                result["hard_votes"].get(model_name, 0),
            )

    st.divider()
    section_heading(
        "Analysis Notes",
        "Feature families used by the dashboard for the current clip.",
        "insight",
    )
    note_cols = st.columns(3, gap="medium")
    with note_cols[0]:
        info_card(
            "Audio Stream",
            "34 Dimensions",
            "MFCC, zero-crossing rate, RMS, and spectral centroid extracted through Librosa.",
        )
    with note_cols[1]:
        info_card(
            "Face Dynamics",
            "104 Dimensions",
            "MediaPipe blendshape statistics capture frame-level facial motion patterns.",
        )
    with note_cols[2]:
        info_card(
            "Visual Embeddings",
            "512 Dimensions",
            "Twenty sampled frames are projected into the ResNet-18 latent space and averaged.",
        )

    st.divider()
    section_heading(
        "Diagnostics",
        "Branch-level feature coverage from the extraction pipeline.",
        "diagnostics",
    )
    diagnostics = result.get("feature_diagnostics", {})
    diag_order = [
        ("audio", "Audio"),
        ("mediapipe", "Face Blendshapes"),
        ("resnet", "ResNet Frames"),
    ]
    diag_cols = st.columns(3, gap="medium")
    for index, (key, label) in enumerate(diag_order):
        branch = diagnostics.get(key, {})
        nonzero_pct = float(branch.get("nonzero_pct", 0.0))
        nonzero_dims = int(branch.get("nonzero_dims", 0))
        total_dims = int(branch.get("total_dims", 0))
        is_fallback = bool(branch.get("all_zero_fallback", False))

        with diag_cols[index]:
            st.markdown(f"**{label}**")
            st.progress(min(max(nonzero_pct / 100.0, 0.0), 1.0))
            st.caption(f"Non-zero dimensions: {nonzero_dims}/{total_dims} ({nonzero_pct:.1f}%)")
            if is_fallback:
                st.error("All-zero fallback detected")
            else:
                st.success("Live features extracted")

    fallback_modalities = result.get("fallback_modalities", [])
    if fallback_modalities:
        st.warning(
            "Fallback active in: "
            + ", ".join(fallback_modalities)
            + ". Final prediction still runs, but reliability is lower."
        )
    else:
        st.success("All feature branches produced live, non-zero values for this clip.")

elif not uploaded:
    st.markdown(
        f"""
        <div class="placeholder-panel">
          <div class="empty-icon">{svg_icon("empty")}</div>
          <div class="placeholder-title">No clip selected</div>
          <div class="placeholder-copy">
            Upload a video clip to run the full deception analysis workflow, including
            ensemble scoring, per-model comparison, and local explanation outputs.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.divider()
st.markdown(
    """
    <div class="footer-copy">
      DeceptiLens | Ensemble: PyTorch Dual-Stream, HistGradientBoosting, Support Vector Machine, Random Forest |
      Feature stack: ResNet-18, MediaPipe, Librosa | Reported 10-fold cross-validation mean accuracy: 81.27%
    </div>
    """,
    unsafe_allow_html=True,
)
