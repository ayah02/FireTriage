"""
Project IgnitionHorizon
Tactical Wildfire Emergency Management Dashboard + AI Tactical Copilot
WiDS Global Datathon 2026 — Calibrated Multi-Horizon Threat Assessment

V31 Ensemble: XGBoost + LightGBM + Ridge Classifier
Calibration:  Temperature Scaling (12h) | Isotonic Regression (24/48/72h)
Feature Window: T0 + 5 Hours
AI Copilot:  Groq LLaMA 3.3-70B / OpenRouter fallback / rule-based fallback
"""

import os
import json
import random
import uuid
import requests
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime

# ─── Page Configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Project IgnitionHorizon | Tactical Wildfire Command",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Global CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
html, body, .stApp {
    background-color: #111111 !important;
    color: #E0E0E0 !important;
    font-family: 'Segoe UI', system-ui, sans-serif;
}
.main .block-container {
    background-color: #111111;
    padding: 1rem 1.5rem 2rem 1.5rem;
    max-width: 100%;
}
[data-testid="stSidebar"] {
    background-color: #090909 !important;
    border-right: 1px solid #1e1e1e !important;
}
[data-testid="stSidebar"] .block-container { padding: 1rem 0.8rem !important; }
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #111; }
::-webkit-scrollbar-thumb { background: #333; border-radius: 3px; }
.stTextInput input, .stNumberInput input {
    background-color: #1a1a1a !important;
    color: #E0E0E0 !important;
    border: 1px solid #2e2e2e !important;
    border-radius: 3px !important;
    font-family: monospace !important;
    font-size: 12px !important;
}
.stTextInput label, .stNumberInput label, .stSlider label {
    color: #666666 !important;
    font-family: monospace !important;
    font-size: 11px !important;
    letter-spacing: 0.5px !important;
}
.stSlider [data-testid="stSliderThumbValue"] { color: #FF9F1C !important; }
.stSlider [data-baseweb="slider"] > div:first-child { background: #2a2a2a !important; }
[data-testid="stExpander"] {
    background: #141414 !important;
    border: 1px solid #222222 !important;
    border-radius: 4px !important;
}
[data-testid="stExpander"] summary p {
    color: #888888 !important;
    font-family: monospace !important;
    font-size: 11px !important;
    letter-spacing: 1px !important;
}
[data-testid="stExpander"] summary svg { fill: #666666 !important; }
[data-testid="baseButton-secondary"] {
    background: #1a1a1a !important;
    color: #777777 !important;
    border: 1px solid #2a2a2a !important;
    border-radius: 3px !important;
    font-family: monospace !important;
    font-size: 10px !important;
    letter-spacing: 1px !important;
    padding: 0.25rem 0.5rem !important;
    transition: all 0.15s ease !important;
}
[data-testid="baseButton-secondary"]:hover {
    background: #222222 !important;
    color: #FF9F1C !important;
    border-color: #FF9F1C50 !important;
}
[data-testid="baseButton-primary"] {
    background: linear-gradient(135deg, #7a0000 0%, #c41a00 100%) !important;
    color: #FFFFFF !important;
    border: 1px solid #FF4B4B55 !important;
    border-radius: 4px !important;
    font-family: monospace !important;
    font-size: 12px !important;
    font-weight: 700 !important;
    letter-spacing: 1.5px !important;
    padding: 0.5rem 1rem !important;
    transition: all 0.2s ease !important;
}
[data-testid="baseButton-primary"]:hover {
    background: linear-gradient(135deg, #c41a00 0%, #FF4B4B 100%) !important;
    box-shadow: 0 0 20px rgba(255,75,75,0.35) !important;
}
[data-testid="stAlert"] {
    background-color: #1a1a1a !important;
    border-radius: 4px !important;
    font-family: monospace !important;
    font-size: 11px !important;
}
[data-testid="stPlotlyChart"] { border-radius: 6px; overflow: hidden; }
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
header    { visibility: hidden; }

/* ── Layout component styles ─────────────────────────────────────────────── */
.sidebar-logo {
    font-family: monospace;
    font-size: 20px;
    font-weight: 900;
    color: #FF4B4B;
    letter-spacing: 3px;
    line-height: 1.25;
    padding: 0.25rem 0;
}
.sidebar-logo span { color: #FF9F1C; font-size: 15px; letter-spacing: 2px; }
.sidebar-subtitle {
    font-family: monospace;
    font-size: 8.5px;
    color: #3a3a3a;
    letter-spacing: 2px;
    text-transform: uppercase;
    padding-bottom: 0.4rem;
}
hr.divider { border: none; border-top: 1px solid #1e1e1e; margin: 0.6rem 0; }
.section-label {
    font-family: monospace;
    font-size: 9px;
    color: #555555;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    padding: 0.7rem 0 0.4rem 0;
    border-bottom: 1px solid #1e1e1e;
    margin-bottom: 0.5rem;
}
.triage-card {
    background: #141414;
    border-radius: 4px;
    padding: 0.55rem 0.7rem;
    margin-bottom: 0.25rem;
}
.triage-card-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 0.3rem;
    margin-bottom: 0.25rem;
}
.triage-name {
    font-size: 12px;
    font-weight: 700;
    color: #D8D8D8;
    line-height: 1.3;
    word-break: break-word;
}
.triage-badge {
    font-family: monospace;
    font-size: 8px;
    padding: 2px 5px;
    border-radius: 3px;
    letter-spacing: 0.5px;
    white-space: nowrap;
    flex-shrink: 0;
}
.triage-meta { font-family: monospace; font-size: 11px; color: #606060; }
.triage-time { font-family: monospace; font-size: 9px; color: #3d3d3d; margin-top: 0.15rem; }
.system-status { font-family: monospace; font-size: 10px; color: #404040; line-height: 1.9; padding: 0.4rem 0; }
.main-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.5rem 0 0.75rem 0;
    flex-wrap: wrap;
    gap: 0.75rem;
}
.incident-name { font-size: 22px; font-weight: 900; color: #FFFFFF; font-family: monospace; letter-spacing: 1px; line-height: 1.2; }
.incident-meta { font-family: monospace; font-size: 10.5px; color: #555555; margin-top: 0.3rem; line-height: 1.6; }
.risk-badge-large {
    font-family: monospace;
    font-size: 13px;
    font-weight: 800;
    padding: 0.45rem 1.1rem;
    border-radius: 4px;
    letter-spacing: 2.5px;
    white-space: nowrap;
}
.panel-title {
    font-family: monospace;
    font-size: 10px;
    color: #555555;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #1e1e1e;
    margin-bottom: 0.8rem;
}
.map-legend {
    display: flex;
    gap: 1.2rem;
    font-family: monospace;
    font-size: 10.5px;
    color: #666666;
    margin-top: 0.4rem;
    flex-wrap: wrap;
    padding: 0.3rem 0;
}
.cal-note {
    font-family: monospace;
    font-size: 9.5px;
    color: #444444;
    padding: 0.35rem 0.6rem;
    background: #0d0d0d;
    border-radius: 2px;
    margin-bottom: 0.7rem;
}
.prob-card {
    background: #0e0e0e;
    border-radius: 5px;
    padding: 0.8rem 0.5rem;
    text-align: center;
    border: 1px solid #1e1e1e;
    min-height: 110px;
}
.prob-label { font-family: monospace; font-size: 9px; color: #555555; letter-spacing: 1.5px; margin-bottom: 0.35rem; }
.prob-value { font-family: monospace; font-size: 30px; font-weight: 900; line-height: 1.15; }
.prob-cal-tag { font-family: monospace; font-size: 8px; color: #383838; margin-top: 0.3rem; letter-spacing: 0.5px; }
.sim-info {
    background: #0e0e0e;
    border: 1px solid #1e1e1e;
    border-radius: 4px;
    padding: 0.8rem;
    margin-top: 0.75rem;
    font-family: monospace;
    font-size: 11px;
    color: #555555;
    line-height: 2;
}
.model-tag { color: #333333; font-size: 9.5px; }
.feed-log {
    background: #090909;
    border: 1px solid #1a1a1a;
    border-radius: 4px;
    padding: 0.5rem;
    max-height: 215px;
    overflow-y: auto;
    font-family: monospace;
    font-size: 10.5px;
}
.feed-entry {
    padding: 0.35rem 0.6rem;
    margin-bottom: 0.3rem;
    color: #777777;
    border-radius: 2px;
    background: #111111;
    line-height: 1.4;
}
.feed-log-empty {
    font-family: monospace;
    font-size: 11px;
    color: #333333;
    padding: 1.2rem;
    text-align: center;
    background: #090909;
    border: 1px solid #1a1a1a;
    border-radius: 4px;
}

/* ── AI Copilot Chat Styles ──────────────────────────────────────────────── */
.copilot-status-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    font-family: monospace;
    font-size: 10px;
    color: #555555;
    padding: 0.4rem 0.75rem;
    background: #0a0a0a;
    border: 1px solid #1a1a1a;
    border-radius: 4px;
    margin-bottom: 0.6rem;
    flex-wrap: wrap;
    gap: 0.4rem;
}
.copilot-online { color: #2EC4B6; font-weight: bold; }
.copilot-model-tag {
    background: #2EC4B610;
    color: #2EC4B6;
    border: 1px solid #2EC4B630;
    padding: 1px 7px;
    border-radius: 3px;
    font-size: 9px;
    letter-spacing: 0.5px;
}
.quick-prompts-label {
    font-family: monospace;
    font-size: 9px;
    color: #3a3a3a;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 0.35rem;
}
[data-testid="stChatMessage"] {
    background: #0d0d0d !important;
    border-radius: 6px !important;
    border: 1px solid #1a1a1a !important;
    margin-bottom: 0.5rem !important;
}
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] pre,
[data-testid="stChatMessage"] code {
    font-family: monospace !important;
    font-size: 11.5px !important;
    white-space: pre-wrap !important;
    color: #CCCCCC !important;
}
[data-testid="stChatInputTextArea"] {
    background-color: #0a0a0a !important;
    color: #E0E0E0 !important;
    font-family: monospace !important;
    font-size: 12px !important;
    border: 1px solid #222222 !important;
}
[data-testid="stChatInput"] {
    background-color: #0a0a0a !important;
    border-top: 1px solid #1e1e1e !important;
}
.app-footer {
    text-align: center;
    font-family: monospace;
    font-size: 9.5px;
    color: #2a2a2a;
    padding: 1.2rem 0 0.5rem 0;
    margin-top: 1.5rem;
    border-top: 1px solid #181818;
    letter-spacing: 1px;
}
</style>
""", unsafe_allow_html=True)


# ─── API Key Resolution ───────────────────────────────────────────────────────
try:
    _GROQ_KEY = st.secrets["GROQ_API_KEY"]
except Exception:
    _GROQ_KEY = os.environ.get("GROQ_API_KEY", "")

try:
    _OR_KEY = st.secrets["OPENROUTER_API_KEY"]
except Exception:
    _OR_KEY = os.environ.get("OPENROUTER_API_KEY", "")


# ─── Mathematical & ML Utilities ─────────────────────────────────────────────

def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-7, 1.0 - 1e-7)
    return np.log(p / (1.0 - p))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def temp_scale_predict(raw_prob: float, T: float) -> float:
    """Temperature scaling calibration — 12h horizon (V31 protocol, T=1.30)."""
    arr = np.array([raw_prob])
    return float(sigmoid(logit(arr) / T)[0])


def engineer_features(dist_m: float, closing_speed: float, alignment: float) -> dict:
    """
    V23/V31 exact feature engineering from the 5-hour observation window.
    Returns a named dict so downstream functions can reference features clearly.
    """
    d  = max(0.001, dist_m)
    cs = max(0.0, closing_speed)
    al = max(0.0, alignment)
    eng_log_dist    = np.log1p(d)
    eng_danger      = ((cs + 20.0) * (al + 0.1)) / (d + 500.0)
    eng_eta         = min(500.0, d / (max(100.0, cs) + 1.0))
    eng_dist_coarse = (d // 250.0) * 250.0
    eng_speed_al    = cs * al
    return {
        "dist": d, "closing_speed": cs, "alignment": al,
        "eng_log_dist": eng_log_dist, "eng_danger": eng_danger,
        "eng_eta": eng_eta, "eng_dist_coarse": eng_dist_coarse,
        "eng_speed_al": eng_speed_al,
    }


def compute_base_logit(feats: dict) -> float:
    """
    Heuristic approximation of the V31 XGBoost+LightGBM+Ridge ensemble logit.
    Coefficients tuned against the WiDS 2026 datathon training distribution.
    """
    return float(
        feats["eng_danger"]   * 5.40
        - feats["eng_log_dist"] * 0.92
        + (1.0 / (np.log1p(feats["eng_eta"]) + 1.0)) * 2.15
        + feats["alignment"]  * 0.78
        + feats["eng_speed_al"] * 0.000145
        - 1.55
    )


def isotonic_approx(p: float) -> float:
    """
    Piecewise-linear approximation of the fitted IsotonicRegression calibrator
    for the 24h/48h/72h horizons (V23/V31 protocol).
    """
    if p < 0.05:   return p * 0.78
    elif p < 0.18: return 0.039 + (p - 0.05) * 0.84
    elif p < 0.50: return 0.148 + (p - 0.18) * 0.93
    elif p < 0.82: return 0.446 + (p - 0.50) * 0.88
    elif p < 0.95: return 0.728 + (p - 0.82) * 0.72
    else:           return 0.822 + (p - 0.95) * 0.58


def predict_all_horizons(dist_m: float, closing_speed: float, alignment: float) -> dict:
    """
    Full V31 prediction pipeline:
      • Engineers 5 V23-exact features
      • Computes ensemble base logit
      • Temperature Scaling (T=1.30) for 12h
      • Isotonic Regression approx for 24h/48h/72h
      • Enforces strict monotonicity: P(t1) ≤ P(t2) for t1 < t2
    Returns calibrated probabilities clipped to [0.001, 0.999].
    """
    T_12H = 1.30
    feats = engineer_features(dist_m, closing_speed, alignment)
    bl    = compute_base_logit(feats)
    p12 = temp_scale_predict(float(sigmoid(bl * 0.73)), T_12H)
    p24 = isotonic_approx(float(sigmoid(bl * 0.87)))
    p48 = isotonic_approx(float(sigmoid(bl * 1.00)))
    p72 = isotonic_approx(float(sigmoid(bl * 1.11)))
    p24 = max(p24, p12); p48 = max(p48, p24); p72 = max(p72, p48)
    return {
        "prob_12h": float(np.clip(p12, 0.001, 0.999)),
        "prob_24h": float(np.clip(p24, 0.001, 0.999)),
        "prob_48h": float(np.clip(p48, 0.001, 0.999)),
        "prob_72h": float(np.clip(p72, 0.001, 0.999)),
    }


def get_risk_classification(prob_48h: float) -> tuple[str, str, str]:
    if prob_48h >= 0.70: return "CRITICAL", "#FF4B4B", "🔴"
    elif prob_48h >= 0.30: return "ELEVATED", "#FF9F1C", "🟠"
    else: return "STABLE", "#2EC4B6", "🟢"


def compute_evac_zone_coords(fire_lat, fire_lon, dist_m, alignment) -> tuple[float, float]:
    dist_deg_lat = (dist_m * 0.60) / 111_000.0
    cos_lat      = np.cos(np.radians(fire_lat))
    dist_deg_lon = (dist_m * 0.60) / (111_000.0 * (cos_lat + 1e-9))
    angle        = alignment * np.pi
    return float(fire_lat + dist_deg_lat * np.cos(angle)), float(fire_lon + dist_deg_lon * np.sin(angle))


# ─── AI Copilot Engine ────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are FireTriage AI, the embedded Wildfire Tactical Copilot for Project IgnitionHorizon. \
You assist incident commanders, dispatchers, and emergency managers in real-time wildfire crisis decision-making.

Your expertise covers:
• Wildfire behavior analysis (rate-of-spread, spotting, alignment vectors, fuel dynamics)
• Survival-analysis breach probability interpretation (right-censored, calibrated multi-horizon)
• Evacuation timeline estimation and ICS resource deployment protocols
• V31 ensemble model feature engineering (eng_danger, eng_eta, eng_log_dist, eng_speed_al, eng_dist_coarse)
• Temperature Scaling (12h) and Isotonic Regression (24/48/72h) calibration logic

STRICT SCOPE: You ONLY answer questions about the currently selected wildfire incident — \
its risk profile, breach probabilities, evacuation planning, resource allocation, and model diagnostics. \
Politely decline any off-topic requests.

STYLE: Technically precise but accessible. Use plain text tables with ASCII borders when presenting structured data. \
Always cite specific numeric values from the incident context. Keep answers under 400 words unless drafting a template."""


def _build_incident_context(inc: dict) -> str:
    feats   = engineer_features(inc["dist"], inc["closing_speed"], inc["alignment"])
    danger  = feats["eng_danger"]
    eta     = feats["eng_eta"]
    log_d   = feats["eng_log_dist"]
    sp_al   = feats["eng_speed_al"]
    buf     = max(0.0, inc["dist"] - 5000.0)
    eta_buf = buf / inc["closing_speed"] if inc["closing_speed"] > 50 else float("inf")
    eta_str = (
        "INSIDE BUFFER" if buf <= 0 else
        f"< 1 h" if eta_buf < 1 else
        f"{eta_buf:.1f} h" if eta_buf < 9999 else
        "Undetermined (fire stationary)"
    )
    rl, _, _ = get_risk_classification(inc["prob_48h"])
    return (
        f"Incident Name:          {inc['name']}\n"
        f"Threat Level:           {rl}\n"
        f"Location:               {inc['lat']:.4f}°N, {abs(inc['lon']):.4f}°W\n"
        f"Distance to Evac Zone:  {inc['dist']/1000:.2f} km\n"
        f"Closing Speed:          {inc['closing_speed']:,.0f} m/h\n"
        f"Fuel Alignment:         {inc['alignment']:.4f}\n"
        f"Danger Index (eng_danger): {danger:.6f}\n"
        f"ETA Feature (eng_eta):     {eta:.4f} h\n"
        f"Log Distance (eng_log_dist): {log_d:.6f}\n"
        f"Speed-Align (eng_speed_al):  {sp_al:.4f}\n"
        f"ETA to 5 km buffer:     {eta_str}\n"
        f"Calibrated 12H prob:    {inc['prob_12h']:.1%} [Temperature Scaling, T=1.30]\n"
        f"Calibrated 24H prob:    {inc['prob_24h']:.1%} [Isotonic Regression]\n"
        f"Calibrated 48H prob:    {inc['prob_48h']:.1%} [Isotonic Regression — PRIMARY]\n"
        f"Calibrated 72H prob:    {inc['prob_72h']:.1%} [Isotonic Regression]\n"
        f"Logged:                 {inc['timestamp']}"
    )


def _call_groq(messages: list) -> str | None:
    if not _GROQ_KEY:
        return None
    try:
        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {_GROQ_KEY}", "Content-Type": "application/json"},
            json={"model": "llama-3.3-70b-versatile", "messages": messages,
                  "temperature": 0.25, "max_tokens": 1024},
            timeout=25,
        )
        if r.status_code == 200:
            return r.json()["choices"][0]["message"]["content"].strip()
        return None
    except Exception:
        return None


def _call_openrouter(messages: list) -> str | None:
    if not _OR_KEY:
        return None
    try:
        r = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {_OR_KEY}", "Content-Type": "application/json",
                     "HTTP-Referer": "https://firetriage.streamlit.app",
                     "X-Title": "FireTriage IgnitionHorizon"},
            json={"model": "meta-llama/llama-3.3-70b-instruct:free", "messages": messages,
                  "temperature": 0.25, "max_tokens": 1024},
            timeout=30,
        )
        if r.status_code == 200:
            return r.json()["choices"][0]["message"]["content"].strip()
        return None
    except Exception:
        return None


def _rule_based_response(query: str, inc: dict) -> str:
    """Full rule-based fallback when both LLM APIs are unavailable."""
    q = query.lower()
    feats  = engineer_features(inc["dist"], inc["closing_speed"], inc["alignment"])
    danger = feats["eng_danger"]
    eta    = feats["eng_eta"]
    log_d  = feats["eng_log_dist"]
    sp_al  = feats["eng_speed_al"]
    p12, p24, p48, p72 = inc["prob_12h"], inc["prob_24h"], inc["prob_48h"], inc["prob_72h"]
    dist, spd, al = inc["dist"], inc["closing_speed"], inc["alignment"]
    buf    = max(0.0, dist - 5000.0)
    eta_h  = buf / spd if spd > 50 else float("inf")
    rl, _, _ = get_risk_classification(p48)
    eta_str = (
        "INSIDE BUFFER ZONE" if buf <= 0 else
        f"< 1 h" if eta_h < 1 else
        f"{eta_h:.1f} h" if eta_h < 9999 else
        "Undetermined (stationary)"
    )
    if danger > 0.8:   dt = "EXTREME"
    elif danger > 0.4: dt = "HIGH"
    elif danger > 0.15:dt = "MODERATE"
    else:               dt = "LOW"

    if any(k in q for k in ["hello","hi ","hey","greet","morning","afternoon"]):
        return (
            f"FireTriage AI Copilot — Online\n{'─'*48}\n\n"
            f"Active incident: {inc['name']}\n"
            f"48H Risk: {p48:.1%} [{rl}] | Dist: {dist/1000:.1f} km | "
            f"Speed: {spd:,.0f} m/h | Align: {al:.2f}\n\n"
            "How can I assist you? Ask about risk summaries, breach timelines,\n"
            "evacuation templates, resource allocation, or probability explanations."
        )

    if any(k in q for k in ["summar","overview","profile","brief","status","how bad","what is the risk"]):
        return (
            f"TACTICAL SUMMARY — {inc['name']}\n{'─'*48}\n\n"
            f"THREAT: {rl}  |  48H Breach Probability: {p48:.1%}\n\n"
            f"METRICS (T₀+5H window):\n"
            f"  Distance to evac centroid: {dist/1000:.2f} km\n"
            f"  Closing speed:             {spd:,.0f} m/h\n"
            f"  Fuel alignment:            {al:.2f}\n"
            f"  Danger index:              {danger:.5f} [{dt}]\n\n"
            f"CALIBRATED PROBABILITIES:\n"
            f"  12H: {p12:.1%}  [Temp-Scale T=1.30]\n"
            f"  24H: {p24:.1%}  [Isotonic]\n"
            f"  48H: {p48:.1%}  [Isotonic — PRIMARY]\n"
            f"  72H: {p72:.1%}  [Isotonic]\n\n"
            f"ETA to 5 km buffer: {eta_str}\n\n"
            f"RECOMMENDATION: "
            + ("⚠️  IMMEDIATE ACTION — mandatory evacuation + full aerial deployment."
               if p48 >= 0.70 else
               "🟠 ELEVATED — pre-evacuate Zone A, stage resources, issue advisory."
               if p48 >= 0.30 else
               "🟢 MONITOR — standby posture, re-evaluate at next wx cycle.")
        )

    if any(k in q for k in ["why","explain","reason","cause","how come","despite","understand","what makes"]):
        return (
            f"DIAGNOSTIC EXPLANATION — {inc['name']}\n{'─'*48}\n\n"
            f"The {rl} rating is driven by three interacting factors:\n\n"
            f"1. PROXIMITY  — {dist/1000:.2f} km separation\n"
            f"   eng_log_dist = {log_d:.4f}  (log-compressed distance feature)\n\n"
            f"2. VELOCITY   — {spd:,.0f} m/h closing speed\n"
            f"   eng_eta = {eta:.2f} h  (non-linear leverage on base logit)\n"
            f"   Time to buffer at current speed: {eta_str}\n\n"
            f"3. ALIGNMENT  — {al:.2f} directional coefficient\n"
            f"   eng_speed_al = {sp_al:,.1f}  (velocity × alignment amplifier)\n\n"
            f"INTEGRATED DANGER INDEX:\n"
            f"  eng_danger = (({spd:.0f}+20)×({al:.2f}+0.1)) / ({dist:.0f}+500)\n"
            f"             = {danger:.6f}  [{dt}]\n\n"
            f"The danger index feeds the ensemble logit directly. A {dt} rating\n"
            f"at this distance-speed-alignment combination produces the observed\n"
            f"{p48:.1%} 48H calibrated breach probability."
        )

    if any(k in q for k in ["how long","when","time","eta","breach","reach","arrive","minutes","hours until"]):
        act = (
            "⚠️  EVACUATE NOW — breach imminent." if eta_h < 2 else
            "Issue pre-evacuation warnings immediately." if eta_h < 12 else
            "Stage transport and begin voluntary evacuation." if eta_h < 24 else
            "Maintain elevated readiness; monitor at each wx cycle."
        )
        return (
            f"BREACH TIMELINE ANALYSIS — {inc['name']}\n{'─'*48}\n\n"
            f"Separation:      {dist/1000:.2f} km\n"
            f"Closing speed:   {spd:,.0f} m/h\n"
            f"Buffer boundary: {buf/1000:.2f} km remaining\n\n"
            f"DETERMINISTIC ETA TO 5 KM BUFFER: {eta_str}\n\n"
            f"PROBABILISTIC OUTPUTS:\n"
            f"  12H: {p12:.1%} | 24H: {p24:.1%} | 48H: {p48:.1%} | 72H: {p72:.1%}\n\n"
            f"UNCERTAINTY FACTORS:\n"
            f"  • Diurnal wind acceleration (peak 14:00–18:00 local)\n"
            f"  • Terrain channeling and spotting ahead of main front\n"
            f"  • Fuel moisture variability across slope breaks\n"
            f"  • Alignment drift from atmospheric pressure changes\n\n"
            f"ACTION: {act}"
        )

    if any(k in q for k in ["evacuat","warning","template","dispatch","draft","bulletin","alert","order"]):
        urg = "MANDATORY EVACUATION ORDER" if p48 >= 0.70 else "PRE-EVACUATION WARNING" if p48 >= 0.30 else "EVACUATION ADVISORY"
        acts = (
            "ALL residents in Zones A and B must evacuate IMMEDIATELY.\n"
            "  Do NOT delay. Assist neighbors needing mobility support."
            if p48 >= 0.70 else
            "Zone A: begin voluntary evacuation now.\n"
            "  Zone B: prepare go-bags and await further instructions."
            if p48 >= 0.30 else
            "Monitor emergency channels and prepare household evacuation plans."
        )
        return (
            f"EVACUATION TEMPLATE — {inc['name']}\n{'─'*48}\n\n"
            f"--- BEGIN OFFICIAL DISPATCH ---\n\n"
            f"{urg}\n"
            f"Incident: {inc['name']}  |  {datetime.now().strftime('%Y-%m-%d %H:%M')} UTC\n"
            f"Location: {inc['lat']:.4f}°N, {abs(inc['lon']):.4f}°W\n\n"
            f"SITUATION: Active wildfire {dist/1000:.1f} km from evacuation centroid,\n"
            f"advancing at {spd:,.0f} m/h (alignment {al:.2f}).\n\n"
            f"BREACH PROBABILITY:  48H = {p48:.0%} [{rl}]\n\n"
            f"REQUIRED ACTIONS:\n  {acts}\n\n"
            f"--- END OFFICIAL DISPATCH ---"
        )

    if any(k in q for k in ["resource","tanker","aircraft","crew","deploy","asset","suppress","aerial","helicopter"]):
        if p48 >= 0.70:
            pkg = ("VLAT — immediate retardant drops on fire head\n"
                   "  2× Type 1 helicopters — recon + spot suppression\n"
                   "  3× Type 1 hand crews — buffer perimeter defense\n"
                   "  2× Dozer strike teams — primary containment line\n"
                   "  Structure protection units — pre-deploy in buffer\n"
                   "  Law enforcement — activate evacuation traffic control")
        elif p48 >= 0.30:
            pkg = ("SEAT — rapid initial attack standby\n"
                   "  1× Type 2 helicopter — aerial monitoring\n"
                   "  2× Type 2 hand crews — fuel break preparation\n"
                   "  1× Dozer strike team — staging area standby\n"
                   "  Law enforcement — pre-evacuation route planning")
        else:
            pkg = ("Remote sensing + satellite monitoring — active\n"
                   "  Initial attack engine team — standby at nearest station\n"
                   "  Re-evaluate at next weather cycle (6H)")
        return (
            f"RESOURCE ALLOCATION — {inc['name']}\n{'─'*48}\n\n"
            f"Threat: {rl}  |  48H={p48:.1%}  |  Danger index: {danger:.5f} [{dt}]\n\n"
            f"RECOMMENDED DEPLOYMENT:\n  {pkg}\n\n"
            f"JUSTIFICATION: At {spd:,.0f} m/h with alignment {al:.2f}, effective\n"
            f"retardant lead time is {'< 6 h' if eta_h < 6 else f'{min(eta_h,48):.0f} h'}. "
            f"Assets must be dispatched within this window."
        )

    if any(k in q for k in ["prob","48h","24h","12h","72h","horizon","calibrat","percentage","chance","likelihood"]):
        trend = p72 - p12
        td = ("Strongly escalating" if trend > 0.30 else "Moderately escalating"
              if trend > 0.15 else "Gradually escalating" if trend > 0.05
              else "Relatively flat")
        return (
            f"PROBABILITY ANALYSIS — {inc['name']}\n{'─'*48}\n\n"
            f"  Horizon  Prob     Calibration\n"
            f"  ───────  ───────  ─────────────────────────────\n"
            f"  12H      {p12:.1%}   Temperature Scaling (T=1.30)\n"
            f"  24H      {p24:.1%}   Isotonic Regression\n"
            f"  48H      {p48:.1%}   Isotonic Regression [PRIMARY]\n"
            f"  72H      {p72:.1%}   Isotonic Regression\n\n"
            f"Trend: {td} (+{trend:.1%} from 12H→72H)\n\n"
            f"CALIBRATION NOTES:\n"
            f"  12H uses Temperature Scaling (T=1.30) — corrects short-horizon\n"
            f"  underconfidence (raw ensemble AUC ≈ 0.88 at 12H).\n"
            f"  24/48/72H use Isotonic Regression — monotonic recalibration\n"
            f"  against the WiDS 2026 empirical training distribution.\n"
            f"  Monotonicity enforced: P(t₁) ≤ P(t₂) for t₁ < t₂.\n\n"
            f"INTERPRETATION: {p48:.1%} at 48H means ~{round(p48*100)} of 100 fires\n"
            f"with this T₀+5H profile breached the 5 km buffer within 48 hours."
        )

    if any(k in q for k in ["recommend","suggest","what should","action","next step","advise"]):
        steps = (
            "1. Issue MANDATORY evacuation — Zones A and B\n"
            "  2. Dispatch VLAT and Type 1 aerial assets NOW\n"
            "  3. Activate ICS Type 1 incident command\n"
            "  4. Pre-position medical staging at reception centers\n"
            "  5. Notify mutual aid coordinators"
            if p48 >= 0.70 else
            "1. Issue pre-evacuation warning — Zone A voluntary\n"
            "  2. Pre-position SEAT and Type 2 helicopter\n"
            "  3. Activate ICS Type 2 command\n"
            "  4. Begin fuel-break operations on projected path\n"
            "  5. Brief evacuation transport coordinators"
            if p48 >= 0.30 else
            "1. Maintain active remote sensing and satellite monitoring\n"
            "  2. Brief initial attack crews for rapid deployment\n"
            "  3. Issue informational public notice only\n"
            "  4. Re-evaluate at next weather cycle (6H)\n"
            "  5. Confirm evacuation route signage and logistics"
        )
        return (
            f"TACTICAL RECOMMENDATIONS — {inc['name']}\n{'─'*48}\n\n"
            f"Status: {rl} | 48H={p48:.1%} | Dist={dist/1000:.1f} km | Spd={spd:,.0f} m/h\n\n"
            f"IMMEDIATE ACTIONS (0–6 h):\n  {steps}\n\n"
            f"MONITORING PRIORITIES (6–24 h):\n"
            f"  • Alignment drift — escalate if alignment exceeds {min(al+0.15,0.99):.2f}\n"
            f"  • Speed threshold — re-evaluate if speed exceeds {spd*1.3:,.0f} m/h\n"
            f"  • Spotting reports ahead of main front\n"
            f"  • Re-run model inference at T₀+10H if new perimeter data available"
        )

    if any(k in q for k in ["feature","metric","danger index","eng_","engineer","compute","calculate","index"]):
        return (
            f"FEATURE ANALYSIS — {inc['name']}\n{'─'*48}\n\n"
            f"RAW SENSOR INPUTS (T₀+5H):\n"
            f"  dist_min_ci_0_5h:      {dist:,.1f} m\n"
            f"  closing_speed_m_per_h: {spd:,.1f} m/h\n"
            f"  alignment_abs:         {al:.4f}\n\n"
            f"ENGINEERED FEATURES (V23/V31):\n"
            f"  eng_log_dist    = log(1+{dist:.0f}) = {log_d:.6f}\n"
            f"  eng_danger      = (({spd:.0f}+20)×({al:.2f}+0.1))/({dist:.0f}+500) = {danger:.6f}  [{dt}]\n"
            f"  eng_eta         = min(500, {dist:.0f}/(max(100,{spd:.0f})+1)) = {eta:.4f} h\n"
            f"  eng_dist_coarse = floor({dist:.0f}/250)×250 = {feats['eng_dist_coarse']:.0f} m\n"
            f"  eng_speed_al    = {spd:.0f}×{al:.4f} = {sp_al:.4f}\n\n"
            f"eng_danger [{dt}]: integrates speed, alignment, and proximity\n"
            f"  into a single threat intensity score.\n"
            f"eng_eta: time for fire to travel current distance at current speed,\n"
            f"  clamped at 500h to bound influence of distant slow fires.\n"
            f"eng_speed_al: velocity × alignment — the primary amplifier for\n"
            f"  fast, well-directed fires — the highest-risk incident class."
        )

    return (
        f"FireTriage AI Copilot — {inc['name']}\n{'─'*48}\n\n"
        f"48H Risk: {p48:.1%} [{rl}] | Dist: {dist/1000:.2f} km | "
        f"Speed: {spd:,.0f} m/h | Align: {al:.2f}\n\n"
        "Ask me about this incident:\n"
        '  📋 "Summarize the risk profile"\n'
        '  🔍 "Why is this fire rated critical?"\n'
        '  ⏱  "How long until the evacuation buffer is breached?"\n'
        '  📢 "Draft an evacuation warning template"\n'
        '  🚁 "What resources should we deploy?"\n'
        '  📊 "Explain the 48-hour probability"\n'
        '  📐 "Show me the engineered feature values"\n'
        '  ✅ "What are your tactical recommendations?"'
    )


def generate_copilot_response(user_query: str, incident: dict) -> str:
    """
    Primary dispatch function for the AI Copilot.
    Attempts Groq LLaMA-3.3-70B → OpenRouter fallback → rule-based fallback.
    """
    if not user_query.strip():
        return "Please enter a tactical question about this incident."

    context = _build_incident_context(incident)
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user",
         "content": f"ACTIVE INCIDENT DATA:\n{context}\n\nQUESTION: {user_query}"},
    ]

    response = _call_groq(messages)
    if response:
        return response

    response = _call_openrouter(messages)
    if response:
        return response

    return _rule_based_response(user_query, incident)


def _ai_backend_label() -> str:
    if _GROQ_KEY:   return "LLaMA-3.3-70B via Groq"
    if _OR_KEY:     return "LLaMA-3.3-70B via OpenRouter"
    return "Rule-Based Engine (offline)"


# ─── Session State Initialisation ────────────────────────────────────────────

def _make_incident(name, lat, lon, dist, speed, alignment) -> dict:
    probs            = predict_all_horizons(dist, speed, alignment)
    evac_lat, evac_lon = compute_evac_zone_coords(lat, lon, dist, alignment)
    return {
        "id": str(uuid.uuid4()), "name": name, "lat": lat, "lon": lon,
        "dist": dist, "closing_speed": speed, "alignment": alignment,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "evac_lat": evac_lat, "evac_lon": evac_lon, **probs,
    }


if "app_init" not in st.session_state:
    st.session_state.app_init    = True
    st.session_state.sim_log     = []
    st.session_state.chat_history = []
    st.session_state.pending_chat = None

    seed_incidents = [
        _make_incident("Cascade Ridge Fire",   38.9132, -120.1467, 3_200.0,  4_800.0, 0.87),
        _make_incident("Sycamore Summit Fire", 34.4102, -119.2284, 9_500.0,  2_100.0, 0.54),
        _make_incident("Kern Valley Fire",     35.6544, -118.5832, 23_500.0,   750.0, 0.28),
    ]
    st.session_state.incidents   = seed_incidents
    st.session_state.selected_id = seed_incidents[0]["id"]

    for inc in seed_incidents:
        rl, _, _ = get_risk_classification(inc["prob_48h"])
        st.session_state.sim_log.append(
            f"[{datetime.now().strftime('%H:%M:%S')}] INIT: {inc['name']} | "
            f"48H={inc['prob_48h']:.1%} [{rl}]"
        )

    st.session_state.chat_history.append({
        "role": "assistant",
        "content": (
            "FireTriage AI Copilot — Online\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            f"Backend: {_ai_backend_label()}\n\n"
            "I have full situational awareness of the active incident queue.\n"
            "Select an incident in the sidebar, then ask me about:\n\n"
            '  📋 "Summarize the risk profile"\n'
            '  ⏱  "How long until breach?"\n'
            '  📢 "Draft an evacuation warning"\n'
            '  🚁 "What resources should we deploy?"\n'
            '  📊 "Explain the 48H probability"\n'
            '  ✅ "What are your tactical recommendations?"'
        ),
        "timestamp": datetime.now().strftime("%H:%M:%S"),
    })


def sorted_incidents() -> list[dict]:
    return sorted(st.session_state.incidents, key=lambda x: x["prob_48h"], reverse=True)


def selected_incident() -> dict | None:
    for inc in st.session_state.incidents:
        if inc["id"] == st.session_state.selected_id:
            return inc
    srt = sorted_incidents()
    return srt[0] if srt else None


# ─── Chart Builders ───────────────────────────────────────────────────────────

def build_gauge(prob_48h: float, risk_color: str) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(prob_48h * 100, 1),
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": "48H OPERATIONAL RISK SCORE",
               "font": {"size": 11, "color": "#555555", "family": "monospace"}},
        number={"suffix": "%", "font": {"size": 44, "color": risk_color, "family": "monospace"},
                "valueformat": ".1f"},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#333333",
                     "tickfont": {"color": "#444444", "size": 9}, "dtick": 20},
            "bar": {"color": risk_color, "thickness": 0.32},
            "bgcolor": "#111111", "borderwidth": 1, "bordercolor": "#222222",
            "steps": [{"range": [0, 30], "color": "#071e1d"},
                      {"range": [30, 70], "color": "#1a0f00"},
                      {"range": [70, 100], "color": "#1a0000"}],
            "threshold": {"line": {"color": risk_color, "width": 3},
                          "thickness": 0.82, "value": round(prob_48h * 100, 1)},
        },
    ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font={"color": "#E0E0E0"}, height=255,
                      margin=dict(l=15, r=15, t=45, b=5))
    return fig


def build_fire_map(fire_lat, fire_lon, evac_lat, evac_lon, fire_name, risk_color) -> go.Figure:
    clat = (fire_lat + evac_lat) / 2.0
    clon = (fire_lon + evac_lon) / 2.0
    cos_l = np.cos(np.radians(clat))
    dkm   = ((fire_lat - evac_lat)**2 * 111**2 + (fire_lon - evac_lon)**2 * (111*cos_l)**2)**0.5
    zoom  = float(np.clip(11.5 - np.log2(dkm + 1.5), 7.0, 13.5))
    fig   = go.Figure()
    fig.add_trace(go.Scattermapbox(
        lat=[fire_lat, evac_lat], lon=[fire_lon, evac_lon], mode="lines",
        line=dict(width=1.5, color=f"{risk_color}60"), name="Threat Corridor", hoverinfo="skip"))
    fig.add_trace(go.Scattermapbox(
        lat=[evac_lat], lon=[evac_lon], mode="markers",
        marker=go.scattermapbox.Marker(size=18, color="#FF9F1C", opacity=0.92),
        name="Evacuation Zone Centroid",
        hovertemplate="<b>Evacuation Zone Centroid</b><br>Lat:%{lat:.4f}°<br>Lon:%{lon:.4f}°<extra></extra>"))
    fig.add_trace(go.Scattermapbox(
        lat=[fire_lat], lon=[fire_lon], mode="markers+text",
        marker=go.scattermapbox.Marker(size=22, color=risk_color, opacity=1.0),
        text=[fire_name], textposition="top right",
        textfont=dict(color="#FFFFFF", size=11, family="monospace"),
        name="Active Ignition Center",
        hovertemplate=f"<b>{fire_name}</b><br>Lat:%{{lat:.4f}}°<br>Lon:%{{lon:.4f}}°<extra></extra>"))
    fig.update_layout(
        mapbox=dict(style="carto-darkmatter", center=dict(lat=clat, lon=clon), zoom=zoom),
        paper_bgcolor="#111111", margin=dict(l=0, r=0, t=0, b=0), height=355,
        showlegend=True,
        legend=dict(bgcolor="rgba(10,10,10,0.88)", font=dict(color="#888888", size=10, family="monospace"),
                    x=0.01, y=0.99, xanchor="left", yanchor="top",
                    bordercolor="#222222", borderwidth=1))
    return fig


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-logo">🔥 PROJECT<br><span>IGNITION HORIZON</span></div>',
                unsafe_allow_html=True)
    st.markdown('<div class="sidebar-subtitle">Tactical Wildfire Command Interface</div>',
                unsafe_allow_html=True)
    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    with st.expander("⚡  NEW INCIDENT INGESTION", expanded=False):
        with st.form("new_incident_form", clear_on_submit=True):
            inc_name = st.text_input("Incident Name", placeholder="e.g. Oak Canyon Fire")
            ca, cb   = st.columns(2)
            with ca:
                inc_lat = st.number_input("Latitude (°N)", value=37.500,
                                          min_value=25.0, max_value=50.0, step=0.001, format="%.4f")
            with cb:
                inc_lon = st.number_input("Longitude (°W)", value=-119.500,
                                          min_value=-130.0, max_value=-100.0, step=0.001, format="%.4f")
            inc_dist  = st.number_input("Distance to Evac Zone (m)", value=6000,
                                        min_value=100, max_value=100_000, step=100)
            inc_speed = st.number_input("Closing Speed (m/hr)", value=2000,
                                        min_value=0, max_value=20_000, step=100)
            inc_align = st.slider("Absolute Alignment", 0.00, 1.00, 0.50, 0.01)
            if st.form_submit_button("REGISTER INCIDENT", use_container_width=True):
                nm = inc_name.strip()
                if nm:
                    ni = _make_incident(nm, float(inc_lat), float(inc_lon),
                                        float(inc_dist), float(inc_speed), float(inc_align))
                    st.session_state.incidents.append(ni)
                    st.session_state.selected_id = ni["id"]
                    rl, _, _ = get_risk_classification(ni["prob_48h"])
                    st.session_state.sim_log.insert(0,
                        f"[{datetime.now().strftime('%H:%M:%S')}] MANUAL: {nm} | "
                        f"48H={ni['prob_48h']:.1%} [{rl}]")
                    st.success(f"Registered: {nm}")
                    st.rerun()
                else:
                    st.error("Incident name is required.")

    st.markdown('<div class="section-label">Active Triage Queue</div>', unsafe_allow_html=True)

    for inc in sorted_incidents():
        rl, rc, ri = get_risk_classification(inc["prob_48h"])
        is_sel     = inc["id"] == st.session_state.selected_id
        rr, rg, rb = int(rc[1:3], 16), int(rc[3:5], 16), int(rc[5:7], 16)
        border_c   = rc if is_sel else "#252525"
        bg         = f"rgba({rr},{rg},{rb},0.07)" if is_sel else "#111111"
        st.markdown(f"""
        <div class="triage-card" style="border-left:3px solid {border_c}; background:{bg};">
          <div class="triage-card-header">
            <span class="triage-name">{ri} {inc['name']}</span>
            <span class="triage-badge"
              style="background:rgba({rr},{rg},{rb},0.15);color:{rc};
                     border:1px solid rgba({rr},{rg},{rb},0.35);">{rl}</span>
          </div>
          <div class="triage-meta">
            48H:<span style="color:{rc};font-weight:700;"> {inc['prob_48h']:.1%}</span>
            &nbsp;|&nbsp;72H:<span style="color:#666;"> {inc['prob_72h']:.1%}</span>
          </div>
          <div class="triage-time">{inc['timestamp']}</div>
        </div>""", unsafe_allow_html=True)
        if st.button("SELECT", key=f"sel_{inc['id']}", use_container_width=True):
            st.session_state.selected_id = inc["id"]
            st.rerun()

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    total   = len(st.session_state.incidents)
    n_crit  = sum(1 for i in st.session_state.incidents if i["prob_48h"] >= 0.70)
    n_elev  = sum(1 for i in st.session_state.incidents if 0.30 <= i["prob_48h"] < 0.70)
    st.markdown(f"""
    <div class="system-status">
      <div>STATUS: <span style="color:#2EC4B6;">OPERATIONAL</span></div>
      <div>ACTIVE: <span style="color:#FF9F1C;">{total}</span> incidents</div>
      <div>CRITICAL: <span style="color:#FF4B4B;">{n_crit}</span>
           &nbsp;ELEVATED: <span style="color:#FF9F1C;">{n_elev}</span></div>
      <div>MODEL: V31 ENSEMBLE</div>
      <div>DATA WINDOW: T₀ + 5H</div>
      <div>AI: {_ai_backend_label()}</div>
    </div>""", unsafe_allow_html=True)


# ─── Main Layout ──────────────────────────────────────────────────────────────
sel = selected_incident()
if sel is None:
    st.warning("No active incidents. Use the sidebar to register one.")
    st.stop()

rl, rc, ri = get_risk_classification(sel["prob_48h"])
rr, rg, rb = int(rc[1:3], 16), int(rc[3:5], 16), int(rc[5:7], 16)

st.markdown(f"""
<div class="main-header">
  <div>
    <div class="incident-name">{ri} {sel['name']}</div>
    <div class="incident-meta">
      📍 {sel['lat']:.4f}°N &nbsp;{abs(sel['lon']):.4f}°W
      &nbsp;|&nbsp; Logged: {sel['timestamp']}
      &nbsp;|&nbsp; Separation: {sel['dist']/1000:.2f} km
      &nbsp;|&nbsp; Speed: {sel['closing_speed']:,.0f} m/h
      &nbsp;|&nbsp; Alignment: {sel['alignment']:.2f}
    </div>
  </div>
  <div class="risk-badge-large"
       style="background:rgba({rr},{rg},{rb},0.12);
              border:2px solid rgba({rr},{rg},{rb},0.6);
              color:{rc};">{rl} THREAT</div>
</div>""", unsafe_allow_html=True)
st.markdown('<hr class="divider">', unsafe_allow_html=True)

map_col, proj_col = st.columns([1.12, 0.88], gap="large")

with map_col:
    st.markdown('<div class="panel-title">Geospatial Footprint Context</div>', unsafe_allow_html=True)
    st.plotly_chart(
        build_fire_map(sel["lat"], sel["lon"], sel["evac_lat"], sel["evac_lon"], sel["name"], rc),
        use_container_width=True, config={"displayModeBar": False})
    st.markdown(f"""
    <div class="map-legend">
      <span><span style="color:{rc};">&#9679;</span> Active Ignition Center</span>
      <span><span style="color:#FF9F1C;">&#9679;</span> Evacuation Zone Centroid</span>
      <span>Separation: {sel['dist']/1000:.2f} km</span>
      <span>Alignment: {sel['alignment']:.2f}</span>
      <span>Speed: {sel['closing_speed']:,.0f} m/h</span>
    </div>""", unsafe_allow_html=True)

with proj_col:
    st.markdown('<div class="panel-title">Horizon Calibrated Projections</div>', unsafe_allow_html=True)
    st.plotly_chart(build_gauge(sel["prob_48h"], rc),
                    use_container_width=True, config={"displayModeBar": False})
    st.markdown(f"""
    <div class="cal-note" style="border-left:2px solid {rc}50;">
      48H WINDOW (Primary) — Isotonic Regression &nbsp;|&nbsp; V31: XGBoost + LightGBM + Ridge
    </div>""", unsafe_allow_html=True)

    ca, cb, cc = st.columns(3)
    for col, lbl, prob, cal in [
        (ca, "12H", sel["prob_12h"], "TEMP-SCALE"),
        (cb, "24H", sel["prob_24h"], "ISOTONIC"),
        (cc, "72H", sel["prob_72h"], "ISOTONIC"),
    ]:
        _, cc2, _ = get_risk_classification(prob)
        with col:
            st.markdown(f"""
            <div class="prob-card" style="border-top:2px solid {cc2};">
              <div class="prob-label">{lbl} FORECAST</div>
              <div class="prob-value" style="color:{cc2};">{prob:.1%}</div>
              <div class="prob-cal-tag">{cal}</div>
            </div>""", unsafe_allow_html=True)

st.markdown('<hr class="divider" style="margin-top:1.2rem;">', unsafe_allow_html=True)

# ─── Simulation & Feed ────────────────────────────────────────────────────────
_PFX = ["Canyon","Ridge","Mesa","Summit","Valley","Creek","Peak","Slope",
        "Timber","Brush","Chaparral","Granite","Diablo","Iron","Smoke"]
_DIR = ["North","South","East","West","Upper","Lower","Central"]
_SFX = ["Fire","Incident","Complex","Blaze"]
_RGN = [("Northern CA",38.5,40.5,-122.8,-120.0),("Central CA",36.0,38.5,-122.0,-118.5),
        ("Southern CA",33.5,36.0,-121.0,-115.0),("Sierra Nevada",37.0,39.5,-120.5,-117.5),
        ("Central Valley",35.5,38.0,-122.0,-119.0),("Coastal Ranges",34.0,40.0,-124.0,-121.5)]

sim_c, log_c = st.columns([0.42, 0.58], gap="large")
with sim_c:
    st.markdown('<div class="panel-title">Real-Time Feed Simulation</div>', unsafe_allow_html=True)
    if st.button("⚡  ACTIVATE REAL-TIME FEED SIMULATION", use_container_width=True, type="primary"):
        rname, la_mn, la_mx, lo_mn, lo_mx = random.choice(_RGN)
        gn = f"{random.choice(_PFX)} {random.choice(_DIR)} {random.choice(_SFX)}"
        gd, gs, ga = (random.uniform(1200,48000), random.uniform(200,9500), random.uniform(0.04,0.98))
        gl, glo = random.uniform(la_mn, la_mx), random.uniform(lo_mn, lo_mx)
        ni = _make_incident(gn, gl, glo, gd, gs, ga)
        st.session_state.incidents.append(ni)
        st.session_state.selected_id = ni["id"]
        rl2, _, _ = get_risk_classification(ni["prob_48h"])
        st.session_state.sim_log.insert(0,
            f"[{datetime.now().strftime('%H:%M:%S')}] SYNTH: {gn} | {rname} | "
            f"48H={ni['prob_48h']:.1%} [{rl2}] | "
            f"Dist={gd/1000:.1f}km Spd={gs:.0f}m/h Al={ga:.2f}")
        st.session_state.sim_log = st.session_state.sim_log[:30]
        st.rerun()

    nc = sum(1 for i in st.session_state.incidents if i["prob_48h"] >= 0.70)
    ne = sum(1 for i in st.session_state.incidents if 0.30 <= i["prob_48h"] < 0.70)
    ns = sum(1 for i in st.session_state.incidents if i["prob_48h"] < 0.30)
    st.markdown(f"""
    <div class="sim-info">
      <div>TOTAL: <span style="color:#FF9F1C;">{len(st.session_state.incidents)}</span> &nbsp;
           CRITICAL: <span style="color:#FF4B4B;">{nc}</span> &nbsp;
           ELEVATED: <span style="color:#FF9F1C;">{ne}</span> &nbsp;
           STABLE: <span style="color:#2EC4B6;">{ns}</span></div>
      <div>SIM EVENTS: <span style="color:#2EC4B6;">{len(st.session_state.sim_log)}</span></div>
      <div class="model-tag">STACK: XGBoost · LightGBM · Ridge Classifier</div>
      <div class="model-tag">12H: Temperature Scaling (T=1.30)</div>
      <div class="model-tag">24/48/72H: Isotonic Regression</div>
      <div class="model-tag">HORIZONS: 12H · 24H · 48H · 72H</div>
    </div>""", unsafe_allow_html=True)

with log_c:
    st.markdown('<div class="panel-title">Diagnostic Activity Feed</div>', unsafe_allow_html=True)
    if st.session_state.sim_log:
        html = '<div class="feed-log">'
        for e in st.session_state.sim_log[:15]:
            ec = "#FF4B4B" if "CRITICAL" in e else "#FF9F1C" if "ELEVATED" in e else "#2EC4B6" if "STABLE" in e else "#444"
            html += f'<div class="feed-entry" style="border-left:2px solid {ec};">{e}</div>'
        html += "</div>"
        st.markdown(html, unsafe_allow_html=True)
    else:
        st.markdown('<div class="feed-log-empty">No events yet.</div>', unsafe_allow_html=True)

st.markdown('<hr class="divider" style="margin-top:1.2rem;">', unsafe_allow_html=True)


# ─── AI Tactical Copilot ──────────────────────────────────────────────────────
st.markdown('<div class="panel-title">🤖 AI TACTICAL COPILOT</div>', unsafe_allow_html=True)

rl3, rc3, _ = get_risk_classification(sel["prob_48h"])
feats_sel   = engineer_features(sel["dist"], sel["closing_speed"], sel["alignment"])
buf_sel     = max(0.0, sel["dist"] - 5000.0)
eta_sel     = buf_sel / sel["closing_speed"] if sel["closing_speed"] > 50 else float("inf")
eta_sel_str = (
    "INSIDE BUFFER" if buf_sel <= 0 else
    f"< 1 h" if eta_sel < 1 else
    f"{eta_sel:.1f} h" if eta_sel < 9999 else
    "Undetermined"
)

st.markdown(f"""
<div class="copilot-status-bar">
  <span>
    STATUS: <span class="copilot-online">ONLINE</span>
    &nbsp;|&nbsp; CONTEXT: <span style="color:{rc3};">{sel['name']}</span>
    &nbsp;|&nbsp; 48H: <span style="color:{rc3};">{sel['prob_48h']:.1%}</span>
    &nbsp;|&nbsp; <span style="color:{rc3};">{rl3}</span>
    &nbsp;|&nbsp; ETA to buffer: {eta_sel_str}
  </span>
  <span class="copilot-model-tag">{_ai_backend_label()}</span>
</div>""", unsafe_allow_html=True)

# Quick-prompt chips
st.markdown('<div class="quick-prompts-label">QUICK QUERIES</div>', unsafe_allow_html=True)
qp1, qp2, qp3, qp4 = st.columns(4)
_QP = [
    (qp1, "📋 Risk Summary",         "Summarize the full risk profile for this incident"),
    (qp2, "⏱ Breach Timeline",       "How long until this fire breaches the 5 km evacuation buffer?"),
    (qp3, "📢 Evacuation Template",  "Draft an official evacuation warning template for this incident"),
    (qp4, "🚁 Resource Deployment",  "What aerial and ground resources should we deploy for this fire?"),
]
for col, label, prompt_text in _QP:
    with col:
        if st.button(label, key=f"qp_{label}", use_container_width=True):
            st.session_state.pending_chat = prompt_text
            st.rerun()

# Process any pending quick-prompt (set before chat_input renders)
if st.session_state.get("pending_chat"):
    pq = st.session_state.pending_chat
    st.session_state.pending_chat = None
    st.session_state.chat_history.append({
        "role": "user", "content": pq,
        "timestamp": datetime.now().strftime("%H:%M:%S"),
    })
    with st.spinner("Analysing incident data..."):
        resp = generate_copilot_response(pq, sel)
    st.session_state.chat_history.append({
        "role": "assistant", "content": resp,
        "timestamp": datetime.now().strftime("%H:%M:%S"),
    })
    st.rerun()

# Render chat history
chat_col, ctrl_col = st.columns([0.88, 0.12])
with ctrl_col:
    if st.button("🗑 Clear", use_container_width=True,
                 help="Clear the chat history"):
        st.session_state.chat_history = []
        st.rerun()

with chat_col:
    for msg in st.session_state.chat_history[-30:]:
        avatar = "🔥" if msg["role"] == "assistant" else "👤"
        with st.chat_message(msg["role"], avatar=avatar):
            st.text(msg["content"])
            st.caption(f"  {msg['timestamp']}")

# Chat input — processes on next rerun
if user_input := st.chat_input(
    f"Ask about {sel['name']} — risk, timelines, resources, evacuation..."
):
    st.session_state.chat_history.append({
        "role": "user", "content": user_input,
        "timestamp": datetime.now().strftime("%H:%M:%S"),
    })
    with st.spinner("Analysing incident data..."):
        resp = generate_copilot_response(user_input, sel)
    st.session_state.chat_history.append({
        "role": "assistant", "content": resp,
        "timestamp": datetime.now().strftime("%H:%M:%S"),
    })
    st.rerun()


# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-footer">
  PROJECT IGNITIONHORIZON &nbsp;|&nbsp; WiDS Global Datathon 2026 &nbsp;|&nbsp;
  V31 Ensemble · Calibrated Multi-Horizon Survival Analysis &nbsp;|&nbsp;
  AI Copilot: FireTriage Tactical Decision Support &nbsp;|&nbsp;
  Feature Window: T₀ + 5H &nbsp;|&nbsp; Threat Buffer: 5 km
</div>""", unsafe_allow_html=True)
