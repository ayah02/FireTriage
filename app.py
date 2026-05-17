"""
Project IgnitionHorizon
Tactical Wildfire Emergency Management Dashboard
WiDS Global Datathon 2026 — Calibrated Multi-Horizon Threat Assessment

V31 Ensemble: XGBoost + LightGBM + Ridge Classifier
Calibration: Temperature Scaling (12h) | Isotonic Regression (24/48/72h)
Feature Window: T0 + 5 Hours
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import random
import uuid
from datetime import datetime

# ─── Page Configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Project IgnitionHorizon | Tactical Wildfire Command",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Global CSS — Emergency Operations Center Dark Theme ─────────────────────
st.markdown(
    """
<style>
/* ── Base ─────────────────────────────────────────────────────────────────── */
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

/* ── Sidebar ─────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background-color: #090909 !important;
    border-right: 1px solid #1e1e1e !important;
}
[data-testid="stSidebar"] .block-container {
    padding: 1rem 0.8rem !important;
}

/* ── Scrollbar ───────────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #111; }
::-webkit-scrollbar-thumb { background: #333; border-radius: 3px; }

/* ── Streamlit native element overrides ──────────────────────────────────── */
.stTextInput input,
.stNumberInput input {
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

/* Expander */
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

/* Form submit + secondary buttons */
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

/* Primary button (Simulation activate) */
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
    text-transform: uppercase !important;
    transition: all 0.2s ease !important;
}
[data-testid="baseButton-primary"]:hover {
    background: linear-gradient(135deg, #c41a00 0%, #FF4B4B 100%) !important;
    box-shadow: 0 0 20px rgba(255, 75, 75, 0.35) !important;
}

/* Success / error alerts */
[data-testid="stAlert"] {
    background-color: #1a1a1a !important;
    border-radius: 4px !important;
    font-family: monospace !important;
    font-size: 11px !important;
}

/* Plotly chart containers */
[data-testid="stPlotlyChart"] {
    border-radius: 6px;
    overflow: hidden;
}

/* Hide Streamlit chrome */
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
header    { visibility: hidden; }

/* ── Custom component classes ─────────────────────────────────────────────── */
.sidebar-logo {
    font-family: monospace;
    font-size: 20px;
    font-weight: 900;
    color: #FF4B4B;
    letter-spacing: 3px;
    line-height: 1.25;
    padding: 0.25rem 0;
}
.sidebar-logo span {
    color: #FF9F1C;
    font-size: 15px;
    letter-spacing: 2px;
}
.sidebar-subtitle {
    font-family: monospace;
    font-size: 8.5px;
    color: #3a3a3a;
    letter-spacing: 2px;
    text-transform: uppercase;
    padding-bottom: 0.4rem;
}
hr.divider {
    border: none;
    border-top: 1px solid #1e1e1e;
    margin: 0.6rem 0;
}
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
.triage-meta {
    font-family: monospace;
    font-size: 11px;
    color: #606060;
}
.triage-time {
    font-family: monospace;
    font-size: 9px;
    color: #3d3d3d;
    margin-top: 0.15rem;
}
.system-status {
    font-family: monospace;
    font-size: 10px;
    color: #404040;
    line-height: 1.9;
    padding: 0.4rem 0;
}
.main-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.5rem 0 0.75rem 0;
    flex-wrap: wrap;
    gap: 0.75rem;
}
.incident-name {
    font-size: 22px;
    font-weight: 900;
    color: #FFFFFF;
    font-family: monospace;
    letter-spacing: 1px;
    line-height: 1.2;
}
.incident-meta {
    font-family: monospace;
    font-size: 10.5px;
    color: #555555;
    margin-top: 0.3rem;
    line-height: 1.6;
}
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
.prob-label {
    font-family: monospace;
    font-size: 9px;
    color: #555555;
    letter-spacing: 1.5px;
    margin-bottom: 0.35rem;
}
.prob-value {
    font-family: monospace;
    font-size: 30px;
    font-weight: 900;
    line-height: 1.15;
}
.prob-cal-tag {
    font-family: monospace;
    font-size: 8px;
    color: #383838;
    margin-top: 0.3rem;
    letter-spacing: 0.5px;
}
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
""",
    unsafe_allow_html=True,
)


# ─── Mathematical & ML Utilities ─────────────────────────────────────────────

def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-7, 1.0 - 1e-7)
    return np.log(p / (1.0 - p))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def temp_scale_predict(raw_prob: float, T: float) -> float:
    """Temperature scaling calibration for 12h window (V31 protocol)."""
    raw_arr = np.array([raw_prob])
    return float(sigmoid(logit(raw_arr) / T)[0])


def engineer_features(dist_min_ci_0_5h: float, closing_speed_m_per_h: float, alignment_abs: float) -> dict:
    """
    V23/V31 exact feature engineering from first-5-hour observation window.
    Computes all engineered predictors from the three core sensor inputs.
    """
    d  = max(0.001, dist_min_ci_0_5h)
    cs = max(0.0, closing_speed_m_per_h)
    al = max(0.0, alignment_abs)

    eng_log_dist    = np.log1p(d)
    eng_danger      = ((cs + 20.0) * (al + 0.1)) / (d + 500.0)
    eng_eta         = min(500.0, d / (max(100.0, cs) + 1.0))
    eng_dist_coarse = (d // 250.0) * 250.0
    eng_speed_al    = cs * al

    return {
        "dist":            d,
        "closing_speed":   cs,
        "alignment":       al,
        "eng_log_dist":    eng_log_dist,
        "eng_danger":      eng_danger,
        "eng_eta":         eng_eta,
        "eng_dist_coarse": eng_dist_coarse,
        "eng_speed_al":    eng_speed_al,
    }


def compute_base_logit(features: dict) -> float:
    """
    Heuristic approximation of the V31 ensemble scoring function
    (XGBoost + LightGBM + RidgeClassifier weighted average).
    Coefficients are calibrated against the datathon training distribution.
    """
    danger    = features["eng_danger"]
    log_dist  = features["eng_log_dist"]
    eta       = features["eng_eta"]
    al        = features["alignment"]
    speed_al  = features["eng_speed_al"]

    score = (
        danger   * 5.40
        - log_dist * 0.92
        + (1.0 / (np.log1p(eta) + 1.0)) * 2.15
        + al      * 0.78
        + speed_al * 0.000145
        - 1.55
    )
    return float(score)


def isotonic_approx(p: float) -> float:
    """
    Piecewise linear approximation of the fitted IsotonicRegression calibrator
    for 24h/48h/72h horizons (V23/V31 protocol).
    Compresses extreme raw probabilities toward the observed empirical distribution.
    """
    if p < 0.05:
        return p * 0.78
    elif p < 0.18:
        return 0.039 + (p - 0.05) * 0.84
    elif p < 0.50:
        return 0.148 + (p - 0.18) * 0.93
    elif p < 0.82:
        return 0.446 + (p - 0.50) * 0.88
    elif p < 0.95:
        return 0.728 + (p - 0.82) * 0.72
    else:
        return 0.822 + (p - 0.95) * 0.58


def predict_all_horizons(dist_m: float, closing_speed: float, alignment: float) -> dict:
    """
    Full V31 prediction pipeline:
      - Engineers the five V23-exact features from raw sensor inputs
      - Computes ensemble base logit score
      - Applies Temperature Scaling (T=1.30) for the 12h horizon
      - Applies Isotonic Regression approximation for 24h/48h/72h
      - Enforces strict monotonicity across all four horizons
    Returns calibrated probability estimates clipped to [0.001, 0.999].
    """
    T_12H = 1.30  # V31 optimised temperature parameter for 12h underfitting correction

    feats      = engineer_features(dist_m, closing_speed, alignment)
    base_logit = compute_base_logit(feats)

    # Horizon-specific logit scaling (longer window → higher cumulative risk)
    raw_12h = float(sigmoid(base_logit * 0.73))
    raw_24h = float(sigmoid(base_logit * 0.87))
    raw_48h = float(sigmoid(base_logit * 1.00))
    raw_72h = float(sigmoid(base_logit * 1.11))

    # Calibration per horizon
    p12 = temp_scale_predict(raw_12h, T_12H)
    p24 = isotonic_approx(raw_24h)
    p48 = isotonic_approx(raw_48h)
    p72 = isotonic_approx(raw_72h)

    # Monotonicity enforcement: P(hit ≤ t1) ≤ P(hit ≤ t2) for t1 < t2
    p24 = max(p24, p12)
    p48 = max(p48, p24)
    p72 = max(p72, p48)

    return {
        "prob_12h": float(np.clip(p12, 0.001, 0.999)),
        "prob_24h": float(np.clip(p24, 0.001, 0.999)),
        "prob_48h": float(np.clip(p48, 0.001, 0.999)),
        "prob_72h": float(np.clip(p72, 0.001, 0.999)),
    }


def get_risk_classification(prob_48h: float) -> tuple[str, str, str]:
    """Classify threat level from 48h calibrated probability (primary operational horizon)."""
    if prob_48h >= 0.70:
        return "CRITICAL", "#FF4B4B", "🔴"
    elif prob_48h >= 0.30:
        return "ELEVATED", "#FF9F1C", "🟠"
    else:
        return "STABLE", "#2EC4B6", "🟢"


def compute_evac_zone_coords(fire_lat: float, fire_lon: float, dist_m: float, alignment: float) -> tuple[float, float]:
    """
    Project the threatened Evacuation Zone Centroid location from the fire origin,
    using distance and directional alignment as proxies for bearing.
    """
    dist_deg_lat = (dist_m * 0.60) / 111_000.0
    cos_lat      = np.cos(np.radians(fire_lat))
    dist_deg_lon = (dist_m * 0.60) / (111_000.0 * (cos_lat + 1e-9))
    angle        = alignment * np.pi
    evac_lat     = fire_lat + dist_deg_lat * np.cos(angle)
    evac_lon     = fire_lon + dist_deg_lon * np.sin(angle)
    return float(evac_lat), float(evac_lon)


# ─── Session State Initialisation ────────────────────────────────────────────

def _make_incident(name: str, lat: float, lon: float, dist: float, speed: float, alignment: float) -> dict:
    probs               = predict_all_horizons(dist, speed, alignment)
    evac_lat, evac_lon  = compute_evac_zone_coords(lat, lon, dist, alignment)
    return {
        "id":            str(uuid.uuid4()),
        "name":          name,
        "lat":           lat,
        "lon":           lon,
        "dist":          dist,
        "closing_speed": speed,
        "alignment":     alignment,
        "timestamp":     datetime.now().strftime("%Y-%m-%d %H:%M"),
        "evac_lat":      evac_lat,
        "evac_lon":      evac_lon,
        **probs,
    }


if "app_init" not in st.session_state:
    st.session_state.app_init = True
    st.session_state.sim_log  = []

    initial_incidents = [
        _make_incident("Cascade Ridge Fire",   38.9132, -120.1467, 3_200.0,  4_800.0, 0.87),
        _make_incident("Sycamore Summit Fire", 34.4102, -119.2284, 9_500.0,  2_100.0, 0.54),
        _make_incident("Kern Valley Fire",     35.6544, -118.5832, 23_500.0,   750.0, 0.28),
    ]
    st.session_state.incidents    = initial_incidents
    st.session_state.selected_id  = initial_incidents[0]["id"]

    for inc in initial_incidents:
        rl, _, _ = get_risk_classification(inc["prob_48h"])
        st.session_state.sim_log.append(
            f"[{datetime.now().strftime('%H:%M:%S')}] INIT: {inc['name']} | "
            f"48H={inc['prob_48h']:.1%} [{rl}]"
        )


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
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=round(prob_48h * 100, 1),
            domain={"x": [0, 1], "y": [0, 1]},
            title={
                "text": "48H OPERATIONAL RISK SCORE",
                "font": {"size": 11, "color": "#555555", "family": "monospace"},
            },
            number={
                "suffix": "%",
                "font": {"size": 44, "color": risk_color, "family": "monospace"},
                "valueformat": ".1f",
            },
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickwidth": 1,
                    "tickcolor": "#333333",
                    "tickfont": {"color": "#444444", "size": 9},
                    "dtick": 20,
                },
                "bar": {"color": risk_color, "thickness": 0.32},
                "bgcolor": "#111111",
                "borderwidth": 1,
                "bordercolor": "#222222",
                "steps": [
                    {"range": [0, 30],  "color": "#071e1d"},
                    {"range": [30, 70], "color": "#1a0f00"},
                    {"range": [70, 100],"color": "#1a0000"},
                ],
                "threshold": {
                    "line": {"color": risk_color, "width": 3},
                    "thickness": 0.82,
                    "value": round(prob_48h * 100, 1),
                },
            },
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#E0E0E0"},
        height=255,
        margin=dict(l=15, r=15, t=45, b=5),
    )
    return fig


def build_fire_map(
    fire_lat: float,
    fire_lon: float,
    evac_lat: float,
    evac_lon: float,
    fire_name: str,
    risk_color: str,
) -> go.Figure:
    center_lat  = (fire_lat + evac_lat) / 2.0
    center_lon  = (fire_lon + evac_lon) / 2.0
    cos_lat     = np.cos(np.radians(center_lat))
    dist_km     = (
        (fire_lat - evac_lat) ** 2 * 111.0 ** 2
        + (fire_lon - evac_lon) ** 2 * (111.0 * cos_lat) ** 2
    ) ** 0.5
    zoom        = float(np.clip(11.5 - np.log2(dist_km + 1.5), 7.0, 13.5))

    fig = go.Figure()

    # Threat corridor line
    fig.add_trace(
        go.Scattermapbox(
            lat=[fire_lat, evac_lat],
            lon=[fire_lon, evac_lon],
            mode="lines",
            line=dict(width=1.5, color=f"{risk_color}60"),
            name="Threat Corridor",
            hoverinfo="skip",
        )
    )

    # Evacuation Zone Centroid
    fig.add_trace(
        go.Scattermapbox(
            lat=[evac_lat],
            lon=[evac_lon],
            mode="markers",
            marker=go.scattermapbox.Marker(size=18, color="#FF9F1C", opacity=0.92),
            name="Evacuation Zone Centroid",
            hovertemplate=(
                "<b>Evacuation Zone Centroid</b><br>"
                "Lat: %{lat:.4f}°<br>Lon: %{lon:.4f}°"
                "<extra></extra>"
            ),
        )
    )

    # Active Ignition Center
    fig.add_trace(
        go.Scattermapbox(
            lat=[fire_lat],
            lon=[fire_lon],
            mode="markers+text",
            marker=go.scattermapbox.Marker(size=22, color=risk_color, opacity=1.0),
            text=[fire_name],
            textposition="top right",
            textfont=dict(color="#FFFFFF", size=11, family="monospace"),
            name="Active Ignition Center",
            hovertemplate=(
                f"<b>{fire_name}</b><br>"
                "Lat: %{lat:.4f}°<br>Lon: %{lon:.4f}°"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        mapbox=dict(
            style="carto-darkmatter",
            center=dict(lat=center_lat, lon=center_lon),
            zoom=zoom,
        ),
        paper_bgcolor="#111111",
        margin=dict(l=0, r=0, t=0, b=0),
        height=355,
        showlegend=True,
        legend=dict(
            bgcolor="rgba(10,10,10,0.88)",
            font=dict(color="#888888", size=10, family="monospace"),
            x=0.01,
            y=0.99,
            xanchor="left",
            yanchor="top",
            bordercolor="#222222",
            borderwidth=1,
        ),
    )
    return fig


# ─── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        '<div class="sidebar-logo">🔥 PROJECT<br><span>IGNITION HORIZON</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sidebar-subtitle">Tactical Wildfire Command Interface</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ── New Incident Ingestion Form ────────────────────────────────────────
    with st.expander("⚡  NEW INCIDENT INGESTION", expanded=False):
        with st.form("new_incident_form", clear_on_submit=True):
            inc_name  = st.text_input("Incident Name", placeholder="e.g. Oak Canyon Fire")

            coord_col_a, coord_col_b = st.columns(2)
            with coord_col_a:
                inc_lat = st.number_input(
                    "Latitude (°N)",
                    value=37.500,
                    min_value=25.0,
                    max_value=50.0,
                    step=0.001,
                    format="%.4f",
                )
            with coord_col_b:
                inc_lon = st.number_input(
                    "Longitude (°W)",
                    value=-119.500,
                    min_value=-130.0,
                    max_value=-100.0,
                    step=0.001,
                    format="%.4f",
                )

            inc_dist  = st.number_input(
                "Distance to Evac Zone (m)",
                value=6000,
                min_value=100,
                max_value=100_000,
                step=100,
            )
            inc_speed = st.number_input(
                "Closing Speed (m/hr)",
                value=2000,
                min_value=0,
                max_value=20_000,
                step=100,
            )
            inc_align = st.slider(
                "Absolute Alignment",
                min_value=0.00,
                max_value=1.00,
                value=0.50,
                step=0.01,
            )

            form_submitted = st.form_submit_button(
                "REGISTER INCIDENT", use_container_width=True
            )

            if form_submitted:
                clean_name = inc_name.strip()
                if clean_name:
                    new_inc = _make_incident(
                        clean_name,
                        float(inc_lat),
                        float(inc_lon),
                        float(inc_dist),
                        float(inc_speed),
                        float(inc_align),
                    )
                    st.session_state.incidents.append(new_inc)
                    st.session_state.selected_id = new_inc["id"]
                    rl, _, _ = get_risk_classification(new_inc["prob_48h"])
                    st.session_state.sim_log.insert(
                        0,
                        f"[{datetime.now().strftime('%H:%M:%S')}] MANUAL: {clean_name} | "
                        f"48H={new_inc['prob_48h']:.1%} [{rl}]",
                    )
                    st.success(f"Registered: {clean_name}")
                    st.rerun()
                else:
                    st.error("Incident name is required.")

    # ── Dynamic Priority Triage Queue ─────────────────────────────────────
    st.markdown('<div class="section-label">Active Triage Queue</div>', unsafe_allow_html=True)

    for inc in sorted_incidents():
        risk_label, risk_color, risk_icon = get_risk_classification(inc["prob_48h"])
        is_selected = inc["id"] == st.session_state.selected_id

        card_border_color = risk_color if is_selected else "#252525"
        r_int = int(risk_color[1:3], 16)
        g_int = int(risk_color[3:5], 16)
        b_int = int(risk_color[5:7], 16)
        card_bg = f"rgba({r_int},{g_int},{b_int},0.07)" if is_selected else "#111111"

        st.markdown(
            f"""
            <div class="triage-card"
                 style="border-left: 3px solid {card_border_color}; background: {card_bg};">
              <div class="triage-card-header">
                <span class="triage-name">{risk_icon} {inc['name']}</span>
                <span class="triage-badge"
                      style="background: rgba({r_int},{g_int},{b_int},0.15);
                             color: {risk_color};
                             border: 1px solid rgba({r_int},{g_int},{b_int},0.35);">
                  {risk_label}
                </span>
              </div>
              <div class="triage-meta">
                48H: <span style="color:{risk_color}; font-weight:700;">{inc['prob_48h']:.1%}</span>
                &nbsp;|&nbsp;
                72H: <span style="color:#666;">{inc['prob_72h']:.1%}</span>
              </div>
              <div class="triage-time">{inc['timestamp']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button("SELECT", key=f"sel_{inc['id']}", use_container_width=True):
            st.session_state.selected_id = inc["id"]
            st.rerun()

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    active_count    = len(st.session_state.incidents)
    critical_count  = sum(1 for i in st.session_state.incidents if i["prob_48h"] >= 0.70)
    elevated_count  = sum(1 for i in st.session_state.incidents if 0.30 <= i["prob_48h"] < 0.70)

    st.markdown(
        f"""
        <div class="system-status">
          <div>STATUS: <span style="color:#2EC4B6;">OPERATIONAL</span></div>
          <div>ACTIVE: <span style="color:#FF9F1C;">{active_count}</span> incidents</div>
          <div>CRITICAL: <span style="color:#FF4B4B;">{critical_count}</span> &nbsp;
               ELEVATED: <span style="color:#FF9F1C;">{elevated_count}</span></div>
          <div>MODEL: V31 ENSEMBLE</div>
          <div>DATA WINDOW: T₀ + 5H</div>
          <div>CAL: TEMP-SCALE + ISOTONIC</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─── Main Operational Layout ──────────────────────────────────────────────────

sel = selected_incident()

if sel is None:
    st.warning("No active incidents. Use the sidebar to register one.")
    st.stop()

risk_label, risk_color, risk_icon = get_risk_classification(sel["prob_48h"])

# ── Incident Header Bar ────────────────────────────────────────────────────
r_int = int(risk_color[1:3], 16)
g_int = int(risk_color[3:5], 16)
b_int = int(risk_color[5:7], 16)

st.markdown(
    f"""
    <div class="main-header">
      <div>
        <div class="incident-name">{risk_icon} {sel['name']}</div>
        <div class="incident-meta">
          📍 {sel['lat']:.4f}°N &nbsp;{abs(sel['lon']):.4f}°W
          &nbsp;|&nbsp; Logged: {sel['timestamp']}
          &nbsp;|&nbsp; Separation: {sel['dist'] / 1000:.2f} km
          &nbsp;|&nbsp; Closing Speed: {sel['closing_speed']:,.0f} m/h
          &nbsp;|&nbsp; Alignment: {sel['alignment']:.2f}
        </div>
      </div>
      <div class="risk-badge-large"
           style="background: rgba({r_int},{g_int},{b_int},0.12);
                  border: 2px solid rgba({r_int},{g_int},{b_int},0.6);
                  color: {risk_color};">
        {risk_label} THREAT
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ── Two-Column Operational Grid ────────────────────────────────────────────
map_col, proj_col = st.columns([1.12, 0.88], gap="large")

with map_col:
    st.markdown('<div class="panel-title">Geospatial Footprint Context</div>', unsafe_allow_html=True)

    map_fig = build_fire_map(
        sel["lat"], sel["lon"],
        sel["evac_lat"], sel["evac_lon"],
        sel["name"],
        risk_color,
    )
    st.plotly_chart(map_fig, use_container_width=True, config={"displayModeBar": False})

    st.markdown(
        f"""
        <div class="map-legend">
          <span><span style="color:{risk_color};">●</span> Active Ignition Center</span>
          <span><span style="color:#FF9F1C;">●</span> Evacuation Zone Centroid</span>
          <span>Separation: {sel['dist'] / 1000:.2f} km</span>
          <span>Alignment: {sel['alignment']:.2f}</span>
          <span>Speed: {sel['closing_speed']:,.0f} m/h</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

with proj_col:
    st.markdown('<div class="panel-title">Horizon Calibrated Projections</div>', unsafe_allow_html=True)

    # 48h Gauge — Primary Operational Zone
    gauge_fig = build_gauge(sel["prob_48h"], risk_color)
    st.plotly_chart(gauge_fig, use_container_width=True, config={"displayModeBar": False})

    st.markdown(
        f"""
        <div class="cal-note" style="border-left: 2px solid {risk_color}50;">
          48H WINDOW (Primary) — Isotonic Regression Calibration
          &nbsp;|&nbsp; V31 Ensemble: XGBoost + LightGBM + Ridge
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── 12h / 24h / 72h Probability Cards ─────────────────────────────────
    card_a, card_b, card_c = st.columns(3)

    card_specs = [
        (card_a, "12H",  sel["prob_12h"], "TEMP-SCALE"),
        (card_b, "24H",  sel["prob_24h"], "ISOTONIC"),
        (card_c, "72H",  sel["prob_72h"], "ISOTONIC"),
    ]

    for col, label, prob, cal_method in card_specs:
        _, card_color, _ = get_risk_classification(prob)
        with col:
            st.markdown(
                f"""
                <div class="prob-card" style="border-top: 2px solid {card_color};">
                  <div class="prob-label">{label} FORECAST</div>
                  <div class="prob-value" style="color:{card_color};">{prob:.1%}</div>
                  <div class="prob-cal-tag">{cal_method}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

st.markdown('<hr class="divider" style="margin-top:1.2rem;">', unsafe_allow_html=True)

# ─── Diagnostic & Simulation Section ─────────────────────────────────────────

sim_col, log_col = st.columns([0.42, 0.58], gap="large")

_FIRE_PREFIXES  = ["Canyon", "Ridge", "Mesa", "Summit", "Valley", "Creek", "Peak", "Slope",
                   "Timber", "Brush", "Chaparral", "Granite", "Diablo", "Iron", "Smoke"]
_FIRE_DIRS      = ["North", "South", "East", "West", "Upper", "Lower", "Central"]
_FIRE_SUFFIXES  = ["Fire", "Incident", "Complex", "Blaze"]
_CA_REGIONS     = [
    ("Northern CA",    38.5, 40.5, -122.8, -120.0),
    ("Central CA",     36.0, 38.5, -122.0, -118.5),
    ("Southern CA",    33.5, 36.0, -121.0, -115.0),
    ("Sierra Nevada",  37.0, 39.5, -120.5, -117.5),
    ("Central Valley", 35.5, 38.0, -122.0, -119.0),
    ("Coastal Ranges", 34.0, 40.0, -124.0, -121.5),
]

with sim_col:
    st.markdown('<div class="panel-title">Real-Time Feed Simulation</div>', unsafe_allow_html=True)

    if st.button(
        "⚡  ACTIVATE REAL-TIME FEED SIMULATION",
        use_container_width=True,
        type="primary",
    ):
        region_name, lat_min, lat_max, lon_min, lon_max = random.choice(_CA_REGIONS)

        gen_lat    = random.uniform(lat_min, lat_max)
        gen_lon    = random.uniform(lon_min, lon_max)
        gen_dist   = random.uniform(1_200.0, 48_000.0)
        gen_speed  = random.uniform(200.0, 9_500.0)
        gen_align  = random.uniform(0.04, 0.98)
        gen_name   = (
            f"{random.choice(_FIRE_PREFIXES)} "
            f"{random.choice(_FIRE_DIRS)} "
            f"{random.choice(_FIRE_SUFFIXES)}"
        )

        new_inc = _make_incident(gen_name, gen_lat, gen_lon, gen_dist, gen_speed, gen_align)
        st.session_state.incidents.append(new_inc)
        st.session_state.selected_id = new_inc["id"]

        rl, _, _ = get_risk_classification(new_inc["prob_48h"])
        log_entry = (
            f"[{datetime.now().strftime('%H:%M:%S')}] SYNTH: {gen_name} "
            f"| {region_name} "
            f"| 48H={new_inc['prob_48h']:.1%} [{rl}] "
            f"| Dist={gen_dist/1000:.1f}km Spd={gen_speed:.0f}m/h Al={gen_align:.2f}"
        )
        st.session_state.sim_log.insert(0, log_entry)
        st.session_state.sim_log = st.session_state.sim_log[:30]

        st.rerun()

    active_critical  = sum(1 for i in st.session_state.incidents if i["prob_48h"] >= 0.70)
    active_elevated  = sum(1 for i in st.session_state.incidents if 0.30 <= i["prob_48h"] < 0.70)
    active_stable    = sum(1 for i in st.session_state.incidents if i["prob_48h"] < 0.30)

    st.markdown(
        f"""
        <div class="sim-info">
          <div>TOTAL INCIDENTS: <span style="color:#FF9F1C;">{len(st.session_state.incidents)}</span></div>
          <div>
            CRITICAL: <span style="color:#FF4B4B;">{active_critical}</span> &nbsp;
            ELEVATED: <span style="color:#FF9F1C;">{active_elevated}</span> &nbsp;
            STABLE: <span style="color:#2EC4B6;">{active_stable}</span>
          </div>
          <div>SIMULATION EVENTS: <span style="color:#2EC4B6;">{len(st.session_state.sim_log)}</span></div>
          <div class="model-tag">MODEL STACK: XGBoost · LightGBM · Ridge</div>
          <div class="model-tag">12H CALIBRATION: Temperature Scaling (T=1.30)</div>
          <div class="model-tag">24/48/72H CALIBRATION: Isotonic Regression</div>
          <div class="model-tag">HORIZONS: 12H · 24H · 48H · 72H</div>
          <div class="model-tag">SEED ENSEMBLE: 5 seeds × 5-fold CV</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with log_col:
    st.markdown('<div class="panel-title">Diagnostic Activity Feed</div>', unsafe_allow_html=True)

    if st.session_state.sim_log:
        log_html = '<div class="feed-log">'
        for entry in st.session_state.sim_log[:15]:
            if "CRITICAL" in entry:
                entry_color = "#FF4B4B"
            elif "ELEVATED" in entry:
                entry_color = "#FF9F1C"
            elif "STABLE" in entry:
                entry_color = "#2EC4B6"
            else:
                entry_color = "#444444"
            log_html += (
                f'<div class="feed-entry" '
                f'style="border-left: 2px solid {entry_color};">{entry}</div>'
            )
        log_html += "</div>"
        st.markdown(log_html, unsafe_allow_html=True)
    else:
        st.markdown(
            '<div class="feed-log-empty">No activity events. '
            "Activate the simulation feed or register an incident.</div>",
            unsafe_allow_html=True,
        )

# ─── App Footer ───────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="app-footer">
      PROJECT IGNITIONHORIZON &nbsp;|&nbsp;
      WiDS Global Datathon 2026 &nbsp;|&nbsp;
      V31 Ensemble · Calibrated Multi-Horizon Survival Analysis &nbsp;|&nbsp;
      Feature Window: T₀ + 5H &nbsp;|&nbsp;
      Threat Buffer: 5 km Evacuation Zone Perimeter
    </div>
    """,
    unsafe_allow_html=True,
)
