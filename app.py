"""
NYCSBUS Streamlit Operations Dashboard
Polished public-safe interview demo for route safety analytics, anomaly review,
and operational decision support.

Run:
    streamlit run app.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

try:
    import pydeck as pdk
except Exception:
    pdk = None


# -----------------------------------------------------------------------------
# Page config
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="NYCSBUS Operations Intelligence",
    page_icon="🚌",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -----------------------------------------------------------------------------
# Styling
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(circle at top left, rgba(37, 99, 235, 0.18), transparent 34%),
            radial-gradient(circle at top right, rgba(14, 165, 233, 0.14), transparent 30%),
            linear-gradient(180deg, #f8fafc 0%, #eef2f7 48%, #f8fafc 100%);
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #111827 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }

    section[data-testid="stSidebar"] * {
        color: #e5e7eb !important;
    }

    .hero-card {
        padding: 2.2rem 2.4rem;
        border-radius: 28px;
        background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 50%, #0369a1 100%);
        color: white;
        box-shadow: 0 24px 70px rgba(15, 23, 42, 0.22);
        margin-bottom: 1.2rem;
    }

    .eyebrow {
        font-size: 0.78rem;
        font-weight: 800;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #bae6fd;
        margin-bottom: 0.45rem;
    }

    .hero-title {
        font-size: 2.65rem;
        line-height: 1.04;
        font-weight: 850;
        margin-bottom: 0.8rem;
    }

    .hero-subtitle {
        max-width: 920px;
        font-size: 1.02rem;
        line-height: 1.65;
        color: #dbeafe;
    }

    .pill-row {
        display: flex;
        gap: 0.6rem;
        flex-wrap: wrap;
        margin-top: 1.2rem;
    }

    .pill {
        padding: 0.42rem 0.72rem;
        border: 1px solid rgba(255, 255, 255, 0.22);
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.10);
        color: #f8fafc;
        font-size: 0.82rem;
        font-weight: 650;
    }

    .metric-card {
        padding: 1.1rem 1.2rem;
        border-radius: 22px;
        background: rgba(255,255,255,0.88);
        border: 1px solid rgba(148, 163, 184, 0.25);
        box-shadow: 0 16px 42px rgba(15, 23, 42, 0.08);
        min-height: 130px;
    }

    .metric-label {
        color: #64748b;
        font-size: 0.78rem;
        font-weight: 800;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }

    .metric-value {
        color: #0f172a;
        font-size: 2.05rem;
        font-weight: 850;
        margin-top: 0.2rem;
    }

    .metric-help {
        color: #475569;
        font-size: 0.86rem;
        margin-top: 0.35rem;
    }

    .section-card {
        padding: 1.25rem 1.35rem;
        border-radius: 24px;
        background: rgba(255,255,255,0.90);
        border: 1px solid rgba(148, 163, 184, 0.25);
        box-shadow: 0 16px 42px rgba(15, 23, 42, 0.07);
        margin-bottom: 1rem;
    }

    .insight-card {
        padding: 1.1rem 1.25rem;
        border-radius: 22px;
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border: 1px solid rgba(148, 163, 184, 0.28);
        box-shadow: 0 12px 34px rgba(15, 23, 42, 0.06);
        height: 100%;
    }

    .insight-title {
        font-size: 0.95rem;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 0.35rem;
    }

    .insight-text {
        color: #475569;
        line-height: 1.55;
        font-size: 0.92rem;
    }

    div[data-testid="stDataFrame"] {
        border-radius: 18px;
        overflow: hidden;
        border: 1px solid rgba(148, 163, 184, 0.28);
    }

    .footer-note {
        color: #64748b;
        font-size: 0.82rem;
        margin-top: 1.25rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
DATA_DIR = Path(__file__).parent

DEFAULT_FILES = {
    "alarms": DATA_DIR / "alarms_from_crashes_demo.csv",
    "hotspots": DATA_DIR / "hotspots_from_crashes_demo.csv",
    "hotspot_analysis": DATA_DIR / "hotspot_analysis.csv",
    "trend": DATA_DIR / "trend_analysis.csv",
    "spatiotemporal": DATA_DIR / "spatiotemporal_alarms.csv",
}

BOROUGH_ORDER = ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"]
RISK_ORDER = ["Critical", "High", "Moderate", "Low"]
NYC_CENTER = [40.7128, -74.0060]


# -----------------------------------------------------------------------------
# Data helpers
# -----------------------------------------------------------------------------
def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [
        str(c).strip().lower().replace(" ", "_").replace("-", "_")
        for c in df.columns
    ]
    return df


def first_existing(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def normalize_borough(series: pd.Series) -> pd.Series:
    fixed = {
        "new york": "Manhattan",
        "manhattan": "Manhattan",
        "kings": "Brooklyn",
        "brooklyn": "Brooklyn",
        "queens": "Queens",
        "bronx": "Bronx",
        "richmond": "Staten Island",
        "staten island": "Staten Island",
    }

    lower = series.astype(str).str.strip().str.lower()
    return lower.map(fixed).fillna(series.astype(str).str.title())


def assign_risk_level(value: float) -> str:
    try:
        value = float(value)
    except Exception:
        return "Unknown"

    if value >= 75:
        return "Critical"
    if value >= 50:
        return "High"
    if value >= 25:
        return "Moderate"
    return "Low"


def prepare_geo(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardizes messy CSV inputs so the Streamlit app does not break if
    column names differ slightly across files.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    df = clean_columns(df)

    # Standardize borough
    borough_col = first_existing(
        df,
        ["borough", "boro", "county", "city_borough", "nyc_borough"],
    )
    if borough_col:
        df["borough"] = normalize_borough(df[borough_col])
    else:
        df["borough"] = "Unknown"

    # Standardize latitude and longitude
    lat_col = first_existing(
        df,
        ["lat", "latitude", "y", "point_latitude", "crash_latitude"],
    )
    lon_col = first_existing(
        df,
        ["lon", "lng", "long", "longitude", "x", "point_longitude", "crash_longitude"],
    )

    if lat_col:
        df["lat"] = pd.to_numeric(df[lat_col], errors="coerce")
    else:
        df["lat"] = np.nan

    if lon_col:
        df["lon"] = pd.to_numeric(df[lon_col], errors="coerce")
    else:
        df["lon"] = np.nan

    # Standardize date
    date_col = first_existing(
        df,
        ["event_date", "date", "crash_date", "week", "created_date", "timestamp"],
    )
    if date_col:
        df["event_date"] = pd.to_datetime(df[date_col], errors="coerce")
    else:
        df["event_date"] = pd.NaT

    # Standardize alarm/signal type
    alarm_col = first_existing(
        df,
        [
            "alarm_type",
            "signal_type",
            "type",
            "category",
            "alert_type",
            "metric",
            "analysis_type",
        ],
    )
    if alarm_col:
        df["alarm_type"] = df[alarm_col].astype(str).str.replace("_", " ").str.title()
    else:
        df["alarm_type"] = "Operational Signal"

    # Standardize severity
    severity_col = first_existing(
        df,
        [
            "severity",
            "severity_score",
            "risk_score",
            "score",
            "alarm_score",
            "crash_count",
            "count",
            "signals",
            "hotspot_score",
        ],
    )

    if severity_col:
        df["severity"] = pd.to_numeric(df[severity_col], errors="coerce")
    else:
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        usable_numeric = [c for c in numeric_cols if c not in ["lat", "lon"]]
        if usable_numeric:
            df["severity"] = pd.to_numeric(df[usable_numeric[0]], errors="coerce")
        else:
            df["severity"] = 0

    df["severity"] = df["severity"].fillna(0)

    # Normalize severity to a 0-100 style score only if the values are tiny or huge.
    max_severity = df["severity"].max()
    min_severity = df["severity"].min()

    if max_severity > 100 or max_severity <= 10:
        if max_severity != min_severity:
            df["severity"] = (
                (df["severity"] - min_severity)
                / (max_severity - min_severity)
                * 100
            )
        else:
            df["severity"] = 50

    df["severity"] = df["severity"].clip(lower=0, upper=100)
    df["risk_level"] = df["severity"].apply(assign_risk_level)

    # Recommended action
    action_col = first_existing(
        df,
        ["recommended_action", "action", "recommendation", "next_step"],
    )
    if action_col:
        df["recommended_action"] = df[action_col].astype(str)
    else:
        df["recommended_action"] = df["risk_level"].map(
            {
                "Critical": "Immediate operations review recommended",
                "High": "Prioritize for route or safety review",
                "Moderate": "Monitor for repeated pattern",
                "Low": "Track as baseline operational signal",
            }
        ).fillna("Review signal")

    return df


@st.cache_data(show_spinner=False)
def load_default_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    alarms_path = (
        DEFAULT_FILES["alarms"]
        if DEFAULT_FILES["alarms"].exists()
        else DEFAULT_FILES["spatiotemporal"]
    )

    hotspots_path = (
        DEFAULT_FILES["hotspots"]
        if DEFAULT_FILES["hotspots"].exists()
        else DEFAULT_FILES["hotspot_analysis"]
    )

    trend_path = DEFAULT_FILES["trend"]

    alarms = (
        prepare_geo(pd.read_csv(alarms_path))
        if alarms_path.exists()
        else pd.DataFrame()
    )

    hotspots = (
        prepare_geo(pd.read_csv(hotspots_path))
        if hotspots_path.exists()
        else pd.DataFrame()
    )

    trends = (
        prepare_geo(pd.read_csv(trend_path))
        if trend_path.exists()
        else pd.DataFrame()
    )

    return alarms, hotspots, trends


def load_uploaded_or_default() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    with st.sidebar:
        st.markdown("### Data source")
        use_demo = st.toggle("Use demo data from repo", value=True)

        if use_demo:
            return load_default_data()

        alarms_file = st.file_uploader("Upload alarms CSV", type=["csv"])
        hotspots_file = st.file_uploader("Upload hotspots CSV", type=["csv"])
        trends_file = st.file_uploader("Upload trends CSV", type=["csv"])

    alarms = prepare_geo(pd.read_csv(alarms_file)) if alarms_file else pd.DataFrame()
    hotspots = prepare_geo(pd.read_csv(hotspots_file)) if hotspots_file else pd.DataFrame()
    trends = prepare_geo(pd.read_csv(trends_file)) if trends_file else pd.DataFrame()

    return alarms, hotspots, trends


def filter_data(
    df: pd.DataFrame,
    boroughs: list[str],
    risk_levels: list[str],
    alarm_types: list[str],
) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()

    if "borough" in out.columns and boroughs:
        out = out[out["borough"].isin(boroughs)]

    if "risk_level" in out.columns and risk_levels:
        out = out[out["risk_level"].isin(risk_levels)]

    if "alarm_type" in out.columns and alarm_types:
        out = out[out["alarm_type"].isin(alarm_types)]

    return out


def metric_card(label: str, value: str, helper: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-help">{helper}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def insight_card(title: str, text: str) -> None:
    st.markdown(
        f"""
        <div class="insight-card">
            <div class="insight-title">{title}</div>
            <div class="insight-text">{text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_map(df: pd.DataFrame) -> None:
    if df.empty or not {"lat", "lon"}.issubset(df.columns):
        st.info("Map will appear once latitude and longitude fields are available.")
        return

    points = df.dropna(subset=["lat", "lon"]).copy()

    if points.empty:
        st.info("No geocoded records available for the current filters.")
        return

    if pdk is None:
        st.map(points.rename(columns={"lat": "latitude", "lon": "longitude"}))
        return

    risk_color = {
        "Critical": [220, 38, 38, 210],
        "High": [239, 68, 68, 185],
        "Moderate": [245, 158, 11, 175],
        "Low": [34, 197, 94, 155],
    }

    points["color"] = points["risk_level"].map(risk_color)
    points["color"] = points["color"].apply(
        lambda x: x if isinstance(x, list) else [59, 130, 246, 155]
    )

    points["radius"] = np.clip(points["severity"].rank(pct=True) * 230, 75, 230)

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=points,
        get_position="[lon, lat]",
        get_fill_color="color",
        get_radius="radius",
        pickable=True,
        opacity=0.78,
    )

    view_state = pdk.ViewState(
        latitude=float(points["lat"].median()),
        longitude=float(points["lon"].median()),
        zoom=10.35,
        pitch=35,
    )

    tooltip = {
        "html": """
        <b>{borough}</b><br/>
        Signal: {alarm_type}<br/>
        Risk: {risk_level}<br/>
        Severity: {severity}<br/>
        Action: {recommended_action}
        """,
        "style": {"backgroundColor": "#0f172a", "color": "white"},
    }

    st.pydeck_chart(
        pdk.Deck(
            initial_view_state=view_state,
            layers=[layer],
            tooltip=tooltip,
        ),
        use_container_width=True,
    )


def render_empty_state() -> None:
    st.warning(
        "No data was found. Keep 'Use demo data from repo' on, or upload a CSV with latitude and longitude fields."
    )


# -----------------------------------------------------------------------------
# Main app
# -----------------------------------------------------------------------------
alarms_df, hotspots_df, trends_df = load_uploaded_or_default()

if alarms_df.empty and hotspots_df.empty:
    render_empty_state()
    st.stop()

base_df = alarms_df if not alarms_df.empty else hotspots_df

with st.sidebar:
    st.markdown("---")
    st.markdown("### Filters")

    if "borough" in base_df.columns:
        present_boroughs = set(base_df["borough"].dropna().unique().tolist())
        available_boroughs = [b for b in BOROUGH_ORDER if b in present_boroughs]

        extra_boroughs = sorted(
            [b for b in present_boroughs if b not in BOROUGH_ORDER and b != "Unknown"]
        )

        available_boroughs = available_boroughs + extra_boroughs

        if not available_boroughs:
            available_boroughs = sorted(base_df["borough"].dropna().unique().tolist())
    else:
        available_boroughs = []

    boroughs = st.multiselect(
        "Borough",
        available_boroughs,
        default=available_boroughs,
    )

    if "risk_level" in base_df.columns:
        present_risks = set(base_df["risk_level"].dropna().unique().tolist())
        available_risks = [r for r in RISK_ORDER if r in present_risks]
        extra_risks = sorted([r for r in present_risks if r not in RISK_ORDER])
        available_risks = available_risks + extra_risks
    else:
        available_risks = RISK_ORDER

    risk_levels = st.multiselect(
        "Risk level",
        available_risks,
        default=available_risks,
    )

    if "alarm_type" in base_df.columns:
        alarm_types_available = sorted(
            base_df["alarm_type"].dropna().unique().tolist()
        )
    else:
        alarm_types_available = []

    alarm_types = st.multiselect(
        "Signal type",
        alarm_types_available,
        default=alarm_types_available,
    )

    st.markdown("---")
    st.markdown("### Interview framing")
    st.caption(
        "Use this as a public-safe demo of the original workflow: metrics, anomaly flags, trend validation, and action-oriented reporting."
    )

filtered = filter_data(base_df, boroughs, risk_levels, alarm_types)


# -----------------------------------------------------------------------------
# Hero
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero-card">
        <div class="eyebrow">NYCSBUS Operations Intelligence</div>
        <div class="hero-title">Route safety signals, anomaly review, and decision-ready operations analytics.</div>
        <div class="hero-subtitle">
            A public-safe Streamlit dashboard that turns transportation data into a repeatable workflow:
            monitor route-level signals, identify hotspots, validate trends, and recommend what an operations team should review next.
        </div>
        <div class="pill-row">
            <span class="pill">Streamlit</span>
            <span class="pill">Python</span>
            <span class="pill">Operational Analytics</span>
            <span class="pill">Anomaly Detection</span>
            <span class="pill">Decision Support</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# KPIs
# -----------------------------------------------------------------------------
total_signals = len(filtered)

if not filtered.empty and "risk_level" in filtered.columns:
    priority_signals = int(
        filtered["risk_level"].isin(["Critical", "High"]).sum()
    )
else:
    priority_signals = 0

borough_count = (
    filtered["borough"].nunique()
    if "borough" in filtered.columns and not filtered.empty
    else 0
)

avg_severity = (
    filtered["severity"].mean()
    if "severity" in filtered.columns and not filtered.empty
    else 0
)

kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    metric_card(
        "Signals in Review",
        f"{total_signals:,}",
        "Filtered operational alerts in the current view",
    )

with kpi2:
    metric_card(
        "Priority Signals",
        f"{priority_signals:,}",
        "Critical and high-risk items to review first",
    )

with kpi3:
    metric_card(
        "Boroughs Covered",
        f"{borough_count:,}",
        "Geographic coverage in the selected slice",
    )

with kpi4:
    metric_card(
        "Avg. Severity",
        f"{avg_severity:,.2f}",
        "Normalized score for risk comparison",
    )


# -----------------------------------------------------------------------------
# Executive summary
# -----------------------------------------------------------------------------
st.markdown("### Executive Summary")

s1, s2, s3 = st.columns(3)

with s1:
    if not filtered.empty and "borough" in filtered.columns:
        top_borough = filtered["borough"].value_counts().idxmax()
    else:
        top_borough = "N/A"

    insight_card(
        "Where attention is concentrated",
        f"The current filter shows the highest number of review signals in <b>{top_borough}</b>. This helps an operations lead quickly decide where to start instead of scanning raw rows.",
    )

with s2:
    if priority_signals:
        priority_text = "Priority items are present and should be reviewed first."
    else:
        priority_text = "No critical or high-risk items appear in the current filter."

    insight_card(
        "What needs action",
        f"{priority_text} The dashboard converts raw signal severity into a practical review queue for non-technical stakeholders.",
    )

with s3:
    insight_card(
        "Why it matters",
        "The workflow is reusable: ingest data, compute signals, validate patterns, and surface decision-ready recommendations through a clean app interface.",
    )


# -----------------------------------------------------------------------------
# Tabs
# -----------------------------------------------------------------------------
tab_overview, tab_map, tab_trends, tab_review, tab_snowflake = st.tabs(
    [
        "Overview",
        "Map",
        "Trend Validation",
        "Review Queue",
        "Snowflake Interview Story",
    ]
)


with tab_overview:
    left, right = st.columns([1.05, 0.95])

    with left:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Signals by borough")

        if not filtered.empty and "borough" in filtered.columns:
            borough_counts = (
                filtered.groupby("borough", as_index=False)
                .size()
                .rename(columns={"size": "signals"})
                .sort_values("signals", ascending=False)
            )

            fig = px.bar(
                borough_counts,
                x="borough",
                y="signals",
                text="signals",
            )

            fig.update_layout(
                height=390,
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis_title="",
                yaxis_title="Signals",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#0f172a"),
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No borough field found.")

        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Risk mix")

        if not filtered.empty and "risk_level" in filtered.columns:
            risk_counts = (
                filtered.groupby("risk_level", as_index=False)
                .size()
                .rename(columns={"size": "signals"})
            )

            fig = px.pie(
                risk_counts,
                names="risk_level",
                values="signals",
                hole=0.58,
            )

            fig.update_layout(
                height=390,
                margin=dict(l=10, r=10, t=10, b=10),
                showlegend=True,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#0f172a"),
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No risk level field found.")

        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Signal type breakdown")

    if (
        not filtered.empty
        and "alarm_type" in filtered.columns
        and "risk_level" in filtered.columns
    ):
        type_counts = (
            filtered.groupby(["alarm_type", "risk_level"], as_index=False)
            .size()
            .rename(columns={"size": "signals"})
        )

        fig = px.bar(
            type_counts,
            x="alarm_type",
            y="signals",
            color="risk_level",
            barmode="group",
        )

        fig.update_layout(
            height=360,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="",
            yaxis_title="Signals",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#0f172a"),
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No signal type field found.")

    st.markdown("</div>", unsafe_allow_html=True)


with tab_map:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Geospatial operations view")
    st.caption(
        "Each point represents a safety or hotspot signal. Larger points indicate higher relative severity."
    )

    build_map(filtered)

    st.markdown("</div>", unsafe_allow_html=True)


with tab_trends:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Trend validation")
    st.caption(
        "This section separates one-time spikes from patterns that deserve sustained attention."
    )

    trend_source = trends_df if not trends_df.empty else filtered
    trend_source = filter_data(trend_source, boroughs, risk_levels, alarm_types)

    if (
        not trend_source.empty
        and "event_date" in trend_source.columns
        and trend_source["event_date"].notna().any()
    ):
        trend_source = trend_source.copy()
        trend_source["week"] = trend_source["event_date"].dt.to_period("W").dt.start_time

        trend_counts = (
            trend_source.groupby(["week", "risk_level"], as_index=False)
            .size()
            .rename(columns={"size": "signals"})
        )

        fig = px.line(
            trend_counts,
            x="week",
            y="signals",
            color="risk_level",
            markers=True,
        )

        fig.update_layout(
            height=420,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="Week",
            yaxis_title="Signals",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#0f172a"),
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(
            "No usable date field found for trend validation. The review queue still works without dates."
        )

    c1, c2, c3 = st.columns(3)

    with c1:
        insight_card(
            "Detect",
            "Identify abnormal route or location-level changes compared with expected behavior.",
        )

    with c2:
        insight_card(
            "Validate",
            "Check whether the signal is persistent or only a one-time spike before escalating it.",
        )

    with c3:
        insight_card(
            "Act",
            "Convert the signal into a clear recommendation for operations review.",
        )

    st.markdown("</div>", unsafe_allow_html=True)


with tab_review:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Operational review queue")
    st.caption("Sorted so the highest-severity items appear first.")

    display_cols = [
        c
        for c in [
            "borough",
            "alarm_type",
            "risk_level",
            "severity",
            "event_date",
            "lat",
            "lon",
            "recommended_action",
        ]
        if c in filtered.columns
    ]

    if not filtered.empty:
        review = (
            filtered.sort_values("severity", ascending=False)
            if "severity" in filtered.columns
            else filtered
        )

        st.dataframe(
            review[display_cols].head(500),
            use_container_width=True,
            hide_index=True,
        )

        csv = review.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download review queue",
            csv,
            file_name="nycsbus_review_queue.csv",
            mime="text/csv",
            use_container_width=True,
        )
    else:
        st.info("No records match the current filters.")

    st.markdown("</div>", unsafe_allow_html=True)


with tab_snowflake:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("How to explain this project to Snowflake")

    st.markdown(
        """
        **Interview story:** I built this as an operations intelligence dashboard, not just a reporting page.
        The app takes raw transportation signals, standardizes the data, computes route or location-level risk,
        flags patterns that deserve review, and gives non-technical stakeholders a clean interface for action.

        **Why it connects to Finance Analytics & AI:** The same workflow can apply to finance data. Instead of route
        safety signals, the inputs could be ARR movement, NRR changes, forecast variance, cost-center spend,
        hiring plan changes, or revenue anomalies. The pattern is the same: ingest data, create reusable metrics,
        detect unusual movement, validate the signal, and surface a recommendation through a Streamlit app.

        **One-liner to use:** This project taught me to think beyond dashboards. A dashboard shows what happened,
        but an intelligence workflow helps the business understand what changed, whether it matters, and what to do next.
        """
    )

    st.markdown("</div>", unsafe_allow_html=True)


st.markdown(
    '<div class="footer-note">Public-safe demo built for interview presentation. Replace demo CSVs with approved internal data sources for production use.</div>',
    unsafe_allow_html=True,
)
