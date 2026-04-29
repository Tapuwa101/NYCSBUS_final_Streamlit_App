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
except Exception:  # pragma: no cover
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
        background: rgba(255,255,255,0.86);
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
        background: rgba(255,255,255,0.88);
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

    .risk-high { color: #dc2626; font-weight: 850; }
    .risk-medium { color: #ea580c; font-weight: 850; }
    .risk-low { color: #16a34a; font-weight: 850; }

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
# Data helpers
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
NYC_CENTER = [40.7128, -74.0060]


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [
        str(c).strip().lower().replace(" ", "_").replace("-", "_") for c in df.columns
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
        "richmond": "Staten Island",
        "staten island": "Staten Island",
        "bronx": "Bronx",
        "brooklyn": "Brooklyn",
        "queens": "Queens",
        "manhattan": "Manhattan",
    }
    return series.astype(str).str.strip().str.lower().map(fixed).fillna(series.astype(str).str.title())


def prepare_geo(df):
    df = df.copy()

    # Standardize latitude and longitude column names
    if "latitude" not in df.columns:
        for col in ["lat", "Latitude", "LAT"]:
            if col in df.columns:
                df["latitude"] = df[col]
                break

    if "longitude" not in df.columns:
        for col in ["lon", "lng", "Longitude", "LON", "LNG"]:
            if col in df.columns:
                df["longitude"] = df[col]
                break

    # Create severity if missing
    if "severity" not in df.columns:
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        if numeric_cols:
            df["severity"] = df[numeric_cols[0]]
        else:
            df["severity"] = 0

    df["severity"] = pd.to_numeric(df["severity"], errors="coerce").fillna(0)

    def assign_risk_level(value):
        try:
            value = float(value)
        except Exception:
            return "Unknown"

        if value >= 75:
            return "Critical"
        elif value >= 50:
            return "High"
        elif value >= 25:
            return "Moderate"
        else:
            return "Low"

    df["risk_level"] = df["severity"].apply(assign_risk_level)

    return df


@st.cache_data(show_spinner=False)
def load_default_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    alarms_path = DEFAULT_FILES["alarms"] if DEFAULT_FILES["alarms"].exists() else DEFAULT_FILES["spatiotemporal"]
    hotspots_path = DEFAULT_FILES["hotspots"] if DEFAULT_FILES["hotspots"].exists() else DEFAULT_FILES["hotspot_analysis"]
    trend_path = DEFAULT_FILES["trend"]

    alarms = prepare_geo(pd.read_csv(alarms_path)) if alarms_path.exists() else pd.DataFrame()
    hotspots = prepare_geo(pd.read_csv(hotspots_path)) if hotspots_path.exists() else pd.DataFrame()
    trends = prepare_geo(pd.read_csv(trend_path)) if trend_path.exists() else pd.DataFrame()
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


def filter_data(df: pd.DataFrame, boroughs: list[str], risk_levels: list[str], alarm_types: list[str]) -> pd.DataFrame:
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
    if df.empty or pdk is None or not {"lat", "lon"}.issubset(df.columns):
        st.info("Map will appear once latitude and longitude fields are available.")
        return

    points = df.dropna(subset=["lat", "lon"]).copy()
    if points.empty:
        st.info("No geocoded records available for the current filters.")
        return

    risk_color = {
        "High": [239, 68, 68, 185],
        "Medium": [245, 158, 11, 170],
        "Low": [34, 197, 94, 155],
    }
    points["color"] = points["risk_level"].map(risk_color).apply(lambda x: x if isinstance(x, list) else [59, 130, 246, 155])
    points["radius"] = np.clip(points["severity"].rank(pct=True) * 220, 70, 220)

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
        latitude=float(points["lat"].median()) if points["lat"].notna().any() else NYC_CENTER[0],
        longitude=float(points["lon"].median()) if points["lon"].notna().any() else NYC_CENTER[1],
        zoom=10.35,
        pitch=35,
    )

    tooltip = {
        "html": "<b>{borough}</b><br/>Alarm: {alarm_type}<br/>Risk: {risk_level}<br/>Severity: {severity}<br/>{recommended_action}",
        "style": {"backgroundColor": "#0f172a", "color": "white"},
    }

    st.pydeck_chart(
        pdk.Deck(
            map_style="mapbox://styles/mapbox/light-v10",
            initial_view_state=view_state,
            layers=[layer],
            tooltip=tooltip,
        ),
        use_container_width=True,
    )


def render_empty_state() -> None:
    st.warning(
        "No data was found. Keep 'Use demo data from repo' on, or upload an alarms CSV with latitude and longitude fields."
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
    available_boroughs = [b for b in BOROUGH_ORDER if b in set(base_df.get("borough", pd.Series(dtype=str)))]
    if not available_boroughs:
        available_boroughs = sorted(base_df.get("borough", pd.Series(dtype=str)).dropna().unique().tolist())

    boroughs = st.multiselect("Borough", available_boroughs, default=available_boroughs)
    risk_levels = st.multiselect("Risk level", ["High", "Medium", "Low"], default=["High", "Medium", "Low"])
    alarm_types_available = sorted(base_df.get("alarm_type", pd.Series(dtype=str)).dropna().unique().tolist())
    alarm_types = st.multiselect("Signal type", alarm_types_available, default=alarm_types_available)

    st.markdown("---")
    st.markdown("### Interview framing")
    st.caption(
        "Use this as a public-safe demo of the original workflow: metrics, anomaly flags, trend validation, and action-oriented reporting."
    )

filtered = filter_data(base_df, boroughs, risk_levels, alarm_types)

# Hero
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

# KPIs
total_signals = len(filtered)
high_risk = int((filtered.get("risk_level", pd.Series(dtype=str)) == "High").sum()) if not filtered.empty else 0
borough_count = filtered["borough"].nunique() if "borough" in filtered.columns and not filtered.empty else 0
avg_severity = filtered["severity"].mean() if "severity" in filtered.columns and not filtered.empty else 0

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
with kpi1:
    metric_card("Signals in Review", f"{total_signals:,}", "Filtered operational alerts in the current view")
with kpi2:
    metric_card("High-Risk Signals", f"{high_risk:,}", "Items that should be prioritized first")
with kpi3:
    metric_card("Boroughs Covered", f"{borough_count:,}", "Geographic coverage in the selected slice")
with kpi4:
    metric_card("Avg. Severity", f"{avg_severity:,.2f}", "Normalized score for risk comparison")

# Executive summary
st.markdown("### Executive Summary")
s1, s2, s3 = st.columns(3)
with s1:
    top_borough = filtered["borough"].value_counts().idxmax() if not filtered.empty and "borough" in filtered.columns else "N/A"
    insight_card(
        "Where attention is concentrated",
        f"The current filter shows the highest number of review signals in <b>{top_borough}</b>. This helps an operations lead quickly decide where to start instead of scanning raw rows.",
    )
with s2:
    priority_text = "High-risk items are present and should be reviewed first." if high_risk else "No high-risk items appear in the current filter."
    insight_card(
        "What needs action",
        f"{priority_text} The dashboard converts raw signal severity into a practical review queue for non-technical stakeholders.",
    )
with s3:
    insight_card(
        "Why it matters",
        "The workflow is reusable: ingest data, compute signals, validate patterns, and surface decision-ready recommendations through a clean app interface.",
    )

# Tabs
tab_overview, tab_map, tab_trends, tab_review, tab_snowflake = st.tabs(
    ["Overview", "Map", "Trend Validation", "Review Queue", "Snowflake Interview Story"]
)

with tab_overview:
    left, right = st.columns([1.05, 0.95])

    with left:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Signals by borough")
        if not filtered.empty and "borough" in filtered.columns:
            borough_counts = filtered.groupby("borough", as_index=False).size().rename(columns={"size": "signals"})
            fig = px.bar(
                borough_counts.sort_values("signals", ascending=False),
                x="borough",
                y="signals",
                text="signals",
                title=None,
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
            risk_counts = filtered.groupby("risk_level", as_index=False).size().rename(columns={"size": "signals"})
            fig = px.donut(
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
    if not filtered.empty and "alarm_type" in filtered.columns:
        type_counts = filtered.groupby(["alarm_type", "risk_level"], as_index=False).size().rename(columns={"size": "signals"})
        fig = px.bar(type_counts, x="alarm_type", y="signals", color="risk_level", barmode="group")
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
        st.info("No alarm type field found.")
    st.markdown("</div>", unsafe_allow_html=True)

with tab_map:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Geospatial operations view")
    st.caption("Each point represents a safety or hotspot signal. Larger points indicate higher relative severity.")
    build_map(filtered)
    st.markdown("</div>", unsafe_allow_html=True)

with tab_trends:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Trend validation")
    st.caption("This section separates one-time spikes from patterns that deserve sustained attention.")

    trend_source = trends_df if not trends_df.empty else filtered
    trend_source = filter_data(trend_source, boroughs, risk_levels, alarm_types)

    if not trend_source.empty and "event_date" in trend_source.columns and trend_source["event_date"].notna().any():
        trend_source = trend_source.copy()
        trend_source["week"] = trend_source["event_date"].dt.to_period("W").dt.start_time
        trend_counts = trend_source.groupby(["week", "risk_level"], as_index=False).size().rename(columns={"size": "signals"})
        fig = px.line(trend_counts, x="week", y="signals", color="risk_level", markers=True)
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
        st.info("No date field found for trend validation. The review queue still works without dates.")

    c1, c2, c3 = st.columns(3)
    with c1:
        insight_card("Detect", "Identify abnormal route or location-level changes compared with expected behavior.")
    with c2:
        insight_card("Validate", "Check whether the signal is persistent or only a one-time spike before escalating it.")
    with c3:
        insight_card("Act", "Convert the signal into a clear recommendation for operations review.")
    st.markdown("</div>", unsafe_allow_html=True)

with tab_review:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Operational review queue")
    st.caption("Sorted so the highest-severity items appear first.")

    display_cols = [
        c
        for c in ["borough", "alarm_type", "risk_level", "severity", "event_date", "lat", "lon", "recommended_action"]
        if c in filtered.columns
    ]
    review = filtered.sort_values("severity", ascending=False) if "severity" in filtered.columns else filtered
    st.dataframe(review[display_cols].head(500), use_container_width=True, hide_index=True)

    csv = review.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download review queue",
        csv,
        file_name="nycsbus_review_queue.csv",
        mime="text/csv",
        use_container_width=True,
    )
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
