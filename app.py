import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="NYCSBUS Safety Operations Dashboard",
    page_icon="🚌",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# STYLING
# ============================================================
st.markdown(
    """
    <style>
        .main {
            background-color: #f7f9fc;
        }

        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }

        .metric-card {
            background: white;
            padding: 1.2rem;
            border-radius: 16px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.06);
            border: 1px solid #e8eef5;
        }

        .section-card {
            background: white;
            padding: 1.4rem;
            border-radius: 18px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.06);
            border: 1px solid #e8eef5;
            margin-bottom: 1rem;
        }

        .title-text {
            font-size: 2.2rem;
            font-weight: 800;
            color: #0f172a;
            margin-bottom: 0.2rem;
        }

        .subtitle-text {
            font-size: 1rem;
            color: #475569;
            margin-bottom: 1.2rem;
        }

        .small-muted {
            color: #64748b;
            font-size: 0.9rem;
        }

        div[data-testid="stMetricValue"] {
            font-size: 1.8rem;
            font-weight: 800;
        }

        div[data-testid="stMetricLabel"] {
            font-size: 0.9rem;
            color: #475569;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# DEMO DATA GENERATION
# ============================================================
@st.cache_data
def generate_demo_data(seed=42):
    """
    Generates realistic demo data for a NYCSBUS safety operations dashboard.
    This lets the app run even if no CSV files are uploaded.
    """
    np.random.seed(seed)

    neighborhoods = [
        "Upper West Side", "Harlem", "South Bronx", "Astoria", "Jackson Heights",
        "Long Island City", "Downtown Brooklyn", "Bed-Stuy", "Crown Heights",
        "Flushing", "Jamaica", "Park Slope", "Bushwick", "Williamsburg"
    ]

    issue_types = [
        "Speeding Risk",
        "Harsh Braking",
        "Late Arrival Pattern",
        "Congestion Delay",
        "Route Deviation",
        "High Incident Density"
    ]

    severity_levels = ["Low", "Medium", "High", "Critical"]

    # NYC-ish coordinate centers
    centers = {
        "Upper West Side": (40.7870, -73.9754),
        "Harlem": (40.8116, -73.9465),
        "South Bronx": (40.8176, -73.9182),
        "Astoria": (40.7644, -73.9235),
        "Jackson Heights": (40.7557, -73.8831),
        "Long Island City": (40.7447, -73.9485),
        "Downtown Brooklyn": (40.6955, -73.9874),
        "Bed-Stuy": (40.6872, -73.9418),
        "Crown Heights": (40.6681, -73.9448),
        "Flushing": (40.7675, -73.8331),
        "Jamaica": (40.7027, -73.7890),
        "Park Slope": (40.6711, -73.9814),
        "Bushwick": (40.6958, -73.9171),
        "Williamsburg": (40.7081, -73.9571),
    }

    rows = []
    today = datetime.today()

    for i in range(850):
        neighborhood = np.random.choice(neighborhoods)
        base_lat, base_lon = centers[neighborhood]

        issue_type = np.random.choice(
            issue_types,
            p=[0.20, 0.18, 0.20, 0.18, 0.12, 0.12]
        )

        severity = np.random.choice(
            severity_levels,
            p=[0.30, 0.42, 0.22, 0.06]
        )

        severity_score_map = {
            "Low": np.random.uniform(10, 35),
            "Medium": np.random.uniform(36, 65),
            "High": np.random.uniform(66, 85),
            "Critical": np.random.uniform(86, 100),
        }

        route_id = f"R-{np.random.randint(1, 116):03d}"
        bus_id = f"BUS-{np.random.randint(1000, 9999)}"

        event_time = today - timedelta(
            days=np.random.randint(0, 30),
            hours=np.random.randint(0, 24),
            minutes=np.random.randint(0, 60)
        )

        rows.append({
            "event_id": f"EVT-{i+1:05d}",
            "event_time": event_time,
            "date": event_time.date(),
            "hour": event_time.hour,
            "route_id": route_id,
            "bus_id": bus_id,
            "neighborhood": neighborhood,
            "issue_type": issue_type,
            "severity": severity,
            "risk_score": round(severity_score_map[severity], 1),
            "lat": base_lat + np.random.normal(0, 0.012),
            "lon": base_lon + np.random.normal(0, 0.012),
            "delay_minutes": max(0, round(np.random.normal(12, 7), 1)),
            "speed_mph": max(5, round(np.random.normal(23, 8), 1)),
            "action_recommended": np.random.choice([
                "Review route timing",
                "Flag for dispatcher review",
                "Monitor next 3 trips",
                "Consider driver coaching",
                "Check recurring congestion zone",
                "Escalate if repeated this week"
            ])
        })

    alarms_df = pd.DataFrame(rows)

    hotspot_df = (
        alarms_df
        .groupby(["neighborhood", "issue_type"], as_index=False)
        .agg(
            event_count=("event_id", "count"),
            avg_risk_score=("risk_score", "mean"),
            avg_delay_minutes=("delay_minutes", "mean"),
            lat=("lat", "mean"),
            lon=("lon", "mean")
        )
    )

    hotspot_df["avg_risk_score"] = hotspot_df["avg_risk_score"].round(1)
    hotspot_df["avg_delay_minutes"] = hotspot_df["avg_delay_minutes"].round(1)

    route_df = (
        alarms_df
        .groupby("route_id", as_index=False)
        .agg(
            total_events=("event_id", "count"),
            avg_risk_score=("risk_score", "mean"),
            critical_events=("severity", lambda x: (x == "Critical").sum()),
            high_events=("severity", lambda x: (x == "High").sum()),
            avg_delay_minutes=("delay_minutes", "mean"),
            lat=("lat", "mean"),
            lon=("lon", "mean")
        )
    )

    route_df["avg_risk_score"] = route_df["avg_risk_score"].round(1)
    route_df["avg_delay_minutes"] = route_df["avg_delay_minutes"].round(1)

    route_df["route_status"] = np.where(
        route_df["avg_risk_score"] >= 75,
        "Needs Immediate Review",
        np.where(
            route_df["avg_risk_score"] >= 55,
            "Monitor Closely",
            "Stable"
        )
    )

    return alarms_df, hotspot_df, route_df


# ============================================================
# DATA LOADING
# ============================================================
def normalize_columns(df):
    """
    Makes the app more forgiving if uploaded files use slightly different column names.
    """
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    rename_map = {
        "latitude": "lat",
        "longitude": "lon",
        "lng": "lon",
        "risk": "risk_score",
        "score": "risk_score",
        "route": "route_id",
        "bus": "bus_id",
        "timestamp": "event_time",
        "time": "event_time",
        "event_type": "issue_type",
        "type": "issue_type",
    }

    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    return df


def load_data():
    st.sidebar.markdown("### Data Source")
    use_demo = st.sidebar.toggle("Use built-in demo data", value=True)

    if use_demo:
        return generate_demo_data()

    st.sidebar.markdown("Upload your dashboard CSV files.")
    uploaded_alarms = st.sidebar.file_uploader("Upload alarms/events CSV", type=["csv"])
    uploaded_hotspots = st.sidebar.file_uploader("Optional: upload hotspots CSV", type=["csv"])
    uploaded_routes = st.sidebar.file_uploader("Optional: upload route summary CSV", type=["csv"])

    if uploaded_alarms is None:
        st.sidebar.warning("No file uploaded. Using demo data for now.")
        return generate_demo_data()

    alarms_df = normalize_columns(pd.read_csv(uploaded_alarms))

    if "event_time" in alarms_df.columns:
        alarms_df["event_time"] = pd.to_datetime(alarms_df["event_time"], errors="coerce")
        alarms_df["date"] = alarms_df["event_time"].dt.date
        alarms_df["hour"] = alarms_df["event_time"].dt.hour
    else:
        alarms_df["event_time"] = pd.Timestamp.today()
        alarms_df["date"] = pd.Timestamp.today().date()
        alarms_df["hour"] = 8

    required_defaults = {
        "event_id": [f"EVT-{i+1:05d}" for i in range(len(alarms_df))],
        "route_id": "Unknown Route",
        "bus_id": "Unknown Bus",
        "neighborhood": "Unknown Area",
        "issue_type": "Operational Alert",
        "severity": "Medium",
        "risk_score": 50,
        "lat": 40.7128,
        "lon": -74.0060,
        "delay_minutes": 0,
        "speed_mph": 0,
        "action_recommended": "Review event"
    }

    for col, default in required_defaults.items():
        if col not in alarms_df.columns:
            alarms_df[col] = default

    if uploaded_hotspots is not None:
        hotspot_df = normalize_columns(pd.read_csv(uploaded_hotspots))
    else:
        hotspot_df = (
            alarms_df
            .groupby(["neighborhood", "issue_type"], as_index=False)
            .agg(
                event_count=("event_id", "count"),
                avg_risk_score=("risk_score", "mean"),
                avg_delay_minutes=("delay_minutes", "mean"),
                lat=("lat", "mean"),
                lon=("lon", "mean")
            )
        )

    if uploaded_routes is not None:
        route_df = normalize_columns(pd.read_csv(uploaded_routes))
    else:
        route_df = (
            alarms_df
            .groupby("route_id", as_index=False)
            .agg(
                total_events=("event_id", "count"),
                avg_risk_score=("risk_score", "mean"),
                critical_events=("severity", lambda x: (x == "Critical").sum()),
                high_events=("severity", lambda x: (x == "High").sum()),
                avg_delay_minutes=("delay_minutes", "mean"),
                lat=("lat", "mean"),
                lon=("lon", "mean")
            )
        )

    if "route_status" not in route_df.columns:
        route_df["route_status"] = np.where(
            route_df["avg_risk_score"] >= 75,
            "Needs Immediate Review",
            np.where(route_df["avg_risk_score"] >= 55, "Monitor Closely", "Stable")
        )

    return alarms_df, hotspot_df, route_df


alarms_df, hotspot_df, route_df = load_data()


# ============================================================
# SIDEBAR FILTERS
# ============================================================
st.sidebar.markdown("---")
st.sidebar.markdown("### Dashboard Filters")

all_issue_types = sorted(alarms_df["issue_type"].dropna().unique())
selected_issue_types = st.sidebar.multiselect(
    "Issue type",
    options=all_issue_types,
    default=all_issue_types
)

all_severities = sorted(alarms_df["severity"].dropna().unique())
selected_severities = st.sidebar.multiselect(
    "Severity",
    options=all_severities,
    default=all_severities
)

all_neighborhoods = sorted(alarms_df["neighborhood"].dropna().unique())
selected_neighborhoods = st.sidebar.multiselect(
    "Neighborhood",
    options=all_neighborhoods,
    default=all_neighborhoods
)

risk_range = st.sidebar.slider(
    "Risk score range",
    min_value=0,
    max_value=100,
    value=(0, 100)
)

filtered_df = alarms_df[
    alarms_df["issue_type"].isin(selected_issue_types)
    & alarms_df["severity"].isin(selected_severities)
    & alarms_df["neighborhood"].isin(selected_neighborhoods)
    & alarms_df["risk_score"].between(risk_range[0], risk_range[1])
].copy()


filtered_hotspots = (
    filtered_df
    .groupby(["neighborhood", "issue_type"], as_index=False)
    .agg(
        event_count=("event_id", "count"),
        avg_risk_score=("risk_score", "mean"),
        avg_delay_minutes=("delay_minutes", "mean"),
        lat=("lat", "mean"),
        lon=("lon", "mean")
    )
)

if len(filtered_hotspots) > 0:
    filtered_hotspots["avg_risk_score"] = filtered_hotspots["avg_risk_score"].round(1)
    filtered_hotspots["avg_delay_minutes"] = filtered_hotspots["avg_delay_minutes"].round(1)


filtered_routes = (
    filtered_df
    .groupby("route_id", as_index=False)
    .agg(
        total_events=("event_id", "count"),
        avg_risk_score=("risk_score", "mean"),
        critical_events=("severity", lambda x: (x == "Critical").sum()),
        high_events=("severity", lambda x: (x == "High").sum()),
        avg_delay_minutes=("delay_minutes", "mean"),
        lat=("lat", "mean"),
        lon=("lon", "mean")
    )
)

if len(filtered_routes) > 0:
    filtered_routes["avg_risk_score"] = filtered_routes["avg_risk_score"].round(1)
    filtered_routes["avg_delay_minutes"] = filtered_routes["avg_delay_minutes"].round(1)
    filtered_routes["route_status"] = np.where(
        filtered_routes["avg_risk_score"] >= 75,
        "Needs Immediate Review",
        np.where(filtered_routes["avg_risk_score"] >= 55, "Monitor Closely", "Stable")
    )


# ============================================================
# HEADER
# ============================================================
st.markdown('<div class="title-text">NYCSBUS Safety Operations Dashboard</div>', unsafe_allow_html=True)
st.markdown(
    """
    <div class="subtitle-text">
    A Streamlit-based analytics tool for monitoring route risk, recurring safety alerts,
    congestion patterns, and neighborhood-level operational hotspots.
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="section-card">
    <b>Purpose:</b> This dashboard helps operations managers quickly identify where safety risks are increasing,
    which routes may need review, and where recurring delays or incident patterns are concentrated.
    The goal is to turn raw transportation data into clear, actionable decisions for dispatchers, managers,
    and safety teams.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# KPI SECTION
# ============================================================
total_events = len(filtered_df)
high_risk_events = len(filtered_df[filtered_df["severity"].isin(["High", "Critical"])])
critical_events = len(filtered_df[filtered_df["severity"] == "Critical"])
avg_risk = filtered_df["risk_score"].mean() if len(filtered_df) > 0 else 0
avg_delay = filtered_df["delay_minutes"].mean() if len(filtered_df) > 0 else 0
routes_monitored = filtered_df["route_id"].nunique() if len(filtered_df) > 0 else 0

kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

with kpi1:
    st.metric("Total Alerts", f"{total_events:,}")

with kpi2:
    st.metric("High/Critical Alerts", f"{high_risk_events:,}")

with kpi3:
    st.metric("Critical Alerts", f"{critical_events:,}")

with kpi4:
    st.metric("Avg Risk Score", f"{avg_risk:.1f}")

with kpi5:
    st.metric("Routes Monitored", f"{routes_monitored:,}")


# ============================================================
# MAIN TABS
# ============================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Executive Overview",
    "Interactive Safety Map",
    "Hotspot Analysis",
    "Route Review",
    "Raw Data"
])


# ============================================================
# TAB 1: EXECUTIVE OVERVIEW
# ============================================================
with tab1:
    st.markdown("## Executive Overview")

    c1, c2 = st.columns([1.1, 1])

    with c1:
        st.markdown("### Alert Volume by Issue Type")

        if len(filtered_df) > 0:
            issue_counts = (
                filtered_df["issue_type"]
                .value_counts()
                .reset_index()
            )
            issue_counts.columns = ["issue_type", "count"]

            fig_issue = px.bar(
                issue_counts,
                x="issue_type",
                y="count",
                text="count",
                title="Most Common Operational Alerts",
                labels={
                    "issue_type": "Issue Type",
                    "count": "Alert Count"
                }
            )
            fig_issue.update_layout(
                xaxis_tickangle=-35,
                height=430,
                margin=dict(l=20, r=20, t=60, b=80)
            )
            st.plotly_chart(fig_issue, use_container_width=True)
        else:
            st.info("No data available for the selected filters.")

    with c2:
        st.markdown("### Severity Breakdown")

        if len(filtered_df) > 0:
            severity_counts = (
                filtered_df["severity"]
                .value_counts()
                .reset_index()
            )
            severity_counts.columns = ["severity", "count"]

            fig_severity = px.pie(
                severity_counts,
                names="severity",
                values="count",
                hole=0.45,
                title="Alerts by Severity"
            )
            fig_severity.update_layout(height=430)
            st.plotly_chart(fig_severity, use_container_width=True)
        else:
            st.info("No severity data available.")

    st.markdown("### What this tells the manager")

    insight_col1, insight_col2, insight_col3 = st.columns(3)

    with insight_col1:
        st.markdown(
            """
            <div class="section-card">
            <b>1. Where risk is concentrated</b><br><br>
            The dashboard highlights neighborhoods and routes with repeated safety signals,
            helping managers avoid relying on one-off incidents.
            </div>
            """,
            unsafe_allow_html=True
        )

    with insight_col2:
        st.markdown(
            """
            <div class="section-card">
            <b>2. Which issues need action</b><br><br>
            Managers can filter by speeding risk, harsh braking, route deviation,
            congestion delays, and late arrival patterns.
            </div>
            """,
            unsafe_allow_html=True
        )

    with insight_col3:
        st.markdown(
            """
            <div class="section-card">
            <b>3. How this supports safety protocols</b><br><br>
            The output can support dispatcher review, driver coaching,
            route timing changes, or targeted monitoring.
            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# TAB 2: INTERACTIVE SAFETY MAP
# ============================================================
with tab2:
    st.markdown("## Interactive Safety Map")
    st.caption("Pan, zoom, hover, and filter alerts to explore where safety and operational risks are occurring.")

    if len(filtered_df) > 0:
        map_df = filtered_df.dropna(subset=["lat", "lon"]).copy()

        fig_map = px.scatter_mapbox(
            map_df,
            lat="lat",
            lon="lon",
            color="severity",
            size="risk_score",
            hover_name="route_id",
            hover_data={
                "bus_id": True,
                "neighborhood": True,
                "issue_type": True,
                "risk_score": True,
                "delay_minutes": True,
                "speed_mph": True,
                "lat": False,
                "lon": False
            },
            zoom=10,
            height=650,
            title="Live-Style View of Safety and Operations Alerts"
        )

        fig_map.update_layout(
            mapbox_style="open-street-map",
            margin=dict(l=0, r=0, t=50, b=0),
            legend_title_text="Severity"
        )

        st.plotly_chart(fig_map, use_container_width=True)

        st.markdown("### Highest Priority Alerts")

        top_alerts = (
            filtered_df
            .sort_values("risk_score", ascending=False)
            .head(10)
            [[
                "event_time",
                "route_id",
                "bus_id",
                "neighborhood",
                "issue_type",
                "severity",
                "risk_score",
                "delay_minutes",
                "action_recommended"
            ]]
        )

        st.dataframe(top_alerts, use_container_width=True, hide_index=True)
    else:
        st.warning("No map data available for the selected filters.")


# ============================================================
# TAB 3: HOTSPOT ANALYSIS
# ============================================================
with tab3:
    st.markdown("## Hotspot Analysis")
    st.caption("This view groups repeated alerts by neighborhood and issue type to identify recurring problem areas.")

    if len(filtered_hotspots) > 0:
        c1, c2 = st.columns([1.1, 1])

        with c1:
            fig_hotspot_map = px.scatter_mapbox(
                filtered_hotspots,
                lat="lat",
                lon="lon",
                size="event_count",
                color="avg_risk_score",
                hover_name="neighborhood",
                hover_data={
                    "issue_type": True,
                    "event_count": True,
                    "avg_risk_score": True,
                    "avg_delay_minutes": True,
                    "lat": False,
                    "lon": False
                },
                zoom=10,
                height=580,
                title="Neighborhood-Level Hotspots"
            )

            fig_hotspot_map.update_layout(
                mapbox_style="open-street-map",
                margin=dict(l=0, r=0, t=50, b=0)
            )

            st.plotly_chart(fig_hotspot_map, use_container_width=True)

        with c2:
            top_hotspots = (
                filtered_hotspots
                .sort_values(["event_count", "avg_risk_score"], ascending=False)
                .head(12)
            )

            fig_hotspot_bar = px.bar(
                top_hotspots,
                x="event_count",
                y="neighborhood",
                color="issue_type",
                orientation="h",
                title="Top Recurring Hotspots",
                labels={
                    "event_count": "Alert Count",
                    "neighborhood": "Neighborhood"
                }
            )
            fig_hotspot_bar.update_layout(
                height=580,
                yaxis=dict(autorange="reversed"),
                margin=dict(l=20, r=20, t=50, b=30)
            )

            st.plotly_chart(fig_hotspot_bar, use_container_width=True)

        st.markdown("### Hotspot Summary Table")
        st.dataframe(
            filtered_hotspots
            .sort_values(["event_count", "avg_risk_score"], ascending=False),
            use_container_width=True,
            hide_index=True
        )

    else:
        st.warning("No hotspot data available for the selected filters.")


# ============================================================
# TAB 4: ROUTE REVIEW
# ============================================================
with tab4:
    st.markdown("## Route Review")
    st.caption("This view helps managers identify routes that may need dispatcher review, schedule adjustment, or targeted coaching.")

    if len(filtered_routes) > 0:
        c1, c2 = st.columns([1.1, 1])

        with c1:
            fig_route_map = px.scatter_mapbox(
                filtered_routes,
                lat="lat",
                lon="lon",
                size="total_events",
                color="route_status",
                hover_name="route_id",
                hover_data={
                    "total_events": True,
                    "avg_risk_score": True,
                    "critical_events": True,
                    "high_events": True,
                    "avg_delay_minutes": True,
                    "route_status": True,
                    "lat": False,
                    "lon": False
                },
                zoom=10,
                height=580,
                title="Route-Level Risk View"
            )

            fig_route_map.update_layout(
                mapbox_style="open-street-map",
                margin=dict(l=0, r=0, t=50, b=0),
                legend_title_text="Route Status"
            )

            st.plotly_chart(fig_route_map, use_container_width=True)

        with c2:
            route_status_counts = (
                filtered_routes["route_status"]
                .value_counts()
                .reset_index()
            )
            route_status_counts.columns = ["route_status", "count"]

            fig_status = px.pie(
                route_status_counts,
                names="route_status",
                values="count",
                hole=0.45,
                title="Routes by Review Status"
            )
            fig_status.update_layout(height=300)
            st.plotly_chart(fig_status, use_container_width=True)

            top_routes = (
                filtered_routes
                .sort_values("avg_risk_score", ascending=False)
                .head(10)
            )

            fig_top_routes = px.bar(
                top_routes,
                x="avg_risk_score",
                y="route_id",
                orientation="h",
                title="Highest Risk Routes",
                labels={
                    "avg_risk_score": "Average Risk Score",
                    "route_id": "Route"
                }
            )
            fig_top_routes.update_layout(
                height=260,
                yaxis=dict(autorange="reversed"),
                margin=dict(l=20, r=20, t=50, b=30)
            )
            st.plotly_chart(fig_top_routes, use_container_width=True)

        st.markdown("### Manager Action List")

        action_df = (
            filtered_routes
            .sort_values(["avg_risk_score", "critical_events", "high_events"], ascending=False)
            .head(15)
            .copy()
        )

        action_df["recommended_next_step"] = np.where(
            action_df["route_status"] == "Needs Immediate Review",
            "Review with dispatcher and safety team",
            np.where(
                action_df["route_status"] == "Monitor Closely",
                "Monitor next trips and compare against baseline",
                "No immediate action needed"
            )
        )

        st.dataframe(
            action_df[[
                "route_id",
                "total_events",
                "avg_risk_score",
                "critical_events",
                "high_events",
                "avg_delay_minutes",
                "route_status",
                "recommended_next_step"
            ]],
            use_container_width=True,
            hide_index=True
        )

    else:
        st.warning("No route data available for the selected filters.")


# ============================================================
# TAB 5: RAW DATA
# ============================================================
with tab5:
    st.markdown("## Raw Data")
    st.caption("Use this section to inspect the filtered records behind the dashboard.")

    st.markdown("### Filtered Alert Records")
    st.dataframe(filtered_df, use_container_width=True, hide_index=True)

    csv = filtered_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Filtered Alerts as CSV",
        data=csv,
        file_name="nycsbus_filtered_alerts.csv",
        mime="text/csv"
    )


# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown(
    """
    <div class="small-muted">
    Built as a Streamlit operations analytics prototype for NYCSBUS-style transportation safety monitoring.
    </div>
    """,
    unsafe_allow_html=True
)
