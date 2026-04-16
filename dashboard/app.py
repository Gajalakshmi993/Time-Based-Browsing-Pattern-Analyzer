"""
dashboard/app.py
─────────────────
Streamlit dashboard for the Browsing Pattern Analyzer.

Run:
    streamlit run dashboard/app.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.config_loader import CFG

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Browsing Pattern Analyzer",
    page_icon="🌐",
    layout="wide",
)

st.title("🌐 Browsing Pattern Analyzer")
st.caption("Time-based behavior analytics from your browser history")

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    days = st.selectbox("Time window", [3, 4, 5], index=2)
    st.markdown("---")
    st.markdown("**Paths (from config.yaml)**")
    st.code(CFG["paths"]["processed_browsing"], language=None)


# ── Load data ─────────────────────────────────────────────────
@st.cache_data
def load(path):
    p = Path(path)
    if p.exists():
        return pd.read_csv(p)
    return pd.DataFrame()


browsing  = load(CFG["paths"]["processed_browsing"])
sessions  = load(CFG["paths"]["sessions"])
features  = load(CFG["paths"]["session_features"])
clusters  = load(CFG["paths"]["cluster_labels"])
anomalies = load(CFG["paths"]["anomaly_scores"])
merged    = load(CFG["paths"]["merged_ram"])


def _no_data(name):
    st.info(f"No data found for **{name}**. Run the pipeline first: `python main.py`")


# ── Tab layout ────────────────────────────────────────────────
tabs = st.tabs([
    "📊 Overview",
    "🕐 Time Patterns",
    "🗂️ Sessions & Clusters",
    "💾 RAM Correlation",
    "🚨 Anomalies",
    "💡 Recommendations",
])

# ══════════════════════════════════════════════
# Tab 1 — Overview
# ══════════════════════════════════════════════
with tabs[0]:
    st.subheader("Top Domains & Categories")
    if browsing.empty:
        _no_data("browsing history")
    else:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total visits",   f"{len(browsing):,}")
        col2.metric("Unique domains",  f"{browsing['domain'].nunique():,}")
        col3.metric("Sessions",        f"{sessions['session_id'].nunique():,}" if not sessions.empty else "—")
        col4.metric("Days covered",    str(browsing["date"].nunique()) if "date" in browsing.columns else "—")

        col_a, col_b = st.columns(2)
        with col_a:
            top_d = browsing["domain"].value_counts().head(15).reset_index()
            top_d.columns = ["domain", "visits"]
            fig = px.bar(top_d, x="visits", y="domain", orientation="h",
                         title="Top 15 Domains", height=400)
            fig.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, use_container_width=True)
        with col_b:
            top_c = browsing["category"].value_counts().reset_index()
            top_c.columns = ["category", "visits"]
            fig2 = px.pie(top_c, names="category", values="visits",
                          title="Category Distribution", height=400)
            st.plotly_chart(fig2, use_container_width=True)


# ══════════════════════════════════════════════
# Tab 2 — Time Patterns
# ══════════════════════════════════════════════
with tabs[1]:
    st.subheader("Hourly & Day-wise Patterns")
    if browsing.empty:
        _no_data("browsing history")
    else:
        if "hour" in browsing.columns:
            hourly = browsing.groupby("hour").size().reset_index(name="visits")
            fig = px.bar(hourly, x="hour", y="visits",
                         title="Visits by Hour of Day",
                         labels={"hour": "Hour (local time)", "visits": "Visits"})
            st.plotly_chart(fig, use_container_width=True)

        if "day_name" in browsing.columns:
            order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
            daily = browsing["day_name"].value_counts().reindex(order, fill_value=0).reset_index()
            daily.columns = ["day", "visits"]
            fig2 = px.bar(daily, x="day", y="visits", title="Visits by Day of Week")
            st.plotly_chart(fig2, use_container_width=True)

        if "day_part" in browsing.columns:
            part = browsing["day_part"].value_counts().reset_index()
            part.columns = ["day_part", "visits"]
            fig3 = px.pie(part, names="day_part", values="visits",
                          title="Activity by Day Part")
            st.plotly_chart(fig3, use_container_width=True)


# ══════════════════════════════════════════════
# Tab 3 — Sessions & Clusters
# ══════════════════════════════════════════════
with tabs[2]:
    st.subheader("Session Clusters")
    if clusters.empty:
        _no_data("cluster data")
    else:
        if "cluster_label" in clusters.columns:
            summary = (
                clusters.groupby("cluster_label")
                .agg(n_sessions=("session_id", "count"),
                     avg_duration_min=("duration_sec", lambda x: round(x.mean()/60, 1)))
                .reset_index()
            )
            st.dataframe(summary, use_container_width=True)

            fig = px.scatter(
                clusters,
                x="duration_sec", y="n_events",
                color="cluster_label",
                hover_data=["top_category", "hour_start"],
                title="Sessions coloured by cluster",
                labels={"duration_sec": "Duration (s)", "n_events": "Events"},
            )
            st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════
# Tab 4 — RAM
# ══════════════════════════════════════════════
with tabs[3]:
    st.subheader("RAM Correlation")
    if merged.empty:
        _no_data("RAM-merged data")
    else:
        if "ram_used_mb" in merged.columns and "category" in merged.columns:
            cat_ram = (
                merged.groupby("category")
                .agg(ram_mean=("ram_used_mb", "mean"),
                     ram_peak=("ram_used_mb", "max"))
                .round(1)
                .sort_values("ram_mean", ascending=False)
                .reset_index()
            )
            fig = px.bar(cat_ram, x="category", y=["ram_mean", "ram_peak"],
                         barmode="group", title="RAM Usage by Category (MB)")
            st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════
# Tab 5 — Anomalies
# ══════════════════════════════════════════════
with tabs[4]:
    st.subheader("Anomalous Sessions")
    if anomalies.empty:
        _no_data("anomaly scores")
    elif "is_anomaly" not in anomalies.columns:
        st.warning("Run the autoencoder module to generate anomaly scores.")
    else:
        n_anom = int(anomalies["is_anomaly"].sum())
        st.metric("Flagged sessions", n_anom)

        fig = px.histogram(
            anomalies, x="reconstruction_error",
            nbins=40, title="Reconstruction Error Distribution",
            labels={"reconstruction_error": "Error (higher = more anomalous)"},
        )
        st.plotly_chart(fig, use_container_width=True)

        if n_anom > 0:
            st.dataframe(
                anomalies[anomalies["is_anomaly"]][
                    ["session_id", "hour_start", "duration_sec",
                     "top_category", "reconstruction_error", "anomaly_reason"]
                ].sort_values("reconstruction_error", ascending=False),
                use_container_width=True,
            )


# ══════════════════════════════════════════════
# Tab 6 — Recommendations
# ══════════════════════════════════════════════
with tabs[5]:
    st.subheader("💡 Actionable Recommendations")
    if not features.empty:
        from src.recommend.engine import generate_recommendations
        from src.analytics.ram_correlation import compute_category_ram_stats

        cat_ram = compute_category_ram_stats(merged) if not merged.empty else None
        recs = generate_recommendations(features, cat_ram)

        for rec in recs:
            st.markdown(f"### {rec['recommendation']}")
            st.caption(f"📌 Evidence: {rec['evidence']}")
            st.divider()
    else:
        _no_data("session features")
