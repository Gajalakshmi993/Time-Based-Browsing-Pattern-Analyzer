"""
analytics/report_generator.py
───────────────────────────────
Assemble final Markdown report from all processed artifacts.

Usage:
    python -m src.analytics.report_generator
"""

from datetime import datetime
from pathlib import Path

import pandas as pd

from src.config_loader import CFG


def generate_report(
    browsing_path:  str | None = None,
    features_path:  str | None = None,
    cluster_path:   str | None = None,
    anomaly_path:   str | None = None,
    recommendations: list | None = None,
    category_ram:   pd.DataFrame | None = None,
    output_path:    str | None = None,
) -> str:
    browsing_path = browsing_path or CFG["paths"]["processed_browsing"]
    features_path = features_path or CFG["paths"]["session_features"]
    cluster_path  = cluster_path  or CFG["paths"]["cluster_labels"]
    anomaly_path  = anomaly_path  or CFG["paths"]["anomaly_scores"]
    output_path   = output_path   or CFG["paths"]["report_out"]

    lines = []
    now   = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines += [
        f"# Browsing Pattern Analysis Report",
        f"**Generated:** {now}  |  **Time window:** last {CFG['collection']['time_window_days']} days",
        "",
    ]

    # ── Section 1: Top domains ────────────────────────────────
    lines.append("## 1. Top Domains & Categories")
    if Path(browsing_path).exists():
        df = pd.read_csv(browsing_path)
        top_domains = df["domain"].value_counts().head(10).reset_index()
        top_domains.columns = ["domain", "visits"]
        top_cats = df["category"].value_counts().head(10).reset_index()
        top_cats.columns = ["category", "visits"]

        lines.append("\n### Top 10 Domains")
        lines.append(top_domains.to_markdown(index=False))
        lines.append("\n### Top 10 Categories")
        lines.append(top_cats.to_markdown(index=False))
    else:
        lines.append("_Data not found._")
    lines.append("")

    # ── Section 2: Time patterns ──────────────────────────────
    lines.append("## 2. Time-Based Patterns")
    if Path(browsing_path).exists():
        df = pd.read_csv(browsing_path)
        if "hour" in df.columns:
            hourly = df.groupby("hour").size().reset_index(name="visits")
            peak   = hourly.loc[hourly["visits"].idxmax()]
            lines.append(f"- **Peak browsing hour:** {int(peak['hour']):02d}:00 ({int(peak['visits'])} visits)")
        if "day_name" in df.columns:
            busiest_day = df["day_name"].value_counts().idxmax()
            lines.append(f"- **Busiest day:** {busiest_day}")
        if "day_part" in df.columns:
            busiest_part = df["day_part"].value_counts().idxmax()
            lines.append(f"- **Most active day-part:** {busiest_part}")
    lines.append("")

    # ── Section 3: Session summary ────────────────────────────
    lines.append("## 3. Session Summary")
    if Path(features_path).exists():
        feat = pd.read_csv(features_path)
        lines += [
            f"- **Total sessions:** {len(feat):,}",
            f"- **Avg session duration:** {feat['duration_sec'].mean()/60:.1f} min",
            f"- **Avg events/session:** {feat['n_events'].mean():.1f}",
            f"- **Avg unique domains/session:** {feat['n_unique_domains'].mean():.1f}",
        ]
    lines.append("")

    # ── Section 4: Clusters ───────────────────────────────────
    lines.append("## 4. Session Clusters")
    if Path(cluster_path).exists():
        clust = pd.read_csv(cluster_path)
        if "cluster_label" in clust.columns:
            summary = (
                clust.groupby("cluster_label")
                .agg(n_sessions=("session_id", "count"),
                     avg_duration_min=("duration_sec", lambda x: round(x.mean()/60, 1)),
                     top_category=("top_category", lambda x: x.mode().iloc[0]))
                .reset_index()
            )
            lines.append(summary.to_markdown(index=False))
    else:
        lines.append("_Cluster data not found._")
    lines.append("")

    # ── Section 5: RAM correlation ────────────────────────────
    lines.append("## 5. RAM Correlation Findings")
    if category_ram is not None and not category_ram.empty:
        top_n = CFG["ram"]["top_n_heavy_categories"]
        lines.append(f"Top {top_n} memory-heavy categories:")
        lines.append(
            category_ram.head(top_n)[
                ["category", "ram_mean_mb", "ram_peak_mb", "n_visits"]
            ].to_markdown(index=False)
        )
    else:
        lines.append("_RAM log data not found or not yet merged._")
    lines.append("")

    # ── Section 6: Anomalies ──────────────────────────────────
    lines.append("## 6. Anomalous Sessions")
    if Path(anomaly_path).exists():
        anom = pd.read_csv(anomaly_path)
        n_flag = anom["is_anomaly"].sum() if "is_anomaly" in anom.columns else 0
        lines.append(f"- **Anomalies detected:** {n_flag}")
        if n_flag > 0 and "anomaly_reason" in anom.columns:
            top_reasons = (
                anom[anom["is_anomaly"]]["anomaly_reason"]
                .value_counts().head(5)
            )
            lines.append("\nTop anomaly reasons:")
            for reason, cnt in top_reasons.items():
                lines.append(f"  - {reason}: {cnt} sessions")
    else:
        lines.append("_Anomaly data not found._")
    lines.append("")

    # ── Section 7: Recommendations ───────────────────────────
    lines.append("## 7. Recommendations")
    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            lines += [
                f"### {i}. {rec['recommendation']}",
                f"> **Evidence:** {rec['evidence']}",
                "",
            ]
    else:
        lines.append("_Run the recommendation engine to populate this section._")

    report_md = "\n".join(lines)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(report_md)
    print(f"[Report] Saved → {output_path}")
    return report_md


if __name__ == "__main__":
    generate_report()
