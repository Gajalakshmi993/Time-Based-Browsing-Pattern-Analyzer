"""
analytics/pattern_analysis.py
───────────────────────────────
Compute hourly, daily, and category-level pattern insights
used by the report and dashboard.

Usage:
    python -m src.analytics.pattern_analysis
"""

import pandas as pd
from pathlib import Path
from src.config_loader import CFG


def hourly_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """Visits per hour, split by category."""
    return (
        df.groupby(["hour", "category"])
        .size()
        .reset_index(name="visits")
        .sort_values(["hour", "visits"], ascending=[True, False])
    )


def daily_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """Visits per date and category."""
    return (
        df.groupby(["date", "category"])
        .size()
        .reset_index(name="visits")
    )


def peak_hours(df: pd.DataFrame, top_n: int = 3) -> pd.Series:
    """Return the top N busiest hours."""
    return df.groupby("hour").size().nlargest(top_n)


def category_time_heatmap(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a pivot table: rows = hour (0-23), cols = category, values = visits.
    Useful for heatmap visualisation in the dashboard.
    """
    pivoted = (
        df.groupby(["hour", "category"])
        .size()
        .unstack(fill_value=0)
    )
    return pivoted


def switching_rate(sessions: pd.DataFrame) -> pd.DataFrame:
    """
    Compute category-switching rate per session.
    High switching = scattered, distracted session.
    """
    if "category_switches" not in sessions.columns or "n_events" not in sessions.columns:
        return sessions
    sessions = sessions.copy()
    sessions["switch_rate"] = (
        sessions["category_switches"] / sessions["n_events"].clip(lower=1)
    )
    return sessions


def run_pattern_analysis(
    browsing_path:  str | None = None,
    sessions_path:  str | None = None,
) -> dict:
    browsing_path = browsing_path or CFG["paths"]["processed_browsing"]
    sessions_path = sessions_path or CFG["paths"]["session_features"]

    df  = pd.read_csv(browsing_path)
    ses = pd.read_csv(sessions_path) if Path(sessions_path).exists() else pd.DataFrame()

    results = {
        "hourly":    hourly_patterns(df),
        "daily":     daily_patterns(df),
        "peak_hours": peak_hours(df),
        "heatmap":   category_time_heatmap(df),
    }
    if not ses.empty:
        results["sessions_with_switch_rate"] = switching_rate(ses)

    print("[Patterns] Peak hours:")
    print(results["peak_hours"].to_string())
    return results


if __name__ == "__main__":
    run_pattern_analysis()
