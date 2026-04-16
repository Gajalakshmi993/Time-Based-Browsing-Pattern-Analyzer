"""
prep/sessionizer.py
───────────────────
Module 3 — Build sessions from cleaned browsing events and compute
per-session summary features used for clustering and DL.

Usage:
    python -m src.prep.sessionizer
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.config_loader import CFG


# ── Core sessionization ───────────────────────────────────────

def assign_sessions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 'session_id' to a sorted browsing DataFrame.

    Algorithm
    ---------
    - Sort by (user/browser, timestamp)
    - time_diff = gap from previous event per user
    - new session if gap > SESSION_GAP or first event of user
    - session_id = cumsum across entire DataFrame (globally unique)
    """
    gap_sec = CFG["sessionization"]["gap_minutes"] * 60

    df = df.sort_values("timestamp").copy()

    # Per-browser diff (acts as a proxy user split when no user_id)
    df["time_diff"] = (
        df.groupby("browser")["timestamp"]
        .diff()
        .dt.total_seconds()
    )

    df["new_session"] = (df["time_diff"] > gap_sec) | (df["time_diff"].isna())
    df["session_id"]  = df["new_session"].cumsum()
    return df


# ── Session-level feature table ───────────────────────────────

def _category_ratios(series: pd.Series) -> dict:
    """Return ratio of each category within a session."""
    counts = series.value_counts(normalize=True)
    return counts.to_dict()


def build_session_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse event-level rows into one row per session with rich features.

    Returns
    -------
    pd.DataFrame  with columns used downstream by clustering and DL.
    """
    min_events = CFG["sessionization"]["min_events_per_session"]

    records = []
    for sid, grp in df.groupby("session_id"):
        if len(grp) < min_events:
            continue

        grp = grp.sort_values("timestamp")
        cats = grp["category"]
        duration_sec = (
            grp["timestamp"].iloc[-1] - grp["timestamp"].iloc[0]
        ).total_seconds()

        # Category switching count
        switches = (cats != cats.shift()).sum() - 1

        ratios = _category_ratios(cats)

        records.append({
            "session_id":       sid,
            "browser":          grp["browser"].iloc[0],
            "session_start":    grp["timestamp"].iloc[0],
            "session_end":      grp["timestamp"].iloc[-1],
            "hour_start":       grp["timestamp_local"].iloc[0].hour
                                if "timestamp_local" in grp.columns
                                else grp["timestamp"].iloc[0].hour,
            "day_part":         grp["day_part"].iloc[0]
                                if "day_part" in grp.columns else "unknown",
            "day_name":         grp["day_name"].iloc[0]
                                if "day_name" in grp.columns else "unknown",
            "n_events":         len(grp),
            "n_unique_domains": grp["domain"].nunique(),
            "n_categories":     cats.nunique(),
            "duration_sec":     duration_sec,
            "events_per_min":   len(grp) / max(duration_sec / 60, 1),
            "category_switches": switches,
            "top_category":     cats.mode().iloc[0] if len(cats) else "unknown",
            # Ratio features for common categories
            "ratio_social":     ratios.get("social_media", 0.0),
            "ratio_video":      ratios.get("video", 0.0),
            "ratio_news":       ratios.get("news", 0.0),
            "ratio_shopping":   ratios.get("shopping", 0.0),
            "ratio_learning":   ratios.get("learning", 0.0),
            "ratio_work":       ratios.get("work", 0.0),
            "ratio_unknown":    ratios.get("unknown", 0.0),
        })

    feat = pd.DataFrame(records)
    return feat


# ── Pipeline entry-point ──────────────────────────────────────

def run_sessionization(
    input_path:   str | None = None,
    sessions_out: str | None = None,
    features_out: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    input_path   = input_path   or CFG["paths"]["processed_browsing"]
    sessions_out = sessions_out or CFG["paths"]["sessions"]
    features_out = features_out or CFG["paths"]["session_features"]

    df = pd.read_csv(input_path, parse_dates=["timestamp"])
    if "timestamp_local" in df.columns:
        df["timestamp_local"] = pd.to_datetime(df["timestamp_local"])

    df = assign_sessions(df)

    feat = build_session_features(df)

    # Save
    Path(sessions_out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(sessions_out, index=False)
    feat.to_csv(features_out, index=False)

    print(f"[Sessionizer] {df['session_id'].nunique():,} sessions found")
    print(f"[Sessionizer] Sessions → {sessions_out}")
    print(f"[Sessionizer] Features → {features_out}")
    return df, feat


if __name__ == "__main__":
    run_sessionization()
