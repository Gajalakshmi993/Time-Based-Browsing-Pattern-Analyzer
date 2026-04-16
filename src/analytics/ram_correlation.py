"""
analytics/ram_correlation.py
─────────────────────────────
Module 4 — Merge browsing events with RAM logs (nearest-timestamp join)
and compute per-session / per-category RAM statistics.

Usage:
    python -m src.analytics.ram_correlation
"""

import pandas as pd
from pathlib import Path

from src.config_loader import CFG


def merge_nearest(
    browsing: pd.DataFrame,
    ram: pd.DataFrame,
    tolerance_sec: int | None = None,
) -> pd.DataFrame:
    """
    Merge browsing events with RAM log using nearest timestamp.

    Parameters
    ----------
    browsing : pd.DataFrame   (must have 'timestamp' column)
    ram      : pd.DataFrame   (must have 'timestamp' column)
    tolerance_sec : int       Max gap allowed for a match.

    Returns
    -------
    pd.DataFrame  Browsing rows enriched with RAM columns.
    """
    tolerance_sec = tolerance_sec or CFG["ram"]["merge_tolerance_sec"]

    b = browsing.copy()
    r = ram.copy()

    b["timestamp"] = pd.to_datetime(b["timestamp"], utc=True)
    r["timestamp"] = pd.to_datetime(r["timestamp"], utc=True)

    # pd.merge_asof requires sorted keys
    b = b.sort_values("timestamp")
    r = r.sort_values("timestamp")

    merged = pd.merge_asof(
        b,
        r[["timestamp", "ram_used_mb", "ram_available_mb",
           "browser_ram_mb", "cpu_percent"]],
        on="timestamp",
        direction="nearest",
        tolerance=pd.Timedelta(seconds=tolerance_sec),
        suffixes=("", "_ram"),
    )
    return merged


def compute_session_ram_stats(merged: pd.DataFrame) -> pd.DataFrame:
    """
    Group merged data by session and compute RAM aggregates.
    """
    stats = (
        merged.groupby("session_id")
        .agg(
            ram_mean_mb=("ram_used_mb", "mean"),
            ram_peak_mb=("ram_used_mb", "max"),
            browser_ram_mean_mb=("browser_ram_mb", "mean"),
            browser_ram_peak_mb=("browser_ram_mb", "max"),
            cpu_mean=("cpu_percent", "mean"),
        )
        .round(2)
        .reset_index()
    )
    return stats


def compute_category_ram_stats(merged: pd.DataFrame) -> pd.DataFrame:
    """Average and peak RAM grouped by category."""
    return (
        merged.groupby("category")
        .agg(
            ram_mean_mb=("ram_used_mb", "mean"),
            ram_peak_mb=("ram_used_mb", "max"),
            browser_ram_mean_mb=("browser_ram_mb", "mean"),
            n_visits=("url", "count"),
        )
        .round(2)
        .sort_values("ram_mean_mb", ascending=False)
        .reset_index()
    )


def flag_ram_spikes(merged: pd.DataFrame) -> pd.DataFrame:
    """Add boolean 'ram_spike' column based on config threshold."""
    threshold = CFG["ram"]["spike_threshold_mb"]
    merged = merged.copy()
    merged["ram_delta"] = merged["ram_used_mb"].diff().fillna(0)
    merged["ram_spike"] = merged["ram_delta"].abs() > threshold
    return merged


def run_ram_analysis(
    sessions_path: str | None = None,
    ram_path:      str | None = None,
    output_path:   str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sessions_path = sessions_path or CFG["paths"]["sessions"]
    ram_path      = ram_path      or CFG["paths"]["raw_ram"]
    output_path   = output_path   or CFG["paths"]["merged_ram"]

    browsing = pd.read_csv(sessions_path)
    ram      = pd.read_csv(ram_path)

    merged       = merge_nearest(browsing, ram)
    merged       = flag_ram_spikes(merged)
    session_ram  = compute_session_ram_stats(merged)
    category_ram = compute_category_ram_stats(merged)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)

    top_n = CFG["ram"]["top_n_heavy_categories"]
    print("[RAM] Top memory-heavy categories:")
    print(category_ram.head(top_n)[["category", "ram_mean_mb", "ram_peak_mb"]].to_string(index=False))

    return merged, session_ram, category_ram


if __name__ == "__main__":
    run_ram_analysis()
