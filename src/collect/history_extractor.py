"""
collect/history_extractor.py
────────────────────────────
Module 1 — Extract browsing history from Chrome / Edge SQLite DB.

Usage:
    python -m src.collect.history_extractor --days 5
"""

import argparse
import os
import platform
import shutil
import sqlite3
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from src.config_loader import CFG

# Chrome stores timestamps as microseconds since 1601-01-01
_CHROME_EPOCH = datetime(1601, 1, 1, tzinfo=timezone.utc)


def _chrome_ts_to_utc(ts: int) -> datetime:
    return _CHROME_EPOCH + timedelta(microseconds=ts)


def _resolve_history_path(browser: str) -> Path:
    """Return the platform-specific history DB path."""
    sys = platform.system().lower()
    key = "linux" if "linux" in sys else ("mac" if "darwin" in sys else "windows")
    raw = CFG["collection"][f"{browser}_history_paths"][key]
    return Path(os.path.expandvars(os.path.expanduser(raw)))


def extract_history(
    browser: str | None = None,
    days: int | None = None,
    output_path: str | None = None,
) -> pd.DataFrame:
    """
    Extract browsing history and save to CSV.

    Parameters
    ----------
    browser : str
        'chrome' or 'edge' (falls back to config default).
    days : int
        How many past days to extract (3, 4, or 5).
    output_path : str
        Where to write the CSV (falls back to config default).

    Returns
    -------
    pd.DataFrame
    """
    browser = browser or CFG["collection"]["browser"]
    days = days or CFG["collection"]["time_window_days"]
    output_path = output_path or CFG["paths"]["raw_browsing"]

    history_db = _resolve_history_path(browser)
    if not history_db.exists():
        raise FileNotFoundError(
            f"History DB not found at {history_db}. "
            "Close the browser or check the path in config.yaml."
        )

    # Copy DB to temp file — browser may hold a lock
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        tmp_path = tmp.name
    shutil.copy2(history_db, tmp_path)

    cutoff_utc = datetime.now(timezone.utc) - timedelta(days=days)
    cutoff_chrome_ts = int(
        (cutoff_utc - _CHROME_EPOCH).total_seconds() * 1_000_000
    )

    query = """
        SELECT
            v.visit_time                AS raw_ts,
            u.url                       AS url,
            u.title                     AS title,
            'chrome'                    AS browser
        FROM visits v
        JOIN urls u ON u.id = v.url
        WHERE v.visit_time >= ?
        ORDER BY v.visit_time
    """

    try:
        conn = sqlite3.connect(tmp_path)
        df = pd.read_sql_query(query, conn, params=(cutoff_chrome_ts,))
        conn.close()
    finally:
        os.unlink(tmp_path)

    if df.empty:
        print(f"[WARN] No history rows found for last {days} days.")
        return df

    df["timestamp"] = df["raw_ts"].apply(_chrome_ts_to_utc)
    df.drop(columns=["raw_ts"], inplace=True)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[OK] Extracted {len(df):,} history rows → {output_path}")
    return df


# ── CLI entry-point ──────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract browser history")
    parser.add_argument("--browser", default=None, choices=["chrome", "edge"])
    parser.add_argument("--days", type=int, default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    extract_history(browser=args.browser, days=args.days, output_path=args.output)
