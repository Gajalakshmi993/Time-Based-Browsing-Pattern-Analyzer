"""
prep/preprocessor.py
────────────────────
Module 2 — Clean URLs, extract domains, map categories, enrich timestamps.

Usage:
    python -m src.prep.preprocessor
"""

import re
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import pytz

from src.config_loader import CFG

# ── Helpers ──────────────────────────────────────────────────

def _strip_query(url: str) -> str:
    """Remove query string and fragment from URL."""
    try:
        p = urlparse(url)
        return p._replace(query="", fragment="").geturl()
    except Exception:
        return url


def _extract_domain(url: str) -> str:
    """Return bare domain (strip www.)."""
    try:
        host = urlparse(url).netloc.lower()
        return re.sub(r"^www\.", "", host)
    except Exception:
        return ""


def _label_day_part(hour: int) -> str:
    for part, (start, end) in CFG["features"]["day_parts"].items():
        if start <= hour <= end:
            return part
    return "late_night"


# ── Main pipeline ─────────────────────────────────────────────

def preprocess(
    raw_path: str | None = None,
    domain_map_path: str | None = None,
    output_path: str | None = None,
) -> pd.DataFrame:
    """
    Full preprocessing pipeline.

    Steps
    -----
    1. Load raw browsing CSV
    2. Drop duplicates / invalid rows
    3. Strip query strings
    4. Extract domain
    5. Map domain → category
    6. Convert timestamps to local timezone
    7. Derive hour, date, day_name, day_part columns

    Returns
    -------
    pd.DataFrame  (also saves to output_path)
    """
    raw_path = raw_path or CFG["paths"]["raw_browsing"]
    domain_map_path = domain_map_path or CFG["paths"]["domain_map"]
    output_path = output_path or CFG["paths"]["processed_browsing"]

    # 1. Load
    df = pd.read_csv(raw_path)
    print(f"[Preprocess] Loaded {len(df):,} rows from {raw_path}")

    # 2. Drop bad rows
    df = df.dropna(subset=["url", "timestamp"])
    df = df[df["url"].str.len() >= CFG["preprocessing"]["min_url_length"]]
    if CFG["preprocessing"]["deduplicate"]:
        df = df.drop_duplicates(subset=["url", "timestamp"])

    # 3. Strip query strings
    if CFG["preprocessing"]["remove_query_strings"]:
        df["url_clean"] = df["url"].apply(_strip_query)
    else:
        df["url_clean"] = df["url"]

    # 4. Domain extraction
    df["domain"] = df["url_clean"].apply(_extract_domain)
    df = df[df["domain"] != ""]   # drop rows where domain extraction failed

    # 5. Category mapping
    if Path(domain_map_path).exists():
        cat_map = pd.read_csv(domain_map_path)
        domain_to_cat = dict(zip(cat_map["domain"], cat_map["category"]))
    else:
        print(f"[WARN] domain_category_map not found at {domain_map_path}. "
              "All categories set to 'unknown'.")
        domain_to_cat = {}

    df["category"] = df["domain"].map(domain_to_cat).fillna("unknown")

    # 6. Timestamp → local time
    tz = pytz.timezone(CFG["preprocessing"]["timezone"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["timestamp_local"] = df["timestamp"].dt.tz_convert(tz)

    # 7. Derived time columns
    df["hour"]      = df["timestamp_local"].dt.hour
    df["date"]      = df["timestamp_local"].dt.date
    df["day_name"]  = df["timestamp_local"].dt.day_name()
    df["day_part"]  = df["hour"].apply(_label_day_part)

    # Sort
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[Preprocess] Saved {len(df):,} clean rows → {output_path}")
    return df


if __name__ == "__main__":
    preprocess()
