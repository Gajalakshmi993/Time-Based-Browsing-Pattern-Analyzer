"""
collect/generate_sample_data.py
────────────────────────────────
Generates realistic synthetic browsing_history.csv and ram_log.csv
for testing the full pipeline without needing a real browser.

Usage:
    python -m src.collect.generate_sample_data --days 5
"""

import argparse
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.config_loader import CFG

# ── Domain pool ───────────────────────────────────────────────
DOMAIN_POOL = {
    "social_media": [
        "instagram.com", "twitter.com", "reddit.com",
        "facebook.com", "linkedin.com", "threads.net",
    ],
    "video": [
        "youtube.com", "netflix.com", "primevideo.com", "twitch.tv",
    ],
    "news": [
        "bbc.com", "ndtv.com", "thehindu.com", "techcrunch.com", "theverge.com",
    ],
    "learning": [
        "coursera.org", "udemy.com", "medium.com",
        "stackoverflow.com", "wikipedia.org", "towardsdatascience.com",
    ],
    "work": [
        "github.com", "docs.google.com", "notion.so",
        "slack.com", "zoom.us", "atlassian.net",
    ],
    "shopping": [
        "amazon.com", "flipkart.com", "myntra.com",
    ],
    "search": ["google.com", "bing.com"],
    "email":  ["mail.google.com", "outlook.live.com"],
}

# Flatten for random sampling
ALL_DOMAINS = [(d, cat) for cat, domains in DOMAIN_POOL.items() for d in domains]

# ── Behaviour archetypes ──────────────────────────────────────
ARCHETYPES = [
    # (hour_start, category_weights, session_length_range)
    {"name": "Morning Work",       "hours": (7, 10),  "cats": {"work": 5, "news": 2, "search": 1}, "len": (5, 20)},
    {"name": "Social Scroll",      "hours": (12, 14), "cats": {"social_media": 6, "video": 2},      "len": (10, 40)},
    {"name": "Afternoon Learning", "hours": (14, 18), "cats": {"learning": 5, "work": 3},            "len": (8, 25)},
    {"name": "Evening Binge",      "hours": (19, 22), "cats": {"video": 6, "social_media": 2},      "len": (20, 60)},
    {"name": "Late Night Social",  "hours": (22, 24), "cats": {"social_media": 8, "news": 1},       "len": (15, 50)},
    {"name": "Quick News",         "hours": (6, 23),  "cats": {"news": 4, "search": 2},             "len": (3, 12)},
    {"name": "Shopping",           "hours": (11, 21), "cats": {"shopping": 6, "search": 2},         "len": (5, 20)},
]


def _weighted_domain(cat_weights: dict) -> tuple[str, str]:
    cats   = list(cat_weights.keys())
    wts    = list(cat_weights.values())
    cat    = random.choices(cats, weights=wts, k=1)[0]
    domain = random.choice(DOMAIN_POOL.get(cat, [("unknown.com", "unknown")]))
    return domain, cat


def _ram_at(ts: datetime, browser_heavy: bool) -> dict:
    base_ram  = random.gauss(4200, 300)
    b_ram     = random.gauss(1200 if browser_heavy else 600, 150)
    return {
        "timestamp":        ts.isoformat(),
        "ram_used_mb":      round(max(2000, base_ram), 1),
        "ram_available_mb": round(max(500, 8192 - base_ram), 1),
        "browser_ram_mb":   round(max(200, b_ram), 1),
        "cpu_percent":      round(random.uniform(5, 60), 1),
    }


def generate(days: int = 5, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    random.seed(seed)
    np.random.seed(seed)

    now    = datetime.now(timezone.utc)
    start  = now - timedelta(days=days)

    browse_rows = []
    ram_rows    = []

    # RAM log: every 10 seconds
    t = start
    while t < now:
        heavy = (18 <= t.hour <= 23)
        ram_rows.append(_ram_at(t, heavy))
        t += timedelta(seconds=10)

    # Browsing: generate sessions per day
    day_cursor = start
    while day_cursor < now:
        n_sessions = random.randint(4, 10)
        for _ in range(n_sessions):
            arch   = random.choice(ARCHETYPES)
            h_lo, h_hi = arch["hours"]
            hour   = random.randint(h_lo, min(h_hi, 23))
            minute = random.randint(0, 59)
            ts     = day_cursor.replace(
                hour=hour, minute=minute, second=0, microsecond=0
            )
            if ts > now:
                continue

            n_events = random.randint(*arch["len"])
            for e in range(n_events):
                domain, cat = _weighted_domain(arch["cats"])
                visit_ts    = ts + timedelta(seconds=e * random.randint(10, 120))
                browse_rows.append({
                    "timestamp": visit_ts.isoformat(),
                    "url":       f"https://{domain}/page_{random.randint(1,9999)}",
                    "title":     f"Page on {domain}",
                    "browser":   "chrome",
                })

        day_cursor += timedelta(days=1)

    browsing_df = pd.DataFrame(browse_rows).drop_duplicates(subset=["url", "timestamp"])
    ram_df      = pd.DataFrame(ram_rows)

    # Save
    out_b = CFG["paths"]["raw_browsing"]
    out_r = CFG["paths"]["raw_ram"]
    Path(out_b).parent.mkdir(parents=True, exist_ok=True)
    browsing_df.to_csv(out_b, index=False)
    ram_df.to_csv(out_r, index=False)

    print(f"[SampleData] {len(browsing_df):,} browse events → {out_b}")
    print(f"[SampleData] {len(ram_df):,} RAM samples     → {out_r}")
    return browsing_df, ram_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic browsing data")
    parser.add_argument("--days", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    generate(days=args.days, seed=args.seed)
