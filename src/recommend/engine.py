"""
recommend/engine.py
────────────────────
Module 7 — Rule + ML signal recommendation engine.

Every recommendation is traceable to a metric threshold defined in config.yaml.

Usage:
    python -m src.recommend.engine
"""

import pandas as pd
from src.config_loader import CFG


# ── Individual rule functions ────────────────────────────────

def _check_social_overuse(feat: pd.DataFrame, cfg: dict) -> list[dict]:
    recs = []
    social_threshold = cfg["social_ratio_threshold"]
    late_start       = cfg["late_night_hour_start"]
    max_social_min   = cfg["max_daily_social_minutes"]

    late_social = feat[
        (feat["ratio_social"] > social_threshold) &
        (feat["hour_start"] >= late_start)
    ]
    if len(late_social) > 0:
        recs.append({
            "rule":      "late_night_social",
            "evidence":  f"{len(late_social)} sessions: ratio_social > {social_threshold} after {late_start}:00",
            "recommendation": (
                "📵 Enable a screen-time limit for social media after 10 PM. "
                "Late-night scrolling disrupts sleep and reduces next-day focus."
            ),
        })

    if "duration_sec" in feat.columns:
        daily_social_min = (
            feat[feat["ratio_social"] > 0.3]["duration_sec"].sum() / 60
        )
        if daily_social_min > max_social_min:
            recs.append({
                "rule":      "excess_daily_social",
                "evidence":  f"~{daily_social_min:.0f} min/day social media (threshold={max_social_min} min)",
                "recommendation": (
                    "⏱ Try time-boxing social media to 30-minute blocks with "
                    "a timer to stay within healthy limits."
                ),
            })
    return recs


def _check_heavy_ram(category_ram: pd.DataFrame, cfg: dict) -> list[dict]:
    recs = []
    if category_ram is None or category_ram.empty:
        return recs
    threshold = cfg["heavy_ram_threshold_mb"]
    heavy = category_ram[category_ram["ram_peak_mb"] > threshold]
    if not heavy.empty:
        cats = ", ".join(heavy["category"].tolist())
        recs.append({
            "rule":      "ram_heavy_categories",
            "evidence":  f"Peak RAM > {threshold} MB for: {cats}",
            "recommendation": (
                f"💾 Categories [{cats}] spike system RAM. "
                "Close unused tabs in these categories or use a tab suspender extension."
            ),
        })
    return recs


def _check_focus_blocks(feat: pd.DataFrame) -> list[dict]:
    recs = []
    if "hour_start" not in feat.columns:
        return recs
    distraction_hours = feat[
        feat["ratio_social"].fillna(0) + feat["ratio_video"].fillna(0) > 0.6
    ]["hour_start"]
    if len(distraction_hours) >= 3:
        peak_hour = int(distraction_hours.mode().iloc[0])
        recs.append({
            "rule":      "distraction_peak_hour",
            "evidence":  f"Distraction sessions cluster around hour {peak_hour:02d}:00",
            "recommendation": (
                f"🎯 Block {peak_hour:02d}:00–{peak_hour+1:02d}:00 as a no-distraction "
                "focus window using browser extensions like LeechBlock or Cold Turkey."
            ),
        })
    return recs


def _check_anomalies(feat: pd.DataFrame) -> list[dict]:
    recs = []
    if "is_anomaly" not in feat.columns:
        return recs
    n_anomalies = feat["is_anomaly"].sum()
    if n_anomalies > 0:
        reasons = feat[feat["is_anomaly"]]["anomaly_reason"].value_counts().head(3)
        reason_str = "; ".join(f"{r} ({c}x)" for r, c in reasons.items())
        recs.append({
            "rule":      "anomalous_sessions_detected",
            "evidence":  f"{n_anomalies} unusual sessions detected. Top reasons: {reason_str}",
            "recommendation": (
                "🚨 Your browsing had unusual sessions (high switching, very late night, "
                "or abnormally long). Review these for productivity or wellbeing impact."
            ),
        })
    return recs


def _check_video_binge(feat: pd.DataFrame) -> list[dict]:
    recs = []
    long_video = feat[
        (feat["ratio_video"].fillna(0) > 0.5) &
        (feat["duration_sec"].fillna(0) > 3600)
    ]
    if len(long_video) >= 2:
        recs.append({
            "rule":      "video_binge_sessions",
            "evidence":  f"{len(long_video)} video sessions > 1 hour",
            "recommendation": (
                "📺 Multiple long video sessions detected. "
                "Use YouTube/Netflix time limits or schedule deliberate watch windows "
                "to prevent unplanned binge cycles."
            ),
        })
    return recs


# ── Main engine ───────────────────────────────────────────────

def generate_recommendations(
    session_features: pd.DataFrame,
    category_ram: pd.DataFrame | None = None,
) -> list[dict]:
    """
    Run all rule checks and return a list of recommendation dicts.
    Each dict has: rule, evidence, recommendation.
    """
    cfg  = CFG["recommendations"]
    recs = []

    recs += _check_social_overuse(session_features, cfg)
    recs += _check_heavy_ram(category_ram or pd.DataFrame(), cfg)
    recs += _check_focus_blocks(session_features)
    recs += _check_anomalies(session_features)
    recs += _check_video_binge(session_features)

    # Deduplicate by rule
    seen = set()
    unique_recs = []
    for r in recs:
        if r["rule"] not in seen:
            unique_recs.append(r)
            seen.add(r["rule"])

    if not unique_recs:
        unique_recs.append({
            "rule":      "no_issues",
            "evidence":  "All metrics within healthy thresholds",
            "recommendation": "✅ Your browsing patterns look healthy! Keep it up.",
        })

    print(f"[Recommender] {len(unique_recs)} recommendations generated")
    return unique_recs


if __name__ == "__main__":
    import sys
    feat = pd.read_csv(CFG["paths"]["session_features"])
    recs = generate_recommendations(feat)
    for i, r in enumerate(recs, 1):
        print(f"\n[{i}] {r['recommendation']}")
        print(f"    Evidence: {r['evidence']}")
