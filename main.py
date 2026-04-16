"""
main.py
────────
Orchestrates all 7 modules end-to-end.

Usage:
    python main.py                   # full run with defaults from config.yaml
    python main.py --days 3          # override time window
    python main.py --skip-collect    # skip extraction (use existing CSVs)
    python main.py --dl lstm         # use LSTM instead of autoencoder
"""

import argparse
import sys
from pathlib import Path

from src.config_loader import CFG


def parse_args():
    p = argparse.ArgumentParser(description="Browsing Pattern Analyzer Pipeline")
    p.add_argument("--days",         type=int, default=None,
                   help="Time window in days (3/4/5)")
    p.add_argument("--skip-collect", action="store_true",
                   help="Skip history extraction (use existing CSVs)")
    p.add_argument("--skip-ram",     action="store_true",
                   help="Skip RAM correlation (no ram_log.csv)")
    p.add_argument("--dl",           default=None, choices=["lstm", "autoencoder", "none"],
                   help="Deep learning mode override")
    p.add_argument("--browser",      default=None, choices=["chrome", "edge"])
    return p.parse_args()


def main():
    args = parse_args()

    # Override config values from CLI
    if args.days:
        CFG["collection"]["time_window_days"] = args.days
    if args.browser:
        CFG["collection"]["browser"] = args.browser
    if args.dl:
        CFG["deep_learning"]["mode"] = args.dl

    print("=" * 60)
    print("  Browsing Pattern Analyzer  —  Full Pipeline")
    print(f"  Window: {CFG['collection']['time_window_days']} days | "
          f"DL mode: {CFG['deep_learning']['mode']}")
    print("=" * 60)

    # ── Module 1: Collect ─────────────────────────────────────
    if not args.skip_collect:
        print("\n[1/7] Extracting browser history...")
        from src.collect.history_extractor import extract_history
        extract_history()
    else:
        print("\n[1/7] Skipping collection (--skip-collect)")

    # ── Module 2: Preprocess ──────────────────────────────────
    print("\n[2/7] Preprocessing...")
    from src.prep.preprocessor import preprocess
    preprocess()

    # ── Module 3: Sessionize ──────────────────────────────────
    print("\n[3/7] Sessionizing...")
    from src.prep.sessionizer import run_sessionization
    _, session_features = run_sessionization()

    # ── Module 4: RAM Correlation ─────────────────────────────
    category_ram = None
    ram_path = Path(CFG["paths"]["raw_ram"])
    if not args.skip_ram and ram_path.exists():
        print("\n[4/7] RAM correlation...")
        from src.analytics.ram_correlation import run_ram_analysis
        _, session_ram, category_ram = run_ram_analysis()
    else:
        print("\n[4/7] Skipping RAM correlation (no ram_log.csv found)")

    # ── Module 5: Clustering ──────────────────────────────────
    print("\n[5/7] Clustering sessions...")
    from src.models.clustering import run_clustering
    clustered = run_clustering()

    # ── Module 6: Deep Learning ───────────────────────────────
    dl_mode = CFG["deep_learning"]["mode"]
    print(f"\n[6/7] Deep learning ({dl_mode})...")
    if dl_mode == "autoencoder":
        from src.models.autoencoder import run_autoencoder
        feat_with_anomalies = run_autoencoder()
    elif dl_mode == "lstm":
        from src.models.lstm_predictor import run_lstm
        run_lstm()
        feat_with_anomalies = clustered
    else:
        print("  Skipped (--dl none)")
        feat_with_anomalies = clustered

    # ── Module 7: Recommendations + Report ───────────────────
    print("\n[7/7] Generating recommendations and report...")
    from src.recommend.engine import generate_recommendations
    from src.analytics.report_generator import generate_report

    recs = generate_recommendations(feat_with_anomalies, category_ram)
    generate_report(
        recommendations=recs,
        category_ram=category_ram,
    )

    print("\n" + "=" * 60)
    print("  Pipeline complete!")
    print(f"  Report → {CFG['paths']['report_out']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
