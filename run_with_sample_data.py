"""
run_with_sample_data.py
────────────────────────
End-to-end smoke test using synthetic data.
No real browser or RAM logger needed.

Usage:
    python run_with_sample_data.py
    python run_with_sample_data.py --dl lstm
"""

import argparse
import sys

from src.config_loader import CFG


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=5)
    parser.add_argument("--dl",   default="autoencoder",
                        choices=["autoencoder", "lstm", "none"])
    args = parser.parse_args()

    CFG["collection"]["time_window_days"] = args.days
    CFG["deep_learning"]["mode"]          = args.dl

    print("=" * 60)
    print("  Browsing Analyzer  —  Sample Data Run")
    print(f"  days={args.days}  dl={args.dl}")
    print("=" * 60)

    # Step 0: generate synthetic data
    print("\n[0/7] Generating synthetic browsing + RAM data...")
    from src.collect.generate_sample_data import generate
    generate(days=args.days)

    # Steps 1–7 via main pipeline (skip real collection)
    sys.argv = ["main.py", "--skip-collect"]
    from main import main as run_pipeline
    run_pipeline()


if __name__ == "__main__":
    main()
