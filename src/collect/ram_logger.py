"""
collect/ram_logger.py
─────────────────────
Module 1 (part 2) — Log RAM metrics at regular intervals using psutil.

Usage:
    python -m src.collect.ram_logger           # runs until Ctrl+C
    python -m src.collect.ram_logger --minutes 60
"""

import argparse
import csv
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil

from src.config_loader import CFG

_STOP = False


def _handle_sigint(sig, frame):
    global _STOP
    print("\n[INFO] Stopping RAM logger...")
    _STOP = True


def _browser_ram_mb() -> float:
    """Sum RSS of all chrome / msedge processes in MB."""
    total = 0.0
    for proc in psutil.process_iter(["name", "memory_info"]):
        try:
            name = proc.info["name"].lower()
            if "chrome" in name or "msedge" in name:
                total += proc.info["memory_info"].rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return round(total / 1024 / 1024, 2)


def run_logger(
    output_path: str | None = None,
    interval_sec: int | None = None,
    duration_minutes: int | None = None,
) -> None:
    """
    Poll RAM metrics and append rows to CSV.

    Parameters
    ----------
    output_path : str
        Destination CSV (appends if exists).
    interval_sec : int
        Polling interval in seconds.
    duration_minutes : int | None
        Stop after N minutes; None = run until Ctrl+C.
    """
    output_path = output_path or CFG["paths"]["raw_ram"]
    interval_sec = interval_sec or CFG["collection"]["ram_poll_interval_sec"]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    file_exists = Path(output_path).exists()

    signal.signal(signal.SIGINT, _handle_sigint)

    deadline = (
        time.time() + duration_minutes * 60 if duration_minutes else float("inf")
    )

    with open(output_path, "a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "timestamp",
                "ram_used_mb",
                "ram_available_mb",
                "browser_ram_mb",
                "cpu_percent",
            ],
        )
        if not file_exists:
            writer.writeheader()

        print(f"[RAM Logger] Writing → {output_path}  (interval={interval_sec}s)")
        while not _STOP and time.time() < deadline:
            vm = psutil.virtual_memory()
            row = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "ram_used_mb": round(vm.used / 1024 / 1024, 2),
                "ram_available_mb": round(vm.available / 1024 / 1024, 2),
                "browser_ram_mb": _browser_ram_mb(),
                "cpu_percent": psutil.cpu_percent(interval=None),
            }
            writer.writerow(row)
            f.flush()
            time.sleep(interval_sec)

    print("[RAM Logger] Done.")


# ── CLI entry-point ──────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Log RAM usage periodically")
    parser.add_argument("--output", default=None)
    parser.add_argument("--interval", type=int, default=None)
    parser.add_argument("--minutes", type=int, default=None)
    args = parser.parse_args()
    run_logger(
        output_path=args.output,
        interval_sec=args.interval,
        duration_minutes=args.minutes,
    )
