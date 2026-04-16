"""config_loader.py — load and expose config.yaml as a typed dict."""

import yaml
from pathlib import Path

_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "config.yaml"


def load_config(path: Path = _CONFIG_PATH) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# Module-level singleton — import CFG anywhere
CFG = load_config()
