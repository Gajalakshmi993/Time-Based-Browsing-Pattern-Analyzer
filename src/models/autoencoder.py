"""
models/autoencoder.py
──────────────────────
Module 6 — Autoencoder-based anomaly detection on session feature vectors.
Unusual sessions (late-night + high RAM + high switching) get high
reconstruction error and are flagged as anomalies.

Usage:
    python -m src.models.autoencoder
"""

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.preprocessing import StandardScaler

from src.config_loader import CFG

# ── Feature set (same as clustering + RAM if available) ───────
AE_FEATURES = [
    "duration_sec",
    "n_events",
    "n_unique_domains",
    "n_categories",
    "events_per_min",
    "category_switches",
    "ratio_social",
    "ratio_video",
    "ratio_news",
    "ratio_shopping",
    "ratio_learning",
    "ratio_work",
    "hour_start",
]


def build_autoencoder(input_dim: int, cfg: dict):
    """Build and compile a simple dense autoencoder."""
    try:
        import tensorflow as tf
        from tensorflow import keras
    except ImportError:
        raise ImportError("Install tensorflow: pip install tensorflow")

    layers     = cfg["hidden_layers"]  # e.g. [32, 16]
    enc_dim    = cfg["encoding_dim"]   # e.g. 8

    inp = keras.Input(shape=(input_dim,))
    x   = inp
    for units in layers:
        x = keras.layers.Dense(units, activation="relu")(x)
    encoded = keras.layers.Dense(enc_dim, activation="relu")(x)
    for units in reversed(layers):
        x = keras.layers.Dense(units, activation="relu")(encoded if x is encoded else x)
    decoded = keras.layers.Dense(input_dim, activation="linear")(x)

    ae = keras.Model(inp, decoded)
    ae.compile(optimizer="adam", loss="mse")
    return ae


def run_autoencoder(
    features_path: str | None = None,
    output_path:   str | None = None,
) -> pd.DataFrame:
    features_path = features_path or CFG["paths"]["session_features"]
    output_path   = output_path   or CFG["paths"]["anomaly_scores"]

    cfg = CFG["deep_learning"]["autoencoder"]
    feat = pd.read_csv(features_path)

    cols = [c for c in AE_FEATURES if c in feat.columns]
    X_raw = feat[cols].fillna(0).values

    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw).astype(np.float32)

    ae = build_autoencoder(X.shape[1], cfg)

    ae.fit(
        X, X,
        epochs=cfg["epochs"],
        batch_size=cfg["batch_size"],
        validation_split=0.1,
        verbose=0,
    )
    print(f"[Autoencoder] Training complete ({cfg['epochs']} epochs)")

    X_pred = ae.predict(X, verbose=0)
    recon_error = np.mean(np.square(X - X_pred), axis=1)

    threshold = np.percentile(recon_error, cfg["anomaly_percentile"])
    feat["reconstruction_error"] = recon_error
    feat["is_anomaly"]           = recon_error > threshold

    # Explain why flagged
    def _explain(row):
        reasons = []
        if row.get("hour_start", 12) < 6:
            reasons.append("late-night")
        if row.get("category_switches", 0) > 10:
            reasons.append("high-switching")
        if row.get("ratio_social", 0) > 0.7:
            reasons.append("social-heavy")
        if row.get("duration_sec", 0) > 7200:
            reasons.append("very-long")
        return ", ".join(reasons) if reasons else "unusual pattern"

    feat["anomaly_reason"] = feat.apply(_explain, axis=1)

    n_flagged = feat["is_anomaly"].sum()
    print(f"[Autoencoder] {n_flagged} anomalous sessions "
          f"(threshold={threshold:.4f})")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    feat.to_csv(output_path, index=False)
    print(f"[Autoencoder] Scores saved → {output_path}")
    return feat


if __name__ == "__main__":
    run_autoencoder()
