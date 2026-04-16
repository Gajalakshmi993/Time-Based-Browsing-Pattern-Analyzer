"""
models/lstm_predictor.py
─────────────────────────
Module 6 (Option B) — LSTM next-category prediction.
Input: sequence of category labels per session
Output: probability of next category / macro-F1

Usage:
    python -m src.models.lstm_predictor
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from src.config_loader import CFG


def _build_sequences(
    df: pd.DataFrame,
    seq_len: int,
    le: LabelEncoder,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build (X, y) windows from the category sequence.
    df must be sorted by timestamp and have 'category' column.
    """
    codes = le.transform(df["category"])
    X, y = [], []
    for i in range(len(codes) - seq_len):
        X.append(codes[i : i + seq_len])
        y.append(codes[i + seq_len])
    return np.array(X), np.array(y)


def run_lstm(
    sessions_path: str | None = None,
) -> dict:
    try:
        import tensorflow as tf
        from tensorflow import keras
    except ImportError:
        raise ImportError("Install tensorflow: pip install tensorflow")

    sessions_path = sessions_path or CFG["paths"]["sessions"]
    cfg           = CFG["deep_learning"]["lstm"]
    seq_len       = cfg["sequence_length"]

    df = pd.read_csv(sessions_path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").dropna(subset=["category"])

    le = LabelEncoder()
    le.fit(df["category"])
    n_classes = len(le.classes_)

    X, y = _build_sequences(df, seq_len, le)
    if len(X) == 0:
        raise ValueError("Not enough data to build LSTM sequences.")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = keras.Sequential([
        keras.layers.Embedding(n_classes, cfg["embedding_dim"], input_length=seq_len),
        keras.layers.LSTM(cfg["hidden_units"], return_sequences=False),
        keras.layers.Dropout(cfg["dropout"]),
        keras.layers.Dense(n_classes, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.fit(
        X_tr, y_tr,
        epochs=cfg["epochs"],
        batch_size=cfg["batch_size"],
        validation_split=0.1,
        verbose=0,
    )

    y_pred = model.predict(X_te, verbose=0).argmax(axis=1)
    report = classification_report(
        y_te, y_pred,
        target_names=le.classes_,
        output_dict=True,
        zero_division=0,
    )

    print("[LSTM] Classification report:")
    print(classification_report(
        y_te, y_pred, target_names=le.classes_, zero_division=0
    ))

    # Baseline: always predict most common class
    baseline_pred = np.full_like(y_te, np.bincount(y_tr).argmax())
    baseline_report = classification_report(
        y_te, baseline_pred,
        target_names=le.classes_,
        output_dict=True,
        zero_division=0,
    )
    print(f"[LSTM] Model macro-F1:    {report['macro avg']['f1-score']:.3f}")
    print(f"[LSTM] Baseline macro-F1: {baseline_report['macro avg']['f1-score']:.3f}")

    return {"model": model, "label_encoder": le, "report": report}


if __name__ == "__main__":
    run_lstm()
