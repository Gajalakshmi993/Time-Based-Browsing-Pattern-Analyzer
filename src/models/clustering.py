"""
models/clustering.py
─────────────────────
Module 5 — Cluster sessions using KMeans / GMM / DBSCAN and produce
human-readable cluster labels.

Usage:
    python -m src.models.clustering
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src.config_loader import CFG

# ── Feature columns used for clustering ──────────────────────
CLUSTER_FEATURES = [
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

# ── Cluster label heuristics ─────────────────────────────────

def _interpret_cluster(sub: pd.DataFrame) -> str:
    """
    Derive a human-readable label from the dominant signals
    in a cluster's rows.
    """
    top_cat    = sub["top_category"].mode().iloc[0] if len(sub) else "unknown"
    avg_hour   = sub["hour_start"].mean()
    avg_dur    = sub["duration_sec"].mean()
    avg_events = sub["n_events"].mean()

    parts = []

    if sub["ratio_social"].mean() > 0.5:
        parts.append("Social Media Loop")
    elif sub["ratio_video"].mean() > 0.4:
        parts.append("Video Binge")
    elif sub["ratio_learning"].mean() > 0.4:
        parts.append("Learning Session")
    elif sub["ratio_work"].mean() > 0.4:
        parts.append("Work / Productivity")
    elif sub["ratio_news"].mean() > 0.35:
        parts.append("News Reading")
    else:
        parts.append(f"Mixed ({top_cat})")

    if avg_hour < 6:
        parts.append("Late Night")
    elif avg_hour < 12:
        parts.append("Morning")
    elif avg_hour < 17:
        parts.append("Afternoon")
    else:
        parts.append("Evening")

    if avg_dur < 300:
        parts.append("Quick")
    elif avg_dur > 3600:
        parts.append("Long")

    return " · ".join(parts)


# ── Clustering algorithms ─────────────────────────────────────

def _fit_kmeans(X: np.ndarray, cfg: dict) -> np.ndarray:
    model = KMeans(
        n_clusters=cfg["n_clusters"],
        random_state=cfg["random_state"],
        n_init=10,
    )
    return model.fit_predict(X)


def _fit_gmm(X: np.ndarray, cfg: dict) -> np.ndarray:
    model = GaussianMixture(
        n_components=cfg["n_clusters"],
        random_state=cfg["random_state"],
    )
    return model.fit_predict(X)


def _fit_dbscan(X: np.ndarray, cfg: dict) -> np.ndarray:
    model = DBSCAN(
        eps=cfg["dbscan_eps"],
        min_samples=cfg["dbscan_min_samples"],
    )
    return model.fit_predict(X)


# ── Main pipeline ─────────────────────────────────────────────

def run_clustering(
    features_path: str | None = None,
    output_path:   str | None = None,
) -> pd.DataFrame:
    features_path = features_path or CFG["paths"]["session_features"]
    output_path   = output_path   or CFG["paths"]["cluster_labels"]

    cfg = CFG["clustering"]
    feat = pd.read_csv(features_path)

    available_cols = [c for c in CLUSTER_FEATURES if c in feat.columns]
    X_raw = feat[available_cols].fillna(0).values

    if cfg["scale_features"]:
        scaler = StandardScaler()
        X = scaler.fit_transform(X_raw)
    else:
        X = X_raw

    algo = cfg["algorithm"]
    if len(feat) < cfg.get("n_clusters", 4):
        raise ValueError(
            f"Only {len(feat)} sessions but n_clusters={cfg['n_clusters']}. "
            "Collect more data or reduce n_clusters in config.yaml."
        )

    if algo == "kmeans":
        labels = _fit_kmeans(X, cfg)
    elif algo == "gmm":
        labels = _fit_gmm(X, cfg)
    elif algo == "dbscan":
        labels = _fit_dbscan(X, cfg)
    else:
        raise ValueError(f"Unknown algorithm '{algo}' in config.yaml")

    feat["cluster"] = labels

    # Silhouette score (skip for DBSCAN noise label -1)
    valid = feat[feat["cluster"] >= 0]
    if len(valid["cluster"].unique()) > 1:
        score = silhouette_score(X[feat["cluster"] >= 0], valid["cluster"])
        print(f"[Clustering] Silhouette score ({algo}): {score:.3f}")

    # Human-readable labels
    label_map = {}
    for c in sorted(feat["cluster"].unique()):
        if c == -1:
            label_map[c] = "Noise (DBSCAN)"
            continue
        sub = feat[feat["cluster"] == c]
        label_map[c] = f"C{c}: {_interpret_cluster(sub)}"

    feat["cluster_label"] = feat["cluster"].map(label_map)

    print("[Clustering] Cluster summary:")
    for c, lbl in label_map.items():
        n = (feat["cluster"] == c).sum()
        print(f"  {lbl:50s}  n={n}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    feat.to_csv(output_path, index=False)
    print(f"[Clustering] Labels saved → {output_path}")
    return feat


if __name__ == "__main__":
    run_clustering()
