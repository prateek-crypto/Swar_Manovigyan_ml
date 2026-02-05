"""
Load and apply feature statistics (mean/std) for z-score normalization at inference.
Must match training: train_av saves feature_stats_av.json next to the checkpoint.
"""

import json
import os
from typing import Any, Dict, Optional

import numpy as np


def get_stats_path_for_checkpoint(checkpoint_path: str) -> str:
    """Return path to feature_stats_av.json in the same directory as the checkpoint."""
    directory = os.path.dirname(checkpoint_path)
    return os.path.join(directory, "feature_stats_av.json") if directory else "feature_stats_av.json"


def load_feature_stats(checkpoint_path: str) -> Optional[Dict[str, Any]]:
    """
    Load feature mean/std from feature_stats_av.json next to the checkpoint.
    Returns None if file not found (e.g. model trained before stats were persisted).
    """
    stats_path = get_stats_path_for_checkpoint(checkpoint_path)
    if not os.path.isfile(stats_path):
        return None
    with open(stats_path, "r") as f:
        return json.load(f)


def save_feature_stats(stats: Dict[str, Any], stats_path: str) -> None:
    """Save feature stats to JSON file."""
    os.makedirs(os.path.dirname(stats_path) if os.path.dirname(stats_path) else ".", exist_ok=True)
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)


def apply_zscore(features: np.ndarray, stats: Dict[str, Any]) -> np.ndarray:
    """
    Apply z-score normalization to features using loaded stats.
    features: shape (T, F) or (N, T, F); F must match len(stats["feature_columns"]).
    stats: dict with "mean" and "std" keyed by feature column name.
    Returns normalized array of same shape, float32.
    """
    cols = stats["feature_columns"]
    mean_arr = np.array([stats["mean"][c] for c in cols], dtype=np.float32)
    std_arr = np.array([stats["std"][c] for c in cols], dtype=np.float32)
    std_arr = np.maximum(std_arr, 1e-8)
    return ((features - mean_arr) / std_arr).astype(np.float32)
