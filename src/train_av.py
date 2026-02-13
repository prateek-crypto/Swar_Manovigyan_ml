"""
Train AV LSTM regressor on processed CSV with arousal and enhanced_valence targets.
Usage:
  python -m src.train_av --csv data/processed/spotify_features_with_emotions.csv --epochs 10 --checkpoint models/av_regressor.keras
"""

import argparse
import json
import logging
import os
import random
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any

from src.models.av_regressor import AVLSTMRegressor

logger = logging.getLogger(__name__)


def _set_global_seeds(seed: int = 42) -> None:
    """Set Python, NumPy, and TensorFlow global seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except Exception:
        pass

# Canonical feature column order (must match inference and app)
FEATURE_COLUMNS: List[str] = [
    'acousticness', 'danceability', 'energy', 'instrumentalness',
    'liveness', 'loudness', 'speechiness', 'tempo', 'valence',
    'arousal', 'enhanced_valence'
]


def build_sequences_for_regression(
    df: pd.DataFrame,
    sequence_length: int = 10,
    stride: int = 1,
    limit_sequences: int | None = None,
    feature_columns: List[str] | None = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Create sequences X (T,F) and y ([arousal, valence]) using existing feature columns.
    Returns (X, y, feature_stats) with feature_stats for inference z-score.
    """
    cols = feature_columns or FEATURE_COLUMNS

    # Prepare features (fillna + standardize)
    df_features = df[cols].fillna(df[cols].median())
    mean_ = df_features.mean().to_dict()
    std_ = (df_features.std() + 1e-8).to_dict()
    features_std = (df_features - df_features.mean()) / (df_features.std() + 1e-8)
    features_np = features_std.values.astype('float32')

    # Labels as numpy, precomputed (avoid DataFrame access in loop)
    labels_np = df[['arousal', 'enhanced_valence']].to_numpy(dtype='float32')

    last_start = len(features_np) - sequence_length
    if last_start < 0:
        raise ValueError(f"Not enough rows ({len(features_np)}) for sequence_length={sequence_length}")

    starts = np.arange(0, last_start + 1, stride)
    if limit_sequences is not None:
        starts = starts[:limit_sequences]

    X = np.empty((len(starts), sequence_length, features_np.shape[1]), dtype='float32')
    y = np.empty((len(starts), 2), dtype='float32')

    for idx, s in enumerate(starts):
        X[idx] = features_np[s:s+sequence_length]
        y[idx] = labels_np[s + sequence_length - 1]

    # Bound labels to [0,1]
    np.clip(y, 0.0, 1.0, out=y)

    feature_stats = {
        "feature_columns": cols,
        "mean": {k: float(v) for k, v in mean_.items()},
        "std": {k: float(v) for k, v in std_.items()},
    }
    return X, y, feature_stats


def split_train_val_test(X: np.ndarray, y: np.ndarray, val_ratio=0.2, test_ratio=0.2, seed=42):
    n = len(X)
    rng = np.random.default_rng(seed)
    indices = np.arange(n)
    rng.shuffle(indices)

    test_size = int(n * test_ratio)
    val_size = int(n * val_ratio)

    test_idx = indices[:test_size]
    val_idx = indices[test_size:test_size+val_size]
    train_idx = indices[test_size+val_size:]

    return (X[train_idx], y[train_idx]), (X[val_idx], y[val_idx]), (X[test_idx], y[test_idx])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, default='data/processed/spotify_features_with_emotions.csv')
    parser.add_argument('--sequence_length', type=int, default=10)
    parser.add_argument('--stride', type=int, default=1, help='Step between sequence starts (increase to reduce sequences)')
    parser.add_argument('--limit_sequences', type=int, default=None, help='Cap the number of sequences to build')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--checkpoint', type=str, default='models/av_regressor.keras')

    args = parser.parse_args()

    # Set global seeds for reproducibility
    _set_global_seeds(42)

    checkpoint_dir = os.path.dirname(args.checkpoint)
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)

    df = pd.read_csv(args.csv)

    # Auto-generate emotion labels if missing columns
    if 'arousal' not in df.columns or 'enhanced_valence' not in df.columns:
        logger.info("CSV missing arousal/enhanced_valence columns. Running EmotionLabeler automatically...")
        from src.utils.data_analysis import EmotionLabeler
        labeler = EmotionLabeler()
        df = labeler.create_emotion_labels(df)
        logger.info("Emotion labels created successfully.")

    X, y, feature_stats = build_sequences_for_regression(
        df,
        sequence_length=args.sequence_length,
        stride=args.stride,
        limit_sequences=args.limit_sequences,
    )
    print(f"Built sequences: X={X.shape}, y={y.shape}")

    # Persist feature stats for inference (z-score)
    stats_path = os.path.join(checkpoint_dir or ".", "feature_stats_av.json")
    with open(stats_path, "w") as f:
        json.dump(feature_stats, f, indent=2)
    print(f"Saved feature stats to {stats_path}")

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_train_val_test(X, y)

    reg = AVLSTMRegressor(input_shape=(X_train.shape[1], X_train.shape[2]), learning_rate=args.lr)
    reg.build()
    reg.fit(X_train, y_train, X_val, y_val, epochs=args.epochs, batch_size=args.batch_size)

    test_loss, test_mae = reg.evaluate(X_test, y_test)
    print(f"Test MSE: {test_loss:.4f} | Test MAE: {test_mae:.4f}")

    reg.save(args.checkpoint)
    print(f"Saved checkpoint to {args.checkpoint}")
    print("Inference: use feature_stats_av.json in the same directory as the checkpoint for z-score normalization.")


if __name__ == '__main__':
    main()
