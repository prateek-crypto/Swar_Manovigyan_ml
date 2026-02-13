"""
Tests for feature_stats utilities (save, load, apply z-score).
"""

import pytest
import numpy as np
import os
import tempfile
import json

from src.utils.feature_stats import (
    get_stats_path_for_checkpoint,
    load_feature_stats,
    save_feature_stats,
    apply_zscore,
)


def test_get_stats_path_with_directory():
    path = get_stats_path_for_checkpoint("models/av_regressor.keras")
    assert path == os.path.join("models", "feature_stats_av.json")


def test_get_stats_path_no_directory():
    path = get_stats_path_for_checkpoint("av_regressor.keras")
    assert path == "feature_stats_av.json"


def test_load_feature_stats_missing_file():
    result = load_feature_stats("/nonexistent/path/model.keras")
    assert result is None


def test_save_and_load_roundtrip():
    stats = {
        "feature_columns": ["a", "b", "c"],
        "mean": {"a": 1.0, "b": 2.0, "c": 3.0},
        "std": {"a": 0.5, "b": 1.0, "c": 0.1},
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "stats.json")
        save_feature_stats(stats, path)
        assert os.path.isfile(path)

        loaded = load_feature_stats(os.path.join(tmpdir, "dummy_model.keras"))
        # load_feature_stats looks for feature_stats_av.json — rename
        stats_av_path = os.path.join(tmpdir, "feature_stats_av.json")
        os.rename(path, stats_av_path)
        loaded = load_feature_stats(os.path.join(tmpdir, "dummy_model.keras"))

        assert loaded is not None
        assert loaded["feature_columns"] == ["a", "b", "c"]
        assert loaded["mean"]["b"] == 2.0


def test_apply_zscore_2d():
    """Test z-score on 2D input (T, F)."""
    stats = {
        "feature_columns": ["f1", "f2"],
        "mean": {"f1": 10.0, "f2": 20.0},
        "std": {"f1": 2.0, "f2": 5.0},
    }
    features = np.array([[10.0, 20.0], [12.0, 25.0]], dtype=np.float32)
    result = apply_zscore(features, stats)

    assert result.shape == (2, 2)
    assert np.isclose(result[0, 0], 0.0, atol=1e-5)  # (10-10)/2 = 0
    assert np.isclose(result[0, 1], 0.0, atol=1e-5)  # (20-20)/5 = 0
    assert np.isclose(result[1, 0], 1.0, atol=1e-5)  # (12-10)/2 = 1
    assert np.isclose(result[1, 1], 1.0, atol=1e-5)  # (25-20)/5 = 1


def test_apply_zscore_3d():
    """Test z-score on 3D input (N, T, F)."""
    stats = {
        "feature_columns": ["f1", "f2", "f3"],
        "mean": {"f1": 0.0, "f2": 0.0, "f3": 0.0},
        "std": {"f1": 1.0, "f2": 2.0, "f3": 0.5},
    }
    features = np.ones((2, 3, 3), dtype=np.float32)
    result = apply_zscore(features, stats)

    assert result.shape == (2, 3, 3)
    assert np.isclose(result[0, 0, 0], 1.0, atol=1e-5)    # 1/1
    assert np.isclose(result[0, 0, 1], 0.5, atol=1e-5)    # 1/2
    assert np.isclose(result[0, 0, 2], 2.0, atol=1e-5)    # 1/0.5


def test_apply_zscore_zero_std():
    """std=0 should be clamped to 1e-8, not cause division by zero."""
    stats = {
        "feature_columns": ["f1"],
        "mean": {"f1": 5.0},
        "std": {"f1": 0.0},
    }
    features = np.array([[5.0], [6.0]], dtype=np.float32)
    result = apply_zscore(features, stats)

    assert np.all(np.isfinite(result)), "Zero std should not cause inf/nan"
