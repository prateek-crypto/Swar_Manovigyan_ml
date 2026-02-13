"""
Tests for DataPreprocessor (sequence creation and balancing) and
precomputed mel-spectrogram loading.
"""

import pytest
import numpy as np
import pandas as pd
import os
import tempfile

from src.utils.data_analysis import DataPreprocessor, EmotionLabeler


# ── DataPreprocessor ───────────────────────────────────────────────────────

@pytest.fixture
def labeled_df():
    """DataFrame that already has emotion labels (simulates post-EmotionLabeler)."""
    np.random.seed(42)
    n = 80
    df = pd.DataFrame({
        "acousticness": np.random.uniform(0, 1, n),
        "danceability": np.random.uniform(0, 1, n),
        "energy": np.random.uniform(0, 1, n),
        "instrumentalness": np.random.uniform(0, 1, n),
        "liveness": np.random.uniform(0, 1, n),
        "loudness": np.random.uniform(-60, 0, n),
        "speechiness": np.random.uniform(0, 1, n),
        "tempo": np.random.uniform(50, 200, n),
        "valence": np.random.uniform(0, 1, n),
    })
    labeler = EmotionLabeler()
    df = labeler.create_emotion_labels(df)
    return df


def test_prepare_features_shapes(labeled_df):
    """Check X, y shapes match expected dimensions."""
    pp = DataPreprocessor()
    X, y = pp.prepare_features(labeled_df, sequence_length=5)

    expected_n = len(labeled_df) - 5 + 1
    assert X.shape == (expected_n, 5, 11)
    assert y.shape == (expected_n,)


def test_prepare_features_scaled(labeled_df):
    """After fit_transform, features should be approximately mean-0 std-1."""
    pp = DataPreprocessor()
    X, _ = pp.prepare_features(labeled_df, sequence_length=5)

    flat = X.reshape(-1, 11)
    means = flat.mean(axis=0)
    stds = flat.std(axis=0)

    # Means should be close to 0, stds close to 1 (within tolerance for finite data)
    assert np.all(np.abs(means) < 0.5), f"Means not near 0: {means}"
    assert np.all(stds > 0.1), f"Stds unexpectedly low: {stds}"


def test_prepare_features_labels_in_range(labeled_df):
    """Labels should be 0-3."""
    pp = DataPreprocessor()
    _, y = pp.prepare_features(labeled_df, sequence_length=5)

    assert np.all(y >= 0) and np.all(y <= 3)
    assert len(np.unique(y)) >= 2, "Should have at least 2 classes"


def test_create_balanced_dataset(labeled_df):
    """Balanced dataset should cap each class."""
    pp = DataPreprocessor()
    X, y = pp.prepare_features(labeled_df, sequence_length=5)

    np.random.seed(42)
    X_bal, y_bal = pp.create_balanced_dataset(X, y, max_samples_per_class=5)

    unique, counts = np.unique(y_bal, return_counts=True)
    assert np.all(counts <= 5), f"Each class should have at most 5 samples: {counts}"
    assert len(X_bal) == len(y_bal)


def test_create_balanced_dataset_preserves_shape(labeled_df):
    """Balanced dataset X should keep (N, T, F) shape."""
    pp = DataPreprocessor()
    X, y = pp.prepare_features(labeled_df, sequence_length=5)

    np.random.seed(42)
    X_bal, _ = pp.create_balanced_dataset(X, y, max_samples_per_class=10)

    assert X_bal.ndim == 3
    assert X_bal.shape[1] == 5
    assert X_bal.shape[2] == 11


def test_prepare_features_handles_missing_values():
    """NaN values in features should be filled (median)."""
    np.random.seed(42)
    n = 30
    df = pd.DataFrame({
        "acousticness": np.random.uniform(0, 1, n),
        "danceability": np.random.uniform(0, 1, n),
        "energy": np.random.uniform(0, 1, n),
        "instrumentalness": np.random.uniform(0, 1, n),
        "liveness": np.random.uniform(0, 1, n),
        "loudness": np.random.uniform(-60, 0, n),
        "speechiness": np.random.uniform(0, 1, n),
        "tempo": np.random.uniform(50, 200, n),
        "valence": np.random.uniform(0, 1, n),
    })
    labeler = EmotionLabeler()
    df = labeler.create_emotion_labels(df)

    # Inject NaNs
    df.loc[0, "acousticness"] = np.nan
    df.loc[5, "energy"] = np.nan
    df.loc[10, "tempo"] = np.nan

    pp = DataPreprocessor()
    X, y = pp.prepare_features(df, sequence_length=5)

    assert np.all(np.isfinite(X)), "No NaN should remain after preprocessing"


# ── Precomputed mel loading ────────────────────────────────────────────────

def test_load_precomputed_mel_and_labels():
    """Test loading .npy mel-spectrograms and matching with CSV labels."""
    from src.train_av_mel import load_precomputed_mel_and_labels

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create fake .npy mel files
        for i in range(4):
            mel = np.random.rand(50 + i * 10, 128).astype(np.float32)
            np.save(os.path.join(tmpdir, f"track_{i}.npy"), mel)

        # Create CSV with matching labels
        csv_path = os.path.join(tmpdir, "labels.csv")
        df = pd.DataFrame({
            "filename": [f"track_{i}" for i in range(4)],
            "arousal": [0.2, 0.5, 0.8, 0.3],
            "enhanced_valence": [0.6, 0.4, 0.7, 0.9],
        })
        df.to_csv(csv_path, index=False)

        mels, labels = load_precomputed_mel_and_labels(tmpdir, csv_path)

        assert len(mels) == 4
        assert labels.shape == (4, 2)
        assert np.all(labels >= 0) and np.all(labels <= 1)
        # Verify shapes match
        for i, m in enumerate(mels):
            assert m.shape[1] == 128
            assert m.shape[0] == 50 + i * 10


def test_load_precomputed_mel_skips_unmatched():
    """Files without CSV labels should be skipped."""
    from src.train_av_mel import load_precomputed_mel_and_labels

    with tempfile.TemporaryDirectory() as tmpdir:
        for name in ["matched", "unmatched1", "unmatched2"]:
            np.save(os.path.join(tmpdir, f"{name}.npy"), np.random.rand(30, 128).astype(np.float32))

        csv_path = os.path.join(tmpdir, "labels.csv")
        pd.DataFrame({
            "filename": ["matched"],
            "arousal": [0.5],
            "enhanced_valence": [0.5],
        }).to_csv(csv_path, index=False)

        mels, labels = load_precomputed_mel_and_labels(tmpdir, csv_path)
        assert len(mels) == 1
        assert labels.shape == (1, 2)


def test_load_precomputed_mel_missing_csv_columns():
    """Should raise ValueError when CSV is missing required columns."""
    from src.train_av_mel import load_precomputed_mel_and_labels

    with tempfile.TemporaryDirectory() as tmpdir:
        np.save(os.path.join(tmpdir, "t.npy"), np.random.rand(20, 128).astype(np.float32))
        csv_path = os.path.join(tmpdir, "bad.csv")
        pd.DataFrame({"filename": ["t"], "arousal": [0.5]}).to_csv(csv_path, index=False)

        with pytest.raises(ValueError, match="missing required columns"):
            load_precomputed_mel_and_labels(tmpdir, csv_path)


def test_load_precomputed_mel_no_npy_files():
    """Should raise ValueError when directory has no .npy files."""
    from src.train_av_mel import load_precomputed_mel_and_labels

    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "labels.csv")
        pd.DataFrame({
            "filename": ["x"],
            "arousal": [0.5],
            "enhanced_valence": [0.5],
        }).to_csv(csv_path, index=False)

        with pytest.raises(ValueError, match="No .npy files"):
            load_precomputed_mel_and_labels(tmpdir, csv_path)


# ── build_sequences_from_mel_files ─────────────────────────────────────────

def test_build_sequences_from_mel_files():
    """Test end-to-end: mel files → sequences + labels."""
    from src.train_av_mel import build_sequences_from_mel_files

    mels = [
        np.random.rand(30, 128).astype(np.float32),
        np.random.rand(50, 128).astype(np.float32),
    ]
    labels = np.array([[0.3, 0.7], [0.8, 0.2]], dtype=np.float32)

    X, y = build_sequences_from_mel_files(mels, labels, sequence_length=10, stride=5)

    assert X.ndim == 3
    assert X.shape[1] == 10
    assert X.shape[2] == 128
    assert y.shape[1] == 2
    assert len(X) == len(y)
    # First file: (30-10)//5 + 1 = 5 seqs; Second: (50-10)//5 + 1 = 9 seqs → 14 total
    assert len(X) == 5 + 9


# ── EmotionLabeler edge cases ─────────────────────────────────────────────

def test_emotion_labeler_formula_consistency():
    """Verify arousal formula: 0.4*energy + 0.3*loudness_norm + 0.3*tempo_norm."""
    labeler = EmotionLabeler()
    df = pd.DataFrame({
        "acousticness": [0.0],
        "danceability": [0.0],
        "energy": [1.0],
        "instrumentalness": [0.0],
        "liveness": [0.0],
        "loudness": [0.0],       # max loudness → loudness_norm = 1.0
        "speechiness": [0.0],
        "tempo": [200.0],        # max tempo → tempo_norm = 1.0
        "valence": [0.0],
    })
    df = labeler.create_emotion_labels(df)

    # arousal = 0.4*1.0 + 0.3*1.0 + 0.3*1.0 = 1.0
    assert np.isclose(df["arousal"].iloc[0], 1.0, atol=0.01)

    # enhanced_valence = 0.6*0 + 0.3*0 + 0.1*(1-0) = 0.1
    assert np.isclose(df["enhanced_valence"].iloc[0], 0.1, atol=0.01)
