"""
Tests for sequence building (tabular and mel-spectrogram).
"""

import pytest
import numpy as np
import pandas as pd
from src.train_av import build_sequences_for_regression
from src.train_av_mel import build_frame_sequences_from_mel


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame with required features."""
    n_samples = 50
    np.random.seed(42)
    df = pd.DataFrame({
        'acousticness': np.random.uniform(0, 1, n_samples),
        'danceability': np.random.uniform(0, 1, n_samples),
        'energy': np.random.uniform(0, 1, n_samples),
        'instrumentalness': np.random.uniform(0, 1, n_samples),
        'liveness': np.random.uniform(0, 1, n_samples),
        'loudness': np.random.uniform(-60, 0, n_samples),
        'speechiness': np.random.uniform(0, 1, n_samples),
        'tempo': np.random.uniform(50, 200, n_samples),
        'valence': np.random.uniform(0, 1, n_samples),
        'arousal': np.random.uniform(0, 1, n_samples),
        'enhanced_valence': np.random.uniform(0, 1, n_samples),
    })
    return df


def test_build_sequences_for_regression(sample_dataframe):
    """Test tabular sequence building."""
    X, y, feature_stats = build_sequences_for_regression(
        sample_dataframe,
        sequence_length=10,
        stride=1,
    )
    
    assert X.shape[0] == y.shape[0], "X and y should have same number of sequences"
    assert X.shape[1] == 10, "Sequence length should be 10"
    assert X.shape[2] == 11, "Feature dimension should be 11"
    assert y.shape[1] == 2, "Labels should be [arousal, valence]"
    
    # Check feature stats
    assert 'mean' in feature_stats
    assert 'std' in feature_stats
    assert 'feature_columns' in feature_stats
    assert len(feature_stats['feature_columns']) == 11
    
    # Check labels are bounded
    assert np.all(y >= 0) and np.all(y <= 1), "Labels should be bounded to [0,1]"


def test_build_sequences_for_regression_stride(sample_dataframe):
    """Test sequence building with stride > 1."""
    X_stride1, _, _ = build_sequences_for_regression(
        sample_dataframe,
        sequence_length=10,
        stride=1,
    )
    X_stride2, _, _ = build_sequences_for_regression(
        sample_dataframe,
        sequence_length=10,
        stride=2,
    )
    
    assert len(X_stride2) < len(X_stride1), "Stride=2 should produce fewer sequences"
    assert len(X_stride2) <= len(X_stride1) // 2 + 1, "Approximate relationship"


def test_build_sequences_for_regression_limit(sample_dataframe):
    """Test sequence building with limit."""
    X, y, _ = build_sequences_for_regression(
        sample_dataframe,
        sequence_length=10,
        stride=1,
        limit_sequences=5,
    )
    
    assert len(X) == 5, "Should limit to 5 sequences"
    assert len(y) == 5


def test_build_frame_sequences_from_mel():
    """Test frame-level sequence building from mel-spectrogram."""
    # Create a synthetic mel-spectrogram (100 frames, 128 mel bands)
    mel = np.random.rand(100, 128).astype(np.float32)
    
    sequences = build_frame_sequences_from_mel(
        mel,
        sequence_length=10,
        stride=1,
    )
    
    assert sequences.shape[0] == 91, "Should have 100-10+1 = 91 sequences"
    assert sequences.shape[1] == 10, "Sequence length should be 10"
    assert sequences.shape[2] == 128, "Feature dimension should be 128"
    
    # Check first sequence matches first 10 frames
    assert np.allclose(sequences[0], mel[0:10]), "First sequence should match first 10 frames"
    assert np.allclose(sequences[1], mel[1:11]), "Second sequence should match frames 1-11"


def test_build_frame_sequences_from_mel_short():
    """Test frame-level sequence building with short audio (should pad)."""
    mel = np.random.rand(5, 128).astype(np.float32)  # Only 5 frames
    
    sequences = build_frame_sequences_from_mel(
        mel,
        sequence_length=10,
        stride=1,
    )
    
    assert sequences.shape[0] == 1, "Should pad and produce 1 sequence"
    assert sequences.shape[1] == 10, "Sequence length should be 10"
    assert np.allclose(sequences[0, :5], mel), "First 5 frames should match original"


def test_build_frame_sequences_from_mel_stride():
    """Test frame-level sequence building with stride > 1."""
    mel = np.random.rand(50, 128).astype(np.float32)
    
    sequences_stride1 = build_frame_sequences_from_mel(mel, sequence_length=10, stride=1)
    sequences_stride2 = build_frame_sequences_from_mel(mel, sequence_length=10, stride=2)
    
    assert len(sequences_stride2) < len(sequences_stride1)
    assert len(sequences_stride2) == 21, "Should have (50-10)/2 + 1 = 21 sequences"
