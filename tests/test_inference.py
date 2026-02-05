"""
Tests for inference pipeline.
"""

import pytest
import numpy as np
import io
import os
import tempfile
import json

pytestmark = [pytest.mark.tensorflow, pytest.mark.audio]

import soundfile as sf
from src.inference_av import predict_from_audio_bytes
from src.models.av_regressor import AVLSTMRegressor
from src.utils.feature_stats import load_feature_stats, apply_zscore, save_feature_stats


@pytest.fixture
def sample_audio_bytes():
    """Generate a synthetic audio signal as bytes."""
    duration = 3.0
    sr = 22050
    t = np.linspace(0, duration, int(sr * duration))
    audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    
    buffer = io.BytesIO()
    sf.write(buffer, audio, sr, format='WAV')
    buffer.seek(0)
    return buffer.read()


@pytest.fixture
def trained_tabular_model(sample_audio_bytes):
    """Create a minimal trained model for tabular inference."""
    # Create dummy training data
    np.random.seed(42)
    X_train = np.random.randn(20, 10, 11).astype(np.float32)
    y_train = np.random.rand(20, 2).astype(np.float32)
    np.clip(y_train, 0, 1, out=y_train)
    
    X_val = np.random.randn(5, 10, 11).astype(np.float32)
    y_val = np.random.rand(5, 2).astype(np.float32)
    np.clip(y_val, 0, 1, out=y_val)
    
    reg = AVLSTMRegressor(input_shape=(10, 11))
    reg.build()
    reg.fit(X_train, y_train, X_val, y_val, epochs=1, batch_size=8, verbose=0)
    
    return reg


@pytest.fixture
def trained_mel_model():
    """Create a minimal trained model for mel-spectrogram inference."""
    np.random.seed(42)
    X_train = np.random.rand(20, 10, 128).astype(np.float32)
    y_train = np.random.rand(20, 2).astype(np.float32)
    np.clip(y_train, 0, 1, out=y_train)
    
    X_val = np.random.rand(5, 10, 128).astype(np.float32)
    y_val = np.random.rand(5, 2).astype(np.float32)
    np.clip(y_val, 0, 1, out=y_val)
    
    reg = AVLSTMRegressor(input_shape=(10, 128))
    reg.build()
    reg.fit(X_train, y_train, X_val, y_val, epochs=1, batch_size=8, verbose=0)
    
    return reg


def test_feature_stats_save_load():
    """Test feature stats save and load."""
    stats = {
        "feature_columns": ['feat1', 'feat2'],
        "mean": {'feat1': 0.5, 'feat2': 1.0},
        "std": {'feat1': 0.2, 'feat2': 0.3},
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        stats_path = os.path.join(tmpdir, "feature_stats_av.json")
        save_feature_stats(stats, stats_path)
        
        loaded = load_feature_stats(stats_path)
        assert loaded['feature_columns'] == stats['feature_columns']
        assert loaded['mean'] == stats['mean']
        assert loaded['std'] == stats['std']


def test_apply_zscore():
    """Test z-score normalization."""
    stats = {
        "feature_columns": ['feat1', 'feat2'],
        "mean": {'feat1': 0.5, 'feat2': 1.0},
        "std": {'feat1': 0.2, 'feat2': 0.3},
    }
    
    # Create sequence (1, 10, 2)
    seq = np.array([[[0.5, 1.0], [0.6, 1.1], [0.4, 0.9]] * 3 + [[0.5, 1.0]]], dtype=np.float32)
    
    normalized = apply_zscore(seq, stats)
    
    # First feature should be normalized: (0.5 - 0.5) / 0.2 = 0.0
    assert np.isclose(normalized[0, 0, 0], 0.0, atol=1e-5)


def test_predict_from_audio_bytes_tabular(sample_audio_bytes, trained_tabular_model):
    """Test inference with tabular model."""
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint = os.path.join(tmpdir, "tabular_model.keras")
        trained_tabular_model.save(checkpoint)
        
        # Create feature stats
        stats = {
            "feature_columns": ['acousticness', 'danceability', 'energy', 'instrumentalness',
                                'liveness', 'loudness', 'speechiness', 'tempo', 'valence',
                                'arousal', 'enhanced_valence'],
            "mean": {col: 0.0 for col in ['acousticness', 'danceability', 'energy', 'instrumentalness',
                                         'liveness', 'loudness', 'speechiness', 'tempo', 'valence',
                                         'arousal', 'enhanced_valence']},
            "std": {col: 1.0 for col in ['acousticness', 'danceability', 'energy', 'instrumentalness',
                                        'liveness', 'loudness', 'speechiness', 'tempo', 'valence',
                                        'arousal', 'enhanced_valence']},
        }
        stats_path = os.path.join(tmpdir, "feature_stats_av.json")
        save_feature_stats(stats, stats_path)
        
        arousal, valence = predict_from_audio_bytes(sample_audio_bytes, checkpoint)
        
        assert isinstance(arousal, float)
        assert isinstance(valence, float)
        assert 0 <= arousal <= 1
        assert 0 <= valence <= 1


def test_predict_from_audio_bytes_mel(sample_audio_bytes, trained_mel_model):
    """Test inference with mel-spectrogram model."""
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint = os.path.join(tmpdir, "mel_model.keras")
        trained_mel_model.save(checkpoint)
        
        arousal, valence = predict_from_audio_bytes(sample_audio_bytes, checkpoint)
        
        assert isinstance(arousal, float)
        assert isinstance(valence, float)
        assert 0 <= arousal <= 1
        assert 0 <= valence <= 1
