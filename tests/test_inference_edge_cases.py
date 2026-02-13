"""
Tests for inference edge cases: multi-sequence mel averaging, feature stats path resolution.
"""

import pytest
import numpy as np
import io
import os
import tempfile

pytestmark = [pytest.mark.tensorflow, pytest.mark.audio]

import soundfile as sf
from src.models.av_regressor import AVLSTMRegressor
from src.inference_av import predict_from_audio_bytes, _MAX_MEL_SEQUENCES_TO_AVG
from src.utils.feature_stats import save_feature_stats


@pytest.fixture
def audio_bytes_long():
    """5-second audio → many mel frames → multiple sequences."""
    sr = 22050
    t = np.linspace(0, 5.0, sr * 5)
    audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV")
    buf.seek(0)
    return buf.read()


@pytest.fixture
def audio_bytes_short():
    """0.5-second audio → fewer frames."""
    sr = 22050
    t = np.linspace(0, 0.5, int(sr * 0.5))
    audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV")
    buf.seek(0)
    return buf.read()


def _make_model_and_save(input_shape, tmpdir, name="model.keras"):
    """Helper: build, quick-train, save a model and return checkpoint path."""
    np.random.seed(42)
    T, F = input_shape
    reg = AVLSTMRegressor(input_shape=input_shape)
    reg.build()
    X = np.random.rand(10, T, F).astype(np.float32)
    y = np.random.rand(10, 2).astype(np.float32)
    reg.fit(X, y, X, y, epochs=1, batch_size=8, verbose=0)
    ckpt = os.path.join(tmpdir, name)
    reg.save(ckpt)
    return ckpt


def test_mel_inference_averages_multiple_sequences(audio_bytes_long):
    """Mel inference should average predictions over multiple sequences."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt = _make_model_and_save((10, 128), tmpdir)
        arousal, valence = predict_from_audio_bytes(audio_bytes_long, ckpt)

        assert 0.0 <= arousal <= 1.0
        assert 0.0 <= valence <= 1.0


def test_mel_inference_short_audio(audio_bytes_short):
    """Short audio should still produce a prediction (padding handles it)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt = _make_model_and_save((10, 128), tmpdir)
        arousal, valence = predict_from_audio_bytes(audio_bytes_short, ckpt)

        assert 0.0 <= arousal <= 1.0
        assert 0.0 <= valence <= 1.0


def test_tabular_inference_with_stats(audio_bytes_long):
    """Tabular path should apply z-score normalisation when stats exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt = _make_model_and_save((10, 11), tmpdir)

        # Save feature stats next to checkpoint
        cols = [
            "acousticness", "danceability", "energy", "instrumentalness",
            "liveness", "loudness", "speechiness", "tempo", "valence",
            "arousal", "enhanced_valence",
        ]
        stats = {
            "feature_columns": cols,
            "mean": {c: 0.0 for c in cols},
            "std": {c: 1.0 for c in cols},
        }
        save_feature_stats(stats, os.path.join(tmpdir, "feature_stats_av.json"))

        arousal, valence = predict_from_audio_bytes(audio_bytes_long, ckpt)
        assert 0.0 <= arousal <= 1.0
        assert 0.0 <= valence <= 1.0


def test_tabular_inference_without_stats(audio_bytes_long):
    """Tabular path should still work when no stats file exists (raw features)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt = _make_model_and_save((10, 11), tmpdir)
        # No feature_stats_av.json saved

        arousal, valence = predict_from_audio_bytes(audio_bytes_long, ckpt)
        assert 0.0 <= arousal <= 1.0
        assert 0.0 <= valence <= 1.0


def test_max_mel_sequences_constant():
    """Verify the constant is reasonable."""
    assert _MAX_MEL_SEQUENCES_TO_AVG >= 1
    assert _MAX_MEL_SEQUENCES_TO_AVG <= 100


def test_deterministic_predictions(audio_bytes_long):
    """Same audio + same model should give same predictions."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt = _make_model_and_save((10, 128), tmpdir)

        a1, v1 = predict_from_audio_bytes(audio_bytes_long, ckpt)
        a2, v2 = predict_from_audio_bytes(audio_bytes_long, ckpt)

        assert np.isclose(a1, a2, atol=1e-4), f"Arousal not deterministic: {a1} vs {a2}"
        assert np.isclose(v1, v2, atol=1e-4), f"Valence not deterministic: {v1} vs {v2}"
