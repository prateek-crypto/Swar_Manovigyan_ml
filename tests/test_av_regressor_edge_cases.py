"""
Tests for AVLSTMRegressor edge cases: error handling, different shapes, callbacks.
"""

import pytest
import numpy as np
import os
import tempfile

pytestmark = pytest.mark.tensorflow

from src.models.av_regressor import AVLSTMRegressor


def test_predict_before_build_raises():
    """Calling predict without building should raise."""
    reg = AVLSTMRegressor(input_shape=(10, 11))
    with pytest.raises(ValueError, match="not built"):
        reg.predict(np.zeros((1, 10, 11)))


def test_save_before_build_raises():
    """Calling save without building should raise."""
    reg = AVLSTMRegressor(input_shape=(10, 11))
    with pytest.raises(ValueError, match="not built"):
        reg.save("/tmp/no_model.keras")


def test_evaluate_before_build_raises():
    """Calling evaluate without building should raise."""
    reg = AVLSTMRegressor(input_shape=(10, 11))
    with pytest.raises(ValueError, match="not built"):
        reg.evaluate(np.zeros((5, 10, 11)), np.zeros((5, 2)))


def test_fit_auto_builds():
    """fit() should auto-build if model is None."""
    reg = AVLSTMRegressor(input_shape=(5, 11))
    assert reg.model is None

    X = np.random.rand(10, 5, 11).astype(np.float32)
    y = np.random.rand(10, 2).astype(np.float32)
    reg.fit(X, y, X, y, epochs=1, batch_size=8, verbose=0)

    assert reg.model is not None


def test_output_bounded_zero_one():
    """Predictions from sigmoid output should always be in [0, 1]."""
    reg = AVLSTMRegressor(input_shape=(10, 11))
    reg.build()

    # Extreme inputs
    X = np.random.randn(20, 10, 11).astype(np.float32) * 100
    preds = reg.predict(X)

    assert np.all(preds >= 0.0), "Predictions should be >= 0"
    assert np.all(preds <= 1.0), "Predictions should be <= 1"


def test_different_sequence_lengths():
    """Model should work with various sequence lengths."""
    for seq_len in [3, 10, 50]:
        reg = AVLSTMRegressor(input_shape=(seq_len, 11))
        model = reg.build()
        assert model.input_shape == (None, seq_len, 11)

        X = np.random.rand(4, seq_len, 11).astype(np.float32)
        preds = reg.predict(X)
        assert preds.shape == (4, 2)


def test_custom_hyperparameters():
    """Test building with custom lstm_units, dropout, learning_rate."""
    reg = AVLSTMRegressor(
        input_shape=(10, 11),
        lstm_units=32,
        dropout_rate=0.5,
        learning_rate=0.01,
    )
    model = reg.build()
    assert model is not None
    assert model.output_shape == (None, 2)


def test_save_load_preserves_predictions():
    """Save and load should produce identical predictions."""
    reg = AVLSTMRegressor(input_shape=(5, 11))
    reg.build()

    X = np.random.rand(3, 5, 11).astype(np.float32)
    pred_before = reg.predict(X)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "model.keras")
        reg.save(path)

        reg2 = AVLSTMRegressor(input_shape=(5, 11))
        reg2.load(path)
        pred_after = reg2.predict(X)

    assert np.allclose(pred_before, pred_after, atol=1e-5)


def test_mel_input_shape():
    """Test with mel-spectrogram dimensions (T, 128)."""
    reg = AVLSTMRegressor(input_shape=(10, 128))
    reg.build()

    X = np.random.rand(4, 10, 128).astype(np.float32)
    preds = reg.predict(X)
    assert preds.shape == (4, 2)
    assert np.all(preds >= 0) and np.all(preds <= 1)
