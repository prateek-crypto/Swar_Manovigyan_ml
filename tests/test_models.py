"""
Tests for model architecture and inference.
"""

import pytest
import numpy as np
import os
import tempfile

pytestmark = pytest.mark.tensorflow

from src.models.av_regressor import AVLSTMRegressor


@pytest.fixture
def sample_tabular_sequences():
    """Generate sample tabular sequences."""
    np.random.seed(42)
    X = np.random.randn(32, 10, 11).astype(np.float32)
    y = np.random.rand(32, 2).astype(np.float32)
    np.clip(y, 0, 1, out=y)
    return X, y


@pytest.fixture
def sample_mel_sequences():
    """Generate sample mel-spectrogram sequences."""
    np.random.seed(42)
    X = np.random.rand(32, 10, 128).astype(np.float32)
    y = np.random.rand(32, 2).astype(np.float32)
    np.clip(y, 0, 1, out=y)
    return X, y


def test_av_regressor_build_tabular():
    """Test AV regressor builds correctly for tabular input."""
    reg = AVLSTMRegressor(input_shape=(10, 11))
    model = reg.build()
    
    assert model is not None
    assert model.input_shape == (None, 10, 11)
    assert model.output_shape == (None, 2)


def test_av_regressor_build_mel():
    """Test AV regressor builds correctly for mel-spectrogram input."""
    reg = AVLSTMRegressor(input_shape=(10, 128))
    model = reg.build()
    
    assert model is not None
    assert model.input_shape == (None, 10, 128)
    assert model.output_shape == (None, 2)


def test_av_regressor_predict_tabular(sample_tabular_sequences):
    """Test AV regressor prediction on tabular sequences."""
    X, y = sample_tabular_sequences
    reg = AVLSTMRegressor(input_shape=(10, 11))
    reg.build()
    
    # Quick training (1 epoch) to initialize weights
    X_train = X[:20]
    y_train = y[:20]
    X_val = X[20:26]
    y_val = y[20:26]
    
    reg.fit(X_train, y_train, X_val, y_val, epochs=1, batch_size=8, verbose=0)
    
    predictions = reg.predict(X[26:])
    assert predictions.shape == (6, 2), f"Expected (6, 2), got {predictions.shape}"
    assert np.all(predictions >= 0) and np.all(predictions <= 1), "Predictions should be [0,1]"


def test_av_regressor_predict_mel(sample_mel_sequences):
    """Test AV regressor prediction on mel-spectrogram sequences."""
    X, y = sample_mel_sequences
    reg = AVLSTMRegressor(input_shape=(10, 128))
    reg.build()
    
    X_train = X[:20]
    y_train = y[:20]
    X_val = X[20:26]
    y_val = y[20:26]
    
    reg.fit(X_train, y_train, X_val, y_val, epochs=1, batch_size=8, verbose=0)
    
    predictions = reg.predict(X[26:])
    assert predictions.shape == (6, 2)
    assert np.all(predictions >= 0) and np.all(predictions <= 1)


def test_av_regressor_save_load(sample_tabular_sequences):
    """Test model save and load."""
    X, y = sample_tabular_sequences
    reg = AVLSTMRegressor(input_shape=(10, 11))
    reg.build()
    
    X_train = X[:20]
    y_train = y[:20]
    X_val = X[20:26]
    y_val = y[20:26]
    
    reg.fit(X_train, y_train, X_val, y_val, epochs=1, batch_size=8, verbose=0)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint = os.path.join(tmpdir, "test_model.keras")
        reg.save(checkpoint)
        assert os.path.exists(checkpoint), "Checkpoint should be saved"
        
        # Load into new regressor
        reg2 = AVLSTMRegressor(input_shape=(10, 11))
        reg2.load(checkpoint)
        
        # Compare predictions
        pred1 = reg.predict(X[26:28])
        pred2 = reg2.predict(X[26:28])
        assert np.allclose(pred1, pred2, atol=1e-5), "Loaded model should produce same predictions"


def test_av_regressor_evaluate(sample_tabular_sequences):
    """Test model evaluation."""
    X, y = sample_tabular_sequences
    reg = AVLSTMRegressor(input_shape=(10, 11))
    reg.build()
    
    X_train = X[:20]
    y_train = y[:20]
    X_val = X[20:26]
    y_val = y[20:26]
    X_test = X[26:]
    y_test = y[26:]
    
    reg.fit(X_train, y_train, X_val, y_val, epochs=1, batch_size=8, verbose=0)
    
    loss, mae = reg.evaluate(X_test, y_test)
    assert isinstance(loss, float)
    assert isinstance(mae, float)
    assert loss >= 0
    assert mae >= 0
