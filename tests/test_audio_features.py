"""
Tests for audio feature extraction (tabular and mel-spectrogram).
"""

import pytest
import numpy as np
import io

pytestmark = pytest.mark.audio

import soundfile as sf
from src.utils.audio_features import (
    extract_mel_spectrogram_sequence,
    extract_tabular_features_sequence,
)


@pytest.fixture
def sample_audio_bytes():
    """Generate a synthetic audio signal as bytes."""
    duration = 3.0  # seconds
    sr = 22050
    t = np.linspace(0, duration, int(sr * duration))
    # Generate a simple sine wave with some harmonics
    audio = np.sin(2 * np.pi * 440 * t) + 0.5 * np.sin(2 * np.pi * 880 * t)
    audio = audio.astype(np.float32)
    
    # Convert to bytes via soundfile
    buffer = io.BytesIO()
    sf.write(buffer, audio, sr, format='WAV')
    buffer.seek(0)
    return buffer.read()


def test_extract_mel_spectrogram_sequence(sample_audio_bytes):
    """Test mel-spectrogram extraction returns correct shape."""
    mel, sr, num_frames = extract_mel_spectrogram_sequence(
        sample_audio_bytes,
        sr=22050,
        n_mels=128,
        target_frames=300,
    )
    
    assert mel.shape == (300, 128), f"Expected (300, 128), got {mel.shape}"
    assert sr == 22050
    assert mel.dtype == np.float32
    assert np.all(mel >= 0) and np.all(mel <= 1), "Mel-spectrogram should be normalized to [0,1]"


def test_extract_mel_spectrogram_sequence_short_audio():
    """Test mel-spectrogram extraction with short audio (should pad)."""
    duration = 0.5  # Very short
    sr = 22050
    t = np.linspace(0, duration, int(sr * duration))
    audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    
    buffer = io.BytesIO()
    sf.write(buffer, audio, sr, format='WAV')
    buffer.seek(0)
    audio_bytes = buffer.read()
    
    mel, _, _ = extract_mel_spectrogram_sequence(
        audio_bytes,
        target_frames=300,
    )
    
    assert mel.shape == (300, 128), "Short audio should be padded to target_frames"


def test_extract_tabular_features_sequence(sample_audio_bytes):
    """Test tabular feature extraction returns correct shape."""
    seq, sr = extract_tabular_features_sequence(
        sample_audio_bytes,
        sequence_length=10,
    )
    
    assert seq.shape == (10, 11), f"Expected (10, 11), got {seq.shape}"
    assert sr == 22050
    assert seq.dtype == np.float32
    
    # Check feature ranges
    assert np.all(seq[:, 0] >= 0) and np.all(seq[:, 0] <= 1), "acousticness should be [0,1]"
    assert np.all(seq[:, 2] >= 0) and np.all(seq[:, 2] <= 1), "energy should be [0,1]"
    assert np.all(seq[:, 5] >= -60) and np.all(seq[:, 5] <= 0), "loudness should be [-60, 0] dB"
    assert np.all(seq[:, 9] >= 0) and np.all(seq[:, 9] <= 1), "arousal should be [0,1]"
    assert np.all(seq[:, 10] >= 0) and np.all(seq[:, 10] <= 1), "enhanced_valence should be [0,1]"


def test_extract_tabular_features_sequence_consistency(sample_audio_bytes):
    """Test that tabular features are consistent across timesteps (tiled)."""
    seq, _ = extract_tabular_features_sequence(
        sample_audio_bytes,
        sequence_length=10,
    )
    
    # All timesteps should be identical (tiled)
    assert np.allclose(seq[0], seq[1]), "Tabular features should be tiled (same vector repeated)"
    assert np.allclose(seq[0], seq[-1]), "First and last timesteps should be identical"


def test_extract_mel_spectrogram_sequence_different_sr():
    """Test mel-spectrogram extraction with different sample rate."""
    duration = 2.0
    sr_orig = 44100
    t = np.linspace(0, duration, int(sr_orig * duration))
    audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    
    buffer = io.BytesIO()
    sf.write(buffer, audio, sr_orig, format='WAV')
    buffer.seek(0)
    audio_bytes = buffer.read()
    
    mel, sr_out, _ = extract_mel_spectrogram_sequence(
        audio_bytes,
        sr=22050,  # Target SR
    )
    
    assert sr_out == 22050, "Should resample to target SR"
    assert mel.shape[1] == 128, "Should have 128 mel bands"
