"""
Tests for mel-spectrogram extraction edge cases and full-length mode.
"""

import pytest
import numpy as np
import io

pytestmark = pytest.mark.audio

import soundfile as sf
from src.utils.audio_features import extract_mel_spectrogram_sequence


@pytest.fixture
def make_audio_bytes():
    """Factory fixture: create WAV bytes from duration + frequency."""
    def _make(duration: float = 2.0, sr: int = 22050, freq: float = 440.0):
        t = np.linspace(0, duration, int(sr * duration))
        audio = np.sin(2 * np.pi * freq * t).astype(np.float32)
        buf = io.BytesIO()
        sf.write(buf, audio, sr, format="WAV")
        buf.seek(0)
        return buf.read()
    return _make


def test_full_length_no_padding(make_audio_bytes):
    """target_frames=None should return the natural length (no crop/pad)."""
    audio_bytes = make_audio_bytes(duration=3.0, sr=22050)
    mel, sr_out, n_frames = extract_mel_spectrogram_sequence(
        audio_bytes, sr=22050, target_frames=None,
    )

    assert mel.shape[0] == n_frames, "Should equal natural frame count"
    assert mel.shape[1] == 128
    assert sr_out == 22050


def test_crop_to_shorter_target(make_audio_bytes):
    """Audio longer than target_frames should be cropped."""
    audio_bytes = make_audio_bytes(duration=5.0)
    mel, _, n_frames = extract_mel_spectrogram_sequence(
        audio_bytes, target_frames=100,
    )
    assert mel.shape[0] == 100
    assert n_frames > 100, "Original should be longer than 100 frames"


def test_pad_to_longer_target(make_audio_bytes):
    """Audio shorter than target_frames should be zero-padded."""
    audio_bytes = make_audio_bytes(duration=0.3)
    mel, _, n_frames = extract_mel_spectrogram_sequence(
        audio_bytes, target_frames=500,
    )
    assert mel.shape[0] == 500
    assert n_frames < 500
    # Tail should be zeros (padded region)
    assert np.allclose(mel[-10:, :], 0.0, atol=1e-6), "Padded region should be zeros"


def test_mel_normalised_zero_to_one(make_audio_bytes):
    """Output should be normalised to [0, 1]."""
    audio_bytes = make_audio_bytes(duration=2.0)
    mel, _, _ = extract_mel_spectrogram_sequence(audio_bytes, target_frames=None)

    assert mel.min() >= 0.0 - 1e-6
    assert mel.max() <= 1.0 + 1e-6


def test_mel_different_n_mels(make_audio_bytes):
    """n_mels parameter should change the feature dimension."""
    audio_bytes = make_audio_bytes(duration=1.0)
    mel, _, _ = extract_mel_spectrogram_sequence(audio_bytes, n_mels=64, target_frames=50)
    assert mel.shape == (50, 64)


def test_mel_stereo_input():
    """Stereo WAV should be handled (averaged to mono)."""
    sr = 22050
    t = np.linspace(0, 1.0, sr)
    stereo = np.stack([
        np.sin(2 * np.pi * 440 * t),
        np.sin(2 * np.pi * 660 * t),
    ], axis=-1).astype(np.float32)

    buf = io.BytesIO()
    sf.write(buf, stereo, sr, format="WAV")
    buf.seek(0)

    mel, _, _ = extract_mel_spectrogram_sequence(buf.read(), target_frames=50)
    assert mel.shape == (50, 128)
    assert np.all(np.isfinite(mel))


def test_mel_resample_from_higher_sr():
    """Audio at 44100 Hz should be resampled to 22050 Hz."""
    sr_orig = 44100
    sr_target = 22050
    t = np.linspace(0, 1.0, sr_orig)
    audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)

    buf = io.BytesIO()
    sf.write(buf, audio, sr_orig, format="WAV")
    buf.seek(0)

    mel, sr_out, _ = extract_mel_spectrogram_sequence(buf.read(), sr=sr_target, target_frames=50)
    assert sr_out == sr_target
    assert mel.shape == (50, 128)
