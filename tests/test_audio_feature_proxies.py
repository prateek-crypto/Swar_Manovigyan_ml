"""
Tests for the new audio feature proxy functions (instrumentalness, liveness, speechiness)
and configurable constants.
"""

import pytest
import numpy as np
import io

pytestmark = pytest.mark.audio

import soundfile as sf
from src.utils.audio_features import (
    _extract_danceability,
    _extract_instrumentalness,
    _extract_liveness,
    _extract_speechiness,
    _safe_extract,
    _safe_extract_tempo,
    VALENCE_CENTROID_LOW,
    VALENCE_CENTROID_HIGH,
    LOUDNESS_MIN,
    LOUDNESS_MAX,
    TEMPO_MIN,
    TEMPO_MAX,
    extract_tabular_features_sequence,
)


@pytest.fixture
def sine_audio():
    """Pure 440 Hz sine wave at 22050 Hz, 2 seconds."""
    sr = 22050
    t = np.linspace(0, 2.0, sr * 2)
    audio = (np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    return audio, sr


@pytest.fixture
def noise_audio():
    """White noise at 22050 Hz, 2 seconds."""
    sr = 22050
    np.random.seed(99)
    audio = np.random.randn(sr * 2).astype(np.float32) * 0.3
    return audio, sr


@pytest.fixture
def speech_like_audio():
    """High-ZCR signal simulating speech-like content."""
    sr = 22050
    t = np.linspace(0, 2.0, sr * 2)
    # Rapidly varying signal → high zero-crossing rate
    audio = (np.sign(np.sin(2 * np.pi * 2000 * t + np.random.randn(len(t)) * 0.5))).astype(np.float32) * 0.3
    return audio, sr


# ── Constants ──────────────────────────────────────────────────────────────

def test_constants_are_correct():
    """Verify module-level constants match expected values."""
    assert LOUDNESS_MIN == -60.0
    assert LOUDNESS_MAX == 0.0
    assert TEMPO_MIN == 50.0
    assert TEMPO_MAX == 200.0
    assert VALENCE_CENTROID_LOW == 500.0
    assert VALENCE_CENTROID_HIGH == 4000.0


# ── _safe_extract ──────────────────────────────────────────────────────────

def test_safe_extract_returns_value_on_success():
    result = _safe_extract(lambda: 0.75, name="test")
    assert result == 0.75


def test_safe_extract_returns_default_on_failure():
    result = _safe_extract(lambda: 1 / 0, name="broken", default=0.42)
    assert result == 0.42


def test_safe_extract_default_is_half():
    result = _safe_extract(lambda: [][0], name="missing")
    assert result == 0.5


# ── Tempo extraction ──────────────────────────────────────────────────────

def test_safe_extract_tempo_returns_float(sine_audio):
    audio, sr = sine_audio
    tempo = _safe_extract_tempo(audio, sr)
    assert isinstance(tempo, float)
    assert tempo > 0


# ── Danceability ─────────────────────────────────────────────────────────

def test_extract_danceability_range(sine_audio):
    audio, sr = sine_audio
    val = _extract_danceability(audio, sr)
    assert 0.0 <= val <= 1.0, f"danceability={val} out of [0,1]"


def test_extract_danceability_single_computation(sine_audio):
    """Calling twice with identical input should return identical results
    (verifies onset_strength is computed once, not twice with drift)."""
    audio, sr = sine_audio
    val1 = _extract_danceability(audio, sr)
    val2 = _extract_danceability(audio, sr)
    assert val1 == val2, f"Danceability not deterministic: {val1} vs {val2}"


# ── Instrumentalness ─────────────────────────────────────────────────────

def test_extract_instrumentalness_range(sine_audio):
    audio, sr = sine_audio
    val = _extract_instrumentalness(audio, sr)
    assert 0.0 <= val <= 1.0, f"instrumentalness={val} out of [0,1]"


def test_extract_instrumentalness_pure_tone_is_high(sine_audio):
    """A pure sine tone (no vocals) should have relatively high instrumentalness."""
    audio, sr = sine_audio
    val = _extract_instrumentalness(audio, sr)
    assert val >= 0.3, f"Expected >=0.3 for pure tone, got {val}"


# ── Liveness ─────────────────────────────────────────────────────────────

def test_extract_liveness_range(sine_audio):
    audio, sr = sine_audio
    val = _extract_liveness(audio, sr)
    assert 0.0 <= val <= 1.0, f"liveness={val} out of [0,1]"


def test_extract_liveness_noise_higher_than_sine(sine_audio, noise_audio):
    """White noise has more high-freq energy → higher liveness than a sine."""
    val_sine = _extract_liveness(*sine_audio)
    val_noise = _extract_liveness(*noise_audio)
    assert val_noise > val_sine, (
        f"Noise liveness ({val_noise}) should exceed sine ({val_sine})"
    )


# ── Speechiness ──────────────────────────────────────────────────────────

def test_extract_speechiness_range(sine_audio):
    audio, sr = sine_audio
    val = _extract_speechiness(audio, sr)
    assert 0.0 <= val <= 1.0, f"speechiness={val} out of [0,1]"


def test_extract_speechiness_pure_tone_is_low(sine_audio):
    """A smooth sine wave should have low ZCR → low speechiness."""
    audio, sr = sine_audio
    val = _extract_speechiness(audio, sr)
    assert val < 0.3, f"Expected <0.3 for pure sine, got {val}"


# ── Full tabular feature vector ──────────────────────────────────────────

def test_tabular_feature_vector_all_eleven_finite():
    """Every element in the 11-D feature vector should be finite."""
    sr = 22050
    t = np.linspace(0, 2.0, sr * 2)
    audio = (np.sin(2 * np.pi * 440 * t) + 0.3 * np.sin(2 * np.pi * 880 * t)).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV")
    buf.seek(0)
    audio_bytes = buf.read()

    seq, _ = extract_tabular_features_sequence(audio_bytes, sequence_length=5)
    assert seq.shape == (5, 11)
    assert np.all(np.isfinite(seq)), "All feature values should be finite"


def test_tabular_feature_order():
    """
    Verify the feature order matches the canonical column order:
    [acousticness, danceability, energy, instrumentalness, liveness,
     loudness, speechiness, tempo, valence, arousal, enhanced_valence]
    """
    sr = 22050
    t = np.linspace(0, 2.0, sr * 2)
    audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV")
    buf.seek(0)

    seq, _ = extract_tabular_features_sequence(buf.read(), sequence_length=1)
    vec = seq[0]

    # idx 0: acousticness [0,1]
    assert 0.0 <= vec[0] <= 1.0, f"acousticness={vec[0]}"
    # idx 1: danceability [0,1]
    assert 0.0 <= vec[1] <= 1.0, f"danceability={vec[1]}"
    # idx 2: energy [0,1]
    assert 0.0 <= vec[2] <= 1.0, f"energy={vec[2]}"
    # idx 3: instrumentalness [0,1]
    assert 0.0 <= vec[3] <= 1.0, f"instrumentalness={vec[3]}"
    # idx 4: liveness [0,1]
    assert 0.0 <= vec[4] <= 1.0, f"liveness={vec[4]}"
    # idx 5: loudness [-60, 0]
    assert -60.0 <= vec[5] <= 0.0, f"loudness={vec[5]}"
    # idx 6: speechiness [0,1]
    assert 0.0 <= vec[6] <= 1.0, f"speechiness={vec[6]}"
    # idx 7: tempo [positive BPM]
    assert vec[7] > 0, f"tempo={vec[7]}"
    # idx 8: valence [0,1]
    assert 0.0 <= vec[8] <= 1.0, f"valence={vec[8]}"
    # idx 9: arousal [computed, roughly 0-1]
    assert np.isfinite(vec[9]), f"arousal={vec[9]}"
    # idx 10: enhanced_valence [computed, roughly 0-1]
    assert np.isfinite(vec[10]), f"enhanced_valence={vec[10]}"


def test_tabular_stereo_audio_handled():
    """Stereo audio should be averaged to mono without error."""
    sr = 22050
    t = np.linspace(0, 1.0, sr)
    left = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    right = np.sin(2 * np.pi * 660 * t).astype(np.float32)
    stereo = np.stack([left, right], axis=-1)

    buf = io.BytesIO()
    sf.write(buf, stereo, sr, format="WAV")
    buf.seek(0)

    seq, _ = extract_tabular_features_sequence(buf.read(), sequence_length=3)
    assert seq.shape == (3, 11)
    assert np.all(np.isfinite(seq))
