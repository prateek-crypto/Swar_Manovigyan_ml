"""
Audio feature extraction utilities for A/V prediction.
Produces mel-spectrogram sequences suitable for LSTM input,
and tabular 11-D feature sequences approximating Spotify audio analysis.
"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import librosa

logger = logging.getLogger(__name__)

# ── Configurable constants for feature extraction ──────────────────────────
# Valence proxy: spectral centroid range (Hz).  Brighter → higher valence.
VALENCE_CENTROID_LOW: float = 500.0
VALENCE_CENTROID_HIGH: float = 4000.0

# Loudness (dBFS) and tempo (BPM) normalisation bounds – must match
# data_analysis.EmotionLabeler / training code.
LOUDNESS_MIN: float = -60.0
LOUDNESS_MAX: float = 0.0
TEMPO_MIN: float = 50.0
TEMPO_MAX: float = 200.0


# ── Mel-spectrogram extraction ─────────────────────────────────────────────

def extract_mel_spectrogram_sequence(
    audio_bytes: bytes,
    sr: int = 22050,
    n_mels: int = 128,
    hop_length: int = 512,
    n_fft: int = 2048,
    target_frames: int | None = 300,
) -> Tuple[np.ndarray, int, int]:
    """
    Convert an audio bytes buffer to a normalised mel-spectrogram sequence.

    Args:
        target_frames: If None, return full-length mel-spectrogram without
                       cropping/padding.  If int, crop/pad to this length.

    Returns:
        features: np.ndarray of shape (T, F) where T=timesteps, F=n_mels
        used_sr:  sample rate used
        num_frames: number of frames before cropping/padding
    """
    audio, _ = _load_audio_from_bytes(audio_bytes, target_sr=sr)

    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Normalise per-file to [0, 1]
    mel_min = mel_db.min()
    mel_max = mel_db.max()
    mel_norm = (mel_db - mel_min) / (mel_max - mel_min + 1e-8)

    # T × F (transpose)
    features = mel_norm.T  # (frames, n_mels)
    original_frames = features.shape[0]

    # Crop or pad to target_frames (if specified)
    if target_frames is not None:
        if features.shape[0] > target_frames:
            features = features[:target_frames, :]
        elif features.shape[0] < target_frames:
            pad_len = target_frames - features.shape[0]
            pad = np.zeros((pad_len, features.shape[1]), dtype=features.dtype)
            features = np.vstack([features, pad])

    return features.astype(np.float32), sr, original_frames


# ── Internal helpers ───────────────────────────────────────────────────────

def _load_audio_from_bytes(audio_bytes: bytes, target_sr: int) -> Tuple[np.ndarray, int]:
    """Load mono audio from bytes at target sample rate."""
    import soundfile as sf
    import io

    with io.BytesIO(audio_bytes) as buf:
        audio, file_sr = sf.read(buf)

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    if file_sr != target_sr:
        audio = librosa.resample(audio, orig_sr=file_sr, target_sr=target_sr)

    return audio.astype(np.float32), target_sr


# ── Tabular (Spotify-like) feature extraction ──────────────────────────────

def extract_tabular_features_sequence(
    audio_bytes: bytes,
    sr: int = 22050,
    sequence_length: int = 10,
) -> Tuple[np.ndarray, int]:
    """
    Build a (sequence_length, 11) feature sequence approximating the tabular
    Spotify-like features used in training, from raw audio.  The single
    feature vector is repeated over time to form a short sequence.

    Feature order matches train_av feature_columns:
    [acousticness, danceability, energy, instrumentalness, liveness,
     loudness, speechiness, tempo, valence, arousal, enhanced_valence]
    """
    audio, _ = _load_audio_from_bytes(audio_bytes, target_sr=sr)

    # ── Energy & loudness ──────────────────────────────────────────────
    rms = librosa.feature.rms(y=audio).flatten()
    rms_mean = float(np.mean(rms))
    energy = float(rms_mean / (np.max(rms) + 1e-8))

    eps = 1e-10
    loudness_db = 20.0 * np.log10(max(rms_mean, eps))
    loudness_db = float(np.clip(loudness_db, LOUDNESS_MIN, LOUDNESS_MAX))

    # ── Tempo ──────────────────────────────────────────────────────────
    tempo = _safe_extract_tempo(audio, sr)

    # ── Acousticness (inverse spectral flatness) ───────────────────────
    acousticness = _safe_extract(
        lambda: float(np.clip(
            1.0 - np.mean(librosa.feature.spectral_flatness(y=audio)),
            0.0, 1.0,
        )),
        name="acousticness",
    )

    # ── Danceability (onset strength regularity) ───────────────────────
    danceability = _extract_danceability(audio, sr)

    # ── Valence (spectral centroid brightness) ─────────────────────────
    valence = _safe_extract(
        lambda: float(np.clip(
            (np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr).flatten())
             - VALENCE_CENTROID_LOW)
            / (VALENCE_CENTROID_HIGH - VALENCE_CENTROID_LOW + 1e-8),
            0.0, 1.0,
        )),
        name="valence",
    )

    # ── Instrumentalness (vocal-band energy ratio) ─────────────────────
    instrumentalness = _extract_instrumentalness(audio, sr)

    # ── Liveness (high-frequency energy ratio as crowd-noise proxy) ────
    liveness = _extract_liveness(audio, sr)

    # ── Speechiness (zero-crossing rate proxy) ─────────────────────────
    speechiness = _extract_speechiness(audio, sr)

    # ── Derived: arousal & enhanced_valence ────────────────────────────
    loudness_norm = (loudness_db - LOUDNESS_MIN) / (LOUDNESS_MAX - LOUDNESS_MIN + 1e-8)
    tempo_norm = (tempo - TEMPO_MIN) / (TEMPO_MAX - TEMPO_MIN + 1e-8)
    arousal = 0.4 * energy + 0.3 * loudness_norm + 0.3 * tempo_norm
    enhanced_valence = 0.6 * valence + 0.3 * danceability + 0.1 * (1.0 - acousticness)

    vec = np.array([
        acousticness,
        danceability,
        energy,
        instrumentalness,
        liveness,
        loudness_db,
        speechiness,
        tempo,
        valence,
        arousal,
        enhanced_valence,
    ], dtype=np.float32)

    seq = np.tile(vec[None, :], (sequence_length, 1))  # (T=sequence_length, 11)
    return seq, sr


# ── Per-feature extraction helpers ─────────────────────────────────────────

def _safe_extract(fn, *, name: str, default: float = 0.5) -> float:
    """Run *fn* and return its result; on any failure, log and return *default*."""
    try:
        return fn()
    except Exception:
        logger.warning("Feature '%s' extraction failed; using default %.2f", name, default)
        return default


def _safe_extract_tempo(audio: np.ndarray, sr: int) -> float:
    """Extract tempo (BPM) using the best available librosa API."""
    try:
        # New location in librosa ≥0.10
        from librosa.feature.rhythm import tempo as _rhythm_tempo  # type: ignore
        tempo_arr = _rhythm_tempo(y=audio, sr=sr, hop_length=512)
        return float(tempo_arr.flatten()[0])
    except Exception:
        pass
    try:
        # Backwards-compatible fallback
        return float(librosa.beat.tempo(y=audio, sr=sr, hop_length=512).flatten()[0])
    except Exception:
        logger.warning("Tempo extraction failed; using default 120.0 BPM")
        return 120.0


def _extract_danceability(audio: np.ndarray, sr: int) -> float:
    """
    Danceability proxy: mean onset strength divided by peak onset strength.
    A higher ratio indicates more rhythmically regular (danceable) audio.
    """
    try:
        onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
        return float(np.clip(
            np.mean(onset_env) / (np.max(onset_env) + 1e-8),
            0.0, 1.0,
        ))
    except Exception:
        logger.warning("Danceability extraction failed; using default 0.5")
        return 0.5


def _extract_instrumentalness(audio: np.ndarray, sr: int) -> float:
    """
    Instrumentalness proxy: ratio of harmonic energy outside the typical
    vocal fundamental range (85–300 Hz) to total harmonic energy.
    High ratio → more instrumental.
    """
    try:
        harmonic = librosa.effects.harmonic(y=audio)
        S = np.abs(librosa.stft(harmonic))
        freqs = librosa.fft_frequencies(sr=sr)
        vocal_mask = (freqs >= 85) & (freqs <= 300)
        vocal_energy = float(S[vocal_mask, :].sum())
        total_energy = float(S.sum()) + 1e-10
        vocal_ratio = vocal_energy / total_energy
        # More vocal → lower instrumentalness
        return float(np.clip(1.0 - vocal_ratio * 3.0, 0.0, 1.0))
    except Exception:
        logger.warning("Instrumentalness extraction failed; using default 0.5")
        return 0.5


def _extract_liveness(audio: np.ndarray, sr: int) -> float:
    """
    Liveness proxy: ratio of high-frequency (>4 kHz) energy to total energy.
    Live recordings with audience noise tend to have relatively more
    high-frequency energy.
    """
    try:
        S = np.abs(librosa.stft(audio))
        freqs = librosa.fft_frequencies(sr=sr)
        high_mask = freqs > 4000
        high_energy = float(S[high_mask, :].sum())
        total_energy = float(S.sum()) + 1e-10
        return float(np.clip(high_energy / total_energy, 0.0, 1.0))
    except Exception:
        logger.warning("Liveness extraction failed; using default 0.1")
        return 0.1


def _extract_speechiness(audio: np.ndarray, sr: int) -> float:
    """
    Speechiness proxy: normalised zero-crossing rate.
    Spoken word / rap tends to have a higher ZCR than tonal music.
    """
    try:
        zcr = librosa.feature.zero_crossing_rate(y=audio).flatten()
        # Typical music ZCR is 0.02–0.10; speech often >0.10
        return float(np.clip(np.mean(zcr) / 0.20, 0.0, 1.0))
    except Exception:
        logger.warning("Speechiness extraction failed; using default 0.1")
        return 0.1
