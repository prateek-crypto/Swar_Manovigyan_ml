"""
Audio feature extraction utilities for A/V prediction.
Produces mel-spectrogram sequences suitable for LSTM input.
"""

from typing import Tuple
import numpy as np
import librosa


def extract_mel_spectrogram_sequence(
    audio_bytes: bytes,
    sr: int = 22050,
    n_mels: int = 128,
    hop_length: int = 512,
    n_fft: int = 2048,
    target_frames: int = 300,
) -> Tuple[np.ndarray, int, int]:
    """
    Convert an audio bytes buffer to a normalized mel-spectrogram sequence.

    Returns
        features: np.ndarray of shape (T, F) where T=timesteps, F=n_mels
        used_sr: sample rate used
        num_frames: number of frames before cropping/padding
    """
    # Load from bytes
    import soundfile as sf
    import io

    with io.BytesIO(audio_bytes) as buf:
        audio, file_sr = sf.read(buf)

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    if file_sr != sr:
        audio = librosa.resample(audio, orig_sr=file_sr, target_sr=sr)

    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Normalize per-file to 0-1
    mel_min = mel_db.min()
    mel_max = mel_db.max()
    mel_norm = (mel_db - mel_min) / (mel_max - mel_min + 1e-8)

    # T x F (transpose)
    features = mel_norm.T  # (frames, n_mels)

    # Crop or pad to target_frames
    if features.shape[0] > target_frames:
        features = features[:target_frames, :]
    elif features.shape[0] < target_frames:
        pad_len = target_frames - features.shape[0]
        pad = np.zeros((pad_len, features.shape[1]), dtype=features.dtype)
        features = np.vstack([features, pad])

    return features.astype(np.float32), sr, mel_norm.shape[1]


def _load_audio_from_bytes(audio_bytes: bytes, target_sr: int) -> Tuple[np.ndarray, int]:
    """
    Internal: load mono audio from bytes at target sample rate.
    """
    import soundfile as sf
    import io

    with io.BytesIO(audio_bytes) as buf:
        audio, file_sr = sf.read(buf)

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    if file_sr != target_sr:
        audio = librosa.resample(audio, orig_sr=file_sr, target_sr=target_sr)

    return audio.astype(np.float32), target_sr


def extract_tabular_features_sequence(
    audio_bytes: bytes,
    sr: int = 22050,
    sequence_length: int = 10,
) -> Tuple[np.ndarray, int]:
    """
    Build a (sequence_length, 11) feature sequence approximating the tabular
    Spotify-like features used in training, from raw audio. The single
    feature vector is repeated over time to form a short sequence.

    Feature order matches train_av feature_columns:
    [acousticness, danceability, energy, instrumentalness, liveness,
     loudness, speechiness, tempo, valence, arousal, enhanced_valence]
    """
    audio, _ = _load_audio_from_bytes(audio_bytes, target_sr=sr)

    # Energy and loudness proxies
    rms = librosa.feature.rms(y=audio).flatten()
    rms_mean = float(np.mean(rms))
    # Normalize RMS to [0,1] per file
    energy = float(rms_mean / (np.max(rms) + 1e-8))

    # Loudness dBFS approx from RMS; clamp to [-60, 0]
    eps = 1e-10
    loudness_db = 20.0 * np.log10(max(rms_mean, eps))
    loudness_db = float(np.clip(loudness_db, -60.0, 0.0))

    # Tempo estimate (BPM)
    try:
        tempo = float(librosa.beat.tempo(y=audio, sr=sr, hop_length=512).flatten()[0])
    except Exception:
        tempo = 120.0

    # Acousticness proxy: inverse spectral flatness (more tonal -> higher acousticness)
    try:
        flatness = float(np.mean(librosa.feature.spectral_flatness(y=audio)))
        acousticness = float(np.clip(1.0 - flatness, 0.0, 1.0))
    except Exception:
        acousticness = 0.5

    # Danceability proxy: onset strength normalized
    try:
        onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
        danceability = float(np.clip(np.mean(onset_env) / (np.max(onset_env) + 1e-8), 0.0, 1.0))
    except Exception:
        danceability = 0.5

    # Valence proxy: normalized spectral centroid (brighter -> higher valence)
    try:
        centroid = librosa.feature.spectral_centroid(y=audio, sr=sr).flatten()
        valence = float(np.clip((np.mean(centroid) - 500.0) / (4000.0 - 500.0 + 1e-8), 0.0, 1.0))
    except Exception:
        valence = 0.5

    # Simple defaults for features without robust audio proxies here
    instrumentalness = 0.0
    liveness = 0.1
    speechiness = 0.1

    # Compute arousal/enhanced_valence with the same weights used in preprocessing
    loudness_norm = (loudness_db - (-60.0)) / (0.0 - (-60.0) + 1e-8)
    tempo_norm = (tempo - 50.0) / (200.0 - 50.0 + 1e-8)
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