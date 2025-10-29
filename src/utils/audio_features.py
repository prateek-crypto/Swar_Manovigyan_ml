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
