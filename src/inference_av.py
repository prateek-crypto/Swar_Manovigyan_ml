"""
A/V inference from audio file or bytes.
Usage:
  python -m src.inference_av --audio_path path/to/audio.wav --checkpoint models/av_regressor.keras
"""

import argparse
import os
import numpy as np
from typing import Tuple

from src.models.av_regressor import AVLSTMRegressor
from src.utils.audio_features import extract_mel_spectrogram_sequence, extract_tabular_features_sequence
from src.utils.feature_stats import load_feature_stats, apply_zscore


def predict_from_audio_bytes(audio_bytes: bytes, checkpoint: str) -> Tuple[float, float]:
    # Load model first to get expected input shape
    model = AVLSTMRegressor(input_shape=(10, 11))  # Default shape, will be overridden
    model.load(checkpoint)

    input_shape = model.model.input_shape  # (None, T, F)
    expected_T = input_shape[1]
    expected_F = input_shape[2]

    # Load feature stats for z-score (tabular path only)
    feature_stats = load_feature_stats(checkpoint) if expected_F == 11 else None

    if expected_F == 128:
        # Mel-spectrogram path (future: train a model with input (T, 128))
        mel, _, _ = extract_mel_spectrogram_sequence(audio_bytes)
        if expected_T is not None:
            if mel.shape[0] > expected_T:
                mel = mel[:expected_T, :]
            elif mel.shape[0] < expected_T:
                pad_len = expected_T - mel.shape[0]
                mel = np.vstack([mel, np.zeros((pad_len, mel.shape[1]), dtype=mel.dtype)])
        X = mel[None, ...]  # (1, T, 128)
    elif expected_F == 11:
        # Tabular path: extract then z-score using training stats
        seq, _ = extract_tabular_features_sequence(audio_bytes, sequence_length=expected_T or 10)
        if feature_stats is not None:
            seq = apply_zscore(seq, feature_stats)
        X = seq[None, ...]  # (1, 10, 11)
    else:
        raise ValueError(f"Unsupported model feature dimension: expected_F={expected_F}")

    av = model.predict(X)[0]
    arousal, valence = float(av[0]), float(av[1])
    return arousal, valence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_path', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, default='models/av_regressor.keras')
    args = parser.parse_args()

    with open(args.audio_path, 'rb') as f:
        audio_bytes = f.read()

    arousal, valence = predict_from_audio_bytes(audio_bytes, args.checkpoint)
    print({"arousal": round(arousal, 4), "valence": round(valence, 4)})


if __name__ == '__main__':
    main()
