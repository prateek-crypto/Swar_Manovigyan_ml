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
from src.utils.audio_features import extract_mel_spectrogram_sequence


def predict_from_audio_bytes(audio_bytes: bytes, checkpoint: str) -> Tuple[float, float]:
    features, _, _ = extract_mel_spectrogram_sequence(audio_bytes)
    X = features[None, ...]  # (1, T, F)

    model = AVLSTMRegressor(input_shape=(X.shape[1], X.shape[2]))
    model.load(checkpoint)

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
