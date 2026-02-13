"""
A/V inference from audio file or bytes.
Usage:
  python -m src.inference_av --audio_path path/to/audio.wav --checkpoint models/av_regressor.keras
"""

import argparse
import logging
import os
import numpy as np
from typing import Tuple

from src.models.av_regressor import AVLSTMRegressor
from src.utils.audio_features import extract_mel_spectrogram_sequence, extract_tabular_features_sequence
from src.utils.feature_stats import load_feature_stats, apply_zscore

logger = logging.getLogger(__name__)

# Maximum number of mel sequences to average for robustness
_MAX_MEL_SEQUENCES_TO_AVG = 10


def predict_from_audio_bytes(audio_bytes: bytes, checkpoint: str) -> Tuple[float, float]:
    """
    Predict arousal and valence from raw audio bytes using a saved checkpoint.

    For mel-spectrogram models: predictions are averaged over up to
    ``_MAX_MEL_SEQUENCES_TO_AVG`` sliding-window sequences for stability.
    """
    # Load model and infer expected input shape
    model = AVLSTMRegressor(input_shape=(1, 1))  # placeholder; overridden by load
    model.load(checkpoint)

    input_shape = model.model.input_shape  # (None, T, F)
    expected_T = input_shape[1]
    expected_F = input_shape[2]

    # Load feature stats for z-score (tabular path only)
    feature_stats = load_feature_stats(checkpoint) if expected_F == 11 else None

    if expected_F == 128:
        # ── Mel-spectrogram path ──────────────────────────────────────
        from src.train_av_mel import build_frame_sequences_from_mel

        mel, _, _ = extract_mel_spectrogram_sequence(audio_bytes, target_frames=None)
        sequences = build_frame_sequences_from_mel(mel, sequence_length=expected_T or 10, stride=1)

        # Average predictions over multiple sequences for robustness
        n_seqs = min(len(sequences), _MAX_MEL_SEQUENCES_TO_AVG)
        if n_seqs == 0:
            raise ValueError("No sequences could be built from the audio file.")

        # Evenly spaced indices to cover the whole track
        indices = np.linspace(0, len(sequences) - 1, n_seqs, dtype=int)
        X_batch = sequences[indices]  # (n_seqs, T, 128)
        preds = model.predict(X_batch)  # (n_seqs, 2)
        av = preds.mean(axis=0)
        logger.info("Mel inference: averaged %d/%d sequences", n_seqs, len(sequences))

    elif expected_F == 11:
        # ── Tabular path ──────────────────────────────────────────────
        seq, _ = extract_tabular_features_sequence(audio_bytes, sequence_length=expected_T or 10)
        if feature_stats is not None:
            seq = apply_zscore(seq, feature_stats)
        X = seq[None, ...]  # (1, 10, 11)
        av = model.predict(X)[0]
    else:
        raise ValueError(f"Unsupported model feature dimension: expected_F={expected_F}")

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
