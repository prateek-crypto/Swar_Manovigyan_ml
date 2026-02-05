"""
Train AV LSTM regressor on mel-spectrograms with frame-level temporal sequences.
This implements true temporal modeling: sequences are built from frames within audio files,
not from different tracks.

Usage:
  # With audio directory and CSV mapping filenames to labels:
  python -m src.train_av_mel --audio_dir data/audio --csv data/processed/spotify_features_with_emotions.csv --epochs 20 --checkpoint models/av_regressor_mel.keras

  # With precomputed mel-spectrograms directory:
  python -m src.train_av_mel --mel_dir data/processed/mel_spectrograms --csv data/processed/spotify_features_with_emotions.csv --epochs 20 --checkpoint models/av_regressor_mel.keras
"""

import argparse
import json
import os
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any, Optional
from pathlib import Path
import librosa
import soundfile as sf
from tqdm import tqdm

from src.models.av_regressor import AVLSTMRegressor


def extract_mel_spectrogram_from_file(
    audio_path: str,
    sr: int = 22050,
    n_mels: int = 128,
    hop_length: int = 512,
    n_fft: int = 2048,
) -> np.ndarray:
    """
    Extract mel-spectrogram from an audio file.
    Returns (T, 128) array where T is number of frames (variable length).
    """
    try:
        audio, file_sr = sf.read(audio_path)
    except Exception as e:
        raise ValueError(f"Failed to load {audio_path}: {e}")

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    if file_sr != sr:
        audio = librosa.resample(audio, orig_sr=file_sr, target_sr=sr)

    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Normalize per-file to [0,1]
    mel_min = mel_db.min()
    mel_max = mel_db.max()
    mel_norm = (mel_db - mel_min) / (mel_max - mel_min + 1e-8)

    # Transpose to (T, F) where T=timesteps, F=n_mels
    features = mel_norm.T.astype(np.float32)
    return features


def build_frame_sequences_from_mel(
    mel_spectrogram: np.ndarray,
    sequence_length: int = 10,
    stride: int = 1,
) -> np.ndarray:
    """
    Build sequences using sliding window over mel-spectrogram frames.
    This implements true temporal modeling: sequences are consecutive frames within one audio file.

    Args:
        mel_spectrogram: (T, 128) array where T is variable number of frames
        sequence_length: Number of consecutive frames per sequence
        stride: Step size between sequence starts

    Returns:
        sequences: (N, sequence_length, 128) array where N is number of sequences
    """
    T = mel_spectrogram.shape[0]
    if T < sequence_length:
        # Pad if audio is too short
        pad_len = sequence_length - T
        pad = np.zeros((pad_len, mel_spectrogram.shape[1]), dtype=mel_spectrogram.dtype)
        mel_spectrogram = np.vstack([mel_spectrogram, pad])
        T = mel_spectrogram.shape[0]

    # Sliding window over frames
    starts = np.arange(0, T - sequence_length + 1, stride)
    sequences = np.empty((len(starts), sequence_length, mel_spectrogram.shape[1]), dtype=np.float32)

    for idx, s in enumerate(starts):
        sequences[idx] = mel_spectrogram[s:s+sequence_length]

    return sequences


def load_audio_files_and_labels(
    audio_dir: str,
    csv_path: str,
    filename_column: str = "filename",
) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Load audio files and extract mel-spectrograms, matching with labels from CSV.

    Args:
        audio_dir: Directory containing audio files
        csv_path: CSV with columns: filename (or specified column), arousal, enhanced_valence
        filename_column: Column name in CSV that contains audio filenames

    Returns:
        mel_spectrograms: List of (T_i, 128) arrays (variable length per file)
        labels: (N, 2) array with [arousal, enhanced_valence] per file
    """
    df = pd.read_csv(csv_path)
    required_cols = [filename_column, 'arousal', 'enhanced_valence']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    audio_dir_path = Path(audio_dir)
    mel_spectrograms = []
    labels = []

    print(f"Loading audio files from {audio_dir}...")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting mel-spectrograms"):
        filename = row[filename_column]
        # Try common extensions
        audio_path = None
        for ext in ['.wav', '.mp3', '.flac', '.ogg', '.m4a']:
            candidate = audio_dir_path / f"{filename}{ext}"
            if candidate.exists():
                audio_path = candidate
                break

        if audio_path is None:
            # Try exact filename match
            candidate = audio_dir_path / filename
            if candidate.exists():
                audio_path = candidate

        if audio_path is None or not audio_path.exists():
            print(f"Warning: Audio file not found for {filename}, skipping.")
            continue

        try:
            mel = extract_mel_spectrogram_from_file(str(audio_path))
            mel_spectrograms.append(mel)
            labels.append([row['arousal'], row['enhanced_valence']])
        except Exception as e:
            print(f"Warning: Failed to process {audio_path}: {e}, skipping.")
            continue

    if len(mel_spectrograms) == 0:
        raise ValueError("No valid audio files found. Check audio_dir and CSV filename matching.")

    labels_array = np.array(labels, dtype=np.float32)
    np.clip(labels_array, 0.0, 1.0, out=labels_array)

    print(f"Loaded {len(mel_spectrograms)} audio files.")
    return mel_spectrograms, labels_array


def build_sequences_from_mel_files(
    mel_spectrograms: List[np.ndarray],
    labels: np.ndarray,
    sequence_length: int = 10,
    stride: int = 1,
    label_strategy: str = "last_frame",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build training sequences from mel-spectrograms with frame-level temporal modeling.

    Args:
        mel_spectrograms: List of (T_i, 128) arrays (variable length per file)
        labels: (N, 2) array with [arousal, enhanced_valence] per file
        sequence_length: Number of consecutive frames per sequence
        stride: Step size between sequence starts
        label_strategy: How to assign labels to sequences:
            - "last_frame": Use the file's label for all sequences from that file
            - "mean": Use mean label across all sequences (not implemented)

    Returns:
        X: (M, sequence_length, 128) sequences
        y: (M, 2) labels where M is total number of sequences
    """
    all_sequences = []
    all_labels = []

    print(f"Building frame-level sequences (sequence_length={sequence_length}, stride={stride})...")
    for mel, label in tqdm(zip(mel_spectrograms, labels), total=len(mel_spectrograms), desc="Building sequences"):
        sequences = build_frame_sequences_from_mel(mel, sequence_length=sequence_length, stride=stride)
        all_sequences.append(sequences)
        # Assign the file's label to all sequences from that file
        if label_strategy == "last_frame":
            seq_labels = np.tile(label[None, :], (len(sequences), 1))
        else:
            raise ValueError(f"Unknown label_strategy: {label_strategy}")
        all_labels.append(seq_labels)

    X = np.vstack(all_sequences)
    y = np.vstack(all_labels)

    print(f"Built {len(X)} sequences from {len(mel_spectrograms)} audio files.")
    print(f"Sequence shape: {X.shape}, Label shape: {y.shape}")
    return X, y


def split_train_val_test(X: np.ndarray, y: np.ndarray, val_ratio=0.2, test_ratio=0.2, seed=42):
    """Split sequences into train/val/test sets."""
    n = len(X)
    rng = np.random.default_rng(seed)
    indices = np.arange(n)
    rng.shuffle(indices)

    test_size = int(n * test_ratio)
    val_size = int(n * val_ratio)

    test_idx = indices[:test_size]
    val_idx = indices[test_size:test_size+val_size]
    train_idx = indices[test_size+val_size:]

    return (X[train_idx], y[train_idx]), (X[val_idx], y[val_idx]), (X[test_idx], y[test_idx])


def main():
    parser = argparse.ArgumentParser(description="Train AV regressor on mel-spectrograms with frame-level sequences")
    parser.add_argument('--audio_dir', type=str, default=None, help='Directory containing audio files')
    parser.add_argument('--mel_dir', type=str, default=None, help='Directory with precomputed .npy mel-spectrograms (alternative to --audio_dir)')
    parser.add_argument('--csv', type=str, required=True, help='CSV with filename, arousal, enhanced_valence columns')
    parser.add_argument('--filename_column', type=str, default='filename', help='Column name in CSV for audio filenames')
    parser.add_argument('--sequence_length', type=int, default=10, help='Number of consecutive frames per sequence')
    parser.add_argument('--stride', type=int, default=1, help='Step size between sequence starts')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--checkpoint', type=str, default='models/av_regressor_mel.keras')
    parser.add_argument('--sr', type=int, default=22050, help='Target sample rate')
    parser.add_argument('--n_mels', type=int, default=128, help='Number of mel bands')
    parser.add_argument('--hop_length', type=int, default=512, help='Hop length for STFT')
    parser.add_argument('--n_fft', type=int, default=2048, help='FFT window size')

    args = parser.parse_args()

    if args.audio_dir is None and args.mel_dir is None:
        raise ValueError("Must provide either --audio_dir or --mel_dir")

    checkpoint_dir = os.path.dirname(args.checkpoint)
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)

    # Load mel-spectrograms and labels
    if args.audio_dir:
        mel_spectrograms, labels = load_audio_files_and_labels(
            args.audio_dir,
            args.csv,
            filename_column=args.filename_column,
        )
    else:
        # Load precomputed mel-spectrograms from .npy files
        raise NotImplementedError("Precomputed mel_dir loading not yet implemented. Use --audio_dir.")

    # Build frame-level sequences
    X, y = build_sequences_from_mel_files(
        mel_spectrograms,
        labels,
        sequence_length=args.sequence_length,
        stride=args.stride,
    )

    # Save mel-spectrogram stats (for potential normalization in future)
    mel_stats = {
        "n_mels": args.n_mels,
        "sr": args.sr,
        "hop_length": args.hop_length,
        "n_fft": args.n_fft,
        "sequence_length": args.sequence_length,
        "input_shape": [args.sequence_length, args.n_mels],
        "note": "Mel-spectrograms are normalized per-file (min-max to [0,1]). No global z-score needed.",
    }
    stats_path = os.path.join(checkpoint_dir or ".", "mel_stats_av.json")
    with open(stats_path, "w") as f:
        json.dump(mel_stats, f, indent=2)
    print(f"Saved mel stats to {stats_path}")

    # Split data
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_train_val_test(X, y)

    # Train model
    reg = AVLSTMRegressor(
        input_shape=(X_train.shape[1], X_train.shape[2]),
        learning_rate=args.lr,
    )
    reg.build()
    print(f"Model input shape: {reg.model.input_shape}")
    reg.fit(X_train, y_train, X_val, y_val, epochs=args.epochs, batch_size=args.batch_size)

    test_loss, test_mae = reg.evaluate(X_test, y_test)
    print(f"Test MSE: {test_loss:.4f} | Test MAE: {test_mae:.4f}")

    reg.save(args.checkpoint)
    print(f"Saved checkpoint to {args.checkpoint}")
    print(f"Mel-spectrogram model trained with frame-level temporal sequences.")


if __name__ == '__main__':
    main()
