"""
Train and evaluate ONLY the LSTM emotion classifier (no CNN, no baselines).

This is a lighter-weight entry point than the full TrainingPipeline.run_complete_training,
useful when you just want a classification model for the Streamlit app.

Usage (from project root):

  python -m src.models.train_lstm_only --epochs 30 --batch_size 32 --sequence_length 10

Outputs:
  - models/lstm_emotion_model.h5
  - models/lstm_emotion_model.keras
  - models/scaler_lstm.joblib
"""

from __future__ import annotations

import argparse
import logging
import os
import random

import joblib
import numpy as np

from .training_pipeline import TrainingPipeline

logger = logging.getLogger(__name__)


def _set_global_seeds(seed: int = 42) -> None:
    """Set Python, NumPy, and TensorFlow global seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except Exception:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Train only the LSTM emotion classifier.")
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/processed/spotify_features_with_emotions.csv",
        help="Path to processed CSV with emotion labels.",
    )
    parser.add_argument(
        "--sequence_length",
        type=int,
        default=10,
        help="Sequence length (timesteps) for LSTM input.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Mini-batch size.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models",
        help="Directory to save trained model and scaler.",
    )
    args = parser.parse_args()

    # Set global seeds for reproducibility
    _set_global_seeds(42)

    os.makedirs(args.output_dir, exist_ok=True)

    pipeline = TrainingPipeline(data_path=args.data_path)
    data = pipeline.load_and_prepare_data(sequence_length=args.sequence_length)

    # Train LSTM only
    lstm_results = pipeline.train_lstm(data, epochs=args.epochs, batch_size=args.batch_size)
    print("\nLSTM-only training complete.")
    print(f"Test accuracy: {lstm_results['test_accuracy']:.4f}")
    print(f"Test loss: {lstm_results['test_loss']:.4f}")

    # Save scaler and LSTM model in a way the app expects
    scaler_path = os.path.join(args.output_dir, "scaler_lstm.joblib")
    joblib.dump(pipeline.preprocessor.scaler, scaler_path)
    print(f"Saved scaler to {scaler_path}")

    if pipeline.lstm_model:
        # Save both .h5 and .keras for app compatibility
        lstm_h5 = os.path.join(args.output_dir, "lstm_emotion_model.h5")
        pipeline.lstm_model.save_model(lstm_h5)
        print(f"Saved LSTM classifier to {lstm_h5}")

        try:
            lstm_keras = os.path.join(args.output_dir, "lstm_emotion_model.keras")
            pipeline.lstm_model.save_model(lstm_keras)
            print(f"Saved LSTM classifier (.keras) to {lstm_keras}")
        except Exception:
            logger.debug("Could not save .keras format; .h5 was saved successfully.")


if __name__ == "__main__":
    main()
