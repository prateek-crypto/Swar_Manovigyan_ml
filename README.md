# Swar_Manovigyan_ML

Emotion-aware music therapy prototype: LSTM-based arousal–valence (A/V) regression from audio with a Streamlit frontend.

## Why
- Most recommenders are popularity/genre-based, not emotion-aware
- Generic mood tags miss nuanced emotions
- Personalization is limited for therapeutic use
- Emotion recognition needs temporal models (audio + signals)

Goal: Dataset-driven A/V model for personalized music therapy.

## Features
- TensorFlow BiLSTM regressor → continuous [arousal, valence] ∈ [0,1]
- Librosa mel-spectrogram features for temporal modeling
- Streamlit app: upload audio, visualize A/V point, see suggestions
- Training and CLI inference scripts

## Project Structure
```
Swar_Manovigyan_ML/
├─ src/
│  ├─ frontend/
│  │  └─ app.py                # Streamlit UI (manual, features, audio upload A/V)
│  ├─ models/
│  │  ├─ av_regressor.py       # TensorFlow AV LSTM regressor
│  │  └─ lstm_model.py         # (Optional) 4-class LSTM classifier support
│  ├─ utils/
│  │  ├─ audio_features.py     # Mel-spectrogram extraction
│  │  └─ data_analysis.py      # Label creation (arousal/enhanced_valence)
│  ├─ train_av.py              # Train AV regressor
│  └─ inference_av.py          # Predict A/V from audio
├─ data/
│  ├─ raw/
│  │  └─ SpotifyFeatures.csv   # Example data (optional)
│  └─ processed/
├─ models/                     # Checkpoints
├─ requirements.txt
└─ README.md
```

## Setup
```bash
python -m venv .venv
. .venv\Scripts\Activate.ps1   # Windows PowerShell
pip install -r requirements.txt
```

## Train
Ensure your processed CSV has `arousal` and `enhanced_valence` columns (generate via `src/utils/data_analysis.py` if needed).
```bash
python -m src.train_av \
  --csv data/processed/spotify_features_with_emotions.csv \
  --epochs 10 \
  --checkpoint models/av_regressor.keras
```

## Inference (CLI)
```bash
python -m src.inference_av --audio_path path/to/audio.wav --checkpoint models/av_regressor.keras
```

## Run UI
```bash
streamlit run src/frontend/app.py
```
- Choose “Audio Upload (A/V)” → upload wav/mp3 → see A/V and recommendations
- Optionally enter a different checkpoint path

## Approach/Methodology/Model

### Preprocessing:
**Features:** Spotify audio features (acousticness, danceability, energy, instrumentalness, liveness, loudness, speechiness, tempo, valence); derived continuous arousal and enhanced valence.

**Arousal computation:** Weighted average of energy (0.4), normalized loudness [−60, 0] → [0,1] (0.3), normalized tempo [50, 200] BPM → [0,1] (0.3).

**Enhanced valence computation:** Weighted average of Spotify valence (0.6), danceability (0.3), (1 − acousticness) (0.1).

**Alternative audio input:** Raw audio → mel-spectrograms (128 mels, 22050 Hz, hop_length=512, n_fft=2048) or tabular proxy features (RMS energy, tempo, spectral centroid, flatness, onset strength) mapped to 11-dimensional sequences.

**Scaling:** Z-score standardization (mean=0, std=1) with median imputation for missing values.

**Sequencing:** Windowed into fixed-length sequences (T=10 timesteps, stride=1); overlapping sliding windows for temporal modeling.

**Normalization:** Labels bounded to [0,1]; mel-spectrograms min-max normalized per-file; sequences padded/cropped to target length.

### Model Layers/Architecture:
**Input Layer:** `(batch, 10, 11)` for tabular features or `(batch, T, 128)` for mel-spectrograms.

**Bidirectional LSTM Layer 1:** 96 units per direction, `return_sequences=True` → `(batch, 10, 192)`.

**Dropout (0.3) + BatchNormalization** after LSTM-1.

**Bidirectional LSTM Layer 2:** 48 units per direction, returns last timestep → `(batch, 96)`.

**Dropout (0.3) + BatchNormalization** after LSTM-2.

**Dense Layer 1:** 128 units, ReLU activation.

**Dropout (0.3)** after Dense-1.

**Dense Layer 2:** 64 units, ReLU activation.

**Output Layer:** Dense(2), sigmoid activation → `[arousal, valence]` ∈ [0,1].

### Training:
**Optimizer:** Adam (learning_rate=1e-3, default).

**Loss:** Mean Squared Error (MSE) for continuous regression.

**Callbacks:** EarlyStopping (monitor='val_loss', patience=8, restore_best_weights=True); ReduceLROnPlateau (monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6).

**Data split:** Train/Val/Test 60/20/20 random split (seed=42).

**Hyperparameters:** Batch size=32, default epochs=20; sequence_length=10, stride=1 (configurable).

### Evaluation:
**Metrics:** Test MSE (primary loss), Mean Absolute Error (MAE); per-dimension MAE for arousal vs valence.

**Visualization:** Arousal–valence scatter plot in 2D emotion space; quadrant mapping (Low/High Arousal × Negative/Positive Valence → 4 emotion classes).

**Inference:** Real-time prediction from uploaded audio (wav/mp3/flac/ogg) via Streamlit UI or CLI; automatic feature extraction and sequence preparation.

## Personalization (Next Steps)
- Fine-tune on user-specific A/V data
- Add physiological signals (multi-modal, multi-task)
- Attention over frames for interpretability
- A/V → recommendation engine with user constraints