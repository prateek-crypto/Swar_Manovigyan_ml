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
├─ models/                     # Checkpoints (not tracked)
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
  --checkpoint models/av_regressor.h5
```

## Inference (CLI)
```bash
python -m src.inference_av --audio_path path/to/audio.wav --checkpoint models/av_regressor.h5
```

## Run UI
```bash
streamlit run src/frontend/app.py
```
- Choose “Audio Upload (A/V)” → upload wav/mp3 → see A/V and recommendations
- Optionally enter a different checkpoint path

## Personalization (Next Steps)
- Fine-tune on user-specific A/V data
- Add physiological signals (multi-modal, multi-task)
- Attention over frames for interpretability
- A/V → recommendation engine with user constraints