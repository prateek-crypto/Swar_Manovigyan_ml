# Structured To-Do List: Swar_Manovigyan_ML (SDLC)

Use this list for efficient software development lifecycle: planning, implementation, testing, and release.

---

## Phase 1: Data & Training

- [ ] **Data**
  - [ ] Ensure `data/raw/SpotifyFeatures.csv` has required columns (genre, acousticness, danceability, energy, …).
  - [ ] Run `src/utils/data_analysis.py` to generate `data/processed/spotify_features_with_emotions.csv` (arousal, enhanced_valence, emotion_label).
  - [ ] Optionally regenerate processed CSV after changing fixed loudness/tempo ranges in `EmotionLabeler`.
- [ ] **AV Regressor (tabular)**
  - [ ] Train: `python -m src.train_av --csv data/processed/spotify_features_with_emotions.csv --epochs 20 --checkpoint models/av_regressor.keras`
  - [ ] Confirm `models/feature_stats_av.json` is created next to the checkpoint.
  - [ ] Run CLI inference: `python -m src.inference_av --audio_path <path> --checkpoint models/av_regressor.keras` and sanity-check A/V output.
- [ ] **AV Regressor (mel-spectrogram) - Frame-level temporal modeling**
  - [ ] Prepare audio files: ensure audio files match filenames in CSV (or add `filename` column to CSV).
  - [ ] Train: `python -m src.train_av_mel --audio_dir data/audio --csv data/processed/spotify_features_with_emotions.csv --epochs 20 --checkpoint models/av_regressor_mel.keras`
  - [ ] Confirm `models/mel_stats_av.json` is created next to the checkpoint.
  - [ ] Run CLI inference: `python -m src.inference_av --audio_path <path> --checkpoint models/av_regressor_mel.keras` and verify frame-level sequences are used.
  - [ ] Compare tabular vs mel model performance on test set.
- [ ] **Classification pipeline (optional)**
  - [ ] Run `src/models/training_pipeline.py` to train LSTM + baselines; confirm `models/scaler_lstm.joblib` and `models/baseline_models/*.joblib` exist.

---

## Phase 2: App & Inference

- [ ] **Paths & config**
  - [ ] Run Streamlit from project root: `streamlit run src/frontend/app.py`.
  - [ ] Verify app finds `models/av_regressor.keras`, `models/feature_stats_av.json`, and (if used) `models/scaler_lstm.joblib`, `models/baseline_models/`.
- [ ] **Manual emotion**
  - [ ] Test “Manual Emotion Selection”: sliders → quadrant + recommendations.
- [ ] **Audio Features Input**
  - [ ] Test “Audio Features Input”: sliders → LSTM/baseline predictions and model comparison; check that recommendations and (if enabled) Azure OpenAI insight appear.
- [ ] **Audio Upload**
  - [ ] Test “Audio Upload (A/V)”: upload wav/mp3 → A/V prediction and quadrant; ensure feature_stats z-score is used (checkpoint dir = `models/`).
- [ ] **Optional Azure OpenAI**
  - [ ] Copy `.env.example` to `.env` and set `AZURE_OPENAI_*` (do not commit `.env`).
  - [ ] Confirm “AI insight” and recommendation blurb appear when env is set and API is reachable.

---

## Phase 3: Testing & Quality

- [ ] **Unit Tests**
  - [ ] Run test suite: `pytest tests/` from project root.
  - [ ] Verify all tests pass: `pytest tests/ -v`
  - [ ] Test coverage: `pytest tests/ --cov=src --cov-report=html` (optional, requires pytest-cov)
- [ ] **Integration Tests**
  - [ ] Test end-to-end training pipeline (tabular): train → save → load → infer.
  - [ ] Test end-to-end training pipeline (mel): train → save → load → infer with frame-level sequences.
  - [ ] Test app with both tabular and mel models.
- [ ] **Code Quality**
  - [ ] Lint/format: run project linter and fix issues.
  - [ ] No secrets in repo: ensure `.env` is in `.gitignore`; only `.env.example` (placeholders) is committed.
- [ ] **Docs**
  - [ ] README: setup, train (tabular + mel), inference, run app; mention optional Azure OpenAI and `.env.example`.
  - [ ] EVALUATION_REPORT: update “Status” for mel-spectrogram training and testing implementation.
- [ ] **Research**
  - [ ] Reproducibility: fix random seeds in train_av, train_av_mel, and training_pipeline; document in README.
  - [ ] Document mel-spectrogram training approach (frame-level temporal sequences) in README and EVALUATION_REPORT.

---

## Phase 4: Release & Extensions

- [ ] **Release**
  - [ ] Tag version; update README with version and citation if needed.
  - [ ] Optional: add `requirements-azure.txt` with `openai`, `python-dotenv` for Azure-only installs.
- [ ] **Future (non-blocking)**
  - [x] ~~Mel-spectrogram model: train on (T, 128) sequences; wire inference and app to use mel path when model expects 128 features.~~ ✅ Implemented in `train_av_mel.py` with frame-level temporal sequences.
  - [ ] User-specific fine-tuning or multi-modal (e.g. physiological signals) as per report “Future Work”.
  - [ ] Attention mechanisms over temporal frames for interpretability.
  - [ ] Ensemble of tabular and mel models for improved robustness.

---

## Quick reference

| Task                    | Command / location |
|-------------------------|--------------------|
| Generate processed CSV  | `python -m src.utils.data_analysis` (from project root; ensure raw CSV path in `if __name__`) |
| Train AV regressor (tabular) | `python -m src.train_av --csv data/processed/spotify_features_with_emotions.csv --checkpoint models/av_regressor.keras` |
| Train AV regressor (mel) | `python -m src.train_av_mel --audio_dir data/audio --csv data/processed/spotify_features_with_emotions.csv --checkpoint models/av_regressor_mel.keras` |
| CLI A/V inference       | `python -m src.inference_av --audio_path <file> --checkpoint models/av_regressor.keras` (or `av_regressor_mel.keras`) |
| Run tests               | `pytest tests/ -v` |
| Run app                 | `streamlit run src/frontend/app.py` |
| Azure OpenAI (optional) | Set `AZURE_OPENAI_*` in `.env`; see `.env.example` |
