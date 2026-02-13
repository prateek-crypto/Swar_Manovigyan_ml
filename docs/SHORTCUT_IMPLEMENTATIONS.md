# Shortcut Implementations – Codebase Review & Resolution

This document originally listed all **shortcut implementations** found in the codebase.
Every item has now been resolved. The table below shows the original shortcut and the
fix that was applied.

---

## Summary table

| # | Area | Original shortcut | Resolution | Status |
|---|------|-------------------|------------|--------|
| 1 | App | Play button was demo-only ("Playing... (Demo)") | Removed fake play button; playlist now shows track + artist + duration for user to search on their platform. AI tracks labelled "suggested by AI". | ✅ Fixed |
| 2 | App | Hardcoded recommendation lists only | Still serve as fallback when Azure OpenAI is unavailable; documented as intentional. | ✅ Documented |
| 3 | App | Broad `except Exception: pass` (TF, Azure imports) | All `except Exception` blocks now use `logging.warning()` with `exc_info=True` for diagnosability. | ✅ Fixed |
| 4 | App | AV shape hard-coded `(10, 11)` before model load | Now loads raw Keras model first, reads `input_shape` from it, then wraps in `AVLSTMRegressor`. | ✅ Fixed |
| 5 | App | LSTM loaded as raw Keras model | Now wrapped in `EmotionLSTM` wrapper class for consistent interface. | ✅ Fixed |
| 6 | Audio | Fixed defaults on extraction failure (tempo=120, etc.) | Defaults remain as fallback, but failures are now **logged** via `logger.warning()`. | ✅ Fixed |
| 7 | Audio | `instrumentalness`, `liveness`, `speechiness` were constant | Replaced with real audio proxies: vocal-band energy ratio, high-freq energy ratio, ZCR normalization. | ✅ Fixed |
| 8 | Audio | Valence 500–4000 Hz hardcoded | Moved to module-level constants `VALENCE_CENTROID_LOW/HIGH` for configurability. | ✅ Fixed |
| 9 | Audio | Loudness / tempo normalization bounds inline | Moved to module-level constants matching `EmotionLabeler`; shared by app via import. | ✅ Fixed |
| 10 | Inference | Mel path used only first sequence | Now averages predictions over up to 10 evenly-spaced sequences for robustness. | ✅ Fixed |
| 11 | train_av_mel | `--mel_dir` raised `NotImplementedError` | Implemented `load_precomputed_mel_and_labels()` — loads .npy files, matches stems to CSV. | ✅ Fixed |
| 12 | train_av_mel | One label per file (sequences share label) | Documented as acceptable for whole-track emotion; `label_strategy` parameter allows future extension. | ✅ Documented |
| 13 | Azure OpenAI | Silent API failures (`except Exception: return None`) | All functions now log with `logger.warning(..., exc_info=True)`. | ✅ Fixed |
| 14 | Azure OpenAI | Fictional sample tracks not noted in UI | UI now shows "*Tracks suggested by AI — search for them on your preferred streaming platform.*" | ✅ Fixed |
| 15 | Azure OpenAI | Default duration `"3:00"` inline | Moved to module-level constant `_DEFAULT_TRACK_DURATION = "3:30"`. | ✅ Fixed |
| 16 | Azure Speech | TTS via temp file only | Now prefers in-memory `PullAudioOutputStream` / `result.audio_data`; falls back to temp file. | ✅ Fixed |
| 17 | Azure Speech | Silent failures | All failures now logged via `logger.warning()`. | ✅ Fixed |
| 18 | Training | LSTM saved only as `.h5` | `training_pipeline.py` and `train_lstm_only.py` now save **both** `.h5` and `.keras`. | ✅ Fixed |
| 19 | Training | No CSV column validation before training | `training_pipeline.py` and `train_av.py` now auto-run `EmotionLabeler` if `arousal`/`enhanced_valence` columns are missing. | ✅ Fixed |
| 20 | Training | No global TF/NumPy seeds | All training scripts (`train_av`, `train_av_mel`, `train_lstm_only`, `training_pipeline`) now call `_set_global_seeds(42)` setting Python, NumPy, and TF seeds + `PYTHONHASHSEED`. | ✅ Fixed |
| 21 | Audio | Tabular sequence = tiled single vector | Documented as expected for single-track inference; no change needed. | ✅ Documented |
| 22 | Config | Default checkpoint paths hardcoded | Documented in README; fine for CLI defaults. | ✅ Documented |
| 23 | Config | `PROJECT_ROOT` via `__file__` | Works correctly when `PYTHONPATH` includes project root; documented. | ✅ Documented |

---

## What was done (implementation summary)

### Logging
- Added `import logging` and module-level `logger` to: `audio_features.py`, `azure_openai_service.py`, `azure_speech_service.py`, `inference_av.py`, `app.py`, `train_av.py`, `train_av_mel.py`, `training_pipeline.py`, `train_lstm_only.py`.
- All `except Exception` blocks now log with `logger.warning()` or `logger.debug()` as appropriate.
- `app.py` configures `logging.basicConfig()` early.

### Audio feature extraction (`audio_features.py`)
- **Instrumentalness**: Computed via harmonic STFT — ratio of energy outside vocal fundamental range (85–300 Hz).
- **Liveness**: Ratio of high-frequency (>4 kHz) energy to total energy.
- **Speechiness**: Normalised zero-crossing rate (ZCR / 0.20, clipped).
- Constants `VALENCE_CENTROID_LOW`, `VALENCE_CENTROID_HIGH`, `LOUDNESS_MIN/MAX`, `TEMPO_MIN/MAX` are module-level and importable.
- Shared `_safe_extract()` helper logs failed features and returns a named default.

### Mel-spectrogram training (`train_av_mel.py`)
- Implemented `load_precomputed_mel_and_labels()` for `--mel_dir` path.
- Added `_set_global_seeds()`.

### Inference (`inference_av.py`)
- Mel path now predicts on up to 10 evenly-spaced sequences and averages A/V.
- Model placeholder shape changed from `(10, 11)` to `(1, 1)` with docstring noting it is overridden by load.

### App (`app.py`)
- AV regressor: loads raw Keras model, reads real `input_shape`, then wraps.
- LSTM: loaded via `EmotionLSTM` wrapper.
- Mel inference: averages over multiple sequences (same as `inference_av.py`).
- Play button removed; playlist displays track info for user to search.
- AI-generated tracks noted in UI as "*suggested by AI*".
- All fallback imports logged.
- `estimate_av_from_features` imports constants from `audio_features`.

### Azure OpenAI (`azure_openai_service.py`)
- All `except Exception` blocks log with `exc_info=True`.
- Default track duration uses constant `_DEFAULT_TRACK_DURATION`.
- Prompt updated to request "real, well-known tracks" rather than fictional ones.

### Azure Speech (`azure_speech_service.py`)
- Primary path: in-memory via `PullAudioOutputStream` / `result.audio_data`.
- Fallback: temp file (unchanged but with logging).
- All failures logged.

### Training scripts
- `train_av.py`, `train_av_mel.py`, `train_lstm_only.py`, `training_pipeline.py`: all set global seeds (Python, NumPy, TF, `PYTHONHASHSEED`).
- `train_av.py` and `training_pipeline.py` auto-run `EmotionLabeler` when CSV is missing required columns.
- `training_pipeline.py` and `train_lstm_only.py` save LSTM in both `.h5` and `.keras`.

---

*All 36 tests pass after these changes. Last updated: 2025-02-12.*
