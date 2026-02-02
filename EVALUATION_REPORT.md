# End-to-End Flow Evaluation: Swar_Manovigyan_ML

This report evaluates the codebase and methodology against the stated architecture and identifies major and minor issues.

---

## Fixes applied (post-evaluation)

- **§2.1** Train/inference normalization: `feature_stats_av.json` is saved next to the checkpoint; `src/utils/feature_stats.py` loads and applies z-score in `inference_av.py` and the app.
- **§2.2** Mel path: Documented as future; inference/app branch on `expected_F == 128` retained for when a mel model is trained.
- **§2.4** Baseline input: App builds (1, 10, 11) from form via `build_sequence_11_from_form()`, applies `scaler_lstm`, flattens to (1, 110) for baselines.
- **§2.5** LSTM path: Correct arousal/enhanced_valence formulas; `scaler_lstm.joblib` persisted by training_pipeline and loaded in app.
- **§3.1** Label definition: `data_analysis.py` uses fixed loudness [−60, 0] and tempo [50, 200].
- **§3.2–3.5** Duplicate load removed; typo fixed; default checkpoint set to `.keras`; app uses project-root paths and clear baseline-load messaging.
- **Optional:** `.env.example` and optional Azure OpenAI integration for therapeutic explanations and recommendation blurbs (see `src/utils/azure_openai_service.py`).

---

## 1. End-to-End Flow Summary

| Stage | Component | Status |
|-------|-----------|--------|
| Data | Raw CSV → EmotionLabeler → processed CSV | ✅ Works |
| Training | build_sequences_for_regression → AVLSTMRegressor + feature_stats | ✅ Works |
| Inference (CLI) | Audio → tabular + z-score → model | ✅ Fixed (§2.1) |
| UI | Streamlit: Manual / Features / Audio Upload | ✅ Fixed (§2.4, §2.5, §3) |

---

## 2. Major Issues

### 2.1 **Train/Inference Feature Normalization Mismatch (Critical)**

**Location:** `train_av.py` vs `inference_av.py` + `audio_features.py`

**Problem:**  
Training uses **z-score standardization** on the 11 tabular features:

```python
# train_av.py build_sequences_for_regression()
features_std = (df_features - df_features.mean()) / (df_features.std() + 1e-8)
features_np = features_std.values.astype('float32')
```

The **mean and std are never saved**. At inference, `extract_tabular_features_sequence()` returns **raw** values (e.g. loudness in dB, tempo in BPM, energy in [0,1]). The model therefore receives inputs on a completely different scale than during training, which severely degrades predictions.

**Fix:**  
- Save training-time feature mean and std (e.g. in a JSON or joblib file next to the checkpoint) and load them in `inference_av.py` and in the app.  
- In inference, apply the same z-score: `(features - mean) / (std + 1e-8)` before building the sequence and calling the model.

---

### 2.2 **Audio Upload: No Mel-Spectrogram Model in Use**

**Location:** Methodology vs implementation

**Problem:**  
The methodology describes a **dual** feature pathway: mel-spectrograms (128 mel bands, 300 frames) and tabular features. The code in `inference_av.py` and `app.py` branches on `expected_F == 128` (mel) vs `expected_F == 11` (tabular). Only the **tabular** path is trained: `train_av.py` uses CSV rows and `build_sequences_for_regression()`, which produces shape `(T, 11)`. There is no training script that trains a model on mel-spectrograms. So the mel path is dead code for the current pipeline, and the “dual pathway” claim is not implemented end-to-end.

**Fix:**  
Either:  
- Document that only the tabular (11-dim) pathway is used and remove or clearly mark the mel-spectrogram branch as “future”; or  
- Add a training path that builds sequences from mel-spectrograms (e.g. from audio files or precomputed mel arrays), train a separate model with input shape `(T, 128)`, and wire it into inference and the app.

---

### 2.3 **“Temporal” Data Is Not Per-Audio Time Series**

**Location:** Data design and methodology wording

**Problem:**  
Sequences are built from **rows of the CSV**: each row is one **track**, and each timestep in a sequence is a **different song**. So the model is trained on “sequence of 10 tracks → predict A/V of the 10th,” not on “time frames within one audio file.” The methodology speaks of “temporal dependencies in audio” and “sequential audio analysis,” which suggests within-track time structure. At inference, one audio file is converted to **one** 11-dim vector and **tiled** to length 10, so every timestep is identical—no temporal variation. Training and inference therefore differ in nature (multi-track context vs single repeated frame).

**Implications:**  
- The LSTM is effectively seeing “context of 10 different tracks” in training but “same frame 10 times” at inference.  
- Wording in the report should be adjusted to “sequence of track-level features” rather than “sequential audio” unless you add true per-audio time-series (e.g. mel frames over time).

---

### 2.4 **Baseline Models: Input Dimension Mismatch in App**

**Location:** `app.py` → `predict_emotion_baseline()` vs `baseline_models.py`

**Problem:**  
Baselines are trained on **flattened sequences**: `X_sequences.reshape(X_sequences.shape[0], -1)` → shape `(N, 10*11) = (N, 110)`. In the app, “Audio Features Input” uses `create_audio_features_form()`, which returns a single vector of **9** features (no arousal/enhanced_valence). That is passed to `predict_emotion_baseline()`, which does `features_flat.reshape(1, -1)` → shape `(1, 9)`. So the baseline models receive **9** features instead of **110**, causing wrong behavior or errors depending on sklearn version.

**Fix:**  
- Either: In the app, build a (1, 10, 11) sequence from the 9 sliders (plus computed arousal and enhanced_valence), flatten to (1, 110), and **optionally** standardize with the same scaler used for baseline training if you introduce one.  
- Or: Train separate “single-vector” baselines on 9 or 11 features per track and use those in the app, and document that “Audio Features Input” baselines use a different input format than the LSTM.

---

### 2.5 **LSTM (Classification) Path in App: Wrong Arousal/Valence and Scaling**

**Location:** `app.py` → `create_audio_features_form()` and `predict_emotion_lstm()`

**Problem:**  
- The form exposes **9** features; the LSTM expects **11** (including arousal and enhanced_valence). The code derives arousal and enhanced_valence inside `predict_emotion_lstm()`.  
- Arousal is computed as `(features[0, 2] + features[0, 5] + features[0, 7]) / 3` (energy, loudness, tempo). The methodology uses **weighted** 0.4·energy + 0.3·loudness_norm + 0.3·tempo_norm with loudness in [−60, 0] and tempo in [50, 200] normalized to [0,1]. Using raw loudness and tempo without normalization makes the scale wrong.  
- The classification LSTM is trained (in `training_pipeline`) on **StandardScaler**-transformed 11-dim sequences. The app never applies that scaler (and the pipeline does not persist it for the classification LSTM). So the LSTM in the app gets unstandardized and incorrectly derived inputs.

**Fix:**  
- Use the same arousal/enhanced_valence formulas as in `data_analysis.py` and `audio_features.py` (normalize loudness and tempo to [0,1], then weighted sum).  
- Persist and load the same scaler used for the classification LSTM and apply it to the (1, 10, 11) sequence before prediction.

---

## 3. Minor Issues

### 3.1 **Arousal/Valence Label Definition: Doc vs Code**

**Location:** `data_analysis.py` EmotionLabeler vs methodology

**Problem:**  
The report states loudness normalized as [−60, 0] → [0, 1] and tempo [50, 200] BPM → [0, 1]. In `data_analysis.py`, arousal uses **min-max normalization over the dataset** for loudness and tempo, not fixed ranges. So labels are dataset-dependent and not aligned with the documented formula; inference uses fixed ranges in `audio_features.py`, creating a slight train-vs-inference label and feature mismatch.

**Fix:**  
Use the same fixed ranges in `data_analysis.py` when creating arousal (and optionally enhanced_valence) so that train labels and inference formulas match the document.

---

### 3.2 **Duplicate AV Regressor Load Block in App**

**Location:** `app.py` `load_models()`

**Problem:**  
Two consecutive `if/elif` blocks both check `os.path.exists('models/av_regressor.keras')` and load the same file. The second block is unreachable.

**Fix:**  
Keep a single block that checks for `av_regressor.keras` (and optionally `av_regressor.h5`) and loads once.

---

### 3.3 **Typo in UI**

**Location:** `app.py` → `display_model_comparison()`

**Problem:**  
Subheader text: `"Mod el Predictions Comparison"` (space in “Model”).

**Fix:**  
Change to `"Model Predictions Comparison"`.

---

### 3.4 **Default Checkpoint Extension**

**Location:** `train_av.py` vs README/App

**Problem:**  
`train_av.py` default is `--checkpoint models/av_regressor.h5`; README and app use `models/av_regressor.keras`. Users running train without `--checkpoint` get a different file than the app expects.

**Fix:**  
Set default checkpoint to `models/av_regressor.keras` (or align README/app with `.h5`).

---

### 3.5 **Baseline Load Failure and CWD**

**Location:** `app.py` `load_models()`

**Problem:**  
If `models/baseline_models/` is missing or empty, `load_models()` still runs and leaves `self.baseline_models.models` empty. The app then shows “No models available” for comparison. Paths are relative to CWD; if Streamlit is run from another directory, model and baseline paths can break.

**Fix:**  
Resolve paths relative to project root (e.g. via `__file__` or an env var) and optionally show a clear message when no baseline models are found.

---

### 3.6 **Inference: Loudness in Tabular Vector**

**Location:** `audio_features.py` → `extract_tabular_features_sequence()`

**Problem:**  
The 11-dim vector stores `loudness_db` in **dB** (e.g. −60 to 0). Training z-scores this column with dataset mean/std. If you later add saved mean/std for inference, this is fine. As long as inference does not z-score, the current bug (§2.1) dominates; once fixed, ensure the same loudness definition (and normalization) is used in training and inference.

---

## 4. Architecture and Concept Summary

| Aspect | Assessment |
|--------|------------|
| **Data pipeline** | Clear: CSV → EmotionLabeler → arousal/enhanced_valence → sequences. |
| **AV regressor** | BiLSTM design and training loop are sound; inference is broken by missing normalization. |
| **Dual pathway (mel + tabular)** | Only tabular is trained; mel path is unimplemented for training. |
| **“Temporal” claims** | Data is per-track sequences, not within-audio time series; inference tiles one vector. |
| **UI modes** | Manual selection is consistent; Audio Features and Audio Upload depend on fixes above. |
| **Therapeutic recommendation** | Quadrant mapping and genre suggestions are consistent with the 4-class design. |

---

## 5. Recommended Fix Order (all addressed)

1. ~~**Critical:** Persist and apply z-score~~ → Done: `feature_stats_av.json` + `src/utils/feature_stats.py`; inference and app apply z-score.  
2. ~~**Critical:** Fix baseline input~~ → Done: (1, 10, 11) from form, scaler, flatten to 110.  
3. ~~**Important:** Align LSTM path and scaler~~ → Done: `scaler_lstm.joblib`; correct A/V formulas in app.  
4. ~~**Important:** Fixed loudness/tempo in data_analysis~~ → Done: EmotionLabeler uses fixed ranges.  
5. ~~**Minor:** Duplicate load, typo, checkpoint, paths~~ → Done.

**Future (optional):** Mel-spectrogram training path; per-user fine-tuning; multi-modal signals. See `TODO_SDLC.md` for a structured development checklist.
