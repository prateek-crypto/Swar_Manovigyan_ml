"""
FastAPI backend for Swar Manovigyan — Emotion-Aware Music Recommendation System.

Wraps existing ML models and Azure integrations into a JSON API consumed by the
Next.js frontend at frontend/.
"""

from __future__ import annotations

import logging
import os
import sys
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Path setup — mirror the Streamlit app so `from models.*` / `from utils.*`
# and `from src.*` all resolve correctly.
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.dirname(_THIS_DIR)
_PROJECT_ROOT = os.path.dirname(_SRC_DIR)

if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

load_dotenv(os.path.join(_PROJECT_ROOT, ".env"))

# ---------------------------------------------------------------------------
# Local imports (same modules the Streamlit app uses)
# ---------------------------------------------------------------------------
from models.baseline_models import BaselineModels
from utils.audio_features import (
    LOUDNESS_MAX,
    LOUDNESS_MIN,
    TEMPO_MAX,
    TEMPO_MIN,
    extract_mel_spectrogram_sequence,
    extract_tabular_features_sequence,
)
from utils.feature_stats import apply_zscore, load_feature_stats

try:
    from utils.azure_openai_service import (
        get_ai_music_styles,
        get_ai_sample_tracks,
        get_recommendation_blurb,
        get_therapeutic_explanation,
        is_azure_openai_available,
    )
except Exception:
    is_azure_openai_available = lambda: False
    get_therapeutic_explanation = lambda *a, **k: None
    get_recommendation_blurb = lambda *a, **k: None
    get_ai_music_styles = lambda *a, **k: None
    get_ai_sample_tracks = lambda *a, **k: None

try:
    from utils.azure_speech_service import (
        is_azure_speech_available,
        synthesize_text_to_wav_bytes,
    )
except Exception:
    is_azure_speech_available = lambda: False
    synthesize_text_to_wav_bytes = lambda *a, **k: None

try:
    import tensorflow as tf
except Exception:
    tf = None

import joblib

logging.basicConfig(level=logging.INFO, format="%(name)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EMOTION_MAPPING: Dict[int, str] = {
    0: "Low Arousal, Negative Valence",
    1: "Low Arousal, Positive Valence",
    2: "High Arousal, Negative Valence",
    3: "High Arousal, Positive Valence",
}

MUSIC_RECOMMENDATIONS: Dict[int, List[str]] = {
    0: ["Sad ballads", "Slow acoustic songs", "Melancholic classical music",
        "Indie folk", "Blues", "Ambient music", "Piano solos"],
    1: ["Meditation music", "Nature sounds", "Soft jazz", "Ambient",
        "Classical music", "Folk music", "Acoustic guitar"],
    2: ["Heavy metal", "Punk rock", "Hard rock", "Rap",
        "Electronic", "Industrial", "Alternative rock"],
    3: ["Pop music", "Dance music", "Upbeat rock", "Funk",
        "Disco", "Reggae", "Ska", "Happy songs"],
}

FALLBACK_TRACKS: Dict[int, List[dict]] = {
    0: [
        {"id": "sad1", "title": "Hurt", "artist": "Johnny Cash", "duration": "3:38"},
        {"id": "sad2", "title": "Mad World", "artist": "Gary Jules", "duration": "3:09"},
        {"id": "sad3", "title": "The Sound of Silence", "artist": "Simon & Garfunkel", "duration": "3:05"},
    ],
    1: [
        {"id": "calm1", "title": "Weightless", "artist": "Marconi Union", "duration": "8:59"},
        {"id": "calm2", "title": "Clair de Lune", "artist": "Claude Debussy", "duration": "4:56"},
        {"id": "calm3", "title": "Teardrop", "artist": "Massive Attack", "duration": "5:30"},
    ],
    2: [
        {"id": "angry1", "title": "Break Stuff", "artist": "Limp Bizkit", "duration": "2:46"},
        {"id": "angry2", "title": "Killing in the Name", "artist": "Rage Against the Machine", "duration": "5:13"},
        {"id": "angry3", "title": "Bodies", "artist": "Drowning Pool", "duration": "3:21"},
    ],
    3: [
        {"id": "happy1", "title": "Happy", "artist": "Pharrell Williams", "duration": "3:53"},
        {"id": "happy2", "title": "Don't Stop Me Now", "artist": "Queen", "duration": "3:29"},
        {"id": "happy3", "title": "Good Vibrations", "artist": "The Beach Boys", "duration": "3:37"},
    ],
}

_MAX_MEL_SEQ_AVG = 10

# ---------------------------------------------------------------------------
# Pydantic request / response schemas
# ---------------------------------------------------------------------------

class AudioFeatures(BaseModel):
    acousticness: float = Field(..., ge=0, le=1)
    danceability: float = Field(..., ge=0, le=1)
    energy: float = Field(..., ge=0, le=1)
    instrumentalness: float = Field(..., ge=0, le=1)
    liveness: float = Field(..., ge=0, le=1)
    loudness: float = Field(..., ge=-60, le=0)
    speechiness: float = Field(..., ge=0, le=1)
    tempo: float = Field(..., ge=50, le=200)
    valence: float = Field(..., ge=0, le=1)


class AnalyzeRequest(BaseModel):
    method: str = Field(..., pattern="^(manual|features|upload)$")
    arousal: Optional[float] = None
    valence: Optional[float] = None
    features: Optional[AudioFeatures] = None


class AnalyzeResponse(BaseModel):
    emotion_label: int
    confidence: float
    arousal: float
    valence: float
    probabilities: Optional[List[float]] = None
    features: Optional[AudioFeatures] = None


class RecommendRequest(BaseModel):
    emotion_label: int = Field(..., ge=0, le=3)
    arousal: float
    valence: float


class TrackOut(BaseModel):
    id: str
    title: str
    artist: str
    duration: str


class RecommendResponse(BaseModel):
    styles: List[str]
    blurb: Optional[str] = None
    tracks: List[TrackOut]
    isAiGenerated: bool


class InsightRequest(BaseModel):
    emotion_label: int = Field(..., ge=0, le=3)
    arousal: float
    valence: float


class InsightResponse(BaseModel):
    text: str


class ModelsRequest(BaseModel):
    features: AudioFeatures


class ModelPrediction(BaseModel):
    model: str
    emotion: str
    confidence: float


class StatusResponse(BaseModel):
    azureOpenAI: bool
    azureSpeech: bool
    lstmModel: bool
    avRegressor: bool
    baselineModels: bool


class TTSRequest(BaseModel):
    text: str


# ---------------------------------------------------------------------------
# Application state — populated on startup
# ---------------------------------------------------------------------------

class _AppState:
    lstm_model = None
    av_model = None
    scaler_lstm = None
    baseline_models: Optional[BaselineModels] = None
    av_checkpoint_path: str = ""

state = _AppState()


def _models_dir() -> str:
    return os.path.join(_PROJECT_ROOT, "models")


def _safe_import_av_regressor():
    try:
        from models.av_regressor import AVLSTMRegressor
        return AVLSTMRegressor, None
    except Exception as e:
        logger.warning("AV regressor import failed: %s", e)
        return None, str(e)


def _load_models() -> None:
    """Load all ML models — mirrors MusicRecommendationApp.load_models()."""
    models_dir = _models_dir()
    state.av_checkpoint_path = os.path.join(models_dir, "av_regressor.keras")

    # LSTM classifier
    lstm_path = os.path.join(models_dir, "lstm_emotion_model.keras")
    if not os.path.isfile(lstm_path):
        lstm_path = os.path.join(models_dir, "lstm_emotion_model.h5")
    if tf is not None and os.path.isfile(lstm_path):
        try:
            state.lstm_model = tf.keras.models.load_model(lstm_path, compile=False)
            logger.info("Loaded LSTM classifier from %s", lstm_path)
        except Exception as e:
            logger.warning("LSTM load failed: %s", e)

    # Scaler
    scaler_path = os.path.join(models_dir, "scaler_lstm.joblib")
    if os.path.isfile(scaler_path):
        state.scaler_lstm = joblib.load(scaler_path)
        logger.info("Loaded scaler from %s", scaler_path)

    # AV regressor
    if tf is not None and os.path.isfile(state.av_checkpoint_path):
        AVCls, err = _safe_import_av_regressor()
        if AVCls is not None:
            try:
                from keras import models as keras_models
                raw = keras_models.load_model(state.av_checkpoint_path, compile=False)
                real_shape = raw.input_shape[1:]
                av = AVCls(input_shape=real_shape)
                av.model = raw
                state.av_model = av
                logger.info("Loaded AV regressor (shape %s) from %s", real_shape, state.av_checkpoint_path)
            except Exception as e:
                logger.warning("AV regressor load failed: %s", e)

    # Baseline models
    state.baseline_models = BaselineModels()
    baseline_dir = os.path.join(models_dir, "baseline_models")
    if os.path.isdir(baseline_dir):
        state.baseline_models.load_models(baseline_dir)
        logger.info("Loaded baseline models from %s", baseline_dir)


# ---------------------------------------------------------------------------
# Lifespan (startup / shutdown)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_models()
    yield

app = FastAPI(
    title="Swar Manovigyan API",
    description="Emotion-Aware Music Recommendation System",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _estimate_av_from_features(feat: AudioFeatures) -> tuple[float, float]:
    """Compute arousal & enhanced_valence from 9 Spotify-like audio features."""
    loudness_norm = (feat.loudness - LOUDNESS_MIN) / (LOUDNESS_MAX - LOUDNESS_MIN + 1e-8)
    tempo_norm = (feat.tempo - TEMPO_MIN) / (TEMPO_MAX - TEMPO_MIN + 1e-8)
    arousal = 0.4 * feat.energy + 0.3 * loudness_norm + 0.3 * tempo_norm
    enhanced_valence = 0.6 * feat.valence + 0.3 * feat.danceability + 0.1 * (1.0 - feat.acousticness)
    return float(arousal), float(enhanced_valence)


def _quadrant_from_av(arousal: float, valence: float) -> int:
    arousal_bin = 1 if arousal > 0.5 else 0
    valence_bin = 1 if valence > 0.5 else 0
    return arousal_bin * 2 + valence_bin


def _build_sequence_11(feat: AudioFeatures, sequence_length: int = 10) -> np.ndarray:
    """Build (1, sequence_length, 11) from 9 audio features."""
    arousal, enhanced_valence = _estimate_av_from_features(feat)
    vec = np.array([
        feat.acousticness, feat.danceability, feat.energy,
        feat.instrumentalness, feat.liveness, feat.loudness,
        feat.speechiness, feat.tempo, feat.valence,
        arousal, enhanced_valence,
    ], dtype=np.float32)
    seq = np.tile(vec[None, :], (sequence_length, 1))
    return seq[None, ...].astype(np.float32)


def _predict_lstm(sequence_11: np.ndarray):
    """Run LSTM prediction. Returns (label, confidence, probs) or (None,None,None)."""
    if state.lstm_model is None:
        return None, None, None
    try:
        X = sequence_11.copy()
        if state.scaler_lstm is not None:
            orig_shape = X.shape
            flat = X.reshape(-1, 11)
            flat = state.scaler_lstm.transform(flat)
            X = flat.reshape(orig_shape).astype(np.float32)
        predictions = state.lstm_model.predict(X, verbose=0)
        label = int(np.argmax(predictions[0]))
        conf = float(np.max(predictions[0]))
        return label, conf, predictions[0].tolist()
    except Exception as e:
        logger.warning("LSTM prediction error: %s", e)
        return None, None, None


def _predict_baseline(sequence_11: np.ndarray, model_name: str = "random_forest"):
    """Run baseline prediction. Returns (label, confidence, probs) or (None,None,None)."""
    if state.baseline_models is None or model_name not in state.baseline_models.models:
        return None, None, None
    try:
        X = sequence_11.copy()
        if state.scaler_lstm is not None:
            flat = X.reshape(-1, 11)
            flat = state.scaler_lstm.transform(flat)
            X = flat.reshape(1, -1)
        else:
            X = X.reshape(1, -1)
        model = state.baseline_models.models[model_name]
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(X)[0]
            label = int(np.argmax(probabilities))
            conf = float(np.max(probabilities))
            return label, conf, probabilities.tolist()
        label = int(model.predict(X)[0])
        return label, 1.0, None
    except Exception as e:
        logger.warning("Baseline %s prediction error: %s", model_name, e)
        return None, None, None


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_emotion(req: AnalyzeRequest):
    """Analyze emotion from manual A/V, audio features, or uploaded audio."""
    try:
        if req.method == "manual":
            if req.arousal is None or req.valence is None:
                raise HTTPException(400, "arousal and valence required for manual method")
            label = _quadrant_from_av(req.arousal, req.valence)
            return AnalyzeResponse(
                emotion_label=label,
                confidence=1.0,
                arousal=req.arousal,
                valence=req.valence,
            )

        if req.method == "features":
            if req.features is None:
                raise HTTPException(400, "features required for features method")
            seq = _build_sequence_11(req.features)
            arousal, valence = _estimate_av_from_features(req.features)

            label, conf, probs = _predict_lstm(seq)
            if label is None:
                label = _quadrant_from_av(arousal, valence)
                conf = 1.0
                probs = None

            return AnalyzeResponse(
                emotion_label=label,
                confidence=conf,
                arousal=arousal,
                valence=valence,
                probabilities=probs,
            )

        raise HTTPException(400, "upload method requires /analyze/upload endpoint")

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("analyze_emotion failed")
        raise HTTPException(500, str(e))


@app.post("/analyze/upload", response_model=AnalyzeResponse)
async def analyze_upload(file: UploadFile = File(...)):
    """Analyze emotion from an uploaded audio file via AV regressor."""
    try:
        audio_bytes = await file.read()
        extracted_seq, _ = extract_tabular_features_sequence(audio_bytes, sequence_length=10)
        extracted = extracted_seq[0]
        extracted_features = AudioFeatures(
            acousticness=float(extracted[0]),
            danceability=float(extracted[1]),
            energy=float(extracted[2]),
            instrumentalness=float(extracted[3]),
            liveness=float(extracted[4]),
            loudness=float(extracted[5]),
            speechiness=float(extracted[6]),
            tempo=float(extracted[7]),
            valence=float(extracted[8]),
        )
        if state.av_model is None:
            raise HTTPException(503, "AV regressor model not loaded")

        input_shape = state.av_model.model.input_shape
        expected_T = input_shape[1]
        expected_F = input_shape[2]
        ckpt = state.av_checkpoint_path
        feature_stats = load_feature_stats(ckpt) if expected_F == 11 else None

        if expected_F == 128:
            from src.train_av_mel import build_frame_sequences_from_mel
            mel, _, _ = extract_mel_spectrogram_sequence(audio_bytes, target_frames=None)
            sequences = build_frame_sequences_from_mel(mel, sequence_length=expected_T or 10, stride=1)
            n_seqs = min(len(sequences), _MAX_MEL_SEQ_AVG)
            if n_seqs == 0:
                raise ValueError("No sequences could be built from the audio.")
            indices = np.linspace(0, len(sequences) - 1, n_seqs, dtype=int)
            X = sequences[indices]
            preds = state.av_model.predict(X)
            av = preds.mean(axis=0)
        elif expected_F == 11:
            seq, _ = extract_tabular_features_sequence(audio_bytes, sequence_length=expected_T or 10)
            if feature_stats is not None:
                seq = apply_zscore(seq, feature_stats)
            X = seq[None, ...]
            av = state.av_model.predict(X)[0]
        else:
            raise ValueError(f"Unsupported model feature dim: expected_F={expected_F}")

        arousal, valence = float(av[0]), float(av[1])
        label = _quadrant_from_av(arousal, valence)

        return AnalyzeResponse(
            emotion_label=label,
            confidence=1.0,
            arousal=arousal,
            valence=valence,
            features=extracted_features,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("analyze_upload failed")
        raise HTTPException(500, str(e))


@app.post("/recommend", response_model=RecommendResponse)
async def recommend(req: RecommendRequest):
    """Get music recommendations for a detected emotion."""
    try:
        emotion_name = EMOTION_MAPPING.get(req.emotion_label, "Unknown")
        styles = MUSIC_RECOMMENDATIONS.get(req.emotion_label, [])
        blurb: Optional[str] = None
        is_ai = False

        if is_azure_openai_available():
            ai_styles = get_ai_music_styles(
                req.emotion_label, req.arousal, req.valence, emotion_name,
            )
            if ai_styles:
                styles = ai_styles
                is_ai = True
            blurb = get_recommendation_blurb(req.emotion_label, styles)

        tracks: List[TrackOut] = []
        if is_azure_openai_available():
            ai_tracks = get_ai_sample_tracks(
                req.emotion_label, req.arousal, req.valence, emotion_name,
            )
            if ai_tracks:
                is_ai = True
                for idx, t in enumerate(ai_tracks, start=1):
                    tracks.append(TrackOut(
                        id=t.get("id") or f"ai_{req.emotion_label}_{idx}",
                        title=t.get("title", f"Track {idx}"),
                        artist=t.get("artist", "Unknown Artist"),
                        duration=t.get("duration", "3:30"),
                    ))

        if not tracks:
            for t in FALLBACK_TRACKS.get(req.emotion_label, []):
                tracks.append(TrackOut(**t))

        return RecommendResponse(
            styles=styles,
            blurb=blurb,
            tracks=tracks,
            isAiGenerated=is_ai,
        )

    except Exception as e:
        logger.exception("recommend failed")
        raise HTTPException(500, str(e))


@app.post("/api/analyze/insight", response_model=InsightResponse)
async def analyze_insight(req: InsightRequest):
    """Get a therapeutic AI insight for the detected emotion."""
    try:
        emotion_name = EMOTION_MAPPING.get(req.emotion_label, "Unknown")
        text = get_therapeutic_explanation(
            req.emotion_label, req.arousal, req.valence, emotion_name,
        )
        if not text:
            raise HTTPException(503, "Azure OpenAI not configured or returned no result")
        return InsightResponse(text=text)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("analyze_insight failed")
        raise HTTPException(500, str(e))


@app.post("/api/analyze/models", response_model=List[ModelPrediction])
async def analyze_models(req: ModelsRequest):
    """Get model comparison predictions from all available models."""
    try:
        seq = _build_sequence_11(req.features)
        results: List[ModelPrediction] = []

        label, conf, _ = _predict_lstm(seq)
        if label is not None:
            results.append(ModelPrediction(
                model="LSTM",
                emotion=EMOTION_MAPPING[label],
                confidence=conf,
            ))

        if state.baseline_models and state.baseline_models.models:
            for name in ["random_forest", "logistic_regression", "svm", "mlp"]:
                if name in state.baseline_models.models:
                    bl, bc, _ = _predict_baseline(seq, name)
                    if bl is not None:
                        results.append(ModelPrediction(
                            model=name.replace("_", " ").title(),
                            emotion=EMOTION_MAPPING[bl],
                            confidence=bc,
                        ))

        return results

    except Exception as e:
        logger.exception("analyze_models failed")
        raise HTTPException(500, str(e))


@app.get("/api/status", response_model=StatusResponse)
async def get_status():
    """Return which services / models are currently loaded."""
    return StatusResponse(
        azureOpenAI=is_azure_openai_available(),
        azureSpeech=is_azure_speech_available(),
        lstmModel=state.lstm_model is not None,
        avRegressor=state.av_model is not None,
        baselineModels=bool(
            state.baseline_models
            and getattr(state.baseline_models, "models", None)
        ),
    )


@app.post("/tts")
async def text_to_speech(req: TTSRequest):
    """Synthesize text-to-speech and return WAV bytes."""
    try:
        if not is_azure_speech_available():
            raise HTTPException(503, "Azure Speech not available")
        wav_bytes = synthesize_text_to_wav_bytes(req.text)
        if wav_bytes is None:
            raise HTTPException(502, "Speech synthesis returned no data")
        return Response(content=wav_bytes, media_type="audio/wav")
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("tts failed")
        raise HTTPException(500, str(e))


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.backend.server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=[_SRC_DIR],
    )
