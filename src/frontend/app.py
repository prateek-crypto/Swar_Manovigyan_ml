"""
Streamlit Frontend for Emotion-Aware Music Recommendation System
"""

import streamlit as st

# Set page config FIRST - this must be the first Streamlit command
st.set_page_config(
    page_title="Emotion-Aware Music Recommendation",
    page_icon="🎵",
    layout="wide"
)

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
# TensorFlow is optional; the app should work with baseline models only
try:
    import tensorflow as tf  # type: ignore
except Exception:
    tf = None  # fallback to baseline-only mode
import joblib
import os
import sys
from datetime import datetime
import random

# Add src to path for imports
_app_dir = os.path.dirname(os.path.abspath(__file__))
_src_dir = os.path.dirname(_app_dir)
sys.path.insert(0, _src_dir)
PROJECT_ROOT = os.path.dirname(_src_dir)

from models.lstm_model import EmotionLSTM
from models.baseline_models import BaselineModels
from models.av_regressor import AVLSTMRegressor
from utils.audio_features import extract_mel_spectrogram_sequence, extract_tabular_features_sequence
from utils.feature_stats import load_feature_stats, apply_zscore
try:
    from utils.azure_openai_service import (
        is_azure_openai_available,
        get_therapeutic_explanation,
        get_recommendation_blurb,
        get_ai_music_styles,
        get_ai_sample_tracks,
    )
except Exception:
    is_azure_openai_available = lambda: False
    get_therapeutic_explanation = lambda *a, **k: None
    get_recommendation_blurb = lambda *a, **k: None
    get_ai_music_styles = lambda *a, **k: None
    get_ai_sample_tracks = lambda *a, **k: None

def _models_dir():
    return os.path.join(PROJECT_ROOT, "models")

class MusicRecommendationApp:
    """
    Streamlit app for emotion-aware music recommendation
    """
    
    def __init__(self):
        self.emotion_mapping = {
            0: 'Low Arousal, Negative Valence',  # Sad, Depressed
            1: 'Low Arousal, Positive Valence',  # Calm, Peaceful
            2: 'High Arousal, Negative Valence', # Angry, Stressed
            3: 'High Arousal, Positive Valence'  # Happy, Excited
        }
        
        self.emotion_descriptions = {
            0: 'Feeling sad, depressed, or melancholic. You might benefit from gentle, soothing music.',
            1: 'Feeling calm, peaceful, or relaxed. Perfect for meditation or quiet activities.',
            2: 'Feeling angry, stressed, or anxious. You might need energetic music to channel emotions.',
            3: 'Feeling happy, excited, or energetic. Great for dancing or celebrating!'
        }
        
        self.music_recommendations = {
            0: [  # Low Arousal, Negative Valence - Sad/Depressed
                "Sad ballads", "Slow acoustic songs", "Melancholic classical music",
                "Indie folk", "Blues", "Ambient music", "Piano solos"
            ],
            1: [  # Low Arousal, Positive Valence - Calm/Peaceful
                "Meditation music", "Nature sounds", "Soft jazz", "Ambient",
                "Classical music", "Folk music", "Acoustic guitar"
            ],
            2: [  # High Arousal, Negative Valence - Angry/Stressed
                "Heavy metal", "Punk rock", "Hard rock", "Rap",
                "Electronic", "Industrial", "Alternative rock"
            ],
            3: [  # High Arousal, Positive Valence - Happy/Excited
                "Pop music", "Dance music", "Upbeat rock", "Funk",
                "Disco", "Reggae", "Ska", "Happy songs"
            ]
        }
        
        self.load_models()
    
    def load_models(self):
        """Load trained models (paths relative to project root)."""
        models_dir = _models_dir()
        self.av_checkpoint_path = os.path.join(models_dir, "av_regressor.keras")
        self.scaler_lstm = None

        try:
            # LSTM (classification)
            lstm_path = os.path.join(models_dir, "lstm_emotion_model.keras")
            if not os.path.isfile(lstm_path):
                lstm_path = os.path.join(models_dir, "lstm_emotion_model.h5")
            if tf is not None and os.path.isfile(lstm_path):
                self.lstm_model = tf.keras.models.load_model(lstm_path, compile=False)
                st.success("LSTM (classification) loaded!")
            else:
                self.lstm_model = None
                if tf is None:
                    st.info("TensorFlow not available. Running without LSTM classifier.")

            # Scaler for LSTM and baselines (11-dim → scaled; same as training)
            scaler_path = os.path.join(models_dir, "scaler_lstm.joblib")
            if os.path.isfile(scaler_path):
                self.scaler_lstm = joblib.load(scaler_path)

            # AV regressor
            if tf is not None and os.path.isfile(self.av_checkpoint_path):
                self.av_model = AVLSTMRegressor(input_shape=(10, 11))
                self.av_model.load(self.av_checkpoint_path)
                st.success("AV LSTM regressor loaded!")
            else:
                self.av_model = None
                st.info("AV regressor not found. Train with `python -m src.train_av`.")

            # Baselines (expect 110-dim flattened sequence)
            self.baseline_models = BaselineModels()
            baseline_dir = os.path.join(models_dir, "baseline_models")
            if os.path.isdir(baseline_dir):
                self.baseline_models.load_models(baseline_dir)
            else:
                st.info("Baseline models not found. Run classification pipeline to train baselines.")

        except Exception as e:
            st.error(f"Error loading models: {str(e)}")
            self.lstm_model = None
            self.baseline_models = None
            self.av_model = None
    
    def create_emotion_input_form(self):
        """Create form for manual emotion input"""
        st.subheader("🎭 How are you feeling?")
        
        col1, col2 = st.columns(2)
        
        with col1:
            arousal = st.slider(
                "Energy Level (Arousal)",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.1,
                help="Low: Calm, relaxed | High: Energetic, excited"
            )
        
        with col2:
            valence = st.slider(
                "Mood Positivity (Valence)",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.1,
                help="Low: Negative emotions | High: Positive emotions"
            )
        
        # Determine emotion quadrant
        arousal_binary = 1 if arousal > 0.5 else 0
        valence_binary = 1 if valence > 0.5 else 0
        emotion_label = arousal_binary * 2 + valence_binary
        
        return emotion_label, arousal, valence

    def estimate_av_from_features(self, features: np.ndarray) -> tuple[float, float]:
        """Estimate arousal/enhanced_valence from slider features (same logic as preprocessing)."""
        energy = float(features[0, 2])
        loudness = float(features[0, 5])  # [-60, 0]
        tempo = float(features[0, 7])     # [50, 200]
        loudness_norm = (loudness - (-60.0)) / (0.0 - (-60.0) + 1e-8)
        tempo_norm = (tempo - 50.0) / (200.0 - 50.0 + 1e-8)
        arousal = 0.4 * energy + 0.3 * loudness_norm + 0.3 * tempo_norm
        valence = float(features[0, 8])
        danceability = float(features[0, 1])
        acousticness = float(features[0, 0])
        enhanced_valence = 0.6 * valence + 0.3 * danceability + 0.1 * (1.0 - acousticness)
        return arousal, enhanced_valence

    def build_sequence_11_from_form(self, features_9: np.ndarray, sequence_length: int = 10) -> np.ndarray:
        """
        Build (1, sequence_length, 11) from form's 9 features.
        Order: acousticness, danceability, energy, instrumentalness, liveness, loudness, speechiness, tempo, valence, arousal, enhanced_valence.
        """
        arousal, enhanced_valence = self.estimate_av_from_features(features_9)
        # Single 11-dim vector
        vec = np.array([
            float(features_9[0, 0]), float(features_9[0, 1]), float(features_9[0, 2]),
            float(features_9[0, 3]), float(features_9[0, 4]), float(features_9[0, 5]),
            float(features_9[0, 6]), float(features_9[0, 7]), float(features_9[0, 8]),
            arousal, enhanced_valence,
        ], dtype=np.float32)
        seq = np.tile(vec[None, :], (sequence_length, 1))  # (10, 11)
        return seq[None, ...].astype(np.float32)  # (1, 10, 11)
    
    def create_audio_features_form(self):
        """Create form for audio features input"""
        st.subheader("🎵 Audio Features Input")
        
        col1, col2 = st.columns(2)
        
        with col1:
            acousticness = st.slider("Acousticness", 0.0, 1.0, 0.5, 0.01)
            danceability = st.slider("Danceability", 0.0, 1.0, 0.5, 0.01)
            energy = st.slider("Energy", 0.0, 1.0, 0.5, 0.01)
            instrumentalness = st.slider("Instrumentalness", 0.0, 1.0, 0.0, 0.01)
            liveness = st.slider("Liveness", 0.0, 1.0, 0.1, 0.01)
        
        with col2:
            loudness = st.slider("Loudness", -60.0, 0.0, -20.0, 1.0)
            speechiness = st.slider("Speechiness", 0.0, 1.0, 0.1, 0.01)
            tempo = st.slider("Tempo (BPM)", 50.0, 200.0, 120.0, 1.0)
            valence = st.slider("Valence", 0.0, 1.0, 0.5, 0.01)
        
        # Create feature vector
        features = np.array([[
            acousticness, danceability, energy, instrumentalness,
            liveness, loudness, speechiness, tempo, valence
        ]])
        
        return features
    
    def predict_emotion_lstm(self, sequence_11: np.ndarray):
        """
        Predict emotion using LSTM model.
        sequence_11: (1, 10, 11) with same scaling as training (apply scaler_lstm if available).
        """
        if self.lstm_model is None:
            return None, None, None
        try:
            X = sequence_11  # (1, 10, 11)
            if self.scaler_lstm is not None:
                # Scaler expects (n_samples, n_features); reshape (1,10,11) -> (10,11), transform, back
                flat = X.reshape(-1, 11)
                flat = self.scaler_lstm.transform(flat)
                X = flat.reshape(1, 10, 11).astype(np.float32)
            predictions = self.lstm_model.predict(X)
            emotion_label = int(np.argmax(predictions[0]))
            confidence = float(np.max(predictions[0]))
            return emotion_label, confidence, predictions[0]
        except Exception as e:
            st.error(f"Error in LSTM prediction: {str(e)}")
            return None, None, None
    
    def predict_av_from_audio(self, audio_bytes: bytes, checkpoint_path: str | None = None):
        """Predict A/V from audio; uses feature_stats for z-score when checkpoint_path has feature_stats_av.json."""
        if self.av_model is None:
            return None
        ckpt = checkpoint_path or getattr(self, "av_checkpoint_path", None)
        if not ckpt:
            ckpt = os.path.join(_models_dir(), "av_regressor.keras")
        try:
            input_shape = self.av_model.model.input_shape
            expected_T = input_shape[1]
            expected_F = input_shape[2]
            feature_stats = load_feature_stats(ckpt) if expected_F == 11 else None

            if expected_F == 128:
                # Mel-spectrogram path: extract full mel, then build frame-level sequence
                from src.train_av_mel import build_frame_sequences_from_mel
                mel, _, _ = extract_mel_spectrogram_sequence(audio_bytes, target_frames=None)  # Get full length
                # Build sequence using sliding window (same as training)
                sequences = build_frame_sequences_from_mel(mel, sequence_length=expected_T or 10, stride=1)
                # Use the first sequence (or average multiple sequences for robustness)
                X = sequences[0:1]  # (1, T, 128) - use first sequence
            elif expected_F == 11:
                seq, _ = extract_tabular_features_sequence(audio_bytes, sequence_length=expected_T or 10)
                if feature_stats is not None:
                    seq = apply_zscore(seq, feature_stats)
                X = seq[None, ...]
            else:
                raise ValueError(f"Unsupported model feature dim: expected_F={expected_F}")

            av = self.av_model.predict(X)[0]
            return float(av[0]), float(av[1])
        except Exception as e:
            st.error(f"Error in AV prediction: {str(e)}")
            return None
    
    def predict_emotion_baseline(self, sequence_11: np.ndarray, model_name='random_forest'):
        """
        Predict emotion using baseline models.
        sequence_11: (1, 10, 11); will be scaled with scaler_lstm then flattened to (1, 110).
        """
        if self.baseline_models is None or model_name not in self.baseline_models.models:
            return None, None, None
        try:
            X = sequence_11  # (1, 10, 11)
            if self.scaler_lstm is not None:
                flat = X.reshape(-1, 11)
                flat = self.scaler_lstm.transform(flat)
                X = flat.reshape(1, -1)  # (1, 110)
            else:
                X = X.reshape(1, -1)
            model = self.baseline_models.models[model_name]
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(X)[0]
                emotion_label = int(np.argmax(probabilities))
                confidence = float(np.max(probabilities))
            else:
                emotion_label = int(model.predict(X)[0])
                confidence = 1.0
                probabilities = None
            return emotion_label, confidence, probabilities
        except Exception as e:
            st.error(f"Error in {model_name} prediction: {str(e)}")
            return None, None, None
    
    def display_emotion_results(self, emotion_label, confidence, arousal=None, valence=None):
        """Display emotion prediction results"""
        st.subheader("🎯 Emotion Analysis Results")
        emotion_name = self.emotion_mapping[emotion_label]
        emotion_desc = self.emotion_descriptions[emotion_label]
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown(f"**Detected Emotion:** {emotion_name}")
            st.markdown(f"**Description:** {emotion_desc}")
            st.markdown(f"**Confidence:** {confidence:.2%}")
            # Optional Azure OpenAI therapeutic explanation
            if arousal is not None and valence is not None and is_azure_openai_available():
                with st.spinner("Generating insight..."):
                    ai_text = get_therapeutic_explanation(emotion_label, arousal, valence, emotion_name)
                if ai_text:
                    st.info(f"💡 **AI insight:** {ai_text}")
        
        with col2:
            # Create emotion visualization
            fig = go.Figure()
            
            # Add arousal-valence point
            if arousal is not None and valence is not None:
                fig.add_trace(go.Scatter(
                    x=[arousal], y=[valence],
                    mode='markers',
                    marker=dict(size=20, color='red', symbol='star'),
                    name='Your Mood'
                ))
            
            # Add quadrant labels
            fig.add_annotation(x=0.25, y=0.25, text="Sad\nDepressed", showarrow=False, font=dict(size=10))
            fig.add_annotation(x=0.75, y=0.25, text="Calm\nPeaceful", showarrow=False, font=dict(size=10))
            fig.add_annotation(x=0.25, y=0.75, text="Angry\nStressed", showarrow=False, font=dict(size=10))
            fig.add_annotation(x=0.75, y=0.75, text="Happy\nExcited", showarrow=False, font=dict(size=10))
            
            fig.update_layout(
                title="Arousal-Valence Space",
                xaxis_title="Arousal (Energy)",
                yaxis_title="Valence (Positivity)",
                xaxis=dict(range=[0, 1]),
                yaxis=dict(range=[0, 1]),
                width=300,
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def display_music_recommendations(self, emotion_label: int, arousal: float | None = None, valence: float | None = None):
        """Display music recommendations based on emotion.

        If Azure OpenAI is available, prefers AI-generated styles and sample tracks,
        with hardcoded defaults as a safe fallback.
        """
        st.subheader("🎵 Music Recommendations")
        emotion_name = self.emotion_mapping.get(emotion_label, "Unknown Emotion")

        # Base (hardcoded) recommendations as fallback
        base_recommendations = self.music_recommendations.get(emotion_label, [])

        # Prefer AI-generated style / playlist suggestions when available
        recommendations = base_recommendations
        if is_azure_openai_available():
            with st.spinner("Asking AI for tailored music styles..."):
                ai_styles = get_ai_music_styles(
                    emotion_label,
                    arousal if arousal is not None else 0.5,
                    valence if valence is not None else 0.5,
                    emotion_name,
                )
            if ai_styles:
                recommendations = ai_styles

        # Optional Azure OpenAI blurb tying the styles to the emotion
        if is_azure_openai_available() and recommendations:
            with st.spinner("Generating recommendation insight..."):
                blurb = get_recommendation_blurb(emotion_label, recommendations)
            if blurb:
                st.caption(f"💡 {blurb}")

        for i, rec in enumerate(recommendations, 1):
            st.markdown(f"{i}. {rec}")
        
        # Create sample playlist (AI first, fallback to demo tracks)
        st.subheader("🎧 Sample Playlist")

        sample_tracks = None
        if is_azure_openai_available():
            with st.spinner("Asking AI for example tracks..."):
                ai_tracks = get_ai_sample_tracks(
                    emotion_label,
                    arousal if arousal is not None else 0.5,
                    valence if valence is not None else 0.5,
                    emotion_name,
                )
            if ai_tracks:
                # Ensure each track has an id for Streamlit buttons
                sample_tracks = []
                for idx, t in enumerate(ai_tracks, start=1):
                    track_id = t.get("id") or f"ai_{emotion_label}_{idx}"
                    sample_tracks.append(
                        {
                            "title": t.get("title", f"Track {idx}"),
                            "artist": t.get("artist", "Unknown Artist"),
                            "duration": t.get("duration", "3:00"),
                            "id": track_id,
                        }
                    )

        # Fallback to demo playlist if AI is unavailable or failed
        if sample_tracks is None:
            sample_tracks = self.generate_sample_tracks(emotion_label)
        
        for track in sample_tracks:
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                st.write(f"🎵 {track['title']} - {track['artist']}")
            with col2:
                st.write(f"⏱️ {track['duration']}")
            with col3:
                if st.button("▶️", key=f"play_{track['id']}"):
                    st.success("Playing... (Demo)")
    
    def generate_sample_tracks(self, emotion_label):
        """Generate sample tracks based on emotion"""
        # This is a demo - in a real app, you'd query a music database
        sample_data = {
            0: [  # Sad/Depressed
                {"title": "Hurt", "artist": "Johnny Cash", "duration": "3:38", "id": "sad1"},
                {"title": "Mad World", "artist": "Gary Jules", "duration": "3:09", "id": "sad2"},
                {"title": "The Sound of Silence", "artist": "Simon & Garfunkel", "duration": "3:05", "id": "sad3"}
            ],
            1: [  # Calm/Peaceful
                {"title": "Weightless", "artist": "Marconi Union", "duration": "8:59", "id": "calm1"},
                {"title": "Clair de Lune", "artist": "Claude Debussy", "duration": "4:56", "id": "calm2"},
                {"title": "Meditation", "artist": "Massive Attack", "duration": "5:20", "id": "calm3"}
            ],
            2: [  # Angry/Stressed
                {"title": "Break Stuff", "artist": "Limp Bizkit", "duration": "2:46", "id": "angry1"},
                {"title": "Killing in the Name", "artist": "Rage Against the Machine", "duration": "5:13", "id": "angry2"},
                {"title": "Bodies", "artist": "Drowning Pool", "duration": "3:21", "id": "angry3"}
            ],
            3: [  # Happy/Excited
                {"title": "Happy", "artist": "Pharrell Williams", "duration": "3:53", "id": "happy1"},
                {"title": "Don't Stop Me Now", "artist": "Queen", "duration": "3:29", "id": "happy2"},
                {"title": "Good Vibrations", "artist": "The Beach Boys", "duration": "3:37", "id": "happy3"}
            ]
        }
        
        return sample_data.get(emotion_label, [])
    
    def display_model_comparison(self, sequence_11: np.ndarray):
        """Display predictions from all models. sequence_11: (1, 10, 11) from build_sequence_11_from_form."""
        st.subheader("🤖 Model Predictions Comparison")
        predictions = {}
        if self.lstm_model:
            lstm_emotion, lstm_conf, _ = self.predict_emotion_lstm(sequence_11)
            if lstm_emotion is not None:
                predictions['LSTM'] = {
                    'emotion': self.emotion_mapping[lstm_emotion],
                    'confidence': lstm_conf
                }
        if self.baseline_models and getattr(self.baseline_models, 'models', None):
            for model_name in ['random_forest', 'logistic_regression', 'svm', 'mlp']:
                if model_name in self.baseline_models.models:
                    emotion, conf, _ = self.predict_emotion_baseline(sequence_11, model_name)
                    if emotion is not None:
                        predictions[model_name.replace('_', ' ').title()] = {
                            'emotion': self.emotion_mapping[emotion],
                            'confidence': conf
                        }
        
        # Display comparison table
        if predictions:
            comparison_data = []
            for model, result in predictions.items():
                comparison_data.append({
                    'Model': model,
                    'Predicted Emotion': result['emotion'],
                    'Confidence': f"{result['confidence']:.2%}"
                })
            
            df = pd.DataFrame(comparison_data)
            st.dataframe(df, use_container_width=True)
        else:
            st.warning("No models available for comparison.")
    
    def run(self):
        """Run the Streamlit app"""
        st.title("🎵 Emotion-Aware Music Recommendation System")
        st.markdown("---")
        
        # Sidebar
        st.sidebar.title("🎛️ Controls")
        
        input_method = st.sidebar.radio(
            "Choose Input Method:",
            ["Manual Emotion Selection", "Audio Features Input", "Audio Upload (A/V)"]
        )
        
        # Main content
        if input_method == "Manual Emotion Selection":
            emotion_label, arousal, valence = self.create_emotion_input_form()
            
            # Display results
            self.display_emotion_results(emotion_label, 1.0, arousal, valence)
            self.display_music_recommendations(emotion_label, arousal, valence)
        
        elif input_method == "Audio Features Input":
            features = self.create_audio_features_form()
            sequence_11 = self.build_sequence_11_from_form(features)

            if st.button("🔮 Predict Emotion", type="primary"):
                with st.spinner("Analyzing your mood..."):
                    if self.lstm_model is not None:
                        lstm_emotion, lstm_conf, lstm_probs = self.predict_emotion_lstm(sequence_11)
                        if lstm_emotion is not None:
                            arousal, valence = self.estimate_av_from_features(features)
                            self.display_emotion_results(lstm_emotion, lstm_conf, arousal, valence)
                            self.display_music_recommendations(lstm_emotion, arousal, valence)
                            self.display_model_comparison(sequence_11)
                        else:
                            st.error("Unable to make prediction with LSTM.")
                    else:
                        arousal, valence = self.estimate_av_from_features(features)
                        arousal_bin = 1 if arousal > 0.5 else 0
                        valence_bin = 1 if valence > 0.5 else 0
                        emotion_label = arousal_bin * 2 + valence_bin
                        self.display_emotion_results(emotion_label, 1.0, arousal, valence)
                        self.display_music_recommendations(emotion_label, arousal, valence)
                        if getattr(self.baseline_models, 'models', None):
                            self.display_model_comparison(sequence_11)
        
        else:  # Audio Upload (A/V)
            st.subheader("🎙️ Upload Audio (wav/mp3)")
            uploaded = st.file_uploader("Upload an audio file", type=["wav", "mp3", "flac", "ogg"])
            models_dir = _models_dir()
            default_ckpt = os.path.join(models_dir, "av_regressor.keras")
            ckpt = st.text_input("Checkpoint path (optional)", value=default_ckpt)
            if uploaded is not None:
                audio_bytes = uploaded.read()
                st.audio(audio_bytes)
                if st.button("🎯 Predict Arousal–Valence", type="primary"):
                    with st.spinner("Extracting features and predicting..."):
                        if ckpt and os.path.isfile(ckpt) and tf is not None and (self.av_model is None or getattr(self, "_last_ckpt", None) != ckpt):
                            try:
                                self.av_model = AVLSTMRegressor(input_shape=(10, 11))
                                self.av_model.load(ckpt)
                                self.av_checkpoint_path = ckpt
                                self._last_ckpt = ckpt
                                st.success("Loaded AV checkpoint.")
                            except Exception as e:
                                st.error(f"Failed to load checkpoint: {e}")
                        av = self.predict_av_from_audio(audio_bytes, checkpoint_path=ckpt or None)
                        if av is not None:
                            arousal, valence = av
                            arousal_binary = 1 if arousal > 0.5 else 0
                            valence_binary = 1 if valence > 0.5 else 0
                            emotion_label = arousal_binary * 2 + valence_binary
                            self.display_emotion_results(emotion_label, 1.0, arousal, valence)
                            self.display_music_recommendations(emotion_label, arousal, valence)
                        else:
                            st.warning("AV model not available. Please train or provide a checkpoint.")
        
        # Footer
        st.markdown("---")
        st.markdown(
            "Built by Adit Jain, Mehir Singh and Ayush Pandey"
        )


def main():
    """Main function to run the app"""
    app = MusicRecommendationApp()
    app.run()

if __name__ == "__main__":
    main()
