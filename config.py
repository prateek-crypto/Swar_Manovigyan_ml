"""
Configuration file for the Emotion-Aware Music Recommendation System
"""

import os

# Data paths
DATA_DIR = 'data'
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
MODELS_DIR = 'models'

# File paths
SPOTIFY_FEATURES_FILE = os.path.join(RAW_DATA_DIR, 'SpotifyFeatures.csv')
PROCESSED_FEATURES_FILE = os.path.join(PROCESSED_DATA_DIR, 'spotify_features_with_emotions.csv')

# Model paths
LSTM_MODEL_PATH = os.path.join(MODELS_DIR, 'lstm_emotion_model.h5')
CNN_MODEL_PATH = os.path.join(MODELS_DIR, 'cnn1d_emotion_model.h5')
BASELINE_MODELS_DIR = os.path.join(MODELS_DIR, 'baseline_models')
TRAINING_RESULTS_PATH = os.path.join(MODELS_DIR, 'training_results.json')

# Model parameters
SEQUENCE_LENGTH = 10
NUM_CLASSES = 4
LSTM_UNITS = 64
DROPOUT_RATE = 0.3
BATCH_SIZE = 32
EPOCHS = 50

# Emotion mapping
EMOTION_MAPPING = {
    0: 'Low Arousal, Negative Valence',  # Sad, Depressed
    1: 'Low Arousal, Positive Valence',  # Calm, Peaceful
    2: 'High Arousal, Negative Valence', # Angry, Stressed
    3: 'High Arousal, Positive Valence'  # Happy, Excited
}

EMOTION_DESCRIPTIONS = {
    0: 'Feeling sad, depressed, or melancholic. You might benefit from gentle, soothing music.',
    1: 'Feeling calm, peaceful, or relaxed. Perfect for meditation or quiet activities.',
    2: 'Feeling angry, stressed, or anxious. You might need energetic music to channel emotions.',
    3: 'Feeling happy, excited, or energetic. Great for dancing or celebrating!'
}

# Music recommendations by emotion
MUSIC_RECOMMENDATIONS = {
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

# Audio features used for emotion classification
AUDIO_FEATURES = [
    'acousticness', 'danceability', 'energy', 'instrumentalness',
    'liveness', 'loudness', 'speechiness', 'tempo', 'valence'
]

# Feature weights for arousal and valence calculation
AROUSAL_WEIGHTS = [0.4, 0.3, 0.3]  # energy, loudness, tempo
VALENCE_WEIGHTS = [0.6, 0.3, 0.1]  # valence, danceability, acousticness

# Training parameters
TRAINING_PARAMS = {
    'test_size': 0.2,
    'val_size': 0.2,
    'random_state': 42,
    'max_samples_per_class': 2000,
    'early_stopping_patience': 10,
    'reduce_lr_patience': 5,
    'min_lr': 1e-7
}

# Streamlit app configuration
STREAMLIT_CONFIG = {
    'page_title': 'Emotion-Aware Music Recommendation',
    'page_icon': '🎵',
    'layout': 'wide'
}

# Visualization settings
PLOT_SETTINGS = {
    'figsize': (12, 8),
    'dpi': 300,
    'style': 'whitegrid',
    'palette': 'viridis'
}
