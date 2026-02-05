"""
Tests for data analysis and emotion labeling.
"""

import pytest
import pandas as pd
import numpy as np
from src.utils.data_analysis import EmotionLabeler


@pytest.fixture
def sample_spotify_df():
    """Create a sample DataFrame with Spotify features."""
    np.random.seed(42)
    n_samples = 100
    df = pd.DataFrame({
        'acousticness': np.random.uniform(0, 1, n_samples),
        'danceability': np.random.uniform(0, 1, n_samples),
        'energy': np.random.uniform(0, 1, n_samples),
        'instrumentalness': np.random.uniform(0, 1, n_samples),
        'liveness': np.random.uniform(0, 1, n_samples),
        'loudness': np.random.uniform(-60, 0, n_samples),
        'speechiness': np.random.uniform(0, 1, n_samples),
        'tempo': np.random.uniform(50, 200, n_samples),
        'valence': np.random.uniform(0, 1, n_samples),
    })
    return df


def test_emotion_labeler_create_labels(sample_spotify_df):
    """Test emotion label creation."""
    labeler = EmotionLabeler()
    df_labeled = labeler.create_emotion_labels(sample_spotify_df.copy())
    
    assert 'arousal' in df_labeled.columns
    assert 'enhanced_valence' in df_labeled.columns
    assert 'arousal_binary' in df_labeled.columns
    assert 'valence_binary' in df_labeled.columns
    assert 'emotion_label' in df_labeled.columns
    
    # Check arousal and enhanced_valence are in [0,1]
    assert np.all(df_labeled['arousal'] >= 0) and np.all(df_labeled['arousal'] <= 1)
    assert np.all(df_labeled['enhanced_valence'] >= 0) and np.all(df_labeled['enhanced_valence'] <= 1)
    
    # Check emotion_label is in [0, 3]
    assert np.all(df_labeled['emotion_label'] >= 0) and np.all(df_labeled['emotion_label'] <= 3)


def test_emotion_labeler_fixed_ranges(sample_spotify_df):
    """Test that fixed loudness and tempo ranges are used."""
    labeler = EmotionLabeler()
    df = sample_spotify_df.copy()
    
    # Set some values outside fixed ranges
    df.loc[0, 'loudness'] = -70  # Below -60
    df.loc[1, 'loudness'] = 10   # Above 0
    df.loc[2, 'tempo'] = 30      # Below 50
    df.loc[3, 'tempo'] = 250     # Above 200
    
    df_labeled = labeler.create_emotion_labels(df)
    
    # Values should be clipped to fixed ranges for normalization
    assert df_labeled.loc[0, 'arousal'] >= 0  # Should still compute arousal
    assert df_labeled.loc[1, 'arousal'] >= 0
    assert df_labeled.loc[2, 'arousal'] >= 0
    assert df_labeled.loc[3, 'arousal'] >= 0


def test_emotion_labeler_quadrant_mapping(sample_spotify_df):
    """Test that emotion labels map to correct quadrants."""
    labeler = EmotionLabeler()
    df_labeled = labeler.create_emotion_labels(sample_spotify_df.copy())
    
    # Check quadrant mapping
    # Quadrant 0: Low Arousal, Negative Valence
    low_arousal_neg_val = df_labeled[
        (df_labeled['arousal_binary'] == 0) & (df_labeled['valence_binary'] == 0)
    ]
    assert np.all(low_arousal_neg_val['emotion_label'] == 0)
    
    # Quadrant 1: Low Arousal, Positive Valence
    low_arousal_pos_val = df_labeled[
        (df_labeled['arousal_binary'] == 0) & (df_labeled['valence_binary'] == 1)
    ]
    assert np.all(low_arousal_pos_val['emotion_label'] == 1)
    
    # Quadrant 2: High Arousal, Negative Valence
    high_arousal_neg_val = df_labeled[
        (df_labeled['arousal_binary'] == 1) & (df_labeled['valence_binary'] == 0)
    ]
    assert np.all(high_arousal_neg_val['emotion_label'] == 2)
    
    # Quadrant 3: High Arousal, Positive Valence
    high_arousal_pos_val = df_labeled[
        (df_labeled['arousal_binary'] == 1) & (df_labeled['valence_binary'] == 1)
    ]
    assert np.all(high_arousal_pos_val['emotion_label'] == 3)


def test_emotion_labeler_analyze_distribution(sample_spotify_df):
    """Test emotion distribution analysis."""
    labeler = EmotionLabeler()
    df_labeled = labeler.create_emotion_labels(sample_spotify_df.copy())
    
    counts = labeler.analyze_emotion_distribution(df_labeled)
    
    assert len(counts) <= 4, "Should have at most 4 emotion classes"
    assert all(idx in [0, 1, 2, 3] for idx in counts.index), "Emotion labels should be 0-3"
    assert counts.sum() == len(df_labeled), "Total counts should match number of samples"
