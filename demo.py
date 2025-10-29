"""
Demo script for the Emotion-Aware Music Recommendation System
Shows how to use the trained models for emotion prediction
"""

import numpy as np
import pandas as pd
import sys
import os

# Add src to path
sys.path.append('src')

from src.models.lstm_model import EmotionLSTM
from src.models.baseline_models import BaselineModels
from config import EMOTION_MAPPING, EMOTION_DESCRIPTIONS, MUSIC_RECOMMENDATIONS

def create_sample_features():
    """Create sample audio features for demonstration"""
    # Sample features representing different emotional states
    samples = {
        'Sad Song': {
            'acousticness': 0.8,
            'danceability': 0.3,
            'energy': 0.2,
            'instrumentalness': 0.1,
            'liveness': 0.1,
            'loudness': -20.0,
            'speechiness': 0.05,
            'tempo': 80.0,
            'valence': 0.2
        },
        'Happy Song': {
            'acousticness': 0.3,
            'danceability': 0.8,
            'energy': 0.9,
            'instrumentalness': 0.0,
            'liveness': 0.2,
            'loudness': -5.0,
            'speechiness': 0.1,
            'tempo': 140.0,
            'valence': 0.9
        },
        'Calm Song': {
            'acousticness': 0.9,
            'danceability': 0.4,
            'energy': 0.1,
            'instrumentalness': 0.8,
            'liveness': 0.05,
            'loudness': -25.0,
            'speechiness': 0.02,
            'tempo': 60.0,
            'valence': 0.6
        },
        'Angry Song': {
            'acousticness': 0.1,
            'danceability': 0.6,
            'energy': 0.95,
            'instrumentalness': 0.0,
            'liveness': 0.3,
            'loudness': -2.0,
            'speechiness': 0.2,
            'tempo': 180.0,
            'valence': 0.1
        }
    }
    
    return samples

def predict_emotion_demo():
    """Demonstrate emotion prediction with sample data"""
    print("="*60)
    print("EMOTION-AWARE MUSIC RECOMMENDATION - DEMO")
    print("="*60)
    
    # Load models
    print("\nLoading models...")
    try:
        # Load LSTM model
        lstm_model = EmotionLSTM(input_shape=(10, 11))
        if os.path.exists('models/lstm_emotion_model.h5'):
            lstm_model.load_model('models/lstm_emotion_model.h5')
            print("✓ LSTM model loaded")
        else:
            print("⚠ LSTM model not found. Please train the model first.")
            lstm_model = None
        
        # Load baseline models
        baseline_models = BaselineModels()
        baseline_models.load_models('models/baseline_models')
        print("✓ Baseline models loaded")
        
    except Exception as e:
        print(f"Error loading models: {e}")
        print("Please run 'python train_model.py' first to train the models.")
        return
    
    # Create sample features
    samples = create_sample_features()
    
    print(f"\nTesting with {len(samples)} sample songs...")
    print("-" * 60)
    
    for song_name, features in samples.items():
        print(f"\n🎵 {song_name}")
        print("-" * 30)
        
        # Convert to numpy array
        feature_array = np.array([list(features.values())])
        
        # Calculate arousal and valence
        arousal = (features['energy'] + (features['loudness'] + 60) / 60 + features['tempo'] / 200) / 3
        valence = (features['valence'] + features['danceability'] + (1 - features['acousticness'])) / 3
        
        print(f"Audio Features:")
        for key, value in features.items():
            print(f"  {key}: {value:.3f}")
        
        print(f"\nCalculated Arousal: {arousal:.3f}")
        print(f"Calculated Valence: {valence:.3f}")
        
        # Predict with LSTM
        if lstm_model:
            try:
                # Create sequence
                sequence_length = 10
                sequence = np.tile(feature_array, (sequence_length, 1)).reshape(1, sequence_length, -1)
                
                # Add arousal and enhanced valence
                extended_sequence = np.zeros((1, sequence_length, 11))
                extended_sequence[0, :, :9] = sequence[0]
                extended_sequence[0, :, 9] = arousal
                extended_sequence[0, :, 10] = valence
                
                emotion_label, confidence, probabilities = lstm_model.predict_emotion(extended_sequence)
                
                print(f"\nLSTM Prediction:")
                print(f"  Emotion: {EMOTION_MAPPING[emotion_label]}")
                print(f"  Confidence: {confidence:.2%}")
                print(f"  Description: {EMOTION_DESCRIPTIONS[emotion_label]}")
                
            except Exception as e:
                print(f"LSTM prediction error: {e}")
        
        # Predict with baseline models
        print(f"\nBaseline Model Predictions:")
        for model_name in ['random_forest', 'logistic_regression', 'svm', 'mlp']:
            if model_name in baseline_models.models:
                try:
                    emotion, conf, _ = baseline_models.predict_emotion_baseline(feature_array, model_name)
                    if emotion is not None:
                        print(f"  {model_name.replace('_', ' ').title()}: {EMOTION_MAPPING[emotion]} ({conf:.2%})")
                except Exception as e:
                    print(f"  {model_name}: Error - {e}")
        
        # Music recommendations
        if lstm_model:
            print(f"\n🎶 Music Recommendations:")
            recommendations = MUSIC_RECOMMENDATIONS[emotion_label]
            for i, rec in enumerate(recommendations[:5], 1):
                print(f"  {i}. {rec}")
        
        print("\n" + "="*60)

def run_streamlit_demo():
    """Instructions for running the Streamlit demo"""
    print("\n" + "="*60)
    print("STREAMLIT DEMO INSTRUCTIONS")
    print("="*60)
    print("\nTo run the interactive Streamlit demo:")
    print("1. Make sure models are trained: python train_model.py")
    print("2. Run the Streamlit app: streamlit run src/frontend/app.py")
    print("3. Open your browser to http://localhost:8501")
    print("\nFeatures available in the Streamlit app:")
    print("• Manual emotion selection with sliders")
    print("• Audio features input for AI prediction")
    print("• Real-time emotion analysis")
    print("• Music recommendations based on detected emotions")
    print("• Model comparison and confidence scores")
    print("• Interactive visualizations")

if __name__ == "__main__":
    # Run the demo
    predict_emotion_demo()
    
    # Show Streamlit instructions
    run_streamlit_demo()
    
    print("\n" + "="*60)
    print("DEMO COMPLETED")
    print("="*60)
    print("\nFor more information, see the README.md file")
    print("To contribute, please check the contributing guidelines")
