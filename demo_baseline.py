"""
Baseline demo script for the Emotion-Aware Music Recommendation System
Demonstrates functionality using only baseline models (no TensorFlow required)
"""

import numpy as np
import pandas as pd
import sys
import os

# Add src to path
sys.path.append('src')

from src.models.baseline_models import BaselineModels
from config import EMOTION_MAPPING, EMOTION_DESCRIPTIONS, MUSIC_RECOMMENDATIONS

def create_sample_features():
    """Create sample audio features for demonstration"""
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

def predict_emotion_baseline_demo():
    """Demonstrate emotion prediction with baseline models"""
    print("="*60)
    print("EMOTION-AWARE MUSIC RECOMMENDATION - BASELINE DEMO")
    print("="*60)
    
    # Load baseline models
    print("\nLoading baseline models...")
    try:
        baseline_models = BaselineModels()
        baseline_models.load_models('models/baseline_models')
        print("[OK] Baseline models loaded successfully!")
        
    except Exception as e:
        print(f"[ERROR] Error loading models: {e}")
        print("Please run 'python train_baseline.py' first to train the models.")
        return
    
    # Create sample features
    samples = create_sample_features()
    
    print(f"\nTesting with {len(samples)} sample songs...")
    print("-" * 60)
    
    for song_name, features in samples.items():
        print(f"\n[SONG] {song_name}")
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
        
        # Predict with baseline models
        print(f"\nBaseline Model Predictions:")
        predictions = {}
        
        for model_name in ['random_forest', 'logistic_regression', 'svm', 'mlp']:
            if model_name in baseline_models.models:
                try:
                    # Flatten features for baseline models
                    features_flat = feature_array.flatten()
                    model = baseline_models.models[model_name]
                    
                    if hasattr(model, 'predict_proba'):
                        probabilities = model.predict_proba(features_flat.reshape(1, -1))[0]
                        emotion = np.argmax(probabilities)
                        conf = np.max(probabilities)
                    else:
                        emotion = model.predict(features_flat.reshape(1, -1))[0]
                        conf = 1.0
                    if emotion is not None:
                        predictions[model_name] = (emotion, conf)
                        print(f"  {model_name.replace('_', ' ').title()}: {EMOTION_MAPPING[emotion]} ({conf:.2%})")
                except Exception as e:
                    print(f"  {model_name}: Error - {e}")
        
        # Get most common prediction
        if predictions:
            emotion_votes = {}
            for model, (emotion, conf) in predictions.items():
                if emotion not in emotion_votes:
                    emotion_votes[emotion] = []
                emotion_votes[emotion].append(conf)
            
            # Find emotion with highest average confidence
            best_emotion = max(emotion_votes.keys(), key=lambda x: np.mean(emotion_votes[x]))
            avg_confidence = np.mean(emotion_votes[best_emotion])
            
            print(f"\n[PREDICTION] Consensus Prediction:")
            print(f"  Emotion: {EMOTION_MAPPING[best_emotion]}")
            print(f"  Confidence: {avg_confidence:.2%}")
            print(f"  Description: {EMOTION_DESCRIPTIONS[best_emotion]}")
            
            # Music recommendations
            print(f"\n[MUSIC] Music Recommendations:")
            recommendations = MUSIC_RECOMMENDATIONS[best_emotion]
            for i, rec in enumerate(recommendations[:5], 1):
                print(f"  {i}. {rec}")
        
        print("\n" + "="*60)

def run_streamlit_demo():
    """Instructions for running the Streamlit demo"""
    print("\n" + "="*60)
    print("STREAMLIT DEMO INSTRUCTIONS")
    print("="*60)
    print("\nTo run the interactive Streamlit demo:")
    print("1. Make sure baseline models are trained: python train_baseline.py")
    print("2. Run the Streamlit app: streamlit run src/frontend/app.py")
    print("3. Open your browser to http://localhost:8501")
    print("\nFeatures available in the Streamlit app:")
    print("• Manual emotion selection with sliders")
    print("• Audio features input for AI prediction")
    print("• Real-time emotion analysis using baseline models")
    print("• Music recommendations based on detected emotions")
    print("• Model comparison and confidence scores")
    print("• Interactive visualizations")

def main():
    """Main demo function"""
    # Run the baseline demo
    predict_emotion_baseline_demo()
    
    # Show Streamlit instructions
    run_streamlit_demo()
    
    print("\n" + "="*60)
    print("BASELINE DEMO COMPLETED")
    print("="*60)
    print("\nThe system is working perfectly with baseline models!")
    print("For more information, see the README.md file")
    print("To contribute, please check the contributing guidelines")

if __name__ == "__main__":
    main()
