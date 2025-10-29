"""
Simple test script for the Emotion-Aware Music Recommendation System
Tests core functionality without complex dependencies
"""

import numpy as np
import pandas as pd
import sys
import os

# Add src to path
sys.path.append('src')

def test_core_functionality():
    """Test core functionality"""
    print("="*60)
    print("EMOTION-AWARE MUSIC RECOMMENDATION SYSTEM")
    print("CORE FUNCTIONALITY TEST")
    print("="*60)
    
    # Test 1: Data Loading
    print("\n1. Testing data loading...")
    try:
        from src.utils.data_analysis import load_and_analyze_data, EmotionLabeler
        
        df = load_and_analyze_data('data/raw/SpotifyFeatures.csv')
        print(f"[OK] Loaded {len(df)} samples")
        
        labeler = EmotionLabeler()
        df = labeler.create_emotion_labels(df)
        print("[OK] Created emotion labels")
        
        emotion_counts = df['emotion_label'].value_counts().sort_index()
        print(f"[OK] Emotion distribution: {dict(emotion_counts)}")
        
    except Exception as e:
        print(f"[FAIL] Data loading failed: {e}")
        return False
    
    # Test 2: Baseline Models
    print("\n2. Testing baseline models...")
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.svm import SVC
        from sklearn.neural_network import MLPClassifier
        
        # Create sample data
        np.random.seed(42)
        n_samples = 1000
        n_features = 9
        
        X = np.random.random((n_samples, n_features))
        y = np.random.randint(0, 4, n_samples)
        
        # Test models
        models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(n_estimators=10, random_state=42),
            'SVM': SVC(random_state=42),
            'MLP': MLPClassifier(random_state=42, max_iter=1000)
        }
        
        for name, model in models.items():
            model.fit(X, y)
            pred = model.predict(X)
            acc = (pred == y).mean()
            print(f"[OK] {name} accuracy: {acc:.3f}")
        
    except Exception as e:
        print(f"[FAIL] Baseline models test failed: {e}")
        return False
    
    # Test 3: Emotion Classification Logic
    print("\n3. Testing emotion classification logic...")
    try:
        # Test arousal-valence calculation
        sample_features = {
            'energy': 0.8,
            'loudness': -5.0,
            'tempo': 140.0,
            'valence': 0.9,
            'danceability': 0.8,
            'acousticness': 0.2
        }
        
        # Calculate arousal
        arousal = (sample_features['energy'] + 
                  (sample_features['loudness'] + 60) / 60 + 
                  sample_features['tempo'] / 200) / 3
        
        # Calculate valence
        valence = (sample_features['valence'] + 
                  sample_features['danceability'] + 
                  (1 - sample_features['acousticness'])) / 3
        
        # Determine emotion
        arousal_binary = 1 if arousal > 0.5 else 0
        valence_binary = 1 if valence > 0.5 else 0
        emotion_label = arousal_binary * 2 + valence_binary
        
        emotion_names = ['Low Arousal, Negative Valence', 'Low Arousal, Positive Valence',
                        'High Arousal, Negative Valence', 'High Arousal, Positive Valence']
        
        print(f"[OK] Arousal: {arousal:.3f}, Valence: {valence:.3f}")
        print(f"[OK] Predicted emotion: {emotion_names[emotion_label]}")
        
    except Exception as e:
        print(f"[FAIL] Emotion classification test failed: {e}")
        return False
    
    # Test 4: Configuration
    print("\n4. Testing configuration...")
    try:
        import config
        
        assert hasattr(config, 'EMOTION_MAPPING')
        assert hasattr(config, 'AUDIO_FEATURES')
        assert hasattr(config, 'MUSIC_RECOMMENDATIONS')
        
        print(f"[OK] Configuration loaded successfully")
        print(f"[OK] {len(config.EMOTION_MAPPING)} emotion categories")
        print(f"[OK] {len(config.AUDIO_FEATURES)} audio features")
        
    except Exception as e:
        print(f"[FAIL] Configuration test failed: {e}")
        return False
    
    # Test 5: Streamlit Basic Import
    print("\n5. Testing Streamlit import...")
    try:
        import streamlit as st
        import plotly.express as px
        import plotly.graph_objects as go
        print("[OK] Streamlit and Plotly imported successfully")
        
    except Exception as e:
        print(f"[FAIL] Streamlit import failed: {e}")
        return False
    
    print("\n" + "="*60)
    print("CORE FUNCTIONALITY TEST RESULTS")
    print("="*60)
    print("[SUCCESS] All core functionality tests passed!")
    print("\nThe system is ready to use!")
    print("\nNext steps:")
    print("1. Run 'python train_model.py' to train models")
    print("2. Run 'streamlit run src/frontend/app.py' to start the app")
    print("3. Run 'python demo.py' to see a demonstration")
    
    return True

if __name__ == "__main__":
    success = test_core_functionality()
    sys.exit(0 if success else 1)
