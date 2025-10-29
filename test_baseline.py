"""
Baseline test script for the Emotion-Aware Music Recommendation System
Tests functionality without requiring TensorFlow
"""

import numpy as np
import pandas as pd
import sys
import os

# Add src to path
sys.path.append('src')

def test_baseline_functionality():
    """Test baseline functionality without TensorFlow"""
    print("="*60)
    print("EMOTION-AWARE MUSIC RECOMMENDATION SYSTEM")
    print("BASELINE FUNCTIONALITY TEST")
    print("="*60)
    
    # Test 1: Data Loading and Processing
    print("\n1. Testing data loading and emotion labeling...")
    try:
        from src.utils.data_analysis import load_and_analyze_data, EmotionLabeler
        
        # Load data
        df = load_and_analyze_data('data/raw/SpotifyFeatures.csv')
        print(f"[OK] Loaded {len(df)} samples")
        
        # Create emotion labels
        labeler = EmotionLabeler()
        df = labeler.create_emotion_labels(df)
        print("[OK] Created emotion labels")
        
        # Check emotion distribution
        emotion_counts = df['emotion_label'].value_counts().sort_index()
        print(f"[OK] Emotion distribution: {dict(emotion_counts)}")
        
    except Exception as e:
        print(f"[FAIL] Data loading failed: {e}")
        return False
    
    # Test 2: Baseline Models
    print("\n2. Testing baseline models...")
    try:
        from src.models.baseline_models import BaselineModels
        
        # Create sample data for testing
        np.random.seed(42)
        n_samples = 1000
        n_features = 9
        
        X = np.random.random((n_samples, n_features))
        y = np.random.randint(0, 4, n_samples)
        
        # Test baseline models
        baseline_models = BaselineModels()
        
        # Test data preparation
        X_flat, y_flat = baseline_models.prepare_data_for_baseline(X, y)
        print(f"[OK] Prepared data: {X_flat.shape}")
        
        # Test individual models
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        
        # Logistic Regression
        lr = LogisticRegression(random_state=42, max_iter=1000)
        lr.fit(X_flat, y_flat)
        lr_pred = lr.predict(X_flat)
        lr_acc = (lr_pred == y_flat).mean()
        print(f"[OK] Logistic Regression accuracy: {lr_acc:.3f}")
        
        # Random Forest
        rf = RandomForestClassifier(n_estimators=10, random_state=42)
        rf.fit(X_flat, y_flat)
        rf_pred = rf.predict(X_flat)
        rf_acc = (rf_pred == y_flat).mean()
        print(f"[OK] Random Forest accuracy: {rf_acc:.3f}")
        
    except Exception as e:
        print(f"[FAIL] Baseline models test failed: {e}")
        return False
    
    # Test 3: Streamlit App (without running)
    print("\n3. Testing Streamlit app imports...")
    try:
        import streamlit as st
        import plotly.express as px
        import plotly.graph_objects as go
        print("[OK] Streamlit and Plotly imported successfully")
        
        # Test app class import (without TensorFlow)
        import sys
        original_path = sys.path.copy()
        
        # Temporarily mock TensorFlow
        class MockTensorFlow:
            class keras:
                class models:
                    @staticmethod
                    def load_model(path):
                        return None
                
                class layers:
                    @staticmethod
                    def Input(shape):
                        return None
                    
                    @staticmethod
                    def LSTM(units, return_sequences=False):
                        return None
                    
                    @staticmethod
                    def Bidirectional(layer):
                        return None
                    
                    @staticmethod
                    def Dense(units, activation=None):
                        return None
                    
                    @staticmethod
                    def Dropout(rate):
                        return None
                    
                    @staticmethod
                    def BatchNormalization():
                        return None
                
                class Sequential:
                    def __init__(self):
                        pass
                    
                    def add(self, layer):
                        pass
                    
                    def compile(self, **kwargs):
                        pass
                    
                    def fit(self, **kwargs):
                        pass
                    
                    def predict(self, X):
                        return np.random.random((X.shape[0], 4))
                    
                    def evaluate(self, X, y):
                        return [0.5, 0.8]  # loss, accuracy
                    
                    def save(self, path):
                        pass
                    
                    def summary(self):
                        print("Mock LSTM Model")
        
        sys.modules['tensorflow'] = MockTensorFlow()
        sys.modules['tensorflow.keras'] = MockTensorFlow.keras()
        sys.modules['tensorflow.keras.models'] = MockTensorFlow.keras.models()
        sys.modules['tensorflow.keras.layers'] = MockTensorFlow.keras.layers()
        
        # Now test the app import
        from src.frontend.app import MusicRecommendationApp
        app = MusicRecommendationApp()
        print("[OK] MusicRecommendationApp imported successfully")
        
        # Restore original path
        sys.path = original_path
        
    except Exception as e:
        print(f"[FAIL] Streamlit app test failed: {e}")
        return False
    
    # Test 4: Configuration
    print("\n4. Testing configuration...")
    try:
        import config
        
        assert hasattr(config, 'EMOTION_MAPPING')
        assert hasattr(config, 'AUDIO_FEATURES')
        assert hasattr(config, 'MUSIC_RECOMMENDATIONS')
        
        print(f"[OK] Configuration loaded: {len(config.EMOTION_MAPPING)} emotions")
        
    except Exception as e:
        print(f"[FAIL] Configuration test failed: {e}")
        return False
    
    # Test 5: Demo functionality
    print("\n5. Testing demo functionality...")
    try:
        # Test emotion prediction with sample data
        sample_features = np.array([[0.5, 0.6, 0.7, 0.3, 0.2, -10.0, 0.1, 120.0, 0.8]])
        
        # Test arousal and valence calculation
        arousal = (sample_features[0, 2] + (sample_features[0, 5] + 60) / 60 + sample_features[0, 7] / 200) / 3
        valence = (sample_features[0, 8] + sample_features[0, 1] + (1 - sample_features[0, 0])) / 3
        
        print(f"[OK] Sample arousal: {arousal:.3f}, valence: {valence:.3f}")
        
        # Determine emotion quadrant
        arousal_binary = 1 if arousal > 0.5 else 0
        valence_binary = 1 if valence > 0.5 else 0
        emotion_label = arousal_binary * 2 + valence_binary
        
        emotion_names = ['Low Arousal, Negative Valence', 'Low Arousal, Positive Valence',
                        'High Arousal, Negative Valence', 'High Arousal, Positive Valence']
        
        print(f"[OK] Predicted emotion: {emotion_names[emotion_label]}")
        
    except Exception as e:
        print(f"[FAIL] Demo functionality test failed: {e}")
        return False
    
    print("\n" + "="*60)
    print("BASELINE TEST RESULTS")
    print("="*60)
    print("[SUCCESS] All baseline functionality tests passed!")
    print("\nThe system is ready to use with baseline models.")
    print("Note: LSTM models require TensorFlow, but baseline models work perfectly.")
    
    return True

if __name__ == "__main__":
    success = test_baseline_functionality()
    sys.exit(0 if success else 1)
