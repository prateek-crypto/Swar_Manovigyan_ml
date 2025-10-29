"""
Test script for the Emotion-Aware Music Recommendation System
Tests basic functionality without requiring trained models
"""

import numpy as np
import pandas as pd
import sys
import os

# Add src to path
sys.path.append('src')

def test_data_loading():
    """Test data loading functionality"""
    print("Testing data loading...")
    
    try:
        from src.utils.data_analysis import load_and_analyze_data, EmotionLabeler
        
        # Test with sample data if main dataset not available
        if os.path.exists('data/raw/SpotifyFeatures.csv'):
            df = load_and_analyze_data('data/raw/SpotifyFeatures.csv')
            print(f"[OK] Loaded {len(df)} samples from SpotifyFeatures.csv")
        else:
            print("⚠ SpotifyFeatures.csv not found, creating sample data...")
            # Create sample data
            np.random.seed(42)
            n_samples = 1000
            
            sample_data = {
                'genre': np.random.choice(['Pop', 'Rock', 'Jazz', 'Classical'], n_samples),
                'artist_name': [f'Artist_{i}' for i in range(n_samples)],
                'track_name': [f'Track_{i}' for i in range(n_samples)],
                'track_id': [f'id_{i}' for i in range(n_samples)],
                'popularity': np.random.randint(0, 100, n_samples),
                'acousticness': np.random.random(n_samples),
                'danceability': np.random.random(n_samples),
                'duration_ms': np.random.randint(60000, 300000, n_samples),
                'energy': np.random.random(n_samples),
                'instrumentalness': np.random.random(n_samples),
                'key': np.random.choice(['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'], n_samples),
                'liveness': np.random.random(n_samples),
                'loudness': np.random.uniform(-60, 0, n_samples),
                'mode': np.random.choice(['Major', 'Minor'], n_samples),
                'speechiness': np.random.random(n_samples),
                'tempo': np.random.uniform(50, 200, n_samples),
                'time_signature': np.random.choice(['3/4', '4/4', '5/4'], n_samples),
                'valence': np.random.random(n_samples)
            }
            
            df = pd.DataFrame(sample_data)
            print(f"[OK] Created sample dataset with {len(df)} samples")
        
        # Test emotion labeling
        labeler = EmotionLabeler()
        df = labeler.create_emotion_labels(df)
        print("[OK] Created emotion labels")
        
        # Test emotion distribution
        emotion_counts = labeler.analyze_emotion_distribution(df)
        print("[OK] Analyzed emotion distribution")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Data loading test failed: {e}")
        return False

def test_model_creation():
    """Test model creation without training"""
    print("\nTesting model creation...")
    
    try:
        from src.models.lstm_model import EmotionLSTM, EmotionCNN1D
        from src.models.baseline_models import BaselineModels
        
        # Test LSTM model creation
        lstm_model = EmotionLSTM(input_shape=(10, 11))
        lstm_model.build_model()
        print("[OK] LSTM model created successfully")
        
        # Test CNN model creation
        cnn_model = EmotionCNN1D(input_shape=(10, 11))
        cnn_model.build_model()
        print("[OK] CNN model created successfully")
        
        # Test baseline models
        baseline_models = BaselineModels()
        print("[OK] Baseline models initialized successfully")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Model creation test failed: {e}")
        return False

def test_preprocessing():
    """Test data preprocessing functionality"""
    print("\nTesting preprocessing...")
    
    try:
        from src.utils.data_analysis import DataPreprocessor
        
        # Create sample data
        np.random.seed(42)
        n_samples = 100
        
        sample_data = {
            'acousticness': np.random.random(n_samples),
            'danceability': np.random.random(n_samples),
            'energy': np.random.random(n_samples),
            'instrumentalness': np.random.random(n_samples),
            'liveness': np.random.random(n_samples),
            'loudness': np.random.uniform(-60, 0, n_samples),
            'speechiness': np.random.random(n_samples),
            'tempo': np.random.uniform(50, 200, n_samples),
            'valence': np.random.random(n_samples),
            'emotion_label': np.random.randint(0, 4, n_samples)
        }
        
        df = pd.DataFrame(sample_data)
        
        # Test preprocessing
        preprocessor = DataPreprocessor()
        # First create emotion labels
        from src.utils.data_analysis import EmotionLabeler
        labeler = EmotionLabeler()
        df = labeler.create_emotion_labels(df)
        X_sequences, y_sequences = preprocessor.prepare_features(df, sequence_length=10)
        
        print(f"[OK] Created {len(X_sequences)} sequences of shape {X_sequences.shape}")
        
        # Test balanced dataset creation
        X_balanced, y_balanced = preprocessor.create_balanced_dataset(X_sequences, y_sequences, max_samples_per_class=20)
        print(f"[OK] Created balanced dataset with {len(X_balanced)} samples")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Preprocessing test failed: {e}")
        return False

def test_streamlit_imports():
    """Test Streamlit app imports"""
    print("\nTesting Streamlit app imports...")
    
    try:
        import streamlit as st
        import plotly.express as px
        import plotly.graph_objects as go
        print("[OK] Streamlit and Plotly imports successful")
        
        # Test app class import
        from src.frontend.app import MusicRecommendationApp
        app = MusicRecommendationApp()
        print("[OK] MusicRecommendationApp class imported successfully")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Streamlit imports test failed: {e}")
        return False

def test_config():
    """Test configuration loading"""
    print("\nTesting configuration...")
    
    try:
        import config
        
        # Test key configurations
        assert hasattr(config, 'EMOTION_MAPPING')
        assert hasattr(config, 'AUDIO_FEATURES')
        assert hasattr(config, 'MUSIC_RECOMMENDATIONS')
        
        print("[OK] Configuration loaded successfully")
        print(f"  - {len(config.EMOTION_MAPPING)} emotion categories")
        print(f"  - {len(config.AUDIO_FEATURES)} audio features")
        print(f"  - {len(config.MUSIC_RECOMMENDATIONS)} recommendation categories")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Configuration test failed: {e}")
        return False

def run_all_tests():
    """Run all tests"""
    print("="*60)
    print("EMOTION-AWARE MUSIC RECOMMENDATION SYSTEM")
    print("SYSTEM TESTS")
    print("="*60)
    
    tests = [
        ("Data Loading", test_data_loading),
        ("Model Creation", test_model_creation),
        ("Preprocessing", test_preprocessing),
        ("Streamlit Imports", test_streamlit_imports),
        ("Configuration", test_config)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"[FAIL] {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST RESULTS SUMMARY")
    print("="*60)
    
    passed = 0
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name:20} : {status}")
        if result:
            passed += 1
    
    print(f"\nPassed: {passed}/{len(results)} tests")
    
    if passed == len(results):
        print("\n🎉 All tests passed! The system is ready to use.")
        print("\nNext steps:")
        print("1. Run 'python train_model.py' to train the models")
        print("2. Run 'streamlit run src/frontend/app.py' to start the app")
        print("3. Run 'python demo.py' to see a demonstration")
    else:
        print(f"\n[WARNING] {len(results) - passed} tests failed. Please check the errors above.")
    
    return passed == len(results)

if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
