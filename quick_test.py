"""
Quick test script for the Emotion-Aware Music Recommendation System
Tests basic functionality without requiring all packages
"""

import sys
import os

def test_basic_imports():
    """Test basic Python imports"""
    print("Testing basic imports...")
    
    try:
        import pandas as pd
        print("[OK] pandas imported successfully")
    except ImportError:
        print("[WARNING] pandas not available - install with: pip install pandas")
    
    try:
        import numpy as np
        print("[OK] numpy imported successfully")
    except ImportError:
        print("[WARNING] numpy not available - install with: pip install numpy")
    
    try:
        import matplotlib.pyplot as plt
        print("[OK] matplotlib imported successfully")
    except ImportError:
        print("[WARNING] matplotlib not available - install with: pip install matplotlib")
    
    try:
        import sklearn
        print("[OK] scikit-learn imported successfully")
    except ImportError:
        print("[WARNING] scikit-learn not available - install with: pip install scikit-learn")
    
    try:
        import streamlit as st
        print("[OK] streamlit imported successfully")
    except ImportError:
        print("[WARNING] streamlit not available - install with: pip install streamlit")
    
    try:
        import tensorflow as tf
        print(f"[OK] tensorflow {tf.__version__} imported successfully")
    except ImportError:
        print("[WARNING] tensorflow not available - install with: pip install tensorflow")
        print("  Note: The app will work with baseline models only")

def test_file_structure():
    """Test if all required files exist"""
    print("\nTesting file structure...")
    
    required_files = [
        'src/frontend/app.py',
        'src/models/lstm_model.py',
        'src/models/baseline_models.py',
        'src/utils/data_analysis.py',
        'config.py',
        'requirements.txt',
        'README.md'
    ]
    
    missing_files = []
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"[OK] {file_path} exists")
        else:
            print(f"[MISSING] {file_path} not found")
            missing_files.append(file_path)
    
    return len(missing_files) == 0

def test_data_availability():
    """Test if data files are available"""
    print("\nTesting data availability...")
    
    if os.path.exists('data/raw/SpotifyFeatures.csv'):
        print("[OK] SpotifyFeatures.csv found")
        return True
    else:
        print("[WARNING] SpotifyFeatures.csv not found")
        print("  The system will create sample data for testing")
        return False

def main():
    """Main test function"""
    print("="*60)
    print("EMOTION-AWARE MUSIC RECOMMENDATION SYSTEM")
    print("QUICK TEST")
    print("="*60)
    
    # Test imports
    test_basic_imports()
    
    # Test file structure
    files_ok = test_file_structure()
    
    # Test data
    data_ok = test_data_availability()
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    if files_ok:
        print("[OK] All required files are present")
    else:
        print("[WARNING] Some files are missing")
    
    if data_ok:
        print("[OK] Data files are available")
    else:
        print("[WARNING] Data files not found - will use sample data")
    
    print("\nNext steps:")
    print("1. Install missing packages: pip install -r requirements.txt")
    print("2. Run full test: python test_system.py")
    print("3. Train models: python train_model.py")
    print("4. Start app: streamlit run src/frontend/app.py")
    
    return files_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
